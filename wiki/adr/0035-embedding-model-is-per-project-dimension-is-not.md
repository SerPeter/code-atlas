# ADR-0035: Embedding model is per project, dimension is not

## Status

Accepted — 2026-08-28.

## Context

One Memgraph holds every project. That is the deployment this system is built for — it is what makes cross-project
search, monorepo sub-projects and worktree indexes work at all.

The embedding lock did not account for it. `get_embedding_config` / `set_embedding_config` wrote `embedding_model` and
`embedding_dimension` onto the single `SchemaVersion` node, with no project scoping, while each project supplied its own
`settings.embeddings.model` from its own `atlas.toml` or `.env`. So the last project to index owned the lock and every
other project was locked out of indexing entirely.

This was not hypothetical. Measured on the production instance on 2026-08-28, six projects deep:

| model                                    | vectors | projects                     |
| ---------------------------------------- | ------- | ---------------------------- |
| `openrouter/google/gemini-embedding-001` | 25,305  | trading-bot + 4 sub-projects |
| `openai/text-embedding-3-small`          | 6,691   | code-atlas                   |

The `SchemaVersion` lock named only the first. `atlas index` on code-atlas had been failing on every run with
`Embedding model changed from 'openrouter/google/gemini-embedding-001' to 'openai/text-embedding-3-small'`.

Three things made it worse than an availability bug:

1. **The offered remedy was destructive across projects.** The error said to run `atlas index --full`, which called
   `clear_all_embeddings()` — `MATCH (n) WHERE n.embedding IS NOT NULL ... REMOVE n.embedding, n.embed_hash`, with no
   project filter. Clearing code-atlas's lock would have silently stripped trading-bot's 25,305 vectors.
2. **Vector search was disabled for the non-owner.** Three separate readers (`health.check_embedding_model`, the MCP
   root switch, `atlas search`) compared the configured model against the _database default_ and turned vector search
   off on mismatch. Every project that was not the lock owner lost semantic search without being told why.
3. **Both spaces were 1536-dimensional.** No dimension check could tell them apart, and nothing recorded which model
   produced a given vector, so a mixed store was undetectable from the data.

## Decision

**Split the lock along the line the storage already draws.**

- **Dimension stays global**, on `SchemaVersion`. Vector indices are one per label for the whole database and carry a
  single dimension. A dimension change genuinely does rebuild indices that every project shares, so it is correct for it
  to clear database-wide — and it now names every project and vector count it is about to destroy, first.
- **Model becomes per project**, on the `Project` node (`get_project_embedding_model` / `set_project_embedding_model`).
  A model decides which space a vector lives in; nothing about that is shared. A model change clears only the changing
  project.
- **`clear_all_embeddings` is replaced by `clear_embeddings(project=None)`**, which returns the number of nodes it
  stripped. A scoped clear also covers `"{root}/{sub}"` sub-projects, because a monorepo's sub-projects share their
  root's model.
- **Every vector is stamped with `embed_model`** as it is written. Which space a vector belongs to must be a property of
  the vector, not an inference from global state that the measurements above show can be wrong.
- **The three vector-search gates compare the project's model**, not the database default.

Rejected: **making one model per database a true invariant** instead. It fixes the deadlock by forbidding the situation,
but it forces every project sharing a store onto one model, and the escape from a wrong choice is still a full re-embed
of everything. The storage does not actually require it — only the dimension does.

Rejected: **per-project vector indices.** Memgraph vector indices are declared per label, not per label and project.
There is nothing to scope.

## Consequences

**Two projects with different models coexist.** Both index, neither destroys the other, and each keeps its own vector
space. Verified on the production graph: after the fix, `atlas index` on code-atlas completed in 71.9s and trading-bot's
counts were unchanged to the vector (22,541 / 1,347 / 1,206 / 141 / 70).

**A project that has vectors but no recorded model adopts the configured one, loudly.** Vectors were written by runs
using that project's own configuration, so the configured model is the right thing to record. The one case this gets
wrong is a model changed while indexing was already failing, so it logs a WARNING naming the count and the remedy rather
than doing it silently.

**Mixed spaces are now knowable but not yet filtered.** `embed_model` is stamped going forward; vectors written before
this ADR carry no stamp. `vector_search` still queries the shared indices and post-filters by project, so a
cross-project query can still compare across spaces. Making retrieval filter on the stamp is deliberately left out of
this change — the stamp is the prerequisite, and it has to exist on a meaningful share of vectors before filtering on it
is anything but a way to hide results.

**A store can hold several models, so a model round-trip is not free.** Going A → B → A re-embeds twice; nothing retains
the A vectors while B is active.

## References

- ATL-135 — the bug; its Context carries the full measurement.
- `graph/client.py` — `get_project_embedding_model`, `set_project_embedding_model`, `get_embedding_models_by_project`,
  `clear_embeddings`, `count_embeddings_by_project`.
- `indexing/orchestrator.py:_check_model_lock` — the split, and the messages that now name the conflicting project.
- ADR-0015 — the embedded backend, which implements the same split over `meta` keys.
