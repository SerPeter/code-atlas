# ADR-0042: Reindex scope and destruction are separate decisions

## Status

Accepted — 2026-08-31.

All five decisions are implemented (ATL-148, ATL-149, ATL-150, ATL-151, ATL-152).

Two things the implementation had to add that the reasoning below does not anticipate:

- A non-destructive `--full` stops reconciling files **deleted from disk**. `_publish_events`' full branch emits one
  `created` per file that exists, so `delete_project_data` was the only thing on that path removing entities for a file
  since deleted. `_reconcile_full_deletions` restores that, and closes the same hole in the ratio-triggered escalation
  to full, which never deleted either.
- `--reset-embeddings` needs the unembedded sweep to **loop**. `find_unembedded_entities` is capped at 5,000 per
  project, which is invisible while the only caller heals a handful of lost events and fatal for a flag that empties the
  whole project: one pass would restore 5,000 of 35,104 and exit 0.

## Context

`atlas index --full` welds together three decisions that have nothing to do with each other:

| axis               | options                                      | reachable today      |
| ------------------ | -------------------------------------------- | -------------------- |
| **A. Enumeration** | delta (git diff vs stored `git_hash`) / all  | `--full` ⇒ all       |
| **B. Gate trust**  | trust `file_hash` / re-parse regardless      | **not controllable** |
| **C. Destruction** | nothing / embeddings only / all project data | `--full` ⇒ all       |

`--full` sets A and C, and B is only reachable as a _side effect_ of C. The flag calls `delete_project_data` — a blanket
`MATCH (n {project_name: $p}) DETACH DELETE n` (`graph/client.py:2219`) — and the file-hash gate then reads `None` for
every file because the nodes are gone. So "re-check everything I have" exists only bundled with "destroy everything I
have and pay to rebuild it."

That bundling is expensive in exactly the situation where a re-check is most wanted: after a change to the _parser or
its configuration_, where every file's bytes are unchanged but the extracted result is now wrong. On the production
graph that is 35,104 embeddings, re-billed through a paid provider, to fix an extraction defect that cost nothing to
re-derive. This was nearly done by hand while investigating
[ADR-0040](./0040-oversized-nodes-are-split-two-different-ways.md)'s chunking path.

The bundling is not needed, because cost already ladders down four layers and money only enters at the bottom:

1. `file_hash` on the `FILE_HASH_LABELS` nodes — did the file's bytes change?
2. `content_hash` per entity, computed pre-truncation — drives the added/modified/unchanged diff.
3. `embed_hash` — hash of the embed text; decides whether the provider is called at all.
4. Dedup (ADR-0036) — `MATCH (n:Entity {embed_hash: h}) WHERE n.embed_model = $model`, so even a changed hash may find a
   vector that already exists.

Skipping layer 1 leaves 2–4 intact. **A re-check where nothing genuinely changed costs zero provider tokens.** Nothing
has to be built for that to be true; it already is.

The reason layer 1 needs skipping at all is that it hashes file _bytes_, while the extracted result also depends on the
parser version and on configuration such as `index.max_source_chars`. The gate's key is narrower than the thing it is
gating. Twelve schema migrations already work around this by calling `generate_clear_file_hashes_ddl` to force a
re-parse, which is the same admission written twelve times.

## Decision

### 1. Three flags, one axis each

- **`--full`** — A=all, B=distrust, C=**nothing**. Enumerate every file, re-read and re-parse each one, and let
  `content_hash` / `embed_hash` decide what is written and what is billed. This is what "full index" is universally read
  to mean.
- **`--reset`** — C=all project data. Today's `--full`, under a name that says so.
- **`--reset-embeddings`** — C=embeddings only. For a model or dimension switch.

Repurposing `--full` is a breaking change to a published flag and is chosen deliberately on an asymmetry: the new
failure mode is "did not destroy when you meant to", which is recoverable by running `--reset`; the old one is
"destroyed 35,104 embeddings and re-billed them", which is not.

### 2. A destructive operation states what it will destroy, before it starts

Any operation on axis C must, **before touching anything**:

- Count what it is about to remove — nodes, relationships, embedded nodes, `EmbedChunk` nodes — and print those counts
  per project.
- Name the exact scope, including which sub-projects of a monorepo are included, because a project prefix match is not
  obvious from the flag.
- State the recovery cost in the terms the user actually pays: how many vectors must be re-embedded and therefore
  re-billed.
- Require explicit confirmation. Non-interactive runs refuse unless `--yes` is passed; there is no prompt to
  default-accept and no timeout that proceeds.

A destructive run that cannot describe its own blast radius must abort rather than proceed on an estimate. This is a
hard requirement, not a UX preference: the failure it prevents is unrecoverable and metered.

### 3. `--reset-embeddings` is scoped, and ordered

`clear_embeddings` already exists (`graph/client.py:4430`) and already strips `embedding`, `embed_hash` and
`embed_model` and deletes `EmbedChunk` nodes. Two properties of it are load-bearing and must be preserved by the flag
that drives it:

- **Scope.** Per project for a _model_ change; database-wide only for a _dimension_ change, where the shared vector
  indices are rebuilt anyway. Clearing every project for one project's model change destroyed other projects' embeddings
  silently once already (ATL-135).
- **Order.** Check the model lock → clear → `ensure_schema`. Today `ensure_schema` runs at `cli.py:633` and rebuilds
  vector indices at the _configured_ dimension, while `_check_model_lock` (`indexing/orchestrator.py:1111`) is not
  reached until `orchestrator.py:2048` — so the indices are rebuilt at the wrong dimension and _then_ the guard aborts
  with a helpful message that arrives too late to matter. A `--reset-embeddings` built on the current order would drive
  into that landmine on every use.

### 4. A no-op re-check should cost only the parse

`_recreate_file_relationships` runs on every upsert (`graph/client.py:1920`), unconditionally. So a re-check that finds
nothing changed still deletes and recreates every edge for every file, and buffers every relationship into the
resolution flush — a full `build_resolution_lookup` plus `resolve_calls` / `resolve_imports` / `resolve_type_refs` pass
over the whole project.

A per-file `rels_hash`, stored beside `file_hash`, closes this: when every entity returns `unchanged` **and** the
relationship fingerprint matches, skip the rewrite. The parse is then the only irreducible cost.

**Amended on implementation (ATL-151), in three places.**

**Skip the write, keep the buffer.** The paragraph above also said "skip buffering", and that is wrong. The buffer is
what [ADR-0026](./0026-resolution-is-replayed-not-batch-final.md) added to fix a measured loss — resolution reads the
graph as it stands at its flush, so a callee upserted by a _later_ batch was never linked, and adding the replay took
CALLS from 9,058 to 9,713 and `find_dead_code` on `src/` from 27 to 15. Skipping the buffer for an unchanged file
reintroduces exactly that, and worse: `--full` is the run that repairs it, so the repair would be the thing broken.
Confirmed by sabotage rather than by argument — implementing the buffer skip makes
`test_a_skipped_file_still_buffers_its_rels_for_a_callee_that_lands_later` fail. `rels_hash` fingerprints the _parse_,
while resolution's output is a function of the parse **and the rest of the graph**, so it is a sound gate for the
per-file write and an unsound one for the buffer.

**The fingerprint is compared pre-detector and stored post-detector.** Detector output is derived from the graph, not
from the file's bytes, and the detectors cannot run until the entity transaction has written the entities they query. So
what is stored covers the merged set the second `rels_only` pass actually writes, while what is compared is the
parser-only set — a file that carried detector edges last run therefore yields a stored hash no pre-detector hash can
equal, and never skips. That asymmetry is the point: only running the detectors could tell you whether they still fire,
and the per-file rewrite is the only thing that revokes a detector edge that has stopped firing.

**A skipped rewrite no longer retracts a stale resolved edge.** A replay MERGEs; it does not retract (ADR-0026), so the
per-file delete was the only thing erasing an edge resolved against a partial graph in some earlier run. Measured on
this repo, three no-op `--full` runs over a graph seeded from empty: IMPORTS 2,322 → 2,367, CALLS 14,235 → 14,236,
DOCUMENTS 214 → 215, and every other type identical. Those 45 IMPORTS are stale edges the old `--full` retracted. This
is deliberate — `--reset` is the operation that converges the graph, and decision 1 already makes it the one you reach
for when you suspect the graph is wrong. What was _not_ acceptable and was fixed rather than documented:
`resolve_doc_links` used `CREATE` where every other resolver MERGEs, silently relying on the per-file delete for
idempotence, so the same three runs took DOCUMENTS from 213 to 432 and would have grown without bound.

Measured effect of the decision itself, no-op `--full` on this repo (450 files, 9,430 entities), before → after:

|                                           | before | after |
| ----------------------------------------- | -----: | ----: |
| files whose relationships were rewritten  |    253 |    60 |
| relationships written                     | 11,324 | 4,115 |
| relationship statements (delete + create) |     44 |    33 |
| wall clock                                |  26.5s | 25.4s |

The surviving 60 files are the ones carrying detector edges, which never skip by construction. The wall clock barely
moves, and the story never claimed it would: this is database churn, and the point is that a re-check stops being
something people avoid.

### 5. The gate key should cover what the gate is gating

`file_hash` becomes `hash(file_bytes + extraction_epoch)`, where the epoch is a deliberate constant bumped when
extraction output changes — not the package version, or a docs-only release forces a global re-parse.

This subsumes the twelve migrations that call `generate_clear_file_hashes_ddl`, and it removes the need to reach for
`--full` after an upgrade at all: the gate invalidates itself because its key changed.

Recorded here rather than deferred silently, because it is the reason `--full` is load-bearing today. It is sequenced
last: it redefines what the gate means, and the flag split delivers most of the value without it.

**Amended on implementation (ATL-152): the key has to be stored in _two_ places, not one.** The paragraph above is right
about `file_hash` and wrong about what that reaches. `file_hash` is gate _2_, and gate 2 is only ever asked about a file
the run already published. Gate 1 is enumeration: `_decide_delta_mode` returns three empty sets when git reports no
change, `_publish_events` then publishes nothing, and the pipeline never starts — so on a project sitting at its stored
HEAD an epoch bump alone reaches no file at all. The twelve migrations' `REMOVE p.git_hash` is therefore _not_ subsumed
and stays; only their `generate_clear_file_hashes_ddl` call goes.

So the resolved key is also stored on the `Project` node, and `_decide_delta_mode` enumerates everything when it differs
from the current one. That is what makes the sentence above true rather than aspirational, and an absent key reading as
"differs" is what carries the one-time re-check every graph indexed before the key existed is owed.

Two consequences worth naming:

- **The key must be stable across runs and processes**, or the failure is not a slow first run: the daemon always trusts
  the gate, so a key that moved would re-parse the whole project forever at watcher cadence and read as a performance
  regression rather than as a key that moved. Lists are sorted, the payload is rendered canonically, and the digest is
  `hashlib` rather than the per-process-salted `hash()`. The computed key is logged at DEBUG on consumer construction,
  because two processes disagreeing about it (an `ATLAS_*` in a `.env` found from a different cwd outranks the target
  project's `atlas.toml`) has no other symptom.
- **This is deliberately not a `SCHEMA_VERSION` bump.** A version whose only real work is `REMOVE p.git_hash` would make
  every existing graph pay `ensure_schema`'s migration branch, which drops every vector and text index unconditionally
  and recreates the vector ones only when embeddings are enabled — the failure this codebase has already had once, and
  an outright `EmbeddingsPresentError` on a lightweight install. Keying gate 1 on the Project node buys the same
  one-time re-check with no index churn, and buys it again for every future epoch bump.

What it does **not** do is repair the ADR-0040 truncation that motivated it. `content_hash` is computed before
truncation, so a cap change re-reads and re-parses every file and then classifies every entity `unchanged`; the stored
`source` stays cut at the old cap. Layer 1 is opened, layer 2 still declines. Repairing that means folding the cap into
`content_hash`, which re-embeds every truncated entity and is a separate decision.

## Consequences

`--full` becomes safe to run and correspondingly boring — CPU and database churn, no provider spend, no data loss. That
is the point: the operation people reach for when something looks wrong should not be the one that costs the most to be
wrong about.

Rejected alternatives:

- **Keep `--full` destructive, add `--recheck` for the safe path.** Non-breaking, and leaves the loaded gun on the table
  under the name everyone reaches for first. The whole defect is that the obvious flag does the expensive thing.
- **Make `--full` prompt instead of splitting it.** A prompt on a flag that means two different things is a prompt users
  learn to accept. Splitting the flag makes the destructive path something you have to _choose_, and confirmation then
  guards a request that was already explicit.
- **Drop embeddings on any model change automatically.** Unnecessary — the dedup lookup keys on
  `(embed_hash, embed_model)`, so a model change misses and re-embeds on its own. Clearing is about reclaiming space and
  about dimension changes, and doing it implicitly is how ATL-135 happened.

Two behaviours worth stating because nothing surfaces them:

- A `--full` run still rewrites `file_hash` for every file it parses, so the first one after an upgrade is also what
  populates hashes for labels that never had them (documents, before v0.10.2).
- Re-checking everything and finding nothing is a _successful_ result with no output to show for it. The run should
  report what it verified, not just what it changed, or it reads as a no-op that did not work.

## References

- ADR-0036 — the graph is the embedding dedup layer; why layer 4 exists
- ADR-0040 — oversized nodes; the extraction defect that motivated a non-destructive re-check
- ATL-135 — clearing every project's embeddings for one project's model change
