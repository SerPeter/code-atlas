# ADR-0042: Reindex scope and destruction are separate decisions

## Status

Proposed — 2026-08-31.

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
relationship fingerprint matches, skip the rewrite and skip buffering. The parse is then the only irreducible cost.

### 5. The gate key should cover what the gate is gating

`file_hash` becomes `hash(file_bytes + extraction_epoch)`, where the epoch is a deliberate constant bumped when
extraction output changes — not the package version, or a docs-only release forces a global re-parse.

This subsumes the twelve migrations that call `generate_clear_file_hashes_ddl`, and it removes the need to reach for
`--full` after an upgrade at all: the gate invalidates itself because its key changed.

Recorded here rather than deferred silently, because it is the reason `--full` is load-bearing today. It is sequenced
last: it redefines what the gate means, and the flag split delivers most of the value without it.

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
