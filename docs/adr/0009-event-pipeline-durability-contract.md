# ADR-0009: Event Pipeline Durability Contract

## Status

Accepted — amends [ADR-0004](./0004-event-driven-tiered-pipeline.md)

## Date

2026-07-12

## Context

A full review (2026-07) traced why the graph never reflected live file edits and found the tiered
pipeline described in ADR-0004 had never durably delivered an update in production use:

- `FileWatcher._flush()` cancelled the debounce/max-wait timer task _currently executing it_ before
  publishing; the resulting `CancelledError` landed inside `bus.publish()`, after `_pending` had
  already been cleared — every timer-fired batch was silently destroyed. Watcher tests passed only
  because their fake event bus never yielded control (`await`), so the pending cancellation was
  never actually delivered — the bug was invisible to the test suite by construction.
- `atlas daemon start` never started a file watcher at all (`include_watcher=False`, contradicting
  its own help text) — a daemon-only deployment had no event producer, period.
- ADR-0004's "Significance Gating" table (docstring/body/signature-percentage heuristics) was never
  implemented as described. What actually gated re-embedding was `content_hash`, computed over
  `name`/`kind`/`visibility`/`signature`/`docstring`/`tags` — **excluding `entity.source`**. A
  body-only edit (the single most common edit) produced an identical hash, was classified
  `unchanged`, and never reached `_batch_update_entities` (the only writer of the stored `source`
  property) or `EmbedDirty` publication. Because the file-level hash was written back regardless,
  this was not a transient staleness window — it was permanent until a full wipe.
- Several other gates dropped events silently: cooldown-deferred changes were ACKed then held only
  in memory (lost on shutdown or a single failed republish); startup PEL reclaim could ACK
  crash-recovery messages before they were actually reprocessed; one deterministically-failing file
  in a batch retried forever with no cap, stalling every subsequent batch behind it; a concurrent
  `atlas index --full` called `EventBus.flush()`, which deleted the Valkey consumer groups a running
  daemon depended on, permanently and silently halting it; stream `XADD ... MAXLEN` trimming could
  discard events that had never been delivered, and `NULL` lag from that trimming was coerced to `0`
  so the drain check reported success anyway.

ADR-0004 accurately describes the tiered _shape_ of the pipeline (Valkey Streams, per-stage batch
windows, dedup-by-key) — that part still stands. What it gets wrong, now that these defects are
fixed, is the significance-gating mechanism and the durability guarantees. This ADR documents what
actually governs event delivery today.

## Decision

### Change detection: binary body-hash, not AST-diff percentage

`content_hash` now includes the entity's full, untruncated `source` text (folded into the same hash
rather than a separate property — `source` textually subsumes name/signature/docstring for every
source-bearing entity, so a second property would just duplicate the same signal). Classification is
binary: hash differs → `modified` (full property update, including `source` and line positions, plus
an `EmbedDirty` publish); hash matches → `unchanged`, skipped entirely. The AST-diff-percentage
significance table from ADR-0004 is removed — it was unimplemented and unreachable in the shipped
code, and the review found no realistic way to make an AST-diff-percentage threshold both cheap and
correct across nine languages. Schema version bumped to 3; the migration clears `file_hash`/`git_hash`
project-wide so every file is reclassified against the new hash formula on first run — a graph
indexed before this ADR cannot be trusted to already hold correct `source`/embeddings for any entity
whose body changed after its last signature/docstring edit.

### ACK invariant

A stream message may be XACKed only when one of:

1. The batch covering it returned successfully and its dedup key is not in the batch's deferred set.
2. It is superseded by a different, still-unacked message with the same dedup key in the same
   consumer's pending entries list.
3. It is structurally unprocessable (undecodable, project-filtered, empty fields) — ACKed
   immediately, logged.
4. It exceeded a poison-retry cap (`_MAX_BATCH_FAILURES = 5`) — parked with an ERROR log naming the
   consumer and dedup key, then ACKed, so one permanently-failing file can no longer stall every
   subsequent batch behind it.

Corollary: for any dedup key with outstanding work, at least one message covering it stays in the
consumer's pending entries list until that work actually succeeds. This replaces ADR-0004's blanket
"failed batches are not acknowledged" — that framing didn't distinguish _per-key_ outcomes within a
batch from the batch as a whole, which is what actually caused the crash-recovery ACK-before-process
defect.

### Group-preserving reset

A full reindex needs to clear stream backlog without destroying a running daemon's consumer groups
(`EventBus.flush()` previously did `DEL` on the stream key, which drops consumer group registrations
along with it). The reset path now trims stream content while leaving consumer groups intact, so
`atlas index --full` run alongside a live daemon no longer permanently kills its consumers.

### File hash withheld until deferred resolution completes

A file's hash is only persisted after its deferred relationships (imports, calls, type references,
cross-file member `parent_type_name` resolution — see
[ADR-0008](./0008-cross-file-relationship-resolution.md)) actually resolve, not immediately after
entity upsert. A crash between upsert and deferred-resolution no longer permanently loses those
edges behind an already-advanced hash gate.

### Explicit poison handling (reverses ADR-0004's "no DLQ" position)

ADR-0004 stated pending-entries-list redelivery avoided the need for a dead-letter mechanism. In
practice, a message that deterministically fails (a file that always throws during parse) is
redelivered forever, consuming batch slots and stalling everything queued behind it — PEL
redelivery handles _transient_ failure, not permanent failure, and the two were conflated. The
5-failure poison cap above is a minimal park-and-log mechanism, not a full dead-letter stream: parked
messages are not retried, not queryable, and require a subsequent file edit (which produces a fresh
message) to re-enter the pipeline.

## Consequences

### Positive

- File watching now durably delivers events end-to-end — verified by an integration test
  (`tests/integration/indexing/test_live_update.py`) that runs the real watcher, real `EventBus`,
  and real consumer, edits only a function body on disk, and asserts the graph's stored source and
  an `EmbedDirty` publication both reflect the edit.
- A single bad file can no longer stall the entire AST pipeline.
- A concurrent full reindex no longer kills a running daemon's consumers.

### Negative

- The binary body-hash rule re-embeds on any body edit, including purely mechanical ones (variable
  renames, whitespace inside the body) that the old (unimplemented) significance table would have
  filtered as `TRIVIAL`/`NONE`. This trades embedding cost for correctness — acceptable given the
  prior mechanism never actually worked, but it is a real increase in embedding-provider call volume
  versus the ADR-0004 design intent.
- Schema v3's forced full reindex is a one-time but real cost for existing installations.

### Risks

- The poison cap (`_MAX_BATCH_FAILURES = 5`) is a judgment call balancing tolerance for transient
  infrastructure blips against how long a genuinely poisoned file stalls its batch neighbors before
  being parked. Needs observability if it proves miscalibrated in practice.
- Stream `maxlen` trimming (now configurable via `redis.stream_maxlen`, default 1,000,000) still
  exists as a safety ceiling — an index large enough to exceed it in one run can still lose
  undelivered events; `NULL` lag now correctly reports "unknown, not drained" rather than coercing to
  `0`, so this failure mode is now visible instead of silent, but not eliminated.

## Alternatives Considered

### Full AST-diff percentage significance gating (completing ADR-0004's original design)

Implement the docstring/body-percentage heuristic as originally specified, across all nine
languages. Rejected: a cheap, correct, cross-language "percentage changed" metric does not exist
without a real AST diff algorithm per language (tree-sitter gives a parse tree, not a diff); the
binary body-hash rule is strictly simpler, provably correct (any body change is detected), and the
embedding-cost tradeoff was judged acceptable for a local developer tool.

### Separate `body_hash` property instead of folding into `content_hash`

Keep `content_hash` as-is and add a second property. Rejected: for every source-bearing entity type,
`source` already contains the name/signature/docstring bytes that `content_hash` separately hashes —
a second hash would just duplicate the same change signal at double the storage/compute cost with no
additional information.

### Full dead-letter stream for poisoned messages

A separate Valkey stream holding parked messages, queryable and manually retriable. Rejected as
disproportionate machinery for a local developer tool — a park-and-log with a clear error message,
recoverable by the next file edit, was judged sufficient; revisit if poisoning turns out to be common
enough to need operator tooling.

## References

- [ADR-0004: Event-Driven Tiered Pipeline](./0004-event-driven-tiered-pipeline.md) — the base
  pipeline shape this ADR amends
- [ADR-0008: Cross-File Relationship Resolution](./0008-cross-file-relationship-resolution.md) —
  the deferred-resolution mechanism whose completion now gates file-hash persistence
- `src/code_atlas/indexing/watcher.py` — `_flush()` self-cancellation fix
- `src/code_atlas/indexing/consumers.py` — ACK ordering, poison cap, cooldown durability
- `src/code_atlas/indexing/daemon.py` — consumer supervision with backoff restart
- `src/code_atlas/events.py` — group-preserving reset, `stream_maxlen`, lag semantics
- `src/code_atlas/parsing/ast.py` — `_compute_content_hash` (schema v3 formula)
- `tests/integration/indexing/test_live_update.py` — end-to-end regression test
