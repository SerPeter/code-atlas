# ADR-0026: Resolution Is Replayed, Not Batch-Final

## Status

Accepted. Amends the deferred-resolution design behind
[ADR-0022](./0022-call-resolution-requires-a-grounded-receiver.md) and
[ADR-0023](./0023-type-directed-call-resolution.md): both reason about how good a resolution strategy is, and neither
notices that a good strategy run against a partial graph is still wrong.

## Context

Cross-file relationships resolve after their batch is upserted, against the graph **as it stands at that flush**.
`set_batch_file_hashes` then records the caller's hash, so the hash gate never re-parses it. A callee upserted by a
_later_ batch is therefore invisible for the life of the index — not "hard to resolve", but never attempted again.

This is not a tail case. In a bulk index most files are parsed before the modules they call into, and the damage was
measured on this repo:

- **`indexing/consumers.py` had zero edges of any type to `events.py`** — no IMPORTS, no CALLS, no USES_TYPE — while
  importing eight names from it and calling `decode_event` in two places.
- **53 modules carried an IMPORTS edge to the root `code_atlas` Package.** The dotted-prefix fallback turned every
  unresolvable import into a plausible-looking edge, which is why nobody noticed: `import_map` was quietly empty for
  those modules, so strategy 1 of `_resolve_one_call` could not fire either.
- **Nine of the twenty-seven `find_dead_code` hits were functions in `events.py`.** Replaying the resolver against the
  finished graph rescued ten of the twenty-seven with real, correct callers.

There are two distinct failures, and only the first is obvious.

1. **Unresolved.** The name matched nothing, because the target did not exist yet.
2. **Resolved against a partial candidate list.** `unverified_receiver` fires when exactly _one_ name match exists, so
   `TierConsumer.run -> EventBus.read_batch` landed on `SqliteEventBus.read_batch` at full weight — the only
   implementation indexed at that moment. This is the worse of the two: a missing edge is visibly missing, while a
   confident edge pointing at the wrong implementation is trusted by `blast_radius`, `find_dead_code` and Leiden alike.
   It is ADR-0022's failure mode arriving through the back door.

## Decision

**Resolvers report what they could not settle, and the consumer replays it.** `resolve_imports`, `resolve_calls` and
`resolve_type_refs` return `ReplayableRels`, split by what a replay costs:

- `unresolved` — matched nothing. Replayed on **every** flush, because a replay that still fails writes nothing at all.
  This is also what lets a running daemon link an existing caller to a function added afterwards.
- `stale_candidates` — resolved, but through the project-wide candidate list. Replayed **once, on the final flush**,
  when the candidate list is finally complete. Each one rewrites the edges it owns, so 40 mid-run replays cost real time
  and every one of them is superseded by the last.

**Only `sibling` and `same_file` are exempt.** Both are settled by the caller's own file, which is upserted in the same
batch — no later batch can change them. Every other strategy reads the project-wide candidate list and is only as good
as the graph at that instant, including `import` (whose `import_map` a replayed `resolve_imports` repairs) and
`project_unique` (which stops being unique the moment a second definition lands).

**The stale-candidate buffer is reindex-mode only.** It is the mode whose ordering causes the problem and the only one
that reliably reaches a final flush; a daemon resolves against an already-complete graph, so retaining it there would be
megabytes held for nothing.

Both buffers are keyed by `(rel_type, from_uid, to_name, receiver, receiver_type)` rather than appended, so re-parsing a
call site replaces its entry. Most entries never resolve — builtins, external libraries — and an unkeyed list would grow
once per re-index.

## Consequences

Measured on this repo, full re-index, before → after:

|                              | before | after |
| ---------------------------- | -----: | ----: |
| CALLS edges                  |  9,058 | 9,713 |
| cross-file CALLS             |  4,066 | 4,720 |
| `consumers.py` → `events.py` |      0 |   116 |
| `find_dead_code` (src/)      |     27 |    15 |
| full index wall-clock        |   197s |  259s |

**The 31% indexing cost is the price of the answer being right**, and it is paid once per project — incremental runs
resolve against a complete graph and replay almost nothing. It is dominated by the final pass, not by the per-flush
replay: gating the per-flush replay off entirely in bulk mode recovered 8 seconds of the 62 and was reverted as
complexity that did not pay.

**A replay MERGEs; it does not retract.** The stale single-candidate edge written earlier survives alongside the
corrected fan-out. That matches ADR-0014's choice to materialize ambiguity rather than discard it, but it means edge
counts drift upward across a run rather than converging exactly.

**The buffer does not survive the process.** Both live in consumer memory, so this fixes within-run staleness only. A
daemon restarted between file A's indexing and file B's arrival still loses that edge — the citations path solves the
same problem durably by persisting `unresolved_citations` on the node, and that is the shape to reuse if this becomes a
real complaint.

**`INHERITS` is deliberately not covered.** It resolves entirely in Cypher with no per-rel result to hand back, and its
failure is a missing edge rather than a wrong one, so it stays as it was.

**The monorepo path does not get the stale-candidate pass on a first index.** `index_project` derives
`reindex_mode = full_reindex or decision.mode == "full"`, so a project seen for the first time qualifies;
`_index_monorepo_inner` derives it from `full_reindex` alone, so a monorepo's first index qualifies only under `--full`.
That asymmetry predates this decision — it also governs the polling policy — and was left alone rather than widened as a
side effect. `atlas index --full` is unaffected either way.

## References

- `graph/client.py` — `ReplayableRels`, `_FILE_LOCAL_STRATEGIES`
- `indexing/consumers.py` — `_retry_rels`, `_stale_candidate_rels`, `_retry_key`
- `tests/integration/indexing/test_consumers.py` — `test_a_callee_indexed_after_its_caller_still_gets_the_edge`,
  `test_a_lone_candidate_is_revisited_when_a_second_one_appears`
- ATL-101, class 3
