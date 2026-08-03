# ADR-0027: The Lexical Strategies Need a Grounded Receiver Too

## Status

Accepted — completes [ADR-0022](./0022-call-resolution-requires-a-grounded-receiver.md) rather than amending it. That
decision established that a name looked up on an ungrounded receiver may not be trusted, and applied it to the
project-wide strategies. It left the two lexical strategies alone, where the same reasoning was needed most.

## Context

`SqliteEventBus.acquire_indexer_lease` and `renew_indexer_lease` were reported dead while being executed on every
embedded-mode `atlas index` run. The call is `bus.acquire_indexer_lease(...)` inside `hold_indexer_lease(bus: Any)` in
`events.py`. `Any` is opaque, so the receiver has no declared type and the rel carries a bare name.

Strategy 3 (same-file match) then found `EventBus.acquire_indexer_lease` — the Valkey twin, which happens to live in
`events.py` too — and returned it as a **single candidate at `confidence: "resolved"` and full weight**. The real
implementation in `backends/sqlite_queue.py` got nothing.

The natural experiment in the same file settles the mechanism. `read_indexer_lease` has the identical untyped `bus.`
receiver, but is _also_ called from `server/health.py` and `indexing/consumers.py`, which define neither implementation.
Those calls reach Strategy 4, fan out, and land `unverified_wide` edges on **both** twins — so the SQLite one is never
reported dead. An untyped receiver alone is not fatal; **co-location with a rival implementation is**, and only for
callers that share the file.

This is ADR-0022's failure mode arriving through a door that ADR left open. Strategies 2 (same-class sibling) and 3
(same-file) are _lexical_ lookups: they answer "what does this name mean here". `bus.foo()` is not a lexical lookup — it
is an attribute of an object whose type the indexer may never have seen. Sharing a file with the caller is a
coincidence, not evidence of identity.

## Decision

**Strategies 2 and 3 fire only for a grounded receiver** — no receiver at all, or one in `_GROUNDED_RECEIVERS`. An
attribute call on anything else falls through to the receiver-type and project-wide strategies, which already damp what
they return.

**`super()` counts as grounded, and `self`/`cls` do not cover it.** `super().foo()` names the caller's own base, so it
is a lexical reference in exactly the sense that matters. Excluding it sent all 8 `super().__init__()` sites to a 47-way
project-wide fanout — 376 edges asserting nothing, and by itself the largest source of noise the gate would have
introduced.

## Consequences

Measured by replaying the resolver over the whole repo against the finished graph, then confirmed end-to-end by a full
re-index:

|                                    | before |                after |
| ---------------------------------- | -----: | -------------------: |
| call rels whose answer changes     |      — | 507 of 20,543 (2.5%) |
| CALLS edges                        |  9,770 |               10,517 |
| `find_dead_code` (src/)            |      5 |                    3 |
| entities losing every inbound edge |      — |                    0 |

**The suppression cost was the thing to measure, and it is zero.** Widening a strategy adds edges, and
`get_dead_code_candidates` counts an inbound CALLS edge regardless of confidence — so a fanout can silence a genuinely
dead entity. Measured: exactly **6** entities gain a first inbound edge, of which **2** are the intended targets and
**0** are genuinely dead. Both known true positives (`GraphClient.run_in_write_transaction`, `mcp._get_app_ctx`) survive
and are still reported.

**Every new edge is honest about itself.** Of the 400 resolutions that change, 362 are `ambiguous` and the 38 that come
back `resolved` are single-candidate `receiver_type` matches — a typed receiver resolving by type instead of by file
coincidence, which is strictly better than what it replaced. `_call_edge_weight` divides by `candidate_count`, so a wide
fanout carries ~0.01 against a resolved edge's 1.0; Leiden, `blast_radius` ranking and centrality already discount it
correctly.

**`find_dead_code` is the one consumer that flattens confidence to a boolean**, and that is deliberate. For a dead-code
report the asymmetry runs one way: a false "alive" costs a reader a few minutes of verification, while a false "dead"
invites deleting live code. Precision over recall is ADR-0022's standing constraint and it argues _for_ counting
ambiguous edges as proof of life. The information is not lost — it is on the edge — so "which entities are reachable
only through ambiguous edges" stays answerable as a reporting question, and should be one if the dead list ever gets
long again.

**Not fixed here:** `super().foo()` still resolves through the same-file heuristic, which is a coincidence for a base
class defined elsewhere. Resolving it properly needs an inheritance-aware lookup that `_CallLookup` does not carry. In
practice these are almost all `__init__`, which the dead-code predicate excludes as a dunder anyway.

## References

- `graph/client.py` — `_GROUNDED_RECEIVERS`, `_resolve_one_call` strategies 2 and 3
- `tests/integration/test_relationship_coverage.py` — `duck-typed-twin-not-stolen-by-co-location`
- ATL-101, gap 5b
