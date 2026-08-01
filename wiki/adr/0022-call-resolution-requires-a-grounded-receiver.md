# ADR-0022: Call Resolution Requires a Grounded Receiver

## Status

Accepted — amends [ADR-0014](./0014-calls-edge-confidence.md), whose `resolved`/`ambiguous` pair this splits along a
second axis, and [ADR-0017](./0017-calls-edge-weights.md), whose weight is derived from that confidence.

## Context

A reader with no knowledge of this codebase, shown only a rendered `summarize_module` outline, flagged
`EmbedCache.clear > FileScope.scan` as "almost certainly wrong — an embedding cache calling a filesystem scanner". It
was wrong, and the graph asserted it with full confidence:

```
EmbedCache.clear -> code_atlas.indexing.orchestrator.FileScope.scan
  {confidence: "resolved", strategy: "project_unique", candidate_count: 1, weight: 1.0}
```

The real call is `await self._redis.scan(...)` — the Valkey client's method, which is not in the graph at all.
`FileScope.scan` was the only project entity named `scan`, so the `project_unique` strategy claimed it.

This is worse than an admitted guess, and invisible at every layer _because_ the resolver was sure:

- `blast_radius`'s `ambiguous_only` cannot flag it — the edge is not ambiguous.
- ADR-0017 gives it the full weight of a real call, feeding community detection and impact ranking.
- The outline's `[k=v]` annotation renders nothing, every property sitting at its neutral value.

The strategy's own comment records that the hazard was known and the guard was removed: "Previously only fired when
exactly 1 candidate existed (ambiguous names like run/close/get were left unresolved to avoid false positives from
external attribute calls such as `asyncio.run()`, `session.run()`)."

## Decision

**Uniqueness within the project is evidence of identity only when the name was looked up in the project's namespace.**
For a bare call `helper()` it must resolve in lexical scope, so uniqueness is real evidence. For `client.scan()` it is
not: the receiver's type may never have been indexed, and the single same-named entity is a coincidence.

The distinguishing fact was already being computed and discarded. `languages/python.py` branches on
`func.type == "identifier"` versus `"attribute"` and then emitted an identical `ParsedRelationship` from both arms. It
now records the receiver expression as a property.

`project_unique` is withheld when a receiver is present and is not a self-reference (`self`, `cls`, `this` —
`self.helper()` is exactly as grounded as `helper()`). Such a match becomes strategy `unverified_receiver`, confidence
`ambiguous`. The edge is kept, per ADR-0014's principle of materializing rather than discarding: it is the best guess
available, it just may not claim to be a fact.

This introduces `candidate_count: 1` with `confidence: "ambiguous"` — previously impossible. That combination is
meaningful and is the point: one name matched, and the match could not be confirmed.

The four other strategies are untouched. Import, sibling, same-file and constructor resolution are lexically grounded,
and they are the bulk of correct output — 4,005, 631, 1,407 and 119 edges respectively against `project_unique`'s 334 in
the reference index.

Recording the receiver _expression_ rather than a bare "is an attribute call" flag is deliberate. Any future
receiver-type inference needs exactly that string, so it makes the next improvement additive instead of a second
migration.

## Consequences

Schema v8. Existing `project_unique` edges are deleted rather than left to be overwritten: they were written `resolved`
with full weight, so nothing downstream distrusts them, and a stale one whose call site has since gone would never be
revisited. Only that strategy is purged — rebuilding the other four would cost a re-index for nothing.

Total CALLS weight falls, because reclassified edges lose the `resolved` weight ADR-0017 assigns them. ADR-0019's
community-detection constants were calibrated on the older distribution and should be re-checked; a large modularity
shift is a signal to retune, not to accept silently.

`self`, `cls` and `this` are a hardcoded set. That covers the languages this indexer parses today, but it is a naming
convention, not a grammar fact, and a language using a different self-reference would silently lose `project_unique`
resolution for its method calls — degrading to `ambiguous`, which is the safe direction.

Only the Python parser records a receiver so far. Every other language keeps today's behaviour, since an absent receiver
reads as a bare call. That is the unsafe direction, and closing it per language is follow-up work.

## Alternatives Considered

**Blacklist known-external method names** (`scan`, `close`, `get`). Rejected — whack-a-mole, and the right list differs
per project.

**Lower the weight of `project_unique` edges.** Rejected: it hides the symptom while the edge still asserts `resolved`,
so `ambiguous_only` and every other consumer keep trusting it.

**Delete the strategy outright.** Rejected — it resolves bare-name calls correctly, which is most of its output.

**Infer the receiver's type.** The correct long-term answer and out of scope here; recording the receiver expression is
the prerequisite, so this decision is a step toward it rather than away.
