# ADR-0025: Structural Protocol Conformance by Method-Set Containment

## Status

Amended by [ADR-0028](./0028-every-resolved-edge-states-its-evidence.md): the conformance edge defined here is a
structural inference, and `inferred: true` is a boolean no consumer scores. It now carries a damped `weight` and
`confidence: "inferred"` so a path scorer can tell it from a declared relationship.

Accepted — revisits and narrows the rejection recorded in [ADR-0023](./0023-type-directed-call-resolution.md)'s
"Alternatives Considered". That rejection stands for the question it was asked; this applies the same technique to a
different question, and the measurement is what separates them.

## Context

`GraphBackend` is this codebase's central abstraction, and the graph held **nothing** implementing it. `GraphClient` and
`SqliteGraphClient` both satisfy it and neither names it, because Python `Protocol` conformance is structural — there is
no base class to record. ADR-0023 verified the consequence directly: zero inheritance edges touch
`GraphBackend`/`GraphClient`/`SqliteGraphClient`.

Measured before this change: **1 of 102** `...`-bodied stub methods had an inbound `IMPLEMENTS`, and 88 of the remaining
101 were `GraphBackend`'s. "What implements this Protocol?" and "which methods implement this stub?" both answered
nothing, for the one abstraction where the answer matters most.

ADR-0023 evaluated method-set containment and rejected it: it fired on 98 of 216 candidate sets at 90/98 precision,
inflated total CALLS weight by 16.1%, and its false positives had a characteristic shape — small test doubles
(`RecordingBus`, `FakeDrainBus`) elected as "the declaration" purely for having the fewest methods.

**That failure mode does not exist here, and the difference is not a matter of degree.** There, containment had to
_infer which class in a candidate set was the interface_. Here the interface identifies itself by inheriting `Protocol`;
containment only answers "does this class satisfy it". Those are different questions, and the smallest-class heuristic
that elected `RecordingBus` is not part of the second one.

## Decision

Emit `IMPLEMENTS` from a class to every self-declared `Protocol` whose non-dunder method set it contains, as a
post-batch project-wide sweep. Derive the method-level edges from the class-level ones rather than matching them
independently, so the two answers cannot disagree.

Measured on this repo before implementing: containment proposes **22 pairs**, of which **20 are correct implementations
on hand review — 100% precision** — and 2 are Protocol-to-Protocol.

| Protocol        | methods | proposed | correct |
| --------------- | ------: | -------: | ------: |
| `Detector`      |       2 |       10 |      10 |
| `SearchGraph`   |       3 |        4 |       3 |
| `GraphExecutor` |       7 |        4 |       3 |
| `GraphBackend`  |      91 |        2 |       2 |
| `EmbedOne`      |       1 |        2 |       2 |

Three guards, each earning its place from that table:

- **The Protocol must declare at least one non-dunder method.** A zero-method Protocol is satisfied by every class in
  the project.
- **A Protocol is never recorded as implementing another Protocol.** Both remaining pairs were this shape
  (`GraphBackend` does satisfy `SearchGraph` and `GraphExecutor`) — true, but not what the question means.
- **`inferred: true` on every edge**, so a consumer can distinguish a structural match from a declared one.

Note what the table does _not_ show: a low method count did not degrade precision. `Detector` matches on two method
names and is right ten times out of ten, and `EmbedOne` on one. The distinctiveness of the names carried it. That is a
property of this codebase rather than a law, which is why the ≥1-method guard is a floor and not a claim that one method
is always enough.

## Consequences

| metric                                    | before   | after          |
| ----------------------------------------- | -------- | -------------- |
| `IMPLEMENTS` edges                        | 11       | **261**        |
| stub methods with an inbound `IMPLEMENTS` | 1 of 102 | **106 of 106** |

"What implements `GraphBackend`?" returns `GraphClient` and `SqliteGraphClient`. "Which methods implement
`GraphBackend.execute`?" returns both concrete `execute` methods.

**No weight impact, by construction rather than by tuning.** `_fetch_community_inputs` reads `MATCH (a)-[r:CALLS]->(b)`
only, so `IMPLEMENTS` never enters the Leiden input and ADR-0019's gamma is untouched. This is the concrete reason the
+16.1% objection from ADR-0023 does not transfer: there, containment fed call resolution and every false positive became
a full-weight call edge. Here it feeds a structural question that nothing weights.

**Not implemented on the SQLite backend.** The containment test needs a set comparison per (protocol, class) pair, which
that schema makes expensive. It returns 0 rather than silently reporting that a project has no protocols.

**The sweep is project-wide, not per-file.** Conformance is a property of two classes that may live anywhere, so it can
only run once every file in the batch is upserted. On this repo that is 5 protocols against 486 classes; the cost scales
with their product and should be re-measured before assuming it holds at ten times this size.

## Alternatives Considered

**Leave it unanswerable, per ADR-0023.** Defensible until measured — and the measurement is what changed the answer.
Refusing to re-open a rejected technique when the question has changed is as much a failure as re-applying it
unexamined.

**Signature matching in addition to names.** Would raise precision further, but precision is already 20 of 20; there is
nothing to buy. Worth revisiting only if a larger codebase produces a false positive.

**Inferring the Protocol itself rather than requiring `Protocol` as a base.** This is exactly what ADR-0023 rejected,
and nothing here rehabilitates it. The self-declaration is what makes the technique safe.
