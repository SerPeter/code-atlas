# ADR-0023: Type-Directed Call Resolution

## Status

Accepted — extends [ADR-0022](./0022-call-resolution-requires-a-grounded-receiver.md), which recorded the receiver
expression and named receiver-type inference as the prerequisite it was a step toward. Amends
[ADR-0014](./0014-calls-edge-confidence.md) and [ADR-0017](./0017-calls-edge-weights.md) again, with two new strategies
and one new confidence path.

Its rejection of method-set containment is **narrowed** by [ADR-0025](./0025-structural-protocol-conformance.md): that
rejection holds for inferring _which class is the interface_, which is what this ADR needed. It does not hold for
testing conformance against a Protocol that declares itself — measured at 20 of 20 versus 90 of 98 here.

## Context

`blast_radius` reported `ambiguous_only` for every caller of `GraphClient.resolve_calls`, because the candidate set was
a Protocol declaration plus its two implementations. The obvious reading was "polymorphic dispatch is misreported as
ambiguity", and the obvious fix was a method-set containment rule to spot the declaration.

Re-measuring inverted both.

- **822 of 915 fanned-out call sites are in `tests/`**, and the top receivers are `graph_client` (543), `client` (263)
  and `conn` (18) — fixtures and locals, not interfaces.
- **Resolving the receiver's declared type sends 772 of 915 sites to exactly ONE concrete implementation, and only 24 to
  the Protocol.**

So ~85% of the population was never polymorphism. It was monomorphic calls on concretely-typed receivers that a
bare-name match spreads across every same-named method in the project.

Two further facts shaped the design, both instances of the same pattern this codebase keeps producing — a fact computed
and then discarded:

- The parser emits `INHERITS -> Protocol` for exactly the five relevant classes, and both write paths drop it because
  `Protocol` is not an in-project TypeDef.
- 1,254 CALLS edges terminated on a Protocol stub body that can never execute.

## Decision

**Reject method-set containment.** It fired on 98 of 216 candidate sets at 90/98 precision, but inflated total CALLS
weight from 3,776.75 to 4,384.75 (+16.1%) **while still pointing at the wrong targets**, because the population is
mostly monomorphic. Its false positives had a characteristic shape: small test doubles (`RecordingBus`, `FakeDrainBus`)
elected as "the declaration" purely for having the fewest methods.

**Phase 1 — record the abstract-base fact instead of discarding it.** A class with a `Protocol`/`ABC` base is flagged
`is_abstract` at parse time, carried through `_CallLookup`, and its methods are dropped from candidate sets. The dotted
`typing.Protocol` form is included; the previous identifier-only guard skipped it.

The filter applies wherever candidates come from a name lookup, not only in the project-wide strategy. Same-file
resolution runs first and was otherwise picking the stub whenever a Protocol lived in the caller's own file.

When removing the stub leaves exactly one implementation, that is a resolution, not a guess: `polymorphic_unique`,
confidence `resolved`, full weight. It must not fall through to ADR-0022's single-candidate branch, which would re-tag
it `unverified_receiver` at half weight — a Protocol declaring that very name **is** the project-namespace evidence that
branch looks for.

**Phase 2 — resolve through the receiver's declared type.** Two sources, covering a measured 90.7% of the affected
sites: parameter annotations, and one-step local construction `x = Foo(...)`. Recovery is deliberately conservative — a
plain identifier annotation resolves, `Store | None` and `list[Store]` do not, and a lowercase callee is not treated as
a constructor. An unrecoverable type falls back to today's behaviour rather than to a guess.

**Also closes a gap ADR-0022 left.** It damped an unverified receiver only when there was exactly one candidate, but
1,498 of 1,506 multi-candidate sites have a non-self receiver — the same unverifiable evidence, undamped. Those now
resolve as `unverified_wide` and take the same halving.

## Consequences

Measured on this repo across both phases:

| metric                 | before   | after    |
| ---------------------- | -------- | -------- |
| ambiguous CALLS edges  | 6,670    | 1,506    |
| resolved CALLS edges   | 6,471    | 7,238    |
| total CALLS edges      | 11,702   | 8,744    |
| edges into stub bodies | 1,254    | 26       |
| total CALLS weight     | 3,776.75 | 3,785.76 |
| modularity             | 0.3645   | 0.4240   |

**Weight moved +0.24%.** That is the whole argument for this approach over containment's +16.1%: it removes false edges
rather than re-weighting real ones, so MAGE's Leiden gamma — normalized by total weight — is undisturbed and ADR-0019's
constants need no retuning. Modularity rose because the removed edges were blurring cluster boundaries.

26 stub-terminating edges remain, where a Protocol method has no implementation in the index at all. The fallback keeps
the interface edge rather than dropping the call entirely, which is the honest answer when there is nothing else to
point at.

Schema v9 clears ambiguous CALLS edges and freshness markers: candidate sets computed before the flag resolved across
stubs, and re-parsing alone would leave the stale edge beside the corrected one.

Python only. Receiver-type recovery rests on annotations and constructor inference measured on this repo's Python; Java,
C# and TypeScript carry declared types more reliably and should generalise better, but that is an expectation and not a
measurement. Every other language keeps today's behaviour, because an absent type reads as unknown.

Inference stops at one step, by choice. Full inference is a different project, and one step already buys 90.7%.

## Alternatives Considered

**Method-set containment.** Rejected on measurement — see Decision. It remains the better option if receiver types ever
become unavailable, since it needs no parse-time data.

**Using `IMPLEMENTS`/`INHERITS` edges to find declarations.** Verified impossible: the graph holds zero inheritance
edges touching `GraphBackend`/`GraphClient`/`SqliteGraphClient`, because Python Protocol conformance is structural — the
implementations declare no bases and `GraphBackend` lives inside `if TYPE_CHECKING:`.

**Marking multi-implementation families `resolved`.** Rejected: 772 of 915 sites call exactly one concrete class, so
asserting resolved edges to every implementation manufactures false edges that `ambiguous_only` can no longer flag —
reintroducing ADR-0022's failure from the other direction.
