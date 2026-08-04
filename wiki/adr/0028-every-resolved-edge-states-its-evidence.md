# ADR-0028: Every Resolved Edge States Its Evidence

## Status

Accepted — extends [ADR-0017](./0017-calls-edge-weights.md), which introduced `weight` for CALLS only, to every edge a
resolver produces. Amends [ADR-0025](./0025-structural-protocol-conformance.md): the conformance edge it defines is a
structural inference and now says so numerically rather than only via a boolean nothing reads.

## Context

A census of the graph asked a question nobody had: which edge types record how sure they are?

| edge                                                                       |  count | weight | confidence | other             |
| -------------------------------------------------------------------------- | -----: | ------ | ---------- | ----------------- |
| CALLS                                                                      | 10,637 | 10,523 | 10,523     | `strategy`        |
| IMPLEMENTS                                                                 |    260 | —      | —          | `inferred` on 249 |
| USES_TYPE                                                                  |  1,640 | —      | —          | nothing           |
| IMPORTS / DEFINES / CONTAINS / INHERITS / OVERRIDES / REFERENCES / EXPORTS | 15,317 | —      | —          | nothing           |

The stated reason weight is CALLS-only is sound as far as it goes: ADR-0017 defines it as `1 / candidate_count`, and
that formula only means something when resolution matched a bare name against a candidate **set**. `IMPORTS` resolves an
exact dotted path; `DEFINES` and `CONTAINS` come straight from the AST. For those, an absent weight is correct — they
are structural facts and 1.0 is truthful.

**The reasoning does not cover two of them, and both are guesses wearing a structural fact's clothes.**

- **`resolve_type_refs` has a three-rung ladder** — import match, same-file, then _project-wide unique TypeDef_. That
  third rung is precisely the shape ADR-0022 demoted for calls, because uniqueness is evidence of identity only when the
  name was looked up in the project's namespace. All 1,640 edges were written by a bare `MERGE (a)-[:USES_TYPE]->(b)`.
  The resolver computes which rung fired and discards it at write time.
- **`resolve_protocol_conformance` derives IMPLEMENTS from method-set containment** (ADR-0025) — declared nowhere in the
  source. It records `inferred: true`, a boolean no consumer scores, and no weight.

This is the fourth instance of one pattern in this codebase: a fact is computed and then dropped. ADR-0022 recovered the
receiver expression, ADR-0023 the receiver type, ADR-0026 the replay classification. Same shape, different resolver.

It surfaced while scoping a fix to `blast_radius`, whose default edge set is `CALLS` only and which is therefore blind
to 18% of `src/` entities — including `GraphClient`, whose 239 dependents it reports as zero. Widening it means
traversing exactly these edge types, and `confidence_score` multiplies `coalesce(r.weight, 1.0)` along the path. Landing
that first would have scored an inferred IMPLEMENTS hop and a project-wide-guess USES_TYPE hop as **maximally certain**,
ranking them above an honestly-marked ambiguous CALLS edge at 0.5. That is ADR-0022's failure rebuilt in the scorer.

## Decision

**Every edge a resolver writes states how it was resolved** — `strategy`, `confidence`, `weight`.

- `resolve_type_refs` stamps its rung. `import` and `same_file` are `resolved` at 1.0; `project_unique` is `ambiguous`
  at 0.5.
- `_link_named_callable` — shared by REFERENCES, REGISTERED_BY and Value-scoped USES_TYPE — stamps `same_file` or
  `import` accordingly. Neither is a guess (its two passes are exactly the rungs ADR-0022's test passes), but saying so
  is what distinguishes them from an edge that recorded nothing.
- `resolve_protocol_conformance` writes `confidence: "inferred"` and a damped weight.

**Reuse the two existing tiers; do not mint a third.** An unverified name match is "at best an even split between the
name and something outside the graph", and that is as true of a type annotation as of a receiver. ADR-0025 measured
conformance at 20 of 20 on this repo, which argues 0.5 is too harsh for it — but ADR-0017 already records that these
numbers are heuristics awaiting evidence, and inventing an unmeasured middle tier to fix an unmeasured one is not
progress. Retune when there is evidence about what ranks well.

**An absent property keeps meaning "structural".** Parser-written edges (DEFINES, CONTAINS, the 5 declared IMPLEMENTS)
stay bare, so `coalesce(r.confidence, 'resolved')` reads them correctly. Only resolvers — which guess — must speak.

## Consequences

For `code-atlas` after a full re-index, **0 of 10,526 CALLS, 1,471 USES_TYPE and 278 REFERENCES edges are unscored**.
The 5 remaining bare IMPLEMENTS are parser-declared, which is the intended state.

**`trace_path` improves immediately.** It already traverses `CALLS|IMPORTS|USES_TYPE` and ranks by `path_weight`, so a
route through a guessed type-use now ranks below one through an import match. That ranking was previously flat.

**`blast_radius` can now be widened safely** — the reason this landed first.

**Measured surprise: `project_unique` fires zero times for USES_TYPE here.** 1,127 import + 344 same_file and nothing
else, so this repo's type edges were all trustworthy already and the tier that motivated the work is empty in it. The
instrumentation still matters — it is a property of this codebase, not of the resolver, and nothing was recording which
it was.

**Cross-project residue is now visible rather than silent.** Edges from other projects sharing the Memgraph instance
(`trading-bot/*`, indexed before this change) show as unscored until re-indexed. That is a reporting improvement, not a
regression: they were always unscored, and were previously indistinguishable from structural facts.

## References

- `graph/client.py` — `_TYPE_REF_FACTS`, `_TYPE_REF_RANK`, `_INFERRED_IMPLEMENTS_WEIGHT`, `_link_named_callable`
- `tests/unit/graph/test_client.py` — `TestNonCallEdgeQuality`
- ADR-0017 (weights), ADR-0022 (grounded receiver), ADR-0025 (structural conformance)
