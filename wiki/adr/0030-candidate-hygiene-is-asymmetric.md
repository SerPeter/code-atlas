# ADR-0030: Candidate Hygiene Is Asymmetric

## Status

Accepted — extends [ADR-0022](./0022-call-resolution-requires-a-grounded-receiver.md) and
[ADR-0027](./0027-lexical-strategies-need-a-grounded-receiver.md), which narrowed _which strategies may trust a name_.
This one narrows _which definitions are eligible to be that name's target in the first place_, one step earlier in the
ladder.

## Context

The CALLS resolver matched a bare name against every same-named Callable in the project. Test definitions were in that
pool, so a call in production code could resolve onto a test double.

Measured on this repo's own index before the change:

| measurement                                                        |     value |
| ------------------------------------------------------------------ | --------: |
| CALLS edges from production code into a test definition            |   **415** |
| distinct production callers affected                               |       223 |
| distinct test definitions absorbing production calls               |        66 |
| production targets whose weight was diluted by a test co-candidate |   **102** |
| average `weight` on those diluted edges                            | **0.205** |
| average `candidate_count` on those diluted edges                   |   **4.7** |

The wrong edge is the visible half. The damaging half is quieter: `candidate_count` is the surviving candidate list's
length and `weight` is `1 / candidate_count` (ADR-0014, ADR-0017), so a same-named fixture does not merely add an edge
that should not exist — **it halves the weight of the edge that should.** That weight is what reaches weighted Leiden
and `blast_radius` ranking, so the consequence is a quiet mis-ranking rather than a visibly wrong answer.

The ingredients were already present and unused. `_CallLookup.name_to_callables` has always been
`name → [(uid, file_path, vis)]`, and `matches_test_pattern` was already imported into `graph/client.py` — but only to
compute `from_test` on the **caller**, never to filter **candidates**.

Graphify (see `.specs/research/2026-08-04_competitor_parsing-and-retrieval.md`) applies the same filter and factors it
into a shared module so both of its resolution entry points stay aligned. It also applies a **symmetric** rule: a test
call site prefers test candidates. That half is not adopted here — see below.

## Decision

Filter test-file definitions out of the candidate pool when the call site is not itself in test code, at the point where
`candidates` is built, before any name-matching strategy reads the list.

Three properties are load-bearing:

1. **Asymmetric.** A test caller filters nothing. "Production code does not depend on test code" is an architectural
   invariant; "tests do not call production code" is the opposite of true — calling the code under test is what a test
   is for. Graphify's symmetric preference is not grounded in an invariant, and it also buys nothing here: a test caller
   co-located with its helper is already resolved by the same-file rung (Strategy 3). The unit test for the asymmetry
   documents exactly that, having first been written the wrong way and passed for the wrong reason.

2. **Early.** Applied where the pool is built, not inside a strategy — the same reason the abstract-stub filter is
   applied there. A filter a strategy can return past is not a filter, and the surviving count _is_ `candidate_count`.

3. **Never empties the pool.** When every definition of a name lives in test code, the candidate list falls back to the
   unfiltered set. Trading a diluted-but-present edge for a silent absence is the failure mode ADR-0014 exists to
   prevent, and the one Graphify's drop-on-ambiguity design walks into.

The test-definition uid set is derived from the **effective** patterns passed to `resolve_calls`, not from
`_DEFAULT_TEST_PATTERNS`, so a project configuring its own `search.test_patterns` filters candidates by the same rule
that decides `from_test`.

## Consequences

- Every downstream rung and the entire weight formula inherit the improvement without changes, because the pool is
  narrowed before any of them run.
- `candidate_count` drops for affected edges, so their `weight` rises. Existing graphs do not re-derive this until the
  next resolution pass.
- Both backends share `_resolve_one_call`, so `GraphClient` and `SqliteGraphBackend` cannot drift; each computes the set
  from its own effective patterns and passes it explicitly. The parameter defaults to an empty frozenset, which disables
  the filter — acceptable only because both production call sites pass it, and unit tests that construct a lookup
  directly want it off.
- **Module-scope callers inherit an existing gap.** Modules are absent from `uid_to_info` — that absence is how
  `_resolve_one_call` detects a module-scope call — so an import-time call in a test file is treated as production code
  and _will_ filter test candidates. This matches how `from_test` already behaves, so it introduces no new
  inconsistency, but it is a real edge case.
- **Constructor candidates (Strategy 5) are not filtered.** `name_to_typedefs` is built outside the lookup and carries
  no pattern context, so a test file defining a class named like a production class still pollutes that last-resort
  rung. Measured pollution was on the Callable path; this is knowingly left open.
- Verified non-vacuous: with the filter disabled, the integration test fails naming the exact wrong edge
  (`patterns.test_doubles.FakeLedger.commit`). The positive LINKED case passed either way, which is why the negative
  assertion is a separate test — the coverage harness's LINKED/MISSING vocabulary cannot express "this edge must never
  appear".

## References

- ATL-103
- [ADR-0014](./0014-calls-edge-confidence.md) — ambiguity is materialized, not dropped
- [ADR-0017](./0017-calls-edge-weights.md) — `weight = 1 / candidate_count`
- [ADR-0022](./0022-call-resolution-requires-a-grounded-receiver.md),
  [ADR-0027](./0027-lexical-strategies-need-a-grounded-receiver.md) — which strategies may trust a name
- `.specs/research/2026-08-04_competitor_parsing-and-retrieval.md` §5 item 1
