# ADR-0017: CALLS Edge Weights and Test Provenance

## Status

Extended by [ADR-0028](./0028-every-resolved-edge-states-its-evidence.md): the weight defined here was scoped to CALLS
because only that resolver matched against a candidate set. That left USES_TYPE's project-wide-uniqueness rung and
inferred IMPLEMENTS indistinguishable from structural facts, which any path scorer reads as certainties.

Accepted — amends [ADR-0014](./0014-calls-edge-confidence.md); itself amended by
[ADR-0019](./0019-module-granularity-community-detection.md), which replaced weighted Leiden with module-granularity
clustering. The weights below are unchanged and still the input — ADR-0019 sums them per module pair instead of handing
them to MAGE. The Empirical Validation section therefore documents a procedure this codebase no longer calls; read it as
the record of why the string `confidence` could not serve as a weight, which remains the reason a numeric `weight`
property exists.

Amended by [ADR-0022](./0022-call-resolution-requires-a-grounded-receiver.md), which reclassifies unverifiable attribute
calls as ambiguous and so lowers total CALLS weight.

## Date

2026-07-30

## Context

[ADR-0014](./0014-calls-edge-confidence.md) gave every CALLS edge a categorical `confidence` (`"resolved"` /
`"ambiguous"`) and a `strategy`. In its Alternatives Considered it explicitly weighed and **rejected** a numeric score:

> Considered a numeric confidence score (e.g. `1.0 / len(candidates)`) instead of a two-value string enum. Rejected as
> premature precision: the underlying signal is genuinely binary (the resolver either found one match or several), and
> inventing a numeric scale on top of it would imply a granularity the heuristics don't actually support.

That reasoning was sound given the consumers of the time. It has since been overtaken by three consumers that need a
magnitude rather than a category, which ADR-0014's own Risks section anticipated ("if this proves noisy in practice, a
future revision could…"):

1. **Weighted community detection.** MAGE's `leiden_community_detection.get()` in the pinned
   `memgraph/memgraph-mage:3.7.2` image accepts a `weight_property`, but reads it via `mg_utility::GetNumericProperty`,
   which falls back to `1.0` for a missing property **and for any non-numeric type, silently and without warning**.
   `confidence` is a string, so passing it would have produced byte-identical communities forever while appearing to
   work.
2. **Blast-radius ranking.** `blast_radius` could flag `ambiguous_only` but had no way to order results, so a test-only
   caller reached through two guessed edges ranked identically to a production caller reached through one certain edge.
3. **Test provenance.** Test callers dominate fan-in for any widely-exercised function. Query-time path filtering
   (`search.test_patterns`) could exclude them from _results_ but could not down-weight them inside a graph algorithm.

## Decision

CALLS edges carry three new properties alongside ADR-0014's `confidence` and `strategy`:

- `candidate_count` (int ≥ 1) — how many candidates the winning strategy returned.
- `from_test` (bool) — whether the **caller** lies in test code, matched with `matches_test_pattern` (the canonical
  predicate, per [ADR-0016](./0016-consistent-test-entity-filtering.md)) against the project's configured
  `search.test_patterns`.
- `weight` (float, strictly positive) — derived as
  `max(BASE / max(candidate_count, 1) * (TEST_DAMPING if from_test else 1), MIN_WEIGHT)` with `BASE = 1.0`,
  `TEST_DAMPING = 0.25`, `MIN_WEIGHT = 1e-6`.

**This honors ADR-0014's objection rather than overriding it.** The edge stores the _raw observations_
(`candidate_count`, `from_test`) as first-class properties; `weight` is a single derived convenience computed in one
function from those facts. Consumers wanting evidence read the facts; consumers needing a scalar (MAGE, which can only
read a persisted number) read `weight`. The weighting formula can therefore be retuned without another reindex, and no
information is destroyed by the derivation.

Supporting decisions:

- **Dedup combination replaces last-write-wins.** N call sites collapse to one edge. The pair now keeps the
  best-evidenced observation (lowest `candidate_count`; ties keep the first seen, so results are order-stable), and
  `from_test` is combined with AND — an edge is test-provenance only if _every_ observed call site was in test code,
  because one production caller makes the edge production-relevant.
- **The positivity floor is load-bearing, not defensive dressing.** Leiden computes `gamma /= sum_of_weights`; a zero
  total yields NaN and silently meaningless communities.
- **IMPORTS edges are left at MAGE's implicit 1.0**, and `BASE` is set to 1.0 to make that a deliberate choice rather
  than an accident: a fully-resolved non-test call is worth exactly one import.
- **Leiden's parallel-edge dedup does not apply to this projection.** Leiden drops rather than sums parallel edges,
  which would defeat per-edge-type weighting. It cannot bite here: `resolve_calls` matches both endpoints as
  `:Callable`, while every IMPORTS edge originates at a Module or Package, so no node pair carries both — including
  under Leiden's undirected view.
- **`blast_radius` scores by product, not min**, so uncertainty compounds: two ambiguous hops are a weaker claim than
  one. The per-entity flag is named `test_only` (no test-free path reaches it within `max_depth`), deliberately distinct
  from the edge-level `from_test`.
- **`trace_path` still sorts by hop count first**, breaking ties by total path weight, so its documented shortest-path
  semantics are unchanged.

`SCHEMA_VERSION` is bumped to **6** with a data migration that clears `file_hash`/`git_hash` and deletes pre-v6
(weightless) CALLS edges. Without it the file-hash gate would skip every unchanged file, `resolve_calls` would never
re-run, and the feature would be permanently invisible on existing indexes — while MAGE read the absent weights as a
uniform 1.0.

## Consequences

### Positive

- Community detection, impact ranking, and path preference all become evidence-aware in one change.
- Test callers stop distorting subsystem boundaries and impact reports without being hidden outright.
- Storing observations rather than only a score means the weighting heuristic is retunable without a reindex.

### Negative

- A schema bump forces a full reindex for every existing user.
- `graph/client.py` now imports `matches_test_pattern` from `search.engine` — a layering inversion. It is acyclic and
  verified at runtime, and the alternative (a second test-matching implementation) is exactly the drift ADR-0016 exists
  to prevent, but the dependency direction is wrong and worth revisiting.
- Two heuristic constants (`TEST_DAMPING`, `MIN_WEIGHT`) now shape community boundaries and are not empirically tuned.

### Risks

- **Leiden's resolution shifts with weights.** `gamma` is divided by total edge weight, so introducing sub-1.0 weights
  changes the effective resolution while `gamma`, `resolution_parameter`, and `_COMMUNITY_NOISE_THRESHOLD` remain
  calibrated on the unweighted graph. Community counts and sizes may move; retuning is a known follow-up.
- **Leiden is non-deterministic between runs**, so a single before/after comparison cannot distinguish a real weighting
  effect from run-to-run variance. Validating the weighting genuinely took effect requires asserting on the projected
  edge properties, not on partition stability.
- Reciprocal CALLS pairs (`a→b` and `b→a`) still collapse under Leiden's undirected view, and which of the two weights
  survives is now observable where before both were 1.0.

## Empirical Validation

Verified live against `memgraph/memgraph-mage:3.7.2` rather than reasoned about, because every failure mode here is
silent. A synthetic barbell graph (two 4-cliques joined by one bridge) was clustered four ways:

| `weight_property` | communities | reads our weight? |
| ----------------- | ----------- | ----------------- |
| omitted           | 7           | **yes**           |
| `"weight"`        | 7           | yes               |
| `"confidence"`    | 2           | no — string       |
| `"no_such_prop"`  | 2           | no — absent       |

Two findings, one of which corrects a natural misreading of the Decision above:

1. **The weighting genuinely takes effect.** Naming `weight` yields a different partition than naming a string or a
   nonexistent property, which is the only way to prove the numeric value is read at all given the silent 1.0 fallback.
   On a realistic graph (three 5-cliques joined by test-only edges at 0.25 against 1.0 intra-cluster) the weighted run
   recovers exactly the three clusters, stably across repeated runs.
2. **Passing `"weight"` explicitly is not what enabled this.** MAGE _defaults_ `weight_property` to the literal string
   `"weight"`, so the pre-change `get(subgraph)` call was already reading a property of that name — it simply did not
   exist, so every edge fell back to 1.0. The behavior change came from **writing** the property, not from passing the
   argument. The explicit argument is retained for readability and because it documents the dependency, but it is
   functionally redundant. The corollary is worth remembering: **any** numeric edge property named `weight` in this
   graph is silently consumed by Leiden, whatever wrote it and whatever it was meant for.

Also observed: `leiden_community_detection.get` **raises** `No communities detected` rather than returning an empty
result when it cannot partition the subgraph (reproduced on the degenerate 0.01-vs-500 weight spread).
`_analyze_communities` now distinguishes that from a genuinely missing procedure instead of advising the caller to check
their Docker image.

## Alternatives Considered

### Reuse `confidence` as the weight property

Rejected, and actively dangerous: `confidence` is a string on CALLS but a float on DOCUMENTS. MAGE would read the string
as 1.0 with no error, producing a silently unweighted result that looks correct.

### Store only the derived `weight`

Rejected — it is precisely what ADR-0014 refused. A bare float discards the observation it came from, makes the
heuristic unauditable, and would require a reindex to retune.

### Compute weights in a pre-pass before each community run

Rejected: the analysis path is read-only by design, and a write query on every `find_communities` call would make an
analysis tool mutate the graph.

## References

- [ADR-0014: CALLS Edge Confidence](./0014-calls-edge-confidence.md) — the ADR this amends
- [ADR-0016: Consistent Test Entity Filtering](./0016-consistent-test-entity-filtering.md) — source of
  `matches_test_pattern`
- `src/code_atlas/graph/client.py` — `_call_edge_weight`, `_combine_call_edge_facts`, `resolve_calls`
- `src/code_atlas/server/analysis.py` — `_analyze_communities`, `trace_path`, `blast_radius`
- MAGE v3.7.2 `leiden_community_detection_module.cpp` and `mg_utils.hpp` `GetNumericProperty` — the silent-1.0-fallback
  behavior this ADR designs around
