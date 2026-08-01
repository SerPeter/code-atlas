# ADR-0014: CALLS Edge Confidence

## Status

Accepted — amends [ADR-0008](./0008-cross-file-relationship-resolution.md); itself amended by
[ADR-0017](./0017-calls-edge-weights.md), which supersedes this ADR's rejection of a numeric confidence score (see its
Alternatives Considered below) now that MAGE's weighted Leiden, blast-radius ranking, and test provenance require a
magnitude rather than a category.

Amended by [ADR-0022](./0022-call-resolution-requires-a-grounded-receiver.md): a lone candidate is no longer sufficient
for `resolved`. An attribute call on a receiver whose type is unknown yields `candidate_count: 1` with
`confidence: "ambiguous"` — a combination this ADR did not anticipate.

## Date

2026-07-17

## Context

`_resolve_one_call`/`resolve_calls` (`graph/client.py`) resolve a parser-emitted bare call name (`to_name`, e.g.
`"run"`) against a project's Callables/TypeDefs through five ordered strategies (import match, same-class sibling,
same-file match, project-wide match, constructor call). ADR-0008 documents this cascade's general "unambiguous-only"
discipline, inherited from the same discipline IMPLEMENTS/cross-file-member resolution use.

For strategies 4 (project-wide match) and 5 (constructor call), "unambiguous-only" meant: if the bare name matched more
than one candidate, the call was left **entirely unresolved** — no CALLS edge was created at all, and the candidate set
itself was discarded, not recorded anywhere. This avoids false positives (`asyncio.run()` vs. `session.run()` are both
named `run`), but it also means a genuinely ambiguous call graph edge disappears rather than surfacing as "there's a
call here, but we're not sure of the target" — the 2026-07-17 competitor research
(`.tasks/research/2026-07-17_competitor_consolidated-insights.md`) found Graphify tags exactly this distinction with
`EXTRACTED`/`INFERRED`/`AMBIGUOUS` edge kinds instead of dropping the ambiguous case, which every downstream consumer of
the call graph (e.g. `blast_radius`, dead-code detection) benefits from being able to see and filter on.

## Decision

`_resolve_one_call` now returns **every** matching candidate for strategies 4 and 5 instead of only firing when exactly
one exists. `resolve_calls` creates a CALLS edge to each candidate, tagging every edge with two properties:

- `confidence`: `"resolved"` when the strategy found exactly one candidate, `"ambiguous"` when it found more than one
  (strategies 1–3 are unambiguous by construction and always resolve, so they're always `"resolved"`).
- `strategy`: which resolution strategy produced the edge — `"import"`, `"sibling"`, `"same_file"`, `"project_unique"`
  (strategy 4, exactly one candidate), `"project_wide"` (strategy 4, multiple candidates), or `"constructor"` (strategy
  5, one or more candidates).

No schema/DDL change is needed — `RelType` properties are set ad hoc in Cypher, matching the existing
`DOCUMENTS.link_type` precedent. Downstream consumers (e.g. `blast_radius`, future dead-code detection) can filter on
`confidence` to distinguish a real call-graph edge from a heuristic guess, instead of the data being unavailable.

## Consequences

### Positive

- Ambiguous calls are now visible in the graph instead of silently vanishing — a caller with an ambiguous `run()` call
  shows up with edges to every candidate, tagged accordingly, rather than looking like it calls nothing.
- Downstream tools (blast_radius, trace_path, future dead-code/complexity analyses) can filter or weight by `confidence`
  instead of treating every CALLS edge as equally certain.
- No schema migration, no new dependency — pure resolver + query change.

### Negative

- Ambiguous names now fan out to multiple CALLS edges per call site, inflating both edge count and naive caller/callee
  counts for commonly-named functions (`run`, `close`, `get`) — consumers that don't check `confidence` will overcount.
- Existing CALLS edges in an already-indexed graph won't have `confidence`/`strategy` until the next full reindex (same
  "parser/resolver behavior changed, existing data didn't" gap noted for prior schema-affecting changes) — not a
  schema-version bump since no DDL changed, but worth knowing when querying an old index.

### Risks

- A project with heavy use of very common short names (`get`, `run`) could see a meaningful CALLS-edge count increase;
  if this proves noisy in practice, a future revision could cap the number of ambiguous candidates materialized per call
  (not done here — no evidence yet that it's needed).

## Alternatives Considered

### Keep dropping ambiguous matches, add a separate "unresolved calls" report

Considered recording dropped candidates in a side channel (log line, separate report) instead of materializing edges.
Rejected: a side channel isn't queryable by `blast_radius`/`trace_path` or any other graph consumer — the whole point is
that the call graph itself should be able to answer "what might this affect," even under uncertainty, without a second
data source to cross-reference.

### Confidence as a float score instead of a resolved/ambiguous enum

Considered a numeric confidence score (e.g. `1.0 / len(candidates)`) instead of a two-value string enum. Rejected as
premature precision: the underlying signal is genuinely binary (the resolver either found one match or several), and
inventing a numeric scale on top of it would imply a granularity the heuristics don't actually support.

## References

- [ADR-0008: Cross-File Relationship Resolution & Qualified-Name Extensions](./0008-cross-file-relationship-resolution.md)
  — the ADR this amends
- `src/code_atlas/graph/client.py` — `_resolve_one_call`, `resolve_calls`
- [[resolve-calls-constructor-gotcha]] — prior CALLS-resolver memory note on the import_map/Callable-scoping trap
- `.tasks/research/2026-07-17_competitor_consolidated-insights.md` — Graphify's EXTRACTED/INFERRED/AMBIGUOUS tagging
