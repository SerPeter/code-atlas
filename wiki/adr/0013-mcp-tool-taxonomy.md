# ADR-0013: MCP Tool Taxonomy — Static Analysis vs. Information Retrieval

## Status

Accepted

## Date

2026-07-17

## Context

The MCP tool surface was growing organically: `analyze_repo` is a single dispatch tool covering five repo/subgraph-wide
report shapes (`structure`, `centrality`, `dependencies`, `patterns`, `quality`), while
`get_node`/`get_context`/`hybrid_search`/`text_search`/`vector_search`/`cypher_query` are separate top-level tools
anchored at specific entities. A 2026-07-17 competitor review (Graphify, oh-my-pi, CodeGraphContext, Synaptic, repowise,
FastContext) surfaced two new tools worth adding — `trace_path` (shortest path between two entities) and `blast_radius`
(transitive closure of callers/callees) — plus a roadmap of further additions (`find_dead_code`,
`find_complexity_hotspots`, community detection, git-derived hotspots). Before adding any of them, the tool-surface
_architecture_ needed deciding: umbrella dispatch tool, named top-level tools, or some hybrid — otherwise every future
analysis re-litigates "does this get its own tool?" from scratch, and the surface either bloats into dozens of
near-duplicate tools or buries genuinely different-shaped operations inside one increasingly overloaded dispatcher.

## Decision

Two tool families, both reading the same underlying graph:

- **Static analysis** (repo/subgraph-wide batch reports, signature `{analysis, project, path, limit}` →
  `{analysis, project, ...keys, query_ms}`) — `analyze_repo`'s existing `Literal`-dispatch stays the umbrella for every
  report shape. New analyses (`dead_code`, `complexity_hotspots`, `communities`, `git_signals`) are added as new
  `Literal` values here first — one place, one report shape, no combinatorial tool growth.
- **Information retrieval** (anchored at specific entities/entity-pairs, signature varies per tool — uid, uid-pair, or
  free-text query, not `project+path+limit`) — `get_node`, `get_context`, `hybrid_search`/`text_search`/`vector_search`,
  `cypher_query`, plus the new `trace_path` and `blast_radius`. These get their own top-level `@mcp.tool` from day one,
  not `analyze_repo` sub-cases, because their signatures are genuinely different and because CodeGraphContext/Synaptic
  both converged on path-tracing/impact-radius as top-level primitives independently — burying them inside a
  `project+path+limit`-shaped dispatcher would fight discoverability for no bloat savings.
- **Shortcut-tool pattern** (resolves the bloat-vs-discoverability tension for the _static-analysis_ family only): a
  shortcut is a thin top-level `@mcp.tool` that delegates to `analyze_repo(graph, "<kind>", ...)` with the analysis
  pre-set — no duplicated logic, just a directly-nameable entry point for a high-value analysis (`find_dead_code`,
  `find_complexity_hotspots`, `find_communities`, `find_hotspots`). Applied selectively, not to every sub-case —
  `structure`/`centrality`/`dependencies`/`patterns`/`quality` stay sub-cases only, matching today; the shortcut pattern
  doesn't apply to the information-retrieval family since those tools are already top-level.

New tools follow this rule: repo/subgraph-wide batch report with a `project+path+limit`-shaped signature → new
`analyze_repo` `Literal` value (+ optional shortcut tool if it's expected to be a common, directly-nameable ask);
anchored at a specific entity/entity-pair with a genuinely different signature → new top-level tool.

## Consequences

### Positive

- One place (`analyze_repo`'s `Literal` union) to add new repo-wide reports — no per-analysis tool proliferation for the
  static-analysis family.
- Information-retrieval tools stay independently discoverable and independently documented (each has its own `@mcp.tool`
  description), matching how agents actually reach for them.
- The shortcut-tool pattern gives high-value static analyses a directly-nameable entry point without duplicating query
  logic — a shortcut is always a thin wrapper, never a second implementation.
- Future tool proposals have a place to go without re-litigating "own tool or dispatch case?" each time.

### Negative

- Two families means two conventions to remember when adding a tool; a contributor unfamiliar with this ADR might
  default to "always add a new top-level tool," re-introducing the bloat this ADR avoids.
- The shortcut pattern is judgment-based ("high-value," "expected to be a common ask") rather than a hard rule —
  different reviewers could disagree on which analyses deserve a shortcut.

### Risks

- If `analyze_repo`'s `Literal` union grows very large (10+ analyses), the umbrella tool's own description becomes
  unwieldy — may need its own follow-up ADR to split further at that point.

## Alternatives Considered

### Alternative 1: Every analysis gets its own top-level tool

- One `@mcp.tool` per analysis (`analyze_structure`, `analyze_centrality`, `find_dead_code`, `find_communities`, ...),
  no umbrella dispatcher.
- Rejected: this is exactly the combinatorial tool growth the taxonomy is meant to avoid — every future repo-wide report
  idea (and there is a long roadmap of them: dead code, complexity, communities, git signals, and more) would mint a new
  tool, most of which share the same `project+path+limit` signature and result envelope shape as the existing five
  `analyze_repo` sub-cases.

### Alternative 2: Fold trace_path/blast_radius into analyze_repo as new Literal values

- Add `"trace_path"`/`"blast_radius"` as `analyze_repo` sub-cases, reusing its `project+path+limit` signature with the
  entity uid(s) passed via an overloaded `path` parameter or new optional fields.
- Rejected: their actual inputs (uid, uid-pair, direction, max_depth, edge_types) don't fit `analyze_repo`'s
  `project+path+limit` shape at all — forcing them in would require overloading existing parameters with different
  meanings per sub-case, which is worse for discoverability and type-safety than giving them their own tool signature.

## References

- `.tasks/research/2026-07-17_competitor_consolidated-insights.md` — the competitor research that prompted this decision
  (Graphify/oh-my-pi/CodeGraphContext/Synaptic/repowise/FastContext).
- [ADR-0014](./0014-calls-edge-confidence.md) — CALLS edge confidence, consumed by `trace_path`/`blast_radius`.
- `src/code_atlas/server/analysis.py`, `src/code_atlas/server/mcp.py` — implementation.
