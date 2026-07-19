# ADR-0016: Consistent Test-Entity Filtering Across analyze_repo

## Status

Accepted

## Date

2026-07-19

## Context

`SearchSettings.test_filter` (default `True`) and `test_patterns` (default `["test_*", "*_test.py", "tests/",
"__tests__/"]`) already existed, and `server/analysis.py`'s `analyze_repo()` dispatcher already had a `test_patterns`
parameter — but it was only actually wired to 2 of 9 sub-analyses (`quality`, `dead_code`). `structure`, `centrality`,
`dependencies`, `patterns`, `complexity`, `communities`, and `git_signals` silently ignored it, so the "exclude tests
by default" setting's intent didn't apply to most of the static-analysis surface.

This wasn't a hypothetical gap — live testing against this repo's own indexed graph surfaced concrete, striking
fallout:

- `analyze_repo(analysis="centrality")` ranked `tests.unit.server.test_mcp._invoke_tool` (a test helper) as the
  **#3 most-central entity in the entire repo** (135 in-degree), ahead of most real production code.
- `find_complexity_hotspots` placed `tests.integration.server.test_mcp.seeded_analysis_graph`, a 180-line test
  fixture, in the **top 5 complexity hotspots** for the whole codebase.
- `find_communities`'s dominant Leiden cluster (2000+ nodes across two separate live runs) was almost entirely
  `__init__` methods from unrelated mock/fake test classes (`_FakeCtx`, `_FakeGraphForContext`, `FakeRedis`, etc.)
  clustered together purely because they're structurally similar shims mimicking the same two real classes — while a
  genuinely real, tight production cluster (`server/mcp.py`'s `_serialize_value`/`_serialize_node`/`_compact_node`
  trio) showed up as a 3-node community, easy to miss entirely.

A first pass threaded `test_patterns` through the remaining 7 sub-analyses and added a per-call `exclude_tests`
override (mirroring `hybrid_search`'s existing `exclude_tests: bool | None` semantics) to `analyze_repo`,
`find_dead_code`, `find_complexity_hotspots`, `find_communities`, and `find_hotspots`. That pass also caught and fixed
a related correctness bug: `get_structure_overview`/`get_centrality_data`/`get_patterns_data`/`get_git_signals_data`
apply `LIMIT` at the Cypher/SQL level, before Python-side filtering ran — naively filtering after a query that
already capped itself at `limit` rows would silently return fewer than requested whenever a filtered entity occupied
one of those slots, instead of backfilling from real candidates beyond the original cutoff. Fixed by padding the
query-level fetch (5x, capped at 200) whenever filtering is active, then re-truncating to the caller's limit
afterward.

That first pass shipped `communities` filtering as a documented, explicit known limitation: entities were dropped
from already-computed community membership lists (post-hoc), not excluded from what Leiden actually clustered on.
This meant test-node connectivity could still act as a bridge gluing otherwise-unrelated production communities
together — filtering hid the symptom (test names in the output) without fixing the cause (test nodes shaping which
production entities got grouped together in the first place). The user asked for the complete fix.

## Decision

**Thread `test_patterns` through all 9 `analyze_repo` sub-analyses**, filtering each one's ranked/listed output
(hub entities, largest modules, complexity hotspots, community members, git-signal hotspots, dependency edges,
inheritance/enum/pattern records) via the single canonical `matches_test_pattern` helper (`search/engine.py`) already
used by `hybrid_search`. Whole-repo aggregate counts that don't rank individual entities (`structure`'s
`label_counts`/`kind_counts`, `patterns`' `visibility_distribution`/`docstring_coverage`) are intentionally left
unfiltered — those describe total repo composition, not "notable" entities, and filtering them would require pushing
filtering into the SQL/Cypher aggregation itself.

**Pad query-level `LIMIT` fetches (5x, capped at 200) whenever `test_patterns` filtering is active**, then
re-truncate every output list to the caller's requested `limit` after filtering — prevents the under-delivery bug
described above across `structure`, `centrality`, `patterns`, `complexity`, and `git_signals`.

**Exclude test entities from Leiden's input graph at the Cypher `WHERE`-clause level, not from its output.** Since
Cypher has no glob/fnmatch operator, this is done as a two-phase query, only when `test_patterns` is non-empty:

1. A cheap node-listing pre-query fetches `{uid, name, file_path}` for all in-scope nodes (same project/path scope,
   already excluding `ExternalPackage`/`ExternalSymbol` since those are excluded from the main query regardless).
2. That list is filtered in Python via the same canonical `matches_test_pattern` used everywhere else, producing a
   set of excluded uids.
3. The excluded uids are passed as a parameterized list into the main Leiden query's `WHERE` clause
   (`AND NOT a.uid IN $excluded_uids AND NOT b.uid IN $excluded_uids`) — the same "exclude before projecting the
   subgraph" technique already used for `ExternalPackage`/`ExternalSymbol` node labels.

Test nodes can no longer appear as `a`/`b` edge endpoints in the subgraph Leiden clusters on at all, so their
connectivity cannot bridge unrelated production communities together. The post-hoc filtering step in the
result-grouping loop is removed as dead code — no test node can reach it anymore. The extra node-listing query only
runs when filtering is actually active (zero overhead on the default path).

Add a per-call `exclude_tests: bool | None = None` override to `analyze_repo`, `find_dead_code`,
`find_complexity_hotspots`, `find_communities`, and `find_hotspots`, resolved identically to `hybrid_search`'s
existing parameter (`settings.test_filter if exclude_tests is None else exclude_tests`) via a new shared
`_resolve_test_patterns` helper in `server/mcp.py`.

## Consequences

### Positive

- `centrality`/`complexity`/`communities`/`structure`/`git_signals` results now reflect production code by default,
  matching what a user actually expects from "code intelligence," not test-scaffolding volume.
- Community detection now produces genuinely more meaningful production clusters — test-mock connectivity can't
  bridge unrelated subsystems into one meaningless giant "community" anymore.
- One canonical pattern-matching implementation (`matches_test_pattern`) across every filtering site — no risk of a
  second, Cypher-regex-translated implementation silently drifting from it.
- An agent that explicitly wants test-inclusive results (e.g. analyzing test-suite health itself) can still get them
  via `exclude_tests=False`, per tool, without touching global settings.

### Negative

- `find_communities` now costs one extra lightweight query round-trip when `test_patterns` filtering is active
  (the default) — a node-listing scan, cheap relative to the Leiden computation itself, but non-zero.
- The 5x/200-cap padding heuristic for `LIMIT`-based analyses is a heuristic, not a guarantee — a request scoped to a
  path with an extremely test-dense candidate pool could still under-deliver in pathological cases. Accepted as a
  large practical improvement over zero headroom, not a formal correctness proof.

### Risks

- `dependencies`' edge filtering and `patterns`' inheritance/detected-pattern filtering match on a qualified-name-
  derived pseudo-path (dots converted to slashes) rather than a real `file_path`, since those records don't carry one
  — this is a reasonable approximation but not identical to `matches_test_pattern`'s real-file-path behavior
  elsewhere; a qualified name shaped unusually could misclassify.

## Alternatives Considered

### Alternative 1: Translate glob patterns into Cypher regex (`=~`) and filter server-side directly

- Cypher/Memgraph supports a regex match operator, and Python's `fnmatch.translate()` converts a glob pattern into a
  regex string — tempting to embed that directly into the Leiden query's `WHERE` clause without a second query
  round-trip.
- Rejected: `fnmatch.translate()`'s output uses Python-`re`-specific regex syntax (e.g. `(?s:...)\Z`) not guaranteed
  portable to Memgraph's regex engine, and a hand-rolled glob-to-Cypher-regex translator would be a second
  pattern-matching implementation that could silently drift from `matches_test_pattern` — exactly the risk the
  two-query approach avoids by reusing one canonical implementation everywhere.

### Alternative 2: Leave communities' post-hoc filtering as a documented, permanent limitation

- The pragmatic first-pass choice — cheaper (no extra query), and still measurably improves the output (a
  test-only community drops below the noise threshold and disappears).
- Rejected once asked to "properly fix" it: post-hoc filtering doesn't stop test-node connectivity from shaping
  which _production_ entities get grouped together by Leiden in the first place, which is the more consequential
  half of the original problem (the 3-node real `mcp.py` community staying buried, not just test names showing up in
  a member list).

## References

- [ADR-0013: MCP Tool Taxonomy](./0013-mcp-tool-taxonomy.md) — `find_communities`/`find_complexity_hotspots`/etc. as
  shortcut tools over `analyze_repo`.
- [ADR-0015: Embedded Backend Option](./0015-embedded-backend-option.md) — `_analyze_communities`'s existing
  `SqliteGraphClient` guard and `ExternalPackage`/`ExternalSymbol` exclusion pattern, reused here for test entities.
- `src/code_atlas/search/engine.py` — `matches_test_pattern`, the single canonical implementation this decision
  reuses rather than duplicating.
- `src/code_atlas/server/analysis.py`, `src/code_atlas/server/mcp.py` — the modules this decision changes.
