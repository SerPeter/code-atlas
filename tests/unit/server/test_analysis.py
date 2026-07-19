"""Unit tests for repository analysis module (mocked GraphBackend — no infrastructure needed).

Query construction now lives on the backend (``GraphClient``/``SqliteGraphClient`` —
see ``graph/protocol.py``'s ``GraphBackend``), so these tests mock the named
backend methods analysis.py calls rather than raw ``graph.execute()``. Query-text
correctness (e.g. "path scoping actually appears in the Cypher") is covered by
``tests/unit/graph/test_client.py`` instead — this file covers analysis.py's own
responsibility: forwarding arguments correctly and shaping/aggregating the
records a backend returns.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from code_atlas.server.analysis import _sid, analyze_repo, blast_radius, generate_diagram, trace_path

# ---------------------------------------------------------------------------
# Dependencies: cross-package coupling
# ---------------------------------------------------------------------------


def _graph_with_imports(
    direct: list[dict[str, str]],
    indirect: list[dict[str, str]] | None = None,
) -> MagicMock:
    """Fake GraphBackend for _analyze_dependencies: module import edges + (empty) external counts."""
    graph = MagicMock()
    graph.get_module_import_edges = AsyncMock(return_value={"direct": direct, "indirect": indirect or []})
    graph.get_dependency_external_counts = AsyncMock(return_value={"ext_packages": [], "ext_symbols": []})
    return graph


async def test_cross_package_coupling_uses_parent_package():
    """Coupling must group by parent package, not the shared top-level segment.

    Module qualified names are import-system dotted paths (post-S2 namespace,
    e.g. 'code_atlas.indexing.consumers'), so the first segment is identical
    for every internal module and deriving 'package' from it filters out all
    real package-to-package coupling.
    """
    graph = _graph_with_imports(
        direct=[
            {"from_mod": "code_atlas.indexing.consumers", "to_mod": "code_atlas.graph.client"},
            {"from_mod": "code_atlas.indexing.orchestrator", "to_mod": "code_atlas.graph.client"},
            {"from_mod": "code_atlas.search.engine", "to_mod": "code_atlas.graph.client"},
        ],
        indirect=[
            {"from_mod": "code_atlas.search.engine", "to_mod": "code_atlas.graph.client"},
        ],
    )

    result = await analyze_repo(graph, "dependencies", "code-atlas")

    coupling = {(e["from"], e["to"]): e["weight"] for e in result["cross_package_coupling"]}
    assert coupling == {
        ("code_atlas.indexing", "code_atlas.graph"): 2,
        ("code_atlas.search", "code_atlas.graph"): 2,
    }


async def test_cross_package_coupling_excludes_intra_package_imports():
    graph = _graph_with_imports(
        direct=[
            {"from_mod": "code_atlas.indexing.consumers", "to_mod": "code_atlas.indexing.watcher"},
            {"from_mod": "code_atlas.indexing.daemon", "to_mod": "code_atlas.indexing.watcher"},
        ],
    )

    result = await analyze_repo(graph, "dependencies", "code-atlas")

    assert result["cross_package_coupling"] == []
    # The module-level edges themselves are still reported
    assert len(result["internal_imports"]) == 2


# ---------------------------------------------------------------------------
# Dependencies / quality: circular-dependency detection must find cycles of
# any length, not just mutual A<->B pairs
# ---------------------------------------------------------------------------


async def test_dependencies_excludes_edges_touching_test_modules():
    """No file_path on module-edge records — filtering matches on the dotted
    qualified module name (e.g. 'tests.unit.foo' -> pseudo-path 'tests/unit/foo')."""
    graph = _graph_with_imports(
        direct=[
            {"from_mod": "tests.unit.test_consumers", "to_mod": "code_atlas.indexing.consumers"},
            {"from_mod": "code_atlas.search.engine", "to_mod": "code_atlas.graph.client"},
        ],
    )

    result = await analyze_repo(graph, "dependencies", "code-atlas", test_patterns=("tests/",))

    edges = {(e["from"], e["to"]) for e in result["internal_imports"]}
    assert edges == {("code_atlas.search.engine", "code_atlas.graph.client")}


async def test_circular_dependencies_detects_cycles_longer_than_two():
    """A->B->C->A is a cycle even though no pair mutually imports each other."""
    graph = _graph_with_imports(
        direct=[
            {"from_mod": "pkg.a", "to_mod": "pkg.b"},
            {"from_mod": "pkg.b", "to_mod": "pkg.c"},
            {"from_mod": "pkg.c", "to_mod": "pkg.a"},
        ],
    )

    result = await analyze_repo(graph, "dependencies", "code-atlas")

    assert result["circular_dependencies"], "3-cycle should be detected"
    cycle_members = set(result["circular_dependencies"][0]["cycle"])
    assert cycle_members == {"pkg.a", "pkg.b", "pkg.c"}


# ---------------------------------------------------------------------------
# Dependencies / structure: path scope must be forwarded to the backend
# ---------------------------------------------------------------------------


async def test_external_imports_forwards_path_scope():
    """external_imports must forward path scoping to get_dependency_external_counts,
    not report whole-project counts."""
    graph = _graph_with_imports(direct=[])

    await analyze_repo(graph, "dependencies", "code-atlas", path="src/foo")

    assert graph.get_dependency_external_counts.call_args[0] == ("code-atlas", "src/foo")


async def test_structure_external_dependencies_forwards_path_scope():
    """_analyze_structure has the same inconsistency as external_imports: fix both."""
    graph = MagicMock()
    graph.get_structure_overview = AsyncMock(
        return_value={"counts": [], "packages": [], "largest_modules": [], "external_deps": []}
    )

    await analyze_repo(graph, "structure", "code-atlas", path="src/foo")

    assert graph.get_structure_overview.call_args[0] == ("code-atlas", "src/foo", 20)


async def test_structure_excludes_test_modules_from_largest_modules():
    graph = MagicMock()
    graph.get_structure_overview = AsyncMock(
        return_value={
            "counts": [],
            "packages": [],
            "largest_modules": [
                {
                    "module": "tests.unit.test_big",
                    "qn": "tests.unit.test_big",
                    "file_path": "tests/test_big.py",
                    "entities": 100,
                },
                {"module": "pkg.real_mod", "qn": "pkg.real_mod", "file_path": "pkg/real_mod.py", "entities": 50},
            ],
            "external_deps": [],
        }
    )

    result = await analyze_repo(graph, "structure", "code-atlas", test_patterns=("test_*.py", "tests/"))

    names = {m["name"] for m in result["largest_modules"]}
    assert names == {"pkg.real_mod"}


async def test_structure_test_filtering_pads_query_limit_and_backfills():
    """A naive limit=2 fetch that's 2/2 test modules would return zero real
    results; padding the query-level limit must backfill a real 3rd candidate."""
    graph = MagicMock()
    graph.get_structure_overview = AsyncMock(
        return_value={
            "counts": [],
            "packages": [],
            "largest_modules": [
                {"module": "tests.a", "qn": "tests.a", "file_path": "tests/a.py", "entities": 100},
                {"module": "tests.b", "qn": "tests.b", "file_path": "tests/b.py", "entities": 90},
                {"module": "pkg.real", "qn": "pkg.real", "file_path": "pkg/real.py", "entities": 80},
            ],
            "external_deps": [],
        }
    )

    result = await analyze_repo(graph, "structure", "code-atlas", limit=2, test_patterns=("tests/",))

    # Query was padded beyond 2 so the 3rd (real) candidate was actually fetched.
    assert graph.get_structure_overview.call_args[0][2] > 2
    assert [m["name"] for m in result["largest_modules"]] == ["pkg.real"]


# ---------------------------------------------------------------------------
# Centrality
# ---------------------------------------------------------------------------


def _centrality_row(name: str, file_path: str, in_degree: int = 1) -> dict[str, object]:
    return {
        "name": name,
        "qn": name,
        "label": "Callable",
        "kind": "function",
        "file_path": file_path,
        "in_degree": in_degree,
        "imported_by": 0,
        "inherited_by": 0,
        "called_by": in_degree,
    }


async def test_centrality_excludes_test_modules_from_hubs_and_leaves():
    graph = MagicMock()
    graph.get_centrality_data = AsyncMock(
        return_value={
            "hubs": [_centrality_row("_invoke_tool", "tests/unit/server/test_mcp.py", in_degree=135)],
            "hub_modules": [
                {"name": "tests.conftest", "qn": "tests.conftest", "file_path": "tests/conftest.py", "imported_by": 50}
            ],
            "leaves": [_centrality_row("test_helper", "tests/helpers.py")],
        }
    )

    result = await analyze_repo(graph, "centrality", "code-atlas", test_patterns=("test_*", "tests/"))

    assert result["hub_entities"] == []
    assert result["hub_modules"] == []
    assert result["leaf_entities"] == []


async def test_centrality_test_filtering_pads_query_limit_and_backfills():
    graph = MagicMock()
    graph.get_centrality_data = AsyncMock(
        return_value={
            "hubs": [
                _centrality_row("test_a", "tests/a.py", in_degree=100),
                _centrality_row("real_hub", "pkg/real.py", in_degree=50),
            ],
            "hub_modules": [],
            "leaves": [],
        }
    )

    result = await analyze_repo(graph, "centrality", "code-atlas", limit=1, test_patterns=("test_*", "tests/"))

    assert graph.get_centrality_data.call_args[0][2] > 1
    assert [h["name"] for h in result["hub_entities"]] == ["real_hub"]


# ---------------------------------------------------------------------------
# Quality: path-scoped fan-in/fan-out must not misclassify out-of-scope
# modules that are only ever edge endpoints
# ---------------------------------------------------------------------------


def _graph_for_quality(entities: list[dict[str, object]], direct: list[dict[str, str]]) -> MagicMock:
    """Fake GraphBackend for _analyze_quality's get_quality_data call."""
    graph = MagicMock()
    graph.get_quality_data = AsyncMock(return_value={"entities": entities, "direct": direct, "indirect": []})
    return graph


async def test_quality_path_scope_does_not_score_out_of_scope_edge_endpoints():
    """An out-of-scope module reached only via an edge must not be scored.

    Without restricting the scored module set to what the path-scoped entity
    query actually matched, an out-of-scope module that's only ever an edge
    endpoint gets a fabricated fan_in/fan_out of 0 on one side, producing a
    false 'rigid' or 'unstable' flag (and a skewed health score) for a module
    the analysis never should have considered.
    """
    graph = _graph_for_quality(
        entities=[{"module": "pkg.in_scope.a", "file_path": "pkg/in_scope/a.py", "entity_count": 1}],
        direct=[
            {"from_mod": "pkg.in_scope.a", "to_mod": "pkg.external.b"},
            {"from_mod": "pkg.external.c", "to_mod": "pkg.in_scope.a"},
        ],
    )

    result = await analyze_repo(graph, "quality", "code-atlas", path="pkg/in_scope")

    rigid_modules = {m["module"] for m in result["instability"]["rigid"]}
    unstable_modules = {m["module"] for m in result["instability"]["unstable"]}
    worst_modules = {m["module"] for m in result["worst_modules"]}
    assert "pkg.external.b" not in rigid_modules
    assert "pkg.external.c" not in unstable_modules
    assert "pkg.external.b" not in worst_modules
    assert "pkg.external.c" not in worst_modules
    # The in-scope module sees both its outbound and inbound edge -> balanced
    assert "pkg.in_scope.a" not in rigid_modules
    assert "pkg.in_scope.a" not in unstable_modules


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------


def _graph_for_patterns(
    inheritance: list[dict[str, str]], enums: list[dict[str, object]], detected: list[dict[str, str]]
) -> MagicMock:
    graph = MagicMock()
    graph.get_patterns_data = AsyncMock(
        return_value={
            "inheritance": inheritance,
            "enums": enums,
            "visibility": [],
            "docstring": [],
            "detected_patterns": detected,
        }
    )
    return graph


async def test_patterns_excludes_test_modules_from_all_three_lists():
    graph = _graph_for_patterns(
        inheritance=[
            {"child": "TestBase", "child_qn": "tests.unit.TestBase", "parent": "Base", "parent_qn": "pkg.Base"},
            {"child": "Real", "child_qn": "pkg.Real", "parent": "Base", "parent_qn": "pkg.Base"},
        ],
        enums=[{"name": "TestEnum", "qn": "tests.unit.TestEnum", "file_path": "tests/unit/foo.py", "members": 3}],
        detected=[
            {"pattern_type": "HANDLES_ROUTE", "name": "test_route", "qn": "tests.unit.test_route", "target_name": "x"}
        ],
    )

    result = await analyze_repo(graph, "patterns", "code-atlas", test_patterns=("test_*", "tests/"))

    assert [i["child"] for i in result["inheritance"]] == ["Real"]
    assert result["enums"] == []
    assert result["detected_patterns"] == []


# ---------------------------------------------------------------------------
# Diagrams: packages must forward path scope
# ---------------------------------------------------------------------------


async def test_diagram_packages_forwards_path_scope():
    """generate_diagram('packages', path=...) must forward path scoping to get_diagram_packages."""
    graph = MagicMock()
    graph.get_diagram_packages = AsyncMock(return_value=[])

    await generate_diagram(graph, "packages", "code-atlas", path="src/foo")

    assert graph.get_diagram_packages.call_args[0] == ("code-atlas", "src/foo", 30)


# ---------------------------------------------------------------------------
# trace_path / blast_radius (information-retrieval family, ADR-0013)
# ---------------------------------------------------------------------------


async def test_trace_path_missing_from_node_returns_not_found():
    graph = MagicMock()
    graph.trace_path_between = AsyncMock(
        return_value={"from_exists": False, "to_exists": True, "found": False, "hop_count": None, "hops": []}
    )

    result = await trace_path(graph, "p:missing", "p:b")

    assert result["code"] == "NOT_FOUND"
    assert "p:missing" in result["error"]


async def test_trace_path_missing_to_node_returns_not_found():
    graph = MagicMock()
    graph.trace_path_between = AsyncMock(
        return_value={"from_exists": True, "to_exists": False, "found": False, "hop_count": None, "hops": []}
    )

    result = await trace_path(graph, "p:a", "p:missing")

    assert result["code"] == "NOT_FOUND"
    assert "p:missing" in result["error"]


async def test_trace_path_no_path_within_depth():
    graph = MagicMock()
    graph.trace_path_between = AsyncMock(
        return_value={"from_exists": True, "to_exists": True, "found": False, "hop_count": None, "hops": []}
    )

    result = await trace_path(graph, "p:a", "p:b", max_depth=3)

    assert result["found"] is False
    assert result["max_depth"] == 3
    assert "message" in result


async def test_trace_path_found_builds_hops_with_confidence():
    graph = MagicMock()
    graph.trace_path_between = AsyncMock(
        return_value={
            "from_exists": True,
            "to_exists": True,
            "found": True,
            "hop_count": 1,
            "hops": [
                {
                    "from": {"uid": "p:a", "name": "a"},
                    "to": {"uid": "p:b", "name": "b"},
                    "edge_type": "CALLS",
                    "confidence": "resolved",
                    "strategy": "import",
                }
            ],
        }
    )

    result = await trace_path(graph, "p:a", "p:b", max_depth=6)

    assert result["found"] is True
    assert result["hop_count"] == 1
    assert result["hops"][0]["confidence"] == "resolved"
    assert result["hops"][0]["edge_type"] == "CALLS"


async def test_blast_radius_not_found():
    graph = MagicMock()
    graph.node_exists = AsyncMock(return_value=False)

    result = await blast_radius(graph, "p:missing")

    assert result["code"] == "NOT_FOUND"


async def test_blast_radius_invalid_direction():
    graph = MagicMock()
    graph.node_exists = AsyncMock(return_value=True)

    result = await blast_radius(graph, "p:a", direction="sideways")

    assert result["code"] == "INVALID_DIRECTION"


async def test_blast_radius_flags_ambiguous_only():
    """An affected entity with no fully-resolved-edge path is flagged ambiguous_only."""
    graph = MagicMock()
    graph.node_exists = AsyncMock(return_value=True)
    graph.compute_blast_radius = AsyncMock(
        return_value=[
            {
                "uid": "p:x",
                "name": "x",
                "qualified_name": "mod.x",
                "label": "Callable",
                "file_path": "mod.py",
                "min_depth": 1,
                "direction": "out",
                "ambiguous_only": False,
            },
            {
                "uid": "p:y",
                "name": "y",
                "qualified_name": "mod.y",
                "label": "Callable",
                "file_path": "mod.py",
                "min_depth": 2,
                "direction": "out",
                "ambiguous_only": True,
            },
        ]
    )

    result = await blast_radius(graph, "p:a", direction="callees", max_depth=3)

    affected = {a["uid"]: a for a in result["affected"]}
    assert affected["p:x"]["ambiguous_only"] is False
    assert affected["p:y"]["ambiguous_only"] is True
    assert result["affected_count"] == 2


async def test_blast_radius_respects_limit_and_reports_truncated():
    graph = MagicMock()
    graph.node_exists = AsyncMock(return_value=True)
    graph.compute_blast_radius = AsyncMock(
        return_value=[
            {
                "uid": f"p:{i}",
                "name": f"n{i}",
                "qualified_name": f"mod.n{i}",
                "label": "Callable",
                "file_path": "mod.py",
                "min_depth": 1,
                "direction": "out",
                "ambiguous_only": False,
            }
            for i in range(3)
        ]
    )

    result = await blast_radius(graph, "p:a", direction="callees", limit=2)

    assert result["affected_count"] == 3
    assert len(result["affected"]) == 2
    assert result["truncated"] is True


# ---------------------------------------------------------------------------
# Dead code (ADR-0013 shortcut: find_dead_code)
# ---------------------------------------------------------------------------


async def test_dead_code_returns_zero_incoming_calls_entities():
    graph = MagicMock()
    graph.get_dead_code_candidates = AsyncMock(
        return_value=[
            {
                "name": "orphan_fn",
                "qn": "pkg.mod.orphan_fn",
                "label": "Callable",
                "kind": "function",
                "file_path": "pkg/mod.py",
                "line_start": 10,
            }
        ]
    )

    result = await analyze_repo(graph, "dead_code", "code-atlas")

    assert result["analysis"] == "dead_code"
    assert result["dead_code_count"] == 1
    assert result["dead_code"][0]["qualified_name"] == "pkg.mod.orphan_fn"


async def test_dead_code_excludes_test_pattern_matches():
    graph = MagicMock()
    graph.get_dead_code_candidates = AsyncMock(
        return_value=[
            {
                "name": "test_something",
                "qn": "tests.test_mod.test_something",
                "label": "Callable",
                "kind": "function",
                "file_path": "tests/test_mod.py",
                "line_start": 1,
            },
            {
                "name": "real_orphan",
                "qn": "pkg.mod.real_orphan",
                "label": "Callable",
                "kind": "function",
                "file_path": "pkg/mod.py",
                "line_start": 5,
            },
        ]
    )

    result = await analyze_repo(graph, "dead_code", "code-atlas", test_patterns=("test_*.py",))

    names = {c["name"] for c in result["dead_code"]}
    assert names == {"real_orphan"}


async def test_dead_code_respects_limit_and_reports_truncated():
    graph = MagicMock()
    graph.get_dead_code_candidates = AsyncMock(
        return_value=[
            {
                "name": f"orphan_{i}",
                "qn": f"pkg.mod.orphan_{i}",
                "label": "Callable",
                "kind": "function",
                "file_path": "pkg/mod.py",
                "line_start": i,
            }
            for i in range(3)
        ]
    )

    result = await analyze_repo(graph, "dead_code", "code-atlas", limit=2)

    assert result["dead_code_count"] == 3
    assert len(result["dead_code"]) == 2
    assert result["truncated"] is True


# ---------------------------------------------------------------------------
# Complexity (ADR-0013 shortcut: find_complexity_hotspots)
# ---------------------------------------------------------------------------


async def test_complexity_returns_loc_span_sorted_hotspots():
    graph = MagicMock()
    graph.get_complexity_hotspots = AsyncMock(
        return_value=[
            {
                "name": "big_fn",
                "qn": "pkg.mod.big_fn",
                "kind": "function",
                "file_path": "pkg/mod.py",
                "line_start": 10,
                "line_end": 210,
                "loc_span": 200,
            }
        ]
    )

    result = await analyze_repo(graph, "complexity", "code-atlas")

    assert result["analysis"] == "complexity"
    assert result["hotspots"][0]["loc_span"] == 200


async def test_complexity_forwards_path_scope():
    graph = MagicMock()
    graph.get_complexity_hotspots = AsyncMock(return_value=[])

    await analyze_repo(graph, "complexity", "code-atlas", path="src/foo")

    assert graph.get_complexity_hotspots.call_args[0] == ("code-atlas", "src/foo", 20)


async def test_complexity_excludes_test_modules_and_backfills_real_hotspot():
    """Regression: a naive fetch=limit would have returned only 1 hotspot (the
    real one buried past the requested limit) after filtering — same failure
    mode observed live where a test fixture ranked in the top-5 hotspots."""
    graph = MagicMock()
    graph.get_complexity_hotspots = AsyncMock(
        return_value=[
            {
                "name": "seeded_analysis_graph",
                "qn": "tests.integration.seeded_analysis_graph",
                "kind": "function",
                "file_path": "tests/integration/test_mcp.py",
                "line_start": 1,
                "line_end": 180,
                "loc_span": 180,
            },
            {
                "name": "real_fn",
                "qn": "pkg.real_fn",
                "kind": "function",
                "file_path": "pkg/mod.py",
                "line_start": 1,
                "line_end": 50,
                "loc_span": 50,
            },
        ]
    )

    result = await analyze_repo(graph, "complexity", "code-atlas", limit=1, test_patterns=("tests/",))

    assert graph.get_complexity_hotspots.call_args[0][2] > 1
    assert [h["name"] for h in result["hotspots"]] == ["real_fn"]


# ---------------------------------------------------------------------------
# Communities (ADR-0013 shortcut: find_communities, MAGE leiden_community_detection)
#
# _analyze_communities still calls graph.execute() directly (deliberate — see
# its docstring), so these tests are unchanged from before the encapsulation.
# ---------------------------------------------------------------------------


def _community_row(uid: str, community_id: int, name: str = "", label: str = "Callable") -> dict[str, object]:
    return {
        "uid": uid,
        "name": name or uid,
        "qn": uid,
        "label": label,
        "file_path": "pkg/mod.py",
        "community_id": community_id,
    }


async def test_communities_groups_and_sorts_by_size_descending():
    graph = MagicMock()
    graph.execute = AsyncMock(
        return_value=[
            _community_row("p:a", 0),
            _community_row("p:b", 0),
            _community_row("p:c", 1),
            _community_row("p:d", 1),
            _community_row("p:e", 1),
        ]
    )

    result = await analyze_repo(graph, "communities", "code-atlas")

    assert result["analysis"] == "communities"
    assert result["community_count"] == 2
    assert [c["community_id"] for c in result["communities"]] == [1, 0]
    assert [c["size"] for c in result["communities"]] == [3, 2]
    query = graph.execute.call_args[0][0]
    assert "leiden_community_detection.get" in query
    assert "project(p)" in query


async def test_communities_query_excludes_external_labels():
    """ExternalPackage/ExternalSymbol must be excluded from both edge endpoints —
    otherwise a widely-referenced external symbol (e.g. collections.abc.Coroutine)
    becomes a false hub that glues unrelated modules into one giant community."""
    graph = MagicMock()
    graph.execute = AsyncMock(return_value=[])

    await analyze_repo(graph, "communities", "code-atlas")

    query = graph.execute.call_args[0][0]
    assert "NOT a:ExternalPackage" in query
    assert "NOT a:ExternalSymbol" in query
    assert "NOT b:ExternalPackage" in query
    assert "NOT b:ExternalSymbol" in query


async def test_communities_drops_singleton_noise():
    graph = MagicMock()
    graph.execute = AsyncMock(
        return_value=[
            _community_row("p:solo", 0),
            _community_row("p:a", 1),
            _community_row("p:b", 1),
        ]
    )

    result = await analyze_repo(graph, "communities", "code-atlas")

    assert result["community_count"] == 1
    assert result["communities"][0]["community_id"] == 1
    assert result["noise_threshold"] == 2


async def test_communities_caps_members_and_communities_by_limit():
    graph = MagicMock()
    graph.execute = AsyncMock(
        return_value=[_community_row(f"p:c1_{i}", 0) for i in range(5)]
        + [_community_row(f"p:c2_{i}", 1) for i in range(3)]
    )

    result = await analyze_repo(graph, "communities", "code-atlas", limit=1)

    assert len(result["communities"]) == 1
    assert result["communities"][0]["community_id"] == 0
    assert len(result["communities"][0]["members"]) == 1


async def test_communities_excludes_test_uids_from_the_leiden_query_itself():
    """Test entities must be excluded from the a/b edge endpoints Leiden clusters
    on (query-level), not filtered from already-computed communities afterward.

    Simulates the two-phase flow: a cheap node-listing pre-query (used to compute
    which uids match test_patterns via the canonical matches_test_pattern), then
    the actual Leiden query — which, in the mock, already omits the test uids,
    matching what Memgraph would do once the NOT a.uid IN $excluded_uids clause
    is applied. If excluded_uids weren't actually computed/passed, the assertion
    on the second call's params below would fail.
    """
    graph = MagicMock()
    node_listing_rows = [
        {"uid": "p:t1", "name": "test_a", "file_path": "tests/test_a.py"},
        {"uid": "p:t2", "name": "test_b", "file_path": "tests/test_b.py"},
        {"uid": "p:r1", "name": "real_a", "file_path": "pkg/real_a.py"},
        {"uid": "p:r2", "name": "real_b", "file_path": "pkg/real_b.py"},
    ]
    leiden_rows = [
        {
            "uid": "p:r1",
            "name": "real_a",
            "qn": "pkg.real_a",
            "label": "Callable",
            "file_path": "pkg/real_a.py",
            "community_id": 1,
        },
        {
            "uid": "p:r2",
            "name": "real_b",
            "qn": "pkg.real_b",
            "label": "Callable",
            "file_path": "pkg/real_b.py",
            "community_id": 1,
        },
    ]
    graph.execute = AsyncMock(side_effect=[node_listing_rows, leiden_rows])

    result = await analyze_repo(graph, "communities", "code-atlas", test_patterns=("tests/",))

    assert graph.execute.call_count == 2
    leiden_call_params = graph.execute.call_args_list[1][0][1]
    assert set(leiden_call_params["excluded_uids"]) == {"p:t1", "p:t2"}
    assert result["community_count"] == 1
    assert {m["name"] for m in result["communities"][0]["members"]} == {"real_a", "real_b"}


async def test_communities_skips_node_listing_query_when_no_test_patterns():
    """No test_patterns means no filtering is needed — must not pay for the
    extra node-listing pre-query when it can't change anything."""
    graph = MagicMock()
    graph.execute = AsyncMock(return_value=[])

    await analyze_repo(graph, "communities", "code-atlas")

    assert graph.execute.call_count == 1


async def test_communities_drops_test_only_community_below_noise_threshold():
    """A community whose only members are test scaffolding must disappear
    entirely once filtered, not survive with an empty/tiny member list."""
    graph = MagicMock()
    graph.execute = AsyncMock(
        side_effect=[
            [
                {"uid": "p:t1", "name": "test_a", "file_path": "tests/test_a.py"},
                {"uid": "p:t2", "name": "test_b", "file_path": "tests/test_b.py"},
                {"uid": "p:r1", "name": "real_a", "file_path": "pkg/real_a.py"},
                {"uid": "p:r2", "name": "real_b", "file_path": "pkg/real_b.py"},
            ],
            [
                {
                    "uid": "p:r1",
                    "name": "real_a",
                    "qn": "pkg.real_a",
                    "label": "Callable",
                    "file_path": "pkg/real_a.py",
                    "community_id": 1,
                },
                {
                    "uid": "p:r2",
                    "name": "real_b",
                    "qn": "pkg.real_b",
                    "label": "Callable",
                    "file_path": "pkg/real_b.py",
                    "community_id": 1,
                },
            ],
        ]
    )

    result = await analyze_repo(graph, "communities", "code-atlas", test_patterns=("tests/",))

    assert result["community_count"] == 1
    assert result["communities"][0]["community_id"] == 1
    assert {m["name"] for m in result["communities"][0]["members"]} == {"real_a", "real_b"}


async def test_communities_returns_procedure_unavailable_error_when_mage_missing():
    graph = MagicMock()
    graph.execute = AsyncMock(side_effect=Exception("Unknown procedure 'leiden_community_detection.get'"))

    result = await analyze_repo(graph, "communities", "code-atlas")

    assert result["code"] == "PROCEDURE_UNAVAILABLE"
    assert "error" in result


async def test_communities_respects_path_scope():
    graph = MagicMock()
    graph.execute = AsyncMock(return_value=[])

    await analyze_repo(graph, "communities", "code-atlas", path="src/foo")

    query, params = graph.execute.call_args[0]
    assert "$path" in query
    assert params["path"] == "src/foo"


# ---------------------------------------------------------------------------
# Git signals (ADR-0013 shortcut: find_hotspots, mined by atlas mine-git-history)
# ---------------------------------------------------------------------------


def _hotspot_row(qn: str, commit_count: int, author_count: int = 2, days: float = 1.0) -> dict[str, object]:
    return {
        "name": qn,
        "qn": qn,
        "file_path": f"pkg/{qn}.py",
        "commit_count": commit_count,
        "author_count": author_count,
        "days_since_last_commit": days,
    }


async def test_git_signals_returns_hotspots():
    graph = MagicMock()
    graph.get_git_signals_data = AsyncMock(
        return_value={
            "hotspots": [_hotspot_row("hot", 42), _hotspot_row("cold", 3)],
            "bus_factor": [],
            "co_change": [],
        }
    )

    result = await analyze_repo(graph, "git_signals", "code-atlas")

    assert result["analysis"] == "git_signals"
    assert result["mined"] is True
    assert [h["qualified_name"] for h in result["hotspots"]] == ["hot", "cold"]


async def test_git_signals_bus_factor_risks():
    graph = MagicMock()
    graph.get_git_signals_data = AsyncMock(
        return_value={
            "hotspots": [_hotspot_row("solo", 10, author_count=1)],
            "bus_factor": [_hotspot_row("solo", 10, author_count=1)],
            "co_change": [],
        }
    )

    result = await analyze_repo(graph, "git_signals", "code-atlas")

    assert len(result["bus_factor_risks"]) == 1
    assert result["bus_factor_risks"][0]["author_count"] == 1


async def test_git_signals_co_change_pairs():
    graph = MagicMock()
    graph.get_git_signals_data = AsyncMock(
        return_value={
            "hotspots": [],
            "bus_factor": [],
            "co_change": [{"a_qn": "pkg.a", "a_path": "pkg/a.py", "b_qn": "pkg.b", "b_path": "pkg/b.py", "count": 5}],
        }
    )

    result = await analyze_repo(graph, "git_signals", "code-atlas")

    assert result["co_change_pairs"] == [
        {"a": "pkg.a", "a_file_path": "pkg/a.py", "b": "pkg.b", "b_file_path": "pkg/b.py", "count": 5}
    ]


async def test_git_signals_not_mined_when_no_data():
    """No hotspots at all means 'atlas mine-git-history' was never run — mined=False,
    not an error, so a caller can distinguish 'never mined' from 'mined, found nothing'.
    """
    graph = MagicMock()
    graph.get_git_signals_data = AsyncMock(return_value={"hotspots": [], "bus_factor": [], "co_change": []})

    result = await analyze_repo(graph, "git_signals", "code-atlas")

    assert result["mined"] is False
    assert result["hotspots"] == []
    assert result["bus_factor_risks"] == []
    assert result["co_change_pairs"] == []


async def test_git_signals_forwards_path_scope():
    graph = MagicMock()
    graph.get_git_signals_data = AsyncMock(return_value={"hotspots": [], "bus_factor": [], "co_change": []})

    await analyze_repo(graph, "git_signals", "code-atlas", path="src/foo")

    assert graph.get_git_signals_data.call_args[0] == ("code-atlas", "src/foo", 20, 1)


async def test_git_signals_excludes_test_files_from_all_three_lists():
    graph = MagicMock()
    graph.get_git_signals_data = AsyncMock(
        return_value={
            "hotspots": [
                {
                    "name": "test_hot",
                    "qn": "tests.test_hot",
                    "file_path": "tests/test_hot.py",
                    "commit_count": 40,
                    "author_count": 2,
                    "days_since_last_commit": 1.0,
                },
                {
                    "name": "real_hot",
                    "qn": "pkg.real_hot",
                    "file_path": "pkg/real_hot.py",
                    "commit_count": 30,
                    "author_count": 2,
                    "days_since_last_commit": 1.0,
                },
            ],
            "bus_factor": [
                {
                    "name": "test_solo",
                    "qn": "tests.test_solo",
                    "file_path": "tests/test_solo.py",
                    "commit_count": 5,
                    "author_count": 1,
                },
            ],
            "co_change": [
                {
                    "a_qn": "tests.test_a",
                    "a_path": "tests/test_a.py",
                    "b_qn": "pkg.b",
                    "b_path": "pkg/b.py",
                    "count": 3,
                },
                {"a_qn": "pkg.c", "a_path": "pkg/c.py", "b_qn": "pkg.d", "b_path": "pkg/d.py", "count": 2},
            ],
        }
    )

    result = await analyze_repo(graph, "git_signals", "code-atlas", test_patterns=("test_*", "tests/"))

    assert [h["qualified_name"] for h in result["hotspots"]] == ["pkg.real_hot"]
    assert result["bus_factor_risks"] == []
    assert result["co_change_pairs"] == [
        {"a": "pkg.c", "a_file_path": "pkg/c.py", "b": "pkg.d", "b_file_path": "pkg/d.py", "count": 2}
    ]


async def test_git_signals_mined_flag_computed_before_filtering():
    """mined signals 'did mine-git-history ever run', not 'are there non-test
    hotspots' — must stay True even if every hotspot happens to be a test file."""
    graph = MagicMock()
    graph.get_git_signals_data = AsyncMock(
        return_value={
            "hotspots": [
                {
                    "name": "test_only",
                    "qn": "tests.test_only",
                    "file_path": "tests/test_only.py",
                    "commit_count": 10,
                    "author_count": 2,
                    "days_since_last_commit": 1.0,
                }
            ],
            "bus_factor": [],
            "co_change": [],
        }
    )

    result = await analyze_repo(graph, "git_signals", "code-atlas", test_patterns=("tests/",))

    assert result["mined"] is True
    assert result["hotspots"] == []


async def test_git_signals_test_filtering_pads_query_limit_and_backfills():
    graph = MagicMock()
    graph.get_git_signals_data = AsyncMock(
        return_value={
            "hotspots": [
                {
                    "name": "test_hot",
                    "qn": "tests.test_hot",
                    "file_path": "tests/test_hot.py",
                    "commit_count": 40,
                    "author_count": 2,
                    "days_since_last_commit": 1.0,
                },
                {
                    "name": "real_hot",
                    "qn": "pkg.real_hot",
                    "file_path": "pkg/real_hot.py",
                    "commit_count": 30,
                    "author_count": 2,
                    "days_since_last_commit": 1.0,
                },
            ],
            "bus_factor": [],
            "co_change": [],
        }
    )

    result = await analyze_repo(graph, "git_signals", "code-atlas", limit=1, test_patterns=("tests/",))

    assert graph.get_git_signals_data.call_args[0][2] > 1
    assert [h["qualified_name"] for h in result["hotspots"]] == ["pkg.real_hot"]


# ---------------------------------------------------------------------------
# Diagrams: imports must not drop edges between already-kept nodes once the
# node cap is hit
# ---------------------------------------------------------------------------


async def test_imports_diagram_keeps_low_weight_edge_between_already_kept_nodes():
    """A lower-weight edge whose endpoints are already kept must survive the
    node cap — it adds no new nodes, so stopping the scan early drops it for
    no reason.
    """
    graph = MagicMock()
    graph.get_module_import_edges = AsyncMock(
        return_value={
            "direct": [
                {"from_mod": "pkg.a", "to_mod": "pkg.b"},
                {"from_mod": "pkg.a", "to_mod": "pkg.b"},
                {"from_mod": "pkg.a", "to_mod": "pkg.b"},
                {"from_mod": "pkg.c", "to_mod": "pkg.d"},
                {"from_mod": "pkg.c", "to_mod": "pkg.d"},
            ],
            "indirect": [{"from_mod": "pkg.b", "to_mod": "pkg.a"}],
        }
    )

    result = await generate_diagram(graph, "imports", "code-atlas", max_nodes=2)

    # pkg.c/pkg.d would blow the cap and are correctly excluded, but the
    # low-weight b->a edge between the two already-kept nodes must remain.
    assert result["node_count"] == 2
    assert result["mermaid"].count("-->") == 2


# ---------------------------------------------------------------------------
# Diagrams: node ID sanitization must not collide distinct qualified names
# ---------------------------------------------------------------------------


def test_sid_avoids_collisions_between_dotted_and_underscored_names():
    """'pkg.data_utils' and 'pkg.data.utils' must not sanitize to the same Mermaid ID."""
    assert _sid("pkg.data_utils") != _sid("pkg.data.utils")


def test_sid_is_deterministic_per_name():
    """The same name must always map to the same ID (declare vs. reference sites)."""
    assert _sid("pkg.mod.Foo") == _sid("pkg.mod.Foo")


# ---------------------------------------------------------------------------
# Diagrams: module_detail must forward max_nodes and never reference a class
# truncated out of the declared node set
# ---------------------------------------------------------------------------


def _graph_for_module_detail(entities: list[dict[str, object]], inherits: list[dict[str, str]]) -> MagicMock:
    """Fake GraphBackend for _diagram_module_detail's get_diagram_module_detail call."""
    graph = MagicMock()
    graph.get_diagram_module_detail = AsyncMock(
        return_value={
            "module": {"name": "mod", "qn": "pkg.mod", "uid": "proj:pkg.mod"},
            "entities": entities,
            "methods": [],
            "inherits": inherits,
        }
    )
    return graph


async def test_module_detail_forwards_max_nodes():
    """max_nodes must reach get_diagram_module_detail — it bounds the entities/
    methods/inheritance queries backend-side, so a module with large classes
    can't blow past the requested output size.
    """
    graph = _graph_for_module_detail(
        entities=[
            {"name": "Foo", "qn": "pkg.mod.Foo", "label": "TypeDef", "kind": "class", "vis": "public", "sig": None}
        ],
        inherits=[],
    )

    await generate_diagram(graph, "module_detail", "code-atlas", path="pkg/mod", max_nodes=5)

    assert graph.get_diagram_module_detail.call_args[0] == ("code-atlas", "pkg/mod", 5)


async def test_module_detail_skips_inheritance_edge_for_truncated_child():
    """A child TypeDef cut off by the max_nodes cap on entities must not get
    an inheritance edge — Mermaid would otherwise silently render an
    unlabeled node for the dangling reference.
    """
    graph = _graph_for_module_detail(
        entities=[
            {"name": "Foo", "qn": "pkg.mod.Foo", "label": "TypeDef", "kind": "class", "vis": "public", "sig": None}
        ],
        inherits=[{"child_qn": "pkg.mod.Bar", "child_name": "Bar", "parent_qn": "pkg.mod.Foo", "parent_name": "Foo"}],
    )

    result = await generate_diagram(graph, "module_detail", "code-atlas", path="pkg/mod", max_nodes=5)

    assert _sid("pkg.mod.Bar") not in result["mermaid"]
