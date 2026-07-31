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

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from code_atlas.backends.sqlite_graph import SqliteGraphClient
from code_atlas.server.analysis import (
    _detect_module_communities,
    _modularity,
    _sid,
    analyze_repo,
    blast_radius,
    generate_diagram,
    trace_path,
)

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


async def test_trace_path_surfaces_path_weight():
    """path_weight is what broke the tie between equal-length paths — surface it
    so the choice is inspectable rather than opaque."""
    graph = MagicMock()
    graph.trace_path_between = AsyncMock(
        return_value={
            "from_exists": True,
            "to_exists": True,
            "found": True,
            "hop_count": 2,
            "hops": [],
            "path_weight": 0.1250001,
        }
    )

    result = await trace_path(graph, "p:a", "p:b")

    assert result["path_weight"] == 0.125


async def test_trace_path_tolerates_a_backend_without_path_weight():
    graph = MagicMock()
    graph.trace_path_between = AsyncMock(
        return_value={"from_exists": True, "to_exists": True, "found": True, "hop_count": 1, "hops": []}
    )

    result = await trace_path(graph, "p:a", "p:b")

    assert result["path_weight"] is None


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


def _affected(uid: str, **overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "uid": uid,
        "name": uid,
        "qualified_name": f"mod.{uid}",
        "label": "Callable",
        "file_path": "mod.py",
        "min_depth": 1,
        "direction": "in",
        "ambiguous_only": False,
        "confidence_score": 1.0,
        "test_only": False,
    }
    entry.update(overrides)
    return entry


async def test_blast_radius_ranks_production_impact_above_test_only_callers():
    """Same depth, same score — the caller only test code reaches sorts last,
    even though its qualified_name would sort first alphabetically."""
    graph = MagicMock()
    graph.node_exists = AsyncMock(return_value=True)
    graph.compute_blast_radius = AsyncMock(
        return_value=[
            _affected("aaa", test_only=True, confidence_score=0.25),
            _affected("zzz"),
        ]
    )

    result = await blast_radius(graph, "p:a")

    assert [a["uid"] for a in result["affected"]] == ["zzz", "aaa"]


async def test_blast_radius_ranks_higher_confidence_first_within_a_depth():
    graph = MagicMock()
    graph.node_exists = AsyncMock(return_value=True)
    graph.compute_blast_radius = AsyncMock(
        return_value=[
            _affected("aaa", confidence_score=0.2, ambiguous_only=True),
            _affected("zzz", confidence_score=1.0),
        ]
    )

    result = await blast_radius(graph, "p:a")

    assert [a["uid"] for a in result["affected"]] == ["zzz", "aaa"]


async def test_blast_radius_keeps_depth_as_the_primary_sort_key():
    """Confidence ranking must not promote a distant high-confidence entity above
    a nearer one — depth is still what "blast radius" means."""
    graph = MagicMock()
    graph.node_exists = AsyncMock(return_value=True)
    graph.compute_blast_radius = AsyncMock(
        return_value=[
            _affected("far", min_depth=3, confidence_score=1.0),
            _affected("near", min_depth=1, confidence_score=0.05, test_only=True),
        ]
    )

    result = await blast_radius(graph, "p:a")

    assert [a["uid"] for a in result["affected"]] == ["near", "far"]


async def test_blast_radius_tolerates_entries_without_the_new_confidence_fields():
    """A backend (or graph) predating the weighting amendment returns entries with
    no confidence_score/test_only — those must rank as neutral, not crash."""
    graph = MagicMock()
    graph.node_exists = AsyncMock(return_value=True)
    legacy = _affected("legacy")
    del legacy["confidence_score"]
    del legacy["test_only"]
    graph.compute_blast_radius = AsyncMock(return_value=[legacy, _affected("scored", confidence_score=0.1)])

    result = await blast_radius(graph, "p:a")

    assert [a["uid"] for a in result["affected"]] == ["legacy", "scored"]


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
# Communities (ADR-0013 shortcut: find_communities)
#
# Clustering happens at MODULE granularity over a graph built in Python, so
# these tests feed the two raw reads _fetch_community_inputs performs (module
# inventory, then module-pair CALLS weights) through graph.execute's side_effect
# and the module-to-module IMPORTS through the get_module_import_edges backend
# method. Only the two reads are Memgraph-specific; everything asserted below
# (aggregation, weighting, clustering, filtering) is pure Python.
# ---------------------------------------------------------------------------


def _mod_row(qn: str, file_path: str = "", name: str = "") -> dict[str, Any]:
    return {
        "uid": f"proj:{qn}",
        "name": name or qn.rsplit(".", 1)[-1],
        "qn": qn,
        "file_path": file_path or (qn.replace(".", "/") + ".py"),
    }


def _call_row(from_path: str, to_path: str, weight: float) -> dict[str, Any]:
    return {"from_path": from_path, "to_path": to_path, "weight": weight}


def _community_graph(
    modules: list[dict[str, Any]],
    calls: list[dict[str, Any]] | None = None,
    direct: list[dict[str, str]] | None = None,
    indirect: list[dict[str, str]] | None = None,
) -> MagicMock:
    graph = MagicMock()
    graph.execute = AsyncMock(side_effect=[modules, calls or []])
    graph.get_module_import_edges = AsyncMock(return_value={"direct": direct or [], "indirect": indirect or []})
    return graph


def _path_of(qn: str) -> str:
    return qn.replace(".", "/") + ".py"


def _clique(qns: list[str], weight: float) -> list[dict[str, Any]]:
    """CALLS rows joining every pair in *qns* (one direction) at *weight*."""
    return [_call_row(_path_of(a), _path_of(b), weight) for i, a in enumerate(qns) for b in qns[i + 1 :]]


async def test_communities_cluster_modules_not_callables():
    """The unit of a subsystem is a module.

    Callable-level CALLS rows are attributed to the module owning each endpoint
    (via file_path) before anything is clustered — at callable granularity the
    call graph is one dense giant component and the partition is useless.
    """
    modules = [_mod_row(qn) for qn in ("pkg.a1", "pkg.a2", "pkg.a3", "pkg.b1", "pkg.b2", "pkg.b3")]
    calls = _clique(["pkg.a1", "pkg.a2", "pkg.a3"], 10.0) + _clique(["pkg.b1", "pkg.b2", "pkg.b3"], 10.0)
    calls.append(_call_row(_path_of("pkg.a1"), _path_of("pkg.b1"), 0.25))
    graph = _community_graph(modules, calls)

    result = await analyze_repo(graph, "communities", "proj")

    assert result["analysis"] == "communities"
    assert result["granularity"] == "module"
    assert result["community_count"] == 2
    assert {m["label"] for c in result["communities"] for m in c["members"]} == {"Module"}
    grouped = [{m["qualified_name"] for m in c["members"]} for c in result["communities"]]
    assert {"pkg.a1", "pkg.a2", "pkg.a3"} in grouped
    assert {"pkg.b1", "pkg.b2", "pkg.b3"} in grouped


async def test_communities_query_sums_the_numeric_calls_weight():
    """A silently-unweighted run is indistinguishable from a weighted one by output.

    The summation happens database-side (one row per module pair), so pin the
    aggregate itself: ``sum`` of the *numeric* ``weight`` property, defaulting to
    ``1.0`` (``_CALL_WEIGHT_BASE``) for edges written before the ADR-0014
    weighting amendment. Summing — not averaging, not counting rows — is what
    makes a module pair joined by many confident calls outrank one joined by a
    single ambiguous or test-provenance call. ``confidence`` is a *string* and
    must never appear as the aggregated property.
    """
    graph = _community_graph([_mod_row("pkg.a")])

    await analyze_repo(graph, "communities", "proj")

    calls_query = graph.execute.call_args_list[1][0][0]
    assert "sum(coalesce(r.weight, 1.0))" in calls_query
    assert "confidence" not in calls_query


async def test_greedy_modularity_is_weight_sensitive():
    """Proof the maximizer reads weights rather than edge presence.

    Same 4-node path either way: with weights 10/10/0.1 the near-zero last hop
    cannot pay for its own community, so all four modules stay together; with
    every edge at 1.0 the identical topology splits in two. If weights were
    dropped anywhere between aggregation and clustering, both calls would return
    the unweighted partition.
    """
    weighted = {("a", "b"): 10.0, ("b", "c"): 10.0, ("c", "d"): 0.1}
    unweighted = dict.fromkeys(weighted, 1.0)

    assert _detect_module_communities({"a", "b", "c", "d"}, weighted) == [["a", "b", "c", "d"]]
    assert _detect_module_communities({"a", "b", "c", "d"}, unweighted) == [["a", "b"], ["c", "d"]]


async def test_communities_are_deterministic_across_identical_calls():
    """Determinism is the reason this isn't MAGE's Leiden.

    Leiden is documented non-deterministic, so two identical calls could return
    different partitions and no two runs could be diffed. Greedy modularity with
    a lexicographic tie-break must be byte-stable.
    """
    modules = [_mod_row(f"pkg.m{i}") for i in range(8)]
    calls = _clique([f"pkg.m{i}" for i in range(4)], 5.0) + _clique([f"pkg.m{i}" for i in range(4, 8)], 5.0)
    calls.append(_call_row(_path_of("pkg.m0"), _path_of("pkg.m4"), 0.1))

    runs = []
    for _ in range(2):
        result = await analyze_repo(_community_graph(list(modules), list(calls)), "communities", "proj")
        result.pop("query_ms")
        runs.append(json.dumps(result, sort_keys=True))

    assert runs[0] == runs[1]


async def test_communities_fold_reciprocal_and_import_edges_into_one_module_pair():
    """a->b, b->a and an a/b IMPORTS edge are one undirected module pair, not three.

    Modularity is defined on undirected graphs; leaving parallel edges separate
    would double-count the same coupling. Their weights add.
    """
    modules = [_mod_row("pkg.a"), _mod_row("pkg.b")]
    calls = [
        _call_row(_path_of("pkg.a"), _path_of("pkg.b"), 3.0),
        _call_row(_path_of("pkg.b"), _path_of("pkg.a"), 2.0),
    ]
    graph = _community_graph(modules, calls, direct=[{"from_mod": "pkg.a", "to_mod": "pkg.b"}])

    result = await analyze_repo(graph, "communities", "proj")

    assert result["edge_count"] == 1
    assert result["community_count"] == 1


async def test_communities_include_module_to_module_imports_without_any_calls():
    """IMPORTS is the other half of the projection.

    ``get_module_import_edges`` already resolves Module->symbol edges back
    through DEFINES to the defining module (the ``indirect`` list) — that
    resolution is what makes IMPORTS genuine module-to-module structure instead
    of a hub through every shared symbol.
    """
    modules = [_mod_row("pkg.a"), _mod_row("pkg.b")]
    graph = _community_graph(modules, indirect=[{"from_mod": "pkg.a", "to_mod": "pkg.b"}])

    result = await analyze_repo(graph, "communities", "proj")

    assert result["edge_count"] == 1
    assert result["community_count"] == 1
    assert {m["qualified_name"] for m in result["communities"][0]["members"]} == {"pkg.a", "pkg.b"}


async def test_communities_ignore_import_edges_pointing_out_of_scope():
    """get_module_import_edges only path-filters the importing side, so a scoped
    call can return edges whose target module isn't in the inventory. Those must
    not conjure a phantom node into the clustered graph."""
    graph = _community_graph(
        [_mod_row("pkg.a"), _mod_row("pkg.b")],
        direct=[{"from_mod": "pkg.a", "to_mod": "other.z"}, {"from_mod": "pkg.a", "to_mod": "pkg.b"}],
    )

    result = await analyze_repo(graph, "communities", "proj")

    assert result["module_count"] == 2
    assert result["edge_count"] == 1


async def test_communities_drop_intra_module_calls():
    """Calls between two entities in the same module say how cohesive that module
    is internally, not which modules belong together."""
    graph = _community_graph(
        [_mod_row("pkg.a"), _mod_row("pkg.b")],
        [_call_row(_path_of("pkg.a"), _path_of("pkg.a"), 50.0)],
    )

    result = await analyze_repo(graph, "communities", "proj")

    assert result["edge_count"] == 0
    assert result["communities"] == []


async def test_communities_exclude_test_modules_from_the_clustered_graph():
    """Test modules must be gone before the graph is built, not filtered from the
    result — otherwise a test module that exercises two unrelated production
    subsystems bridges them into one community (see ADR-0016).

    Here tests/test_bridge.py exercises all of pkg.a* and pkg.b* — heavier
    coupling than either production pair has with itself, so with it in the graph
    the whole thing is one community and the two subsystems disappear.
    """
    modules = [_mod_row(qn) for qn in ("pkg.a1", "pkg.a2", "pkg.b1", "pkg.b2")]
    modules.append(_mod_row("tests.test_bridge", file_path="tests/test_bridge.py", name="test_bridge"))
    calls = [
        _call_row(_path_of("pkg.a1"), _path_of("pkg.a2"), 1.0),
        _call_row(_path_of("pkg.b1"), _path_of("pkg.b2"), 1.0),
    ] + [_call_row("tests/test_bridge.py", _path_of(qn), 5.0) for qn in ("pkg.a1", "pkg.a2", "pkg.b1", "pkg.b2")]

    unfiltered = await analyze_repo(_community_graph(list(modules), list(calls)), "communities", "proj")
    assert unfiltered["module_count"] == 5
    assert unfiltered["community_count"] == 1, "fixture must actually bridge, or the test proves nothing"

    filtered = await analyze_repo(
        _community_graph(list(modules), list(calls)), "communities", "proj", test_patterns=("tests/",)
    )

    assert filtered["module_count"] == 4
    assert filtered["community_count"] == 2
    assert all(not m["file_path"].startswith("tests/") for c in filtered["communities"] for m in c["members"])


async def test_communities_drop_isolated_modules_as_noise():
    """Config/manifest pseudo-modules and unreferenced leaves have no edges at
    all — they'd otherwise flood the output as singleton 'communities'."""
    graph = _community_graph(
        [_mod_row("pkg.a"), _mod_row("pkg.b"), _mod_row("pyproject_toml", file_path="pyproject.toml")],
        [_call_row(_path_of("pkg.a"), _path_of("pkg.b"), 4.0)],
    )

    result = await analyze_repo(graph, "communities", "proj")

    assert result["module_count"] == 3
    assert result["community_count"] == 1
    assert result["noise_threshold"] == 2
    assert {m["qualified_name"] for m in result["communities"][0]["members"]} == {"pkg.a", "pkg.b"}


async def test_communities_caps_members_and_communities_by_limit():
    modules = [_mod_row(f"pkg.a{i}") for i in range(5)] + [_mod_row(f"pkg.b{i}") for i in range(3)]
    calls = _clique([f"pkg.a{i}" for i in range(5)], 10.0) + _clique([f"pkg.b{i}" for i in range(3)], 10.0)
    graph = _community_graph(modules, calls)

    result = await analyze_repo(graph, "communities", "proj", limit=1)

    assert len(result["communities"]) == 1
    assert result["communities"][0]["size"] == 5, "size reports full membership, limit only caps what's listed"
    assert len(result["communities"][0]["members"]) == 1


async def test_communities_report_an_unclusterable_scope_as_empty_not_an_error():
    graph = _community_graph([_mod_row("pkg.a")])

    result = await analyze_repo(graph, "communities", "proj")

    assert result["communities"] == []
    assert "code" not in result, "a scope with no module-to-module edges is not an error condition"
    assert "no communities detected" in result["note"].lower()


async def test_communities_respect_path_scope():
    graph = _community_graph([_mod_row("pkg.a")])

    await analyze_repo(graph, "communities", "proj", path="src/foo")

    module_query, module_params = graph.execute.call_args_list[0][0]
    calls_query, _ = graph.execute.call_args_list[1][0]
    assert "m.file_path STARTS WITH $path" in module_query
    assert "a.file_path STARTS WITH $path AND b.file_path STARTS WITH $path" in calls_query
    assert module_params["path"] == "src/foo"
    graph.get_module_import_edges.assert_awaited_once_with("proj", "src/foo")


async def test_communities_report_modularity_of_the_full_partition():
    """Q is computed before the noise cut so it stays comparable across calls —
    dropping singletons would otherwise inflate it as the repo grows leaves."""
    modules = [_mod_row("pkg.a"), _mod_row("pkg.b"), _mod_row("pkg.c"), _mod_row("pkg.d")]
    calls = [
        _call_row(_path_of("pkg.a"), _path_of("pkg.b"), 10.0),
        _call_row(_path_of("pkg.c"), _path_of("pkg.d"), 10.0),
        _call_row(_path_of("pkg.b"), _path_of("pkg.c"), 0.1),
    ]
    graph = _community_graph(modules, calls)

    result = await analyze_repo(graph, "communities", "proj")

    assert result["community_count"] == 2
    expected_edges = {("pkg.a", "pkg.b"): 10.0, ("pkg.c", "pkg.d"): 10.0, ("pkg.b", "pkg.c"): 0.1}
    assert result["modularity"] == round(_modularity([["pkg.a", "pkg.b"], ["pkg.c", "pkg.d"]], expected_edges), 4)


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


# ---------------------------------------------------------------------------
# Module summary (analyze_repo(analysis="module_summary"))
# ---------------------------------------------------------------------------


def _summary_entity(
    qn: str,
    *,
    name: str | None = None,
    label: str = "Callable",
    kind: str = "function",
    vis: str = "public",
    sig: str | None = None,
    docstring: str | None = None,
    line_start: int = 1,
    line_end: int | None = None,
    file_path: str = "pkg/mod.py",
    parent_qn: str | None = None,
) -> dict[str, Any]:
    return {
        "uid": f"proj:{qn}",
        "name": name or qn.rsplit(".", 1)[-1],
        "qn": qn,
        "label": label,
        "kind": kind,
        "vis": vis,
        "sig": sig,
        "docstring": docstring,
        "line_start": line_start,
        "line_end": line_end,
        "file_path": file_path,
        "parent_qn": parent_qn,
    }


def _graph_for_module_summary(**overrides: Any) -> MagicMock:
    """Fake GraphBackend returning a canned get_module_summary payload."""
    payload: dict[str, list[dict[str, Any]]] = {
        "modules": [],
        "entities": [],
        "internal_edges": [],
        "fan_in": [],
        "fan_out": [],
        "docs": [],
    }
    payload.update(overrides)
    graph = MagicMock()
    graph.get_module_summary = AsyncMock(return_value=payload)
    return graph


async def test_module_summary_requires_a_path():
    """Fan-in/fan-out mean "exactly one endpoint outside path" — with no path
    everything is in scope and the boundary would be empty by construction.
    """
    graph = _graph_for_module_summary()

    result = await analyze_repo(graph, "module_summary", "proj")

    assert result["code"] == "PATH_REQUIRED"
    graph.get_module_summary.assert_not_awaited()


async def test_module_summary_not_found_when_path_matches_nothing():
    graph = _graph_for_module_summary()

    result = await analyze_repo(graph, "module_summary", "proj", path="pkg/nope")

    assert result["code"] == "NOT_FOUND"
    assert "pkg/nope" in result["error"]


async def test_module_summary_scales_limit_into_entity_and_edge_budgets():
    """analyze_repo's shared limit (<=100) is a per-section knob, not an entity
    budget — module_summary multiplies it before it reaches the backend.
    """
    graph = _graph_for_module_summary(modules=[{"qn": "pkg.mod", "name": "mod", "file_path": "pkg/mod.py"}])

    await analyze_repo(graph, "module_summary", "proj", path="pkg", limit=20)

    assert graph.get_module_summary.call_args[0] == ("proj", "pkg", 200, 600)


async def test_module_summary_emits_signature_visibility_span_and_first_doc_line_only():
    graph = _graph_for_module_summary(
        modules=[{"qn": "pkg.mod", "name": "mod", "file_path": "pkg/mod.py", "docstring": "Module blurb.\n\nMore."}],
        entities=[
            _summary_entity(
                "pkg.mod.run",
                sig="def run(self,\n        x: int) -> str",
                docstring="Do the thing.\n\nLong explanation nobody needs here.",
                line_start=10,
                line_end=42,
            ),
            _summary_entity("pkg.mod._helper", vis="private", sig="def _helper()", line_start=50),
        ],
    )

    outline = (await analyze_repo(graph, "module_summary", "proj", path="pkg"))["outline"]

    assert "+ def run(self, x: int) -> str L10-42  # Do the thing." in outline
    assert "- def _helper() L50" in outline
    assert "Long explanation nobody needs here." not in outline
    assert "More." not in outline
    assert "# Module blurb." in outline


async def test_module_summary_indents_class_members_under_their_class():
    graph = _graph_for_module_summary(
        modules=[{"qn": "pkg.mod", "name": "mod", "file_path": "pkg/mod.py", "docstring": None}],
        entities=[
            _summary_entity("pkg.mod.Widget", label="TypeDef", kind="class", sig=None, line_start=1, line_end=30),
            _summary_entity("pkg.mod.Widget.draw", kind="method", sig="def draw(self)", parent_qn="pkg.mod.Widget"),
            _summary_entity("pkg.mod.free_fn", sig="def free_fn()", parent_qn="pkg.mod"),
        ],
    )

    outline = (await analyze_repo(graph, "module_summary", "proj", path="pkg"))["outline"]

    assert "  + class Widget L1-30" in outline
    assert "    + def draw(self) L1" in outline
    assert "  + def free_fn() L1" in outline


async def test_module_summary_collapses_adjacency_and_relativizes_names():
    graph = _graph_for_module_summary(
        modules=[
            {"qn": "pkg.a", "name": "a", "file_path": "pkg/a.py", "docstring": None},
            {"qn": "pkg.b", "name": "b", "file_path": "pkg/b.py", "docstring": None},
        ],
        entities=[_summary_entity("pkg.a.caller", sig="def caller()", file_path="pkg/a.py")],
        internal_edges=[
            {"from_qn": "pkg.a.caller", "to_qn": "pkg.b.one", "rel_type": "CALLS", "props": {}},
            {"from_qn": "pkg.a.caller", "to_qn": "pkg.b.two", "rel_type": "CALLS", "props": {}},
        ],
    )

    result = await analyze_repo(graph, "module_summary", "proj", path="pkg")

    assert "NAMES below are relative to pkg" in result["outline"]
    # One line per source, not one per edge.
    assert "    a.caller > b.one, b.two" in result["outline"]
    assert result["internal_edge_count"] == 2


async def test_module_summary_reports_fan_in_and_fan_out_boundary():
    graph = _graph_for_module_summary(
        modules=[{"qn": "pkg.mod", "name": "mod", "file_path": "pkg/mod.py", "docstring": None}],
        entities=[_summary_entity("pkg.mod.api", sig="def api()")],
        fan_in=[
            {
                "from_qn": "other.cli.main",
                "from_name": "main",
                "from_path": "other/cli.py",
                "from_label": "Callable",
                "to_qn": "pkg.mod.api",
                "rel_type": "CALLS",
                "props": {},
            }
        ],
        fan_out=[
            {
                "from_qn": "pkg.mod.api",
                "to_qn": "requests",
                "to_name": "requests",
                "to_path": None,
                "to_label": "ExternalPackage",
                "rel_type": "IMPORTS",
                "props": {},
            }
        ],
    )

    result = await analyze_repo(graph, "module_summary", "proj", path="pkg")

    # Single in-scope module, so the shared prefix is its full qn and in-scope
    # names shrink to bare locals; out-of-scope names stay fully qualified.
    assert "NAMES below are relative to pkg.mod" in result["outline"]
    assert "FAN-IN" in result["outline"]
    assert "    api < other.cli.main" in result["outline"]
    assert "FAN-OUT" in result["outline"]
    # External targets are marked so an agent does not hunt for them in the repo.
    assert "    api > requests*" in result["outline"]
    assert (result["fan_in_count"], result["fan_out_count"]) == (1, 1)


async def test_module_summary_passes_through_unknown_edge_properties():
    """Edge annotations are value-driven, not a hardcoded key list: neutral values
    (confidence='resolved', weight 1, false flags) cost no tokens, and any other
    property — including ones added after this code was written — is rendered.
    """
    graph = _graph_for_module_summary(
        modules=[{"qn": "pkg.mod", "name": "mod", "file_path": "pkg/mod.py", "docstring": None}],
        entities=[_summary_entity("pkg.mod.a", sig="def a()")],
        internal_edges=[
            {
                "from_qn": "pkg.mod.a",
                "to_qn": "pkg.mod.b",
                "rel_type": "CALLS",
                "props": {"confidence": "ambiguous", "strategy": "name_match", "candidate_count": 3},
            },
            {
                "from_qn": "pkg.mod.a",
                "to_qn": "pkg.mod.c",
                "rel_type": "CALLS",
                "props": {"confidence": "resolved", "weight": 1.0, "from_test": False},
            },
            {
                "from_qn": "pkg.mod.a",
                "to_qn": "pkg.mod.d",
                "rel_type": "CALLS",
                "props": {"weight": 0.25, "from_test": True, "some_future_prop": "xyz"},
            },
        ],
    )

    outline = (await analyze_repo(graph, "module_summary", "proj", path="pkg"))["outline"]

    assert "b[candidate_count=3 confidence=ambiguous strategy=name_match]" in outline
    assert ", c," in outline
    assert "c[" not in outline
    assert "d[from_test=True some_future_prop=xyz weight=0.25]" in outline


async def test_module_summary_filters_test_callers_but_not_in_scope_entities():
    """The caller named the path explicitly, so in-scope entities are never
    filtered (summarizing a test package must work); test scaffolding is only
    dropped from the boundary lists, where it otherwise swamps fan-in.
    """
    graph = _graph_for_module_summary(
        modules=[{"qn": "tests.unit.test_thing", "name": "test_thing", "file_path": "tests/unit/test_thing.py"}],
        entities=[
            _summary_entity(
                "tests.unit.test_thing.test_it",
                sig="def test_it()",
                file_path="tests/unit/test_thing.py",
            )
        ],
        fan_in=[
            {
                "from_qn": "tests.unit.test_other.test_x",
                "from_name": "test_x",
                "from_path": "tests/unit/test_other.py",
                "from_label": "Callable",
                "to_qn": "tests.unit.test_thing.test_it",
                "rel_type": "CALLS",
                "props": {},
            },
            {
                "from_qn": "pkg.mod.prod_caller",
                "from_name": "prod_caller",
                "from_path": "pkg/mod.py",
                "from_label": "Callable",
                "to_qn": "tests.unit.test_thing.test_it",
                "rel_type": "CALLS",
                "props": {},
            },
        ],
    )

    result = await analyze_repo(graph, "module_summary", "proj", path="tests/unit", test_patterns=("test_*", "tests/"))

    assert result["entity_count"] == 1
    assert "def test_it()" in result["outline"]
    assert result["fan_in_count"] == 1
    assert "pkg.mod.prod_caller" in result["outline"]
    assert "test_other" not in result["outline"]


async def test_module_summary_flags_truncation_at_the_entity_cap():
    entities = [_summary_entity(f"pkg.mod.f{i}", sig=f"def f{i}()", line_start=i) for i in range(10)]
    graph = _graph_for_module_summary(
        modules=[{"qn": "pkg.mod", "name": "mod", "file_path": "pkg/mod.py", "docstring": None}],
        entities=entities,
    )

    result = await analyze_repo(graph, "module_summary", "proj", path="pkg", limit=1)

    assert result["truncated"] is True
    assert "TRUNCATED" in result["outline"]


async def test_module_summary_dedupes_rows_duplicated_by_the_defines_join():
    """The backend's OPTIONAL MATCH / LEFT JOIN on DEFINES can emit a row per
    parent; the same uid must not be rendered twice.
    """
    row = _summary_entity("pkg.mod.f", sig="def f()")
    graph = _graph_for_module_summary(
        modules=[{"qn": "pkg.mod", "name": "mod", "file_path": "pkg/mod.py", "docstring": None}],
        entities=[row, dict(row, parent_qn="pkg.mod")],
    )

    result = await analyze_repo(graph, "module_summary", "proj", path="pkg")

    assert result["entity_count"] == 1
    assert result["outline"].count("def f()") == 1


async def test_module_summary_renders_linked_docs():
    graph = _graph_for_module_summary(
        modules=[{"qn": "pkg.mod", "name": "mod", "file_path": "pkg/mod.py", "docstring": None}],
        entities=[_summary_entity("pkg.mod.f", sig="def f()")],
        docs=[
            {
                "doc_qn": "note:f-gotchas",
                "doc_name": "f-gotchas",
                "doc_label": "Note",
                "to_qn": "pkg.mod.f",
                "link_type": "anchor",
            }
        ],
    )

    outline = (await analyze_repo(graph, "module_summary", "proj", path="pkg"))["outline"]

    assert "DOCS" in outline
    assert "  f < note:f-gotchas(anchor)" in outline


async def test_module_summary_renders_a_citation_the_right_way_round():
    """``<`` means "documented by". A citation edge is doc → code like every
    other DOCUMENTS edge, so the cited ADR is the reference, not the target."""
    graph = _graph_for_module_summary(
        modules=[{"qn": "pkg.mod", "name": "mod", "file_path": "pkg/mod.py", "docstring": None}],
        entities=[_summary_entity("pkg.mod.f", sig="def f()")],
        docs=[
            {
                "doc_qn": "wiki/adr/0014-calls-edge-confidence.md",
                "doc_name": "0014-calls-edge-confidence.md",
                "doc_label": "DocFile",
                "to_qn": "pkg.mod.f",
                "link_type": "citation",
            }
        ],
    )

    outline = (await analyze_repo(graph, "module_summary", "proj", path="pkg"))["outline"]

    assert "  f < wiki/adr/0014-calls-edge-confidence.md(citation)" in outline


# ---------------------------------------------------------------------------
# Module summary over the real SQLite backend
#
# Placed here rather than in tests/unit/backends/ because it is the analysis
# function that is under test end-to-end; SqliteGraphClient is just a real
# backend to run its SQL against (same direction as the _analyze_communities
# import already in tests/unit/backends/test_sqlite_graph.py). Memgraph parity
# for the same scenario is covered by tests/integration/server/test_mcp.py.
# ---------------------------------------------------------------------------


async def _seed_sqlite_scope(client: SqliteGraphClient) -> None:
    conn = await client._get_conn()
    nodes = [
        ("proj:pkg.mod", "Module", "pkg.mod", "pkg/mod.py", "mod", None, {"docstring": "Scope module."}),
        (
            "proj:pkg.mod.Widget",
            "TypeDef",
            "pkg.mod.Widget",
            "pkg/mod.py",
            "Widget",
            "class",
            {"visibility": "public", "line_start": 5, "line_end": 40, "docstring": "A widget.\nDetails."},
        ),
        (
            "proj:pkg.mod.Widget.draw",
            "Callable",
            "pkg.mod.Widget.draw",
            "pkg/mod.py",
            "draw",
            "method",
            {"visibility": "public", "line_start": 10, "line_end": 20, "signature": "def draw(self) -> None"},
        ),
        (
            "proj:pkg.mod._hidden",
            "Callable",
            "pkg.mod._hidden",
            "pkg/mod.py",
            "_hidden",
            "function",
            {"visibility": "private", "line_start": 45, "line_end": 47, "signature": "def _hidden()"},
        ),
        (
            "proj:other.cli.main",
            "Callable",
            "other.cli.main",
            "other/cli.py",
            "main",
            "function",
            {"visibility": "public", "line_start": 1, "line_end": 3},
        ),
        ("proj:ext/requests", "ExternalPackage", "requests", None, "requests", None, {}),
    ]
    for uid, label, qn, file_path, name, kind, props in nodes:
        await conn.execute(
            "INSERT INTO nodes(uid, labels, project_name, qualified_name, file_path, name, kind, props_json) "
            "VALUES (?, ?, 'proj', ?, ?, ?, ?, ?)",
            (uid, label, qn, file_path, name, kind, json.dumps(props)),
        )
    edges = [
        ("proj:pkg.mod", "proj:pkg.mod.Widget", "DEFINES", {}),
        ("proj:pkg.mod.Widget", "proj:pkg.mod.Widget.draw", "DEFINES", {}),
        ("proj:pkg.mod", "proj:pkg.mod._hidden", "DEFINES", {}),
        ("proj:pkg.mod.Widget.draw", "proj:pkg.mod._hidden", "CALLS", {"confidence": "ambiguous"}),
        ("proj:other.cli.main", "proj:pkg.mod.Widget.draw", "CALLS", {"confidence": "resolved"}),
        ("proj:pkg.mod.Widget.draw", "proj:ext/requests", "IMPORTS", {}),
    ]
    for from_uid, to_uid, rel_type, props in edges:
        await conn.execute(
            "INSERT INTO edges(from_uid, to_uid, rel_type, props_json) VALUES (?, ?, ?, ?)",
            (from_uid, to_uid, rel_type, json.dumps(props)),
        )
    await conn.commit()


async def test_module_summary_sqlite_backend_end_to_end(tmp_path):
    client = SqliteGraphClient(tmp_path / "graph.sqlite3")
    await client.ensure_schema()
    await _seed_sqlite_scope(client)

    result = await analyze_repo(client, "module_summary", "proj", path="pkg/")

    outline = result["outline"]
    assert result["entity_count"] == 3
    assert "pkg.mod (pkg/mod.py)" in outline
    assert "# Scope module." in outline
    # Class members indented under the class, private marker preserved.
    assert "  + class Widget L5-40  # A widget." in outline
    assert "    + def draw(self) -> None L10-20" in outline
    assert "  - def _hidden() L45-47" in outline
    # Intra-scope edge with its ADR-0014 confidence annotation.
    assert "Widget.draw > _hidden[confidence=ambiguous]" in outline
    # Boundary: an external caller in, an external package out.
    assert "Widget.draw < other.cli.main" in outline
    assert "Widget.draw > requests*" in outline
    assert (result["fan_in_count"], result["fan_out_count"]) == (1, 1)
    await client.close()


async def test_module_summary_sqlite_backend_reports_not_found(tmp_path):
    client = SqliteGraphClient(tmp_path / "graph.sqlite3")
    await client.ensure_schema()

    result = await analyze_repo(client, "module_summary", "proj", path="nope/")

    assert result["code"] == "NOT_FOUND"
    await client.close()
