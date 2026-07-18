"""Unit tests for repository analysis module (mocked graph client — no infrastructure needed)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from code_atlas.server.analysis import _format_path_hops, _sid, analyze_repo, blast_radius, generate_diagram, trace_path

# ---------------------------------------------------------------------------
# Dependencies: cross-package coupling
# ---------------------------------------------------------------------------


def _graph_with_imports(
    direct: list[dict[str, str]],
    indirect: list[dict[str, str]] | None = None,
) -> MagicMock:
    """Fake GraphClient whose execute() returns the four _analyze_dependencies result sets in call order."""
    graph = MagicMock()
    graph.execute = AsyncMock(side_effect=[direct, indirect or [], [], []])
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
# Dependencies / structure: external import counts must honor path scope
# ---------------------------------------------------------------------------


async def test_external_imports_respects_path_scope():
    """external_imports must be scoped like internal_imports, not report whole-project counts."""
    graph = _graph_with_imports(direct=[])

    await analyze_repo(graph, "dependencies", "code-atlas", path="src/foo")

    ext_pkg_query = graph.execute.call_args_list[2][0][0]
    ext_sym_query = graph.execute.call_args_list[3][0][0]
    assert "$path" in ext_pkg_query
    assert "$path" in ext_sym_query


async def test_structure_external_dependencies_respects_path_scope():
    """_analyze_structure has the same inconsistency as external_imports: fix both."""
    graph = MagicMock()
    graph.execute = AsyncMock(return_value=[])

    await analyze_repo(graph, "structure", "code-atlas", path="src/foo")

    ext_query = graph.execute.call_args_list[3][0][0]
    assert "$path" in ext_query


# ---------------------------------------------------------------------------
# Quality: path-scoped fan-in/fan-out must not misclassify out-of-scope
# modules that are only ever edge endpoints
# ---------------------------------------------------------------------------


def _graph_for_quality(entities: list[dict[str, object]], direct: list[dict[str, str]]) -> MagicMock:
    """Fake GraphClient for _analyze_quality's 3 execute() calls: entities, direct, indirect."""
    graph = MagicMock()
    graph.execute = AsyncMock(side_effect=[entities, direct, []])
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
# Diagrams: packages must honor path scope
# ---------------------------------------------------------------------------


async def test_diagram_packages_applies_path_filter():
    """generate_diagram('packages', path=...) must scope the query, not just accept the param."""
    graph = MagicMock()
    graph.execute = AsyncMock(return_value=[])

    await generate_diagram(graph, "packages", "code-atlas", path="src/foo")

    query, params = graph.execute.call_args[0]
    assert "$path" in query
    assert params["path"] == "src/foo"


# ---------------------------------------------------------------------------
# trace_path / blast_radius (information-retrieval family, ADR-0013)
# ---------------------------------------------------------------------------


class _FakeRel(dict):
    """Minimal stand-in for a neo4j Relationship object: dict props + .type."""

    def __init__(self, type_: str, **props: object) -> None:
        super().__init__(**props)
        self.type = type_


def test_format_path_hops_includes_confidence_and_strategy():
    nodes = [{"uid": "p:a", "name": "a"}, {"uid": "p:b", "name": "b"}]
    rels = [_FakeRel("CALLS", confidence="resolved", strategy="import")]

    hops = _format_path_hops(nodes, rels)

    assert hops == [
        {
            "from": {"uid": "p:a", "name": "a"},
            "to": {"uid": "p:b", "name": "b"},
            "edge_type": "CALLS",
            "confidence": "resolved",
            "strategy": "import",
        }
    ]


def test_format_path_hops_omits_confidence_when_absent():
    """A non-CALLS edge (e.g. IMPORTS) has no confidence/strategy property to surface."""
    nodes = [{"uid": "p:a", "name": "a"}, {"uid": "p:b", "name": "b"}]
    rels = [_FakeRel("IMPORTS")]

    hops = _format_path_hops(nodes, rels)

    assert "confidence" not in hops[0]
    assert "strategy" not in hops[0]
    assert hops[0]["edge_type"] == "IMPORTS"


def test_format_path_hops_multi_hop():
    nodes = [{"uid": "p:a", "name": "a"}, {"uid": "p:b", "name": "b"}, {"uid": "p:c", "name": "c"}]
    rels = [_FakeRel("CALLS"), _FakeRel("CALLS")]

    hops = _format_path_hops(nodes, rels)

    assert len(hops) == 2
    assert hops[0]["to"]["uid"] == "p:b"
    assert hops[1]["from"]["uid"] == "p:b"


async def test_trace_path_missing_from_node_returns_not_found():
    graph = MagicMock()
    graph.execute = AsyncMock(return_value=[{"from_exists": False, "to_exists": True}])

    result = await trace_path(graph, "p:missing", "p:b")

    assert result["code"] == "NOT_FOUND"
    assert "p:missing" in result["error"]


async def test_trace_path_missing_to_node_returns_not_found():
    graph = MagicMock()
    graph.execute = AsyncMock(return_value=[{"from_exists": True, "to_exists": False}])

    result = await trace_path(graph, "p:a", "p:missing")

    assert result["code"] == "NOT_FOUND"
    assert "p:missing" in result["error"]


async def test_trace_path_no_path_within_depth():
    graph = MagicMock()
    graph.execute = AsyncMock(
        side_effect=[
            [{"from_exists": True, "to_exists": True}],
            [],
        ]
    )

    result = await trace_path(graph, "p:a", "p:b", max_depth=3)

    assert result["found"] is False
    assert result["max_depth"] == 3
    assert "message" in result


async def test_trace_path_found_builds_hops_with_confidence():
    nodes = [{"uid": "p:a", "name": "a"}, {"uid": "p:b", "name": "b"}]
    rels = [_FakeRel("CALLS", confidence="resolved", strategy="import")]
    graph = MagicMock()
    graph.execute = AsyncMock(
        side_effect=[
            [{"from_exists": True, "to_exists": True}],
            [{"path_nodes": nodes, "path_rels": rels, "hops": 1}],
        ]
    )

    result = await trace_path(graph, "p:a", "p:b", max_depth=6)

    assert result["found"] is True
    assert result["hop_count"] == 1
    assert result["hops"][0]["confidence"] == "resolved"
    assert result["hops"][0]["edge_type"] == "CALLS"


async def test_blast_radius_not_found():
    graph = MagicMock()
    graph.execute = AsyncMock(return_value=[{"exists": False}])

    result = await blast_radius(graph, "p:missing")

    assert result["code"] == "NOT_FOUND"


async def test_blast_radius_invalid_direction():
    graph = MagicMock()
    graph.execute = AsyncMock(return_value=[{"exists": True}])

    result = await blast_radius(graph, "p:a", direction="sideways")

    assert result["code"] == "INVALID_DIRECTION"


async def test_blast_radius_flags_ambiguous_only():
    """An affected entity with no fully-resolved-edge path is flagged ambiguous_only."""
    graph = MagicMock()
    graph.execute = AsyncMock(
        side_effect=[
            [{"exists": True}],
            [
                {"uid": "p:x", "name": "x", "qn": "mod.x", "label": "Callable", "file_path": "mod.py", "min_depth": 1},
                {"uid": "p:y", "name": "y", "qn": "mod.y", "label": "Callable", "file_path": "mod.py", "min_depth": 2},
            ],
            [{"uid": "p:x"}],  # only x is reachable via an all-resolved path
        ]
    )

    result = await blast_radius(graph, "p:a", direction="callees", max_depth=3)

    affected = {a["uid"]: a for a in result["affected"]}
    assert affected["p:x"]["ambiguous_only"] is False
    assert affected["p:y"]["ambiguous_only"] is True
    assert result["affected_count"] == 2


async def test_blast_radius_respects_limit_and_reports_truncated():
    graph = MagicMock()
    all_raw = [
        {
            "uid": f"p:{i}",
            "name": f"n{i}",
            "qn": f"mod.n{i}",
            "label": "Callable",
            "file_path": "mod.py",
            "min_depth": 1,
        }
        for i in range(3)
    ]
    graph.execute = AsyncMock(side_effect=[[{"exists": True}], all_raw, []])

    result = await blast_radius(graph, "p:a", direction="callees", limit=2)

    assert result["affected_count"] == 3
    assert len(result["affected"]) == 2
    assert result["truncated"] is True


# ---------------------------------------------------------------------------
# Dead code (ADR-0013 shortcut: find_dead_code)
# ---------------------------------------------------------------------------


async def test_dead_code_returns_zero_incoming_calls_entities():
    graph = MagicMock()
    graph.execute = AsyncMock(
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
    query = graph.execute.call_args[0][0]
    assert "NOT ()-[:CALLS]->(n)" in query
    assert "NOT n.name STARTS WITH '__'" in query


async def test_dead_code_excludes_test_pattern_matches():
    graph = MagicMock()
    graph.execute = AsyncMock(
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
    graph.execute = AsyncMock(
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
    graph.execute = AsyncMock(
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
    query = graph.execute.call_args[0][0]
    assert "line_end - n.line_start" in query
    assert "ORDER BY loc_span DESC" in query


async def test_complexity_respects_path_scope():
    graph = MagicMock()
    graph.execute = AsyncMock(return_value=[])

    await analyze_repo(graph, "complexity", "code-atlas", path="src/foo")

    query, params = graph.execute.call_args[0]
    assert "$path" in query
    assert params["path"] == "src/foo"


# ---------------------------------------------------------------------------
# Communities (ADR-0013 shortcut: find_communities, MAGE leiden_community_detection)
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
    graph.execute = AsyncMock(
        side_effect=[
            [_hotspot_row("hot", 42), _hotspot_row("cold", 3)],
            [],
            [],
        ]
    )

    result = await analyze_repo(graph, "git_signals", "code-atlas")

    assert result["analysis"] == "git_signals"
    assert result["mined"] is True
    assert [h["qualified_name"] for h in result["hotspots"]] == ["hot", "cold"]
    query = graph.execute.call_args_list[0][0][0]
    assert "git_commit_count" in query


async def test_git_signals_bus_factor_risks():
    graph = MagicMock()
    graph.execute = AsyncMock(
        side_effect=[
            [_hotspot_row("solo", 10, author_count=1)],
            [_hotspot_row("solo", 10, author_count=1)],
            [],
        ]
    )

    result = await analyze_repo(graph, "git_signals", "code-atlas")

    assert len(result["bus_factor_risks"]) == 1
    assert result["bus_factor_risks"][0]["author_count"] == 1
    query = graph.execute.call_args_list[1][0][0]
    assert "git_author_count" in query


async def test_git_signals_co_change_pairs():
    graph = MagicMock()
    graph.execute = AsyncMock(
        side_effect=[
            [],
            [],
            [{"a_qn": "pkg.a", "a_path": "pkg/a.py", "b_qn": "pkg.b", "b_path": "pkg/b.py", "count": 5}],
        ]
    )

    result = await analyze_repo(graph, "git_signals", "code-atlas")

    assert result["co_change_pairs"] == [
        {"a": "pkg.a", "a_file_path": "pkg/a.py", "b": "pkg.b", "b_file_path": "pkg/b.py", "count": 5}
    ]
    query = graph.execute.call_args_list[2][0][0]
    assert "CO_CHANGES_WITH" in query


async def test_git_signals_not_mined_when_no_data():
    """No hotspots at all means 'atlas mine-git-history' was never run — mined=False,
    not an error, so a caller can distinguish 'never mined' from 'mined, found nothing'.
    """
    graph = MagicMock()
    graph.execute = AsyncMock(side_effect=[[], [], []])

    result = await analyze_repo(graph, "git_signals", "code-atlas")

    assert result["mined"] is False
    assert result["hotspots"] == []
    assert result["bus_factor_risks"] == []
    assert result["co_change_pairs"] == []


async def test_git_signals_respects_path_scope():
    graph = MagicMock()
    graph.execute = AsyncMock(side_effect=[[], [], []])

    await analyze_repo(graph, "git_signals", "code-atlas", path="src/foo")

    hotspot_query, hotspot_params = graph.execute.call_args_list[0][0]
    assert "$path" in hotspot_query
    assert hotspot_params["path"] == "src/foo"
    co_change_query = graph.execute.call_args_list[2][0][0]
    assert "$path" in co_change_query


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
    graph.execute = AsyncMock(
        side_effect=[
            [
                {"from_mod": "pkg.a", "to_mod": "pkg.b"},
                {"from_mod": "pkg.a", "to_mod": "pkg.b"},
                {"from_mod": "pkg.a", "to_mod": "pkg.b"},
                {"from_mod": "pkg.c", "to_mod": "pkg.d"},
                {"from_mod": "pkg.c", "to_mod": "pkg.d"},
            ],
            [{"from_mod": "pkg.b", "to_mod": "pkg.a"}],
        ]
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
# Diagrams: module_detail must bound methods/inheritance queries by max_nodes
# and never reference a class truncated out of the declared node set
# ---------------------------------------------------------------------------


def _graph_for_module_detail(entities: list[dict[str, object]], inherits: list[dict[str, str]]) -> MagicMock:
    """Fake GraphClient for _diagram_module_detail's 4 execute() calls: module, entities, methods, inherits."""
    graph = MagicMock()
    graph.execute = AsyncMock(
        side_effect=[
            [{"name": "mod", "qn": "pkg.mod", "uid": "proj:pkg.mod"}],
            entities,
            [],
            inherits,
        ]
    )
    return graph


async def test_module_detail_methods_and_inherits_respect_max_nodes():
    """Both the methods and inheritance queries must be bounded by max_nodes,
    not just the top-level entities query, so a module with large classes
    can't blow past the requested output size.
    """
    graph = _graph_for_module_detail(
        entities=[
            {"name": "Foo", "qn": "pkg.mod.Foo", "label": "TypeDef", "kind": "class", "vis": "public", "sig": None}
        ],
        inherits=[],
    )

    await generate_diagram(graph, "module_detail", "code-atlas", path="pkg/mod", max_nodes=5)

    methods_query = graph.execute.call_args_list[2][0][0]
    inherits_query = graph.execute.call_args_list[3][0][0]
    assert "LIMIT" in methods_query.upper()
    assert "LIMIT" in inherits_query.upper()


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
