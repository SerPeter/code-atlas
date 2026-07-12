"""Unit tests for repository analysis module (mocked graph client — no infrastructure needed)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from code_atlas.server.analysis import _sid, analyze_repo, generate_diagram

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
