"""The map payload (ATL-117 / v1.1 design port).

``build_module_graph`` is patched rather than faked through Cypher: the clustering is
already covered by the analysis tests, and what needs proving here is the *view's* own
behaviour — direction and evidence survive to the client, every count the sidebar
derives can close, and the entity level folds without hiding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from code_atlas.server.analysis import ModuleGraph
from code_atlas.server.web.services import MapViewService

if TYPE_CHECKING:
    from code_atlas.graph.protocol import GraphBackend


class _Graph:
    """A backend stand-in for the reads the payload makes besides the module graph."""

    def __init__(self, *, entities: int = 1000, summary: dict[str, Any] | None = None):
        self._entities = entities
        self._summary = summary or {"entities": [], "internal_edges": []}

    async def get_project_status(self, project_name: str | None = None) -> list[dict[str, Any]]:
        return [{"n": {"name": "demo", "entity_count": self._entities}}]

    async def get_structure_overview(self, project: str, path: str, limit: int) -> dict[str, Any]:
        return {
            "counts": [{"label": "Module", "kind": "module", "cnt": 9}],
            "largest_modules": [{"file_path": "src/app/big.py", "name": "app.big", "cnt": 40}],
            "packages": [],
        }

    async def get_module_summary(self, project: str, path: str, limit: int, edge_limit: int) -> dict[str, Any]:
        return self._summary


def _service(graph: _Graph | None = None) -> MapViewService:
    return MapViewService(cast("GraphBackend", graph or _Graph()), "demo")


def _modules(*names: str) -> dict[str, dict[str, Any]]:
    return {
        n: {"uid": f"u:{n}", "name": n.rsplit(".", 1)[-1], "qn": n, "file_path": f"src/{n.replace('.', '/')}.py"}
        for n in names
    }


def _patch(monkeypatch, module_graph: ModuleGraph, external: list[dict[str, Any]] | None = None) -> None:
    async def _fake_graph(graph, project, path, *, test_patterns=()):
        return module_graph

    async def _fake_external(graph, project):
        return external or []

    monkeypatch.setattr("code_atlas.server.analysis.build_module_graph", _fake_graph)
    monkeypatch.setattr("code_atlas.server.analysis.fetch_first_hop_external", _fake_external)


class TestModuleLevel:
    async def test_the_partition_is_used_verbatim(self, monkeypatch):
        """The map and `find_communities` must never disagree about the same project."""
        _patch(
            monkeypatch,
            ModuleGraph(
                modules=_modules("app.a", "app.b", "web.c", "web.d"),
                edges={("app.a", "app.b"): 3.0, ("web.c", "web.d"): 2.0},
                directed={("app.a", "app.b"): 3.0, ("web.c", "web.d"): 2.0},
                partition=[["app.a", "app.b"], ["web.c", "web.d"]],
            ),
        )

        result = await _service().map()

        assert {n.id: n.community for n in result.nodes} == {
            "app.a": 0,
            "app.b": 0,
            "web.c": 1,
            "web.d": 1,
        }

    async def test_community_counts_sum_to_the_module_total(self, monkeypatch):
        """Acceptance check 1: the sidebar's arithmetic derives from one table and closes.

        Singleton communities must ride along — leaving them out made the stated total
        disagree with the drawn count by exactly their number.
        """
        _patch(
            monkeypatch,
            ModuleGraph(
                modules=_modules("a", "b", "loner"),
                edges={("a", "b"): 1.0},
                directed={("a", "b"): 1.0},
                partition=[["a", "b"], ["loner"]],
            ),
        )

        result = await _service().map()

        assert sum(c.count for c in result.communities) == result.module_total == 3

    async def test_edge_direction_is_the_dependency_not_the_alphabet(self, monkeypatch):
        """`s` depends on `t`, whichever way the names happen to sort."""
        _patch(
            monkeypatch,
            ModuleGraph(
                modules=_modules("zeta", "alpha"),
                edges={("alpha", "zeta"): 4.0},
                partition=[["alpha", "zeta"]],
                directed={("zeta", "alpha"): 4.0},
            ),
        )

        [edge] = (await _service().map()).edges

        assert (edge.s, edge.t) == ("zeta", "alpha")

    async def test_a_mutual_dependency_renders_as_two_edges(self, monkeypatch):
        _patch(
            monkeypatch,
            ModuleGraph(
                modules=_modules("a", "b"),
                edges={("a", "b"): 3.0},
                partition=[["a", "b"]],
                directed={("a", "b"): 1.0, ("b", "a"): 2.0},
            ),
        )

        edges = (await _service().map()).edges

        assert {(e.s, e.t) for e in edges} == {("a", "b"), ("b", "a")}

    async def test_evidence_survives_to_the_client(self, monkeypatch):
        """Unknown is not zero — an edge nobody looked up must still say so."""
        _patch(
            monkeypatch,
            ModuleGraph(
                modules=_modules("a", "b", "c"),
                edges={("a", "b"): 1.0, ("b", "c"): 1.0},
                directed={("a", "b"): 1.0, ("b", "c"): 1.0},
                partition=[["a", "b", "c"]],
                evidence={("a", "b"): "structural"},
            ),
        )

        by_pair = {(e.s, e.t): e.ev for e in (await _service().map()).edges}

        assert by_pair == {("a", "b"): "structural", ("b", "c"): "unknown"}

    async def test_weights_are_scaled_into_the_design_band(self, monkeypatch):
        """The canvas thickness formula expects 1..3; raw aggregates span four orders."""
        _patch(
            monkeypatch,
            ModuleGraph(
                modules=_modules("a", "b", "c"),
                edges={("a", "b"): 126.79, ("b", "c"): 0.0027},
                directed={("a", "b"): 126.79, ("b", "c"): 0.0027},
                partition=[["a", "b", "c"]],
            ),
        )

        weights = {(e.s, e.t): e.w for e in (await _service().map()).edges}

        assert weights[("a", "b")] == 3.0
        assert 1.0 <= weights[("b", "c")] < weights[("a", "b")]

    async def test_labels_are_breadcrumbs_never_bare_filenames(self, monkeypatch):
        """`conftest` four times on one map identifies nothing."""
        _patch(
            monkeypatch,
            ModuleGraph(
                modules=_modules("app.parsing.ast", "app.schema"),
                edges={},
                directed={},
                partition=[["app.parsing.ast", "app.schema"]],
            ),
        )

        labels = {n.id: n.label for n in (await _service().map()).nodes}

        assert labels["app.parsing.ast"] == "parsing › ast.py"  # noqa: RUF001  # the breadcrumb separator
        assert labels["app.schema"] == "app › schema.py"  # noqa: RUF001

    async def test_tests_are_hidden_by_default_and_counted(self, monkeypatch):
        modules = _modules("app.core")
        modules["tests.test_core"] = {
            "uid": "u:t",
            "name": "test_core",
            "qn": "tests.test_core",
            "file_path": "tests/test_core.py",
        }
        _patch(
            monkeypatch,
            ModuleGraph(
                modules=modules,
                edges={},
                directed={},
                partition=[["app.core", "tests.test_core"]],
            ),
        )

        hidden = await _service().map()
        shown = await _service().map(show_tests=True)

        assert {n.id for n in hidden.nodes} == {"app.core"}
        assert hidden.test_count == 1
        assert {n.id: n.kind for n in shown.nodes} == {"app.core": "code", "tests.test_core": "test"}

    async def test_truncation_keeps_communities_whole(self, monkeypatch):
        """A half-drawn subsystem is misleading, not merely partial — the gap is invisible."""
        big = [f"big{i}" for i in range(8)]
        small = [f"small{i}" for i in range(3)]
        _patch(
            monkeypatch,
            ModuleGraph(modules=_modules(*big, *small), edges={}, directed={}, partition=[big, small]),
        )

        result = await _service().map(node_limit=8)

        assert {n.id for n in result.nodes} == set(big)
        assert result.truncated is True

    async def test_an_unindexed_project_is_not_an_empty_map(self, monkeypatch):
        _patch(monkeypatch, ModuleGraph(modules={}, edges={}, directed={}, partition=[]))

        result = await _service().map()

        assert result.nodes == ()
        assert not result.caveat.is_complete, "an empty map must say why, not render as a clean blank"

    async def test_every_node_is_laid_out_inside_the_canvas(self, monkeypatch):
        _patch(
            monkeypatch,
            ModuleGraph(
                modules=_modules("a", "b", "c", "d"),
                edges={("a", "b"): 1.0},
                directed={("a", "b"): 1.0},
                partition=[["a", "b"], ["c", "d"]],
            ),
        )

        result = await _service().map()

        for node in result.nodes:
            assert 0.0 <= node.x <= 1000.0
            assert 0.0 <= node.y <= 1000.0


class TestMultiProject:
    async def test_cross_project_imports_appear_when_both_are_loaded(self, monkeypatch):
        """The dialog's own promise: cross-project dependencies appear when more than
        one project is loaded — as structural edges, since they are IMPORTS."""
        graphs = {
            "demo": ModuleGraph(modules=_modules("app.a"), edges={}, directed={}, partition=[["app.a"]]),
            "other": ModuleGraph(modules=_modules("lib.b"), edges={}, directed={}, partition=[["lib.b"]]),
        }

        async def _fake_graph(graph, project, path, *, test_patterns=()):
            return graphs[project]

        async def _fake_external(graph, project):
            if project == "demo":
                return [{"from_mod": "app.a", "to_mod": "lib.b", "to_project": "other"}]
            return []

        monkeypatch.setattr("code_atlas.server.analysis.build_module_graph", _fake_graph)
        monkeypatch.setattr("code_atlas.server.analysis.fetch_first_hop_external", _fake_external)

        result = await _service().map(projects=("demo", "other"))

        assert {n.id for n in result.nodes} == {"demo:app.a", "other:lib.b"}
        [edge] = result.edges
        assert (edge.s, edge.t, edge.ev) == ("demo:app.a", "other:lib.b", "structural")


class TestEntityLevel:
    def _summary(self) -> dict[str, Any]:
        def row(qn: str, label: str, kind: str = "") -> dict[str, Any]:
            return {
                "uid": f"u:{qn}",
                "qn": qn,
                "name": qn.rsplit(".", 1)[-1],
                "label": label,
                "kind": kind,
                "file_path": "src/app/mod.py",
            }

        return {
            "entities": [
                row("app.mod", "Module", "module"),
                row("app.mod.Klass", "TypeDef", "class"),
                row("app.mod.Klass.run", "Callable", "method"),
                row("app.mod.helper", "Callable", "function"),
                row("app.mod.LIMIT", "Value", "constant"),
            ],
            "internal_edges": [
                {
                    "from_qn": "app.mod.helper",
                    "to_qn": "app.mod.Klass.run",
                    "rel_type": "CALLS",
                    "props": {"strategy": "import", "confidence": "resolved", "weight": 1.0},
                }
            ],
        }

    async def test_methods_fold_into_their_class_and_calls_are_rewired(self):
        """Folding hides nodes without hiding dependencies."""
        result = await _service(_Graph(summary=self._summary())).entity_map("src/app/mod.py")

        ids = {n.id for n in result.nodes}
        assert "app.mod.Klass.run" not in ids
        assert ("app.mod.helper", "app.mod.Klass") in {(e.s, e.t) for e in result.edges}

    async def test_the_tally_counts_the_whole_inventory_not_the_drawn_subset(self):
        """A folded method still appears in its row — kind tallies never shrink."""
        result = await _service(_Graph(summary=self._summary())).entity_map("src/app/mod.py")

        tally = {t.id: t for t in result.tally}
        assert tally["method"].in_module == 1
        assert tally["method"].drawn == 0
        assert result.in_module == 5

    async def test_expand_methods_draws_them(self):
        result = await _service(_Graph(summary=self._summary())).entity_map("src/app/mod.py", expand_methods=True)

        assert "app.mod.Klass.run" in {n.id for n in result.nodes}
        assert result.collapsed is False

    async def test_containment_anchors_every_entity(self):
        """An entity with no resolved call must read as contained, not isolated."""
        result = await _service(_Graph(summary=self._summary())).entity_map("src/app/mod.py")

        anchored = {e.t for e in result.edges if e.ev == "structural"}
        assert "app.mod.LIMIT" in anchored, "the constant has no call edge, only its definer"
        # Every non-module node is reachable from something.
        targets = {e.t for e in result.edges} | {e.s for e in result.edges}
        for node in result.nodes:
            if node.kind != "module":
                assert node.id in targets

    async def test_an_empty_scope_names_itself(self):
        result = await _service(_Graph(summary={"entities": [], "internal_edges": []})).entity_map("src/app/empty.py")

        assert result.nodes == ()
        assert "empty.py" in result.caveat.note


class TestDegradedBackend:
    """Community detection needs raw Cypher, which SQLite does not serve."""

    async def test_the_sqlite_backend_says_so_rather_than_drawing_half_a_map(self, monkeypatch):
        from code_atlas.backends.sqlite_graph import SqliteGraphClient

        fake = cast("GraphBackend", object.__new__(SqliteGraphClient))
        service = MapViewService(fake, "demo")

        async def _explode(*args, **kwargs):
            raise AssertionError("the backend must be rejected before any query runs")

        monkeypatch.setattr("code_atlas.server.analysis.build_module_graph", _explode)

        result = await service.map()

        assert not result.is_available
        assert "SQLite" in result.unavailable
        assert result.nodes == ()


class TestMapEndpoint:
    def test_the_api_serves_the_payload_the_island_renders(self, monkeypatch):
        pytest.importorskip("litestar")
        from litestar.testing import TestClient

        from code_atlas.server.web.app import create_app

        _patch(
            monkeypatch,
            ModuleGraph(
                modules=_modules("app.api.users", "app.api.orders"),
                edges={("app.api.users", "app.api.orders"): 4.0},
                directed={("app.api.users", "app.api.orders"): 4.0},
                partition=[["app.api.users", "app.api.orders"]],
            ),
        )

        with TestClient(app=create_app(cast("GraphBackend", _Graph()), "demo")) as client:
            page = client.get("/").text
            payload = client.get("/map/api").json()

        assert 'id="map-aside"' in page, "the island's rail mount must exist"
        assert 'id="map-main"' in page, "the island's canvas mount must exist"
        assert "/static/map.js" in page
        assert len(payload["nodes"]) == 2
        assert payload["edges"][0]["ev"] == "unknown"
