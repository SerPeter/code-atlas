"""The community map view (ATL-117).

``build_module_graph`` is patched rather than faked through Cypher: the clustering is
already covered by the analysis tests, and what needs proving here is the *view's* own
behaviour — that it uses the shared partition verbatim, truncates without lying, and
marks what came from outside the project.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from code_atlas.server.analysis import ModuleGraph
from code_atlas.server.web.services import MapViewService

if TYPE_CHECKING:
    from code_atlas.graph.protocol import GraphBackend


class _Graph:
    """A backend stand-in. Every read the map makes is patched, so nothing runs here."""

    async def close(self) -> None: ...


def _service(project: str = "demo") -> MapViewService:
    return MapViewService(cast("GraphBackend", _Graph()), project)


def _modules(*names: str) -> dict[str, dict[str, Any]]:
    return {n: {"uid": f"u:{n}", "name": n.rsplit(".", 1)[-1], "qn": n, "file_path": f"{n}.py"} for n in names}


def _patch(monkeypatch, module_graph: ModuleGraph, external: list[dict[str, Any]] | None = None) -> None:
    async def _fake_graph(graph, project, path, *, test_patterns=()):
        return module_graph

    async def _fake_external(graph, project):
        return external or []

    monkeypatch.setattr("code_atlas.server.analysis.build_module_graph", _fake_graph)
    monkeypatch.setattr("code_atlas.server.analysis.fetch_first_hop_external", _fake_external)


class TestMapView:
    async def test_the_partition_is_used_verbatim(self, monkeypatch):
        """The map and `find_communities` must never disagree about the same project.

        They share `build_module_graph`, so this pins that the view reports what the
        shared path returned rather than re-clustering on its own.
        """
        _patch(
            monkeypatch,
            ModuleGraph(
                modules=_modules("app.a", "app.b", "web.c", "web.d"),
                edges={("app.a", "app.b"): 3.0, ("web.c", "web.d"): 2.0},
                partition=[["app.a", "app.b"], ["web.c", "web.d"]],
            ),
        )

        result = await _service().map()

        assert [c.size for c in result.communities] == [2, 2]
        assert {n.id: n.community for n in result.nodes} == {
            "app.a": 0,
            "app.b": 0,
            "web.c": 1,
            "web.d": 1,
        }

    async def test_a_community_is_named_by_its_shared_prefix(self):
        """ "community 3" tells a reader nothing they can act on."""
        from code_atlas.server.web.services import _community_label

        assert _community_label(["app.api.users", "app.api.orders"]) == "app.api"
        assert _community_label(["solo"]) == "solo"
        assert _community_label(["alpha.one", "beta.two"]) == "alpha.one"

    async def test_edge_weight_survives_to_the_view_model(self, monkeypatch):
        """Thickness must reflect the stored ADR-0017 weight, not an edge count."""
        _patch(
            monkeypatch,
            ModuleGraph(
                modules=_modules("a", "b", "c"),
                edges={("a", "b"): 12.5, ("b", "c"): 0.25},
                partition=[["a", "b", "c"]],
            ),
        )

        result = await _service().map()

        assert {(e.source, e.target): e.weight for e in result.edges} == {
            ("a", "b"): 12.5,
            ("b", "c"): 0.25,
        }

    async def test_cross_community_edges_are_marked(self, monkeypatch):
        _patch(
            monkeypatch,
            ModuleGraph(
                modules=_modules("a", "b", "x", "y"),
                edges={("a", "b"): 1.0, ("b", "x"): 1.0},
                partition=[["a", "b"], ["x", "y"]],
            ),
        )

        result = await _service().map()

        crossing = {(e.source, e.target) for e in result.edges if e.crosses_community}
        assert crossing == {("b", "x")}

    async def test_first_hop_external_nodes_are_distinguishable(self, monkeypatch):
        _patch(
            monkeypatch,
            ModuleGraph(
                modules=_modules("app.a", "app.b"),
                edges={("app.a", "app.b"): 1.0},
                partition=[["app.a", "app.b"]],
            ),
            external=[{"from_mod": "app.a", "to_mod": "shared.util", "to_project": "other"}],
        )

        result = await _service().map()

        external = [n for n in result.nodes if n.is_external]
        assert [n.id for n in external] == ["shared.util"]
        assert external[0].project == "other", "an external node must name the project that owns it"
        assert result.external_count == 1
        assert ("app.a", "shared.util") in {(e.source, e.target) for e in result.edges}

    async def test_external_nodes_can_be_switched_off(self, monkeypatch):
        _patch(
            monkeypatch,
            ModuleGraph(
                modules=_modules("app.a", "app.b"),
                edges={("app.a", "app.b"): 1.0},
                partition=[["app.a", "app.b"]],
            ),
            external=[{"from_mod": "app.a", "to_mod": "shared.util", "to_project": "other"}],
        )

        result = await _service().map(include_external=False)

        assert result.external_count == 0

    async def test_an_external_edge_from_a_dropped_module_is_not_drawn(self, monkeypatch):
        """Truncation must not leave an edge pointing at a node the map never added."""
        _patch(
            monkeypatch,
            ModuleGraph(
                modules=_modules("a", "b"),
                edges={},
                partition=[["a", "b"]],
            ),
            external=[{"from_mod": "not-on-the-map", "to_mod": "shared.util", "to_project": "other"}],
        )

        result = await _service().map()

        assert result.external_count == 0
        node_ids = {n.id for n in result.nodes}
        for edge in result.edges:
            assert edge.source in node_ids
            assert edge.target in node_ids

    async def test_truncation_keeps_communities_whole(self, monkeypatch):
        """A half-drawn subsystem is misleading, not merely partial — the gap is invisible."""
        big = [f"big{i}" for i in range(8)]
        small = [f"small{i}" for i in range(3)]
        _patch(
            monkeypatch,
            ModuleGraph(modules=_modules(*big, *small), edges={}, partition=[big, small]),
        )

        result = await _service().map(node_limit=8)

        drawn = {n.id for n in result.nodes}
        assert drawn == set(big), "the community that fit is complete; the one that did not is absent"
        assert result.truncated is True
        assert "8" in result.caveat.note or "11" in result.caveat.note

    async def test_a_complete_map_says_it_is_complete(self, monkeypatch):
        _patch(monkeypatch, ModuleGraph(modules=_modules("a", "b"), edges={}, partition=[["a", "b"]]))

        result = await _service().map()

        assert result.truncated is False

    async def test_an_unindexed_project_is_not_an_empty_map(self, monkeypatch):
        _patch(monkeypatch, ModuleGraph(modules={}, edges={}, partition=[]))

        result = await _service().map()

        assert result.nodes == ()
        assert not result.caveat.is_complete, "an empty map must say why, not render as a clean blank"

    async def test_every_node_carries_a_position(self, monkeypatch):
        """Positions are precomputed server-side; a node without one lands on the origin pile."""
        _patch(
            monkeypatch,
            ModuleGraph(
                modules=_modules("a", "b", "c", "d"),
                edges={("a", "b"): 1.0},
                partition=[["a", "b"], ["c", "d"]],
            ),
        )

        result = await _service().map()

        origin = [n.id for n in result.nodes if n.x == 0.0 and n.y == 0.0]
        assert not origin, f"{origin} were not laid out"


class TestDegradedBackend:
    """Community detection needs MAGE, which SQLite does not have."""

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
    def test_the_page_renders_the_canvas_and_its_subsystems(self, monkeypatch):
        pytest.importorskip("litestar")
        from litestar.testing import TestClient

        from code_atlas.server.web.app import create_app

        _patch(
            monkeypatch,
            ModuleGraph(
                modules=_modules("app.api.users", "app.api.orders", "web.ui"),
                edges={("app.api.users", "app.api.orders"): 4.0},
                partition=[["app.api.users", "app.api.orders"], ["web.ui"]],
            ),
        )

        with TestClient(app=create_app(cast("GraphBackend", _Graph()), "demo")) as client:
            body = client.get("/map/").text
            payload = client.get("/map/api").json()

        assert "map-canvas" in body
        assert "app.api" in body, "the subsystem list must name communities, not number them"
        assert "/static/vendor/sigma-3.0.3.min.js" in body, "the renderer is vendored, never a CDN"
        assert len(payload["nodes"]) == 3
        assert payload["edges"][0]["weight"] == 4.0

    def test_the_page_reports_an_unavailable_backend(self):
        pytest.importorskip("litestar")
        from litestar.testing import TestClient

        from code_atlas.backends.sqlite_graph import SqliteGraphClient
        from code_atlas.server.web.app import create_app

        # Rendered through the real app so the template's degraded branch is exercised.
        graph = cast("GraphBackend", object.__new__(SqliteGraphClient))
        with TestClient(app=create_app(graph, "demo")) as client:
            response = client.get("/map/")

        assert response.status_code == 200
        assert "Map unavailable" in response.text
