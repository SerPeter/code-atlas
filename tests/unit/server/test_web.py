"""Unit tests for the `atlas ui` web layer (ATL-115).

No database and no network: the service layer depends on the ``GraphBackend`` protocol,
so a fake satisfies it. That is the point of the three-layer split — the HTTP layer can
be exercised without Memgraph, and the service layer without Litestar.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from litestar.testing import TestClient

from code_atlas.server.web.app import create_app
from code_atlas.server.web.services import ProjectNotIndexedError, ProjectViewService

if TYPE_CHECKING:
    from code_atlas.graph.protocol import GraphBackend


class FakeGraph:
    """Enough of ``GraphBackend`` for the overview path.

    Faithful to the real shapes: ``get_structure_overview`` returns the keyed lists
    ``_analyze_structure`` reads, not a convenient invention.
    """

    def __init__(self, *, projects: list[dict[str, Any]] | None = None, counts: list[dict[str, Any]] | None = None):
        self._projects = (
            projects
            if projects is not None
            else [
                {
                    "project": "demo",
                    "entities": 42,
                    "indexed_at": "2026-08-08T00:00:00Z",
                    "git_hash": "abc123def456789",
                },
                {"project": "other", "entities": 7},
            ]
        )
        # `kind` is required, not optional: _analyze_structure reads r["kind"] directly.
        self._counts = (
            counts
            if counts is not None
            else [
                {"label": "Callable", "kind": "function", "cnt": 30},
                {"label": "Module", "kind": "module", "cnt": 9},
                {"label": "TypeDef", "kind": "class", "cnt": 3},
            ]
        )

    async def get_project_status(self, project_name: str | None = None) -> list[dict[str, Any]]:
        return self._projects

    async def get_structure_overview(self, project: str, path: str, limit: int) -> dict[str, list[dict[str, Any]]]:
        return {"counts": self._counts, "largest_modules": [], "packages": []}

    async def close(self) -> None: ...


def _service(graph: FakeGraph, project: str) -> ProjectViewService:
    """A service over *graph* — see :func:`_client` for why the cast is deliberate."""
    return ProjectViewService(cast("GraphBackend", graph), project)


def _client(graph: FakeGraph, project: str) -> TestClient:
    """A test client over *graph*.

    The cast is deliberate. ``GraphBackend`` is a 90-method Protocol; a fake that
    satisfied all of it would be a second backend implementation, which is exactly what
    a fake exists to avoid. This one implements the handful of methods the overview path
    actually calls, and any method it is missing fails loudly as an AttributeError —
    which is how the incomplete first version of it was caught.
    """
    return TestClient(app=create_app(cast("GraphBackend", graph), project))


class TestProjectViewService:
    """The service owns the use case and knows nothing about HTTP."""

    async def test_overview_reports_the_current_project(self):
        overview = await _service(FakeGraph(), "demo").overview()

        assert overview.project == "demo"
        assert overview.entity_count == 42
        assert overview.module_count == 9
        assert overview.label_counts["Callable"] == 30

    async def test_other_projects_are_listed_but_not_loaded(self):
        """Scope is one project; others are reachable, not rendered.

        This is what keeps every query bounded — the alternative is an unbounded read
        over a graph already ~30k nodes for eight projects.
        """
        overview = await _service(FakeGraph(), "demo").overview()

        current = [p for p in overview.other_projects if p.is_current]
        assert [p.name for p in current] == ["demo"]
        assert {p.name for p in overview.other_projects} == {"demo", "other"}

    async def test_an_unindexed_project_is_not_an_empty_one(self):
        """The distinction ATL-110 drew in the CLI, held here too."""
        with pytest.raises(ProjectNotIndexedError):
            await _service(FakeGraph(), "never-indexed").overview()

    async def test_an_indexed_but_empty_project_carries_a_caveat(self):
        graph = FakeGraph(projects=[{"project": "demo", "entities": 0}], counts=[])
        overview = await _service(graph, "demo").overview()

        assert overview.entity_count == 0
        assert not overview.caveat.is_complete, "an empty index must say so, not render as a clean zero"

    async def test_absent_metadata_stays_absent(self):
        """Empty string and None both mean "not recorded" and must not render as data."""
        graph = FakeGraph(projects=[{"project": "demo", "entities": 1, "indexed_at": "", "git_hash": None}])
        overview = await _service(graph, "demo").overview()

        assert overview.indexed_at is None
        assert overview.git_hash is None


class TestWebApp:
    """The HTTP layer, against a fake backend."""

    def test_health_reports_the_served_project(self):
        with _client(FakeGraph(), "demo") as client:
            response = client.get("/healthz/")

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "project": "demo"}

    def test_the_index_page_renders_the_project(self):
        with _client(FakeGraph(), "demo") as client:
            response = client.get("/")

        assert response.status_code == 200
        assert "demo" in response.text

    def test_the_api_returns_the_same_view_model(self):
        """One implementation behind both renderings — a divergence here is the failure
        mode the service layer exists to prevent."""
        with _client(FakeGraph(), "demo") as client:
            payload = client.get("/api/overview").json()

        assert payload["project"] == "demo"
        assert payload["entity_count"] == 42
        assert payload["module_count"] == 9

    def test_an_unindexed_project_says_how_to_fix_it(self):
        with _client(FakeGraph(), "nope") as client:
            response = client.get("/")

        assert response.status_code == 404
        assert "atlas index" in response.text

    def test_the_api_404s_for_an_unindexed_project(self):
        with _client(FakeGraph(), "nope") as client:
            response = client.get("/api/overview")

        assert response.status_code == 404
