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
from code_atlas.server.web.services import (
    EntityNotFoundError,
    ProjectNotIndexedError,
    ProjectViewService,
    SearchViewService,
)

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

    async def get_module_summary(
        self, project: str, path: str, limit: int, edge_limit: int
    ) -> dict[str, list[dict[str, Any]]]:
        # Shapes match the real backends: every edge row carries a decoded `props`
        # dict, which is where ADR-0028's evidence lives.
        return {
            "modules": [],
            "entities": [],
            "internal_edges": [
                {
                    "from_qn": "app.caller",
                    "to_qn": "app.target",
                    "rel_type": "CALLS",
                    "props": {"strategy": "import", "confidence": "resolved", "weight": 1.0, "line": 12},
                }
            ],
            "fan_in": [
                {
                    "from_qn": "other.guesser",
                    "to_qn": "app.target",
                    "rel_type": "CALLS",
                    "props": {"strategy": "project_wide", "confidence": "ambiguous", "weight": 0.25},
                }
            ],
            "fan_out": [
                {"from_qn": "app.target", "to_qn": "app.helper", "rel_type": "CALLS", "props": {}},
            ],
            "docs": [],
        }

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


class _Node:
    """Stands in for a CompactNode."""

    def __init__(self, uid: str, name: str, qn: str, **kw: Any) -> None:
        self.uid, self.name, self.qualified_name = uid, name, qn
        self.kind = kw.get("kind", "function")
        self.file_path = kw.get("file_path", "app.py")
        self.line_start = kw.get("line_start", 1)
        self.line_end = kw.get("line_end", 2)
        self.signature = kw.get("signature", "")
        self.docstring = kw.get("docstring", "")
        self.labels = kw.get("labels", ["Callable"])


class _Context:
    def __init__(self, target: _Node, callers: list[_Node], callees: list[_Node]) -> None:
        self.target, self.callers, self.callees = target, callers, callees
        self.parent = None
        self.docs: list[_Node] = []
        self.siblings: list[_Node] = []


def _search_service(graph: FakeGraph, project: str = "demo") -> SearchViewService:
    from code_atlas.settings import SearchSettings

    return SearchViewService(cast("GraphBackend", graph), project, search_settings=SearchSettings())


class TestEntityDetailEvidence:
    """Every edge shown carries the claim behind it (ATL-116, ADR-0028).

    A caller found by matching an import and one found by matching a bare name across
    the project are very different claims, and a picture that renders them identically
    is worse than a list — it looks authoritative.
    """

    @staticmethod
    def _patch_context(monkeypatch, context):
        async def _fake_expand(graph, uid, **kwargs):
            return context

        monkeypatch.setattr("code_atlas.search.engine.expand_context", _fake_expand)

    async def test_a_resolved_edge_carries_its_strategy_and_line(self, monkeypatch):
        target = _Node("u:target", "target", "app.target")
        self._patch_context(monkeypatch, _Context(target, [_Node("u:caller", "caller", "app.caller")], []))

        detail = await _search_service(FakeGraph()).detail("u:target")

        [caller] = detail.callers
        assert caller.evidence is not None
        assert caller.evidence.strategy == "import"
        assert caller.evidence.confidence == "resolved"
        assert caller.evidence.line == 12
        assert not caller.evidence.is_guess

    async def test_an_ambiguous_edge_is_marked_as_a_guess(self, monkeypatch):
        """The distinction the whole view exists for."""
        target = _Node("u:target", "target", "app.target")
        self._patch_context(monkeypatch, _Context(target, [_Node("u:g", "guesser", "other.guesser")], []))

        detail = await _search_service(FakeGraph()).detail("u:target")

        [caller] = detail.callers
        assert caller.evidence is not None
        assert caller.evidence.is_guess, "an ambiguous edge must be distinguishable from a resolved one"
        assert caller.evidence.weight == 0.25

    async def test_an_edge_with_no_recorded_props_is_structural_not_resolved(self, monkeypatch):
        """Absent evidence means structural (ADR-0029), which is a fact, not a guess."""
        target = _Node("u:target", "target", "app.target")
        self._patch_context(monkeypatch, _Context(target, [], [_Node("u:h", "helper", "app.helper")]))

        detail = await _search_service(FakeGraph()).detail("u:target")

        [callee] = detail.callees
        assert callee.evidence is not None
        assert callee.evidence.is_structural
        assert not callee.evidence.is_guess

    async def test_a_missing_entity_is_not_an_empty_one(self, monkeypatch):
        async def _none(graph, uid, **kwargs):
            return None

        monkeypatch.setattr("code_atlas.search.engine.expand_context", _none)

        with pytest.raises(EntityNotFoundError):
            await _search_service(FakeGraph()).detail("u:nope")


class TestSearchHonesty:
    """A list on screen reads as the whole answer, so it must say when it is not."""

    @staticmethod
    def _patch_search(monkeypatch, results):
        async def _fake(graph, embed, settings, query, **kwargs):
            limit = kwargs.get("limit", 20)
            return results[:limit]

        monkeypatch.setattr("code_atlas.search.engine.hybrid_search", _fake)

    def _hits(self, n: int) -> list[Any]:
        from code_atlas.search.engine import SearchResult

        return [
            SearchResult(
                uid=f"u:{i}",
                name=f"e{i}",
                qualified_name=f"app.e{i}",
                kind="function",
                file_path="app.py",
                line_start=i,
                line_end=i,
                signature="",
                docstring="",
                labels=["Callable"],
                rrf_score=1.0 / (i + 1),
                sources={"bm25": i},
            )
            for i in range(n)
        ]

    async def test_more_results_than_the_page_are_reported_without_inventing_a_count(self, monkeypatch):
        self._patch_search(monkeypatch, self._hits(500))

        page = await _search_service(FakeGraph()).search("e", limit=5)

        assert len(page.hits) == 5
        assert page.more_available is True
        # There is deliberately no `total`: the search fetched limit+1 and knows only
        # that more exist. Reporting the fetch size as a count is the ATL-111 bug.
        assert not hasattr(page, "total")

    async def test_a_complete_page_says_so(self, monkeypatch):
        self._patch_search(monkeypatch, self._hits(3))

        page = await _search_service(FakeGraph()).search("e", limit=5)

        assert len(page.hits) == 3
        assert page.more_available is False

    async def test_an_empty_query_does_not_hit_the_engine(self, monkeypatch):
        async def _explode(*a, **k):
            raise AssertionError("hybrid_search must not run for an empty query")

        monkeypatch.setattr("code_atlas.search.engine.hybrid_search", _explode)

        page = await _search_service(FakeGraph()).search("   ")

        assert page.hits == ()
        assert page.more_available is False
