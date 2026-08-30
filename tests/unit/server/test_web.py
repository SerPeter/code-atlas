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
    ArchitectureViewService,
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

    def __init__(
        self,
        *,
        projects: list[dict[str, Any]] | None = None,
        counts: list[dict[str, Any]] | None = None,
        imports: list[tuple[str, str]] | None = None,
    ):
        # `direct` only: the architecture view measures declared dependencies, and
        # counting the transitive `indirect` rows too would double-count every path the
        # closure already covers.
        self._imports = imports if imports is not None else [("app", "service"), ("service", "repo")]
        # Malformed rows a test wants to inject verbatim, past the (from, to) shorthand.
        self.extra_import_rows: list[dict[str, Any]] = []
        self._projects = (
            projects
            if projects is not None
            else [
                {
                    "name": "demo",
                    "entity_count": 42,
                    "last_indexed_at": "2026-08-08T00:00:00Z",
                    "git_hash": "abc123def456789",
                },
                {"name": "other", "entity_count": 7},
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
        # `[{"n": <node>}]` with the node's OWN property names — what both backends really
        # return. An earlier version of this fake flattened the row and renamed the keys
        # to `project`/`entities`/`indexed_at`, none of which exist on a Project node. The
        # service read those names, matched nothing, and 404'd every real install as "not
        # indexed" — while every test here passed.
        return [{"n": p} for p in self._projects]

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

    def with_snapshots(self, runs: list[float] | list[tuple[float, int]]) -> FakeGraph:
        """Attach a recorded architecture history to the Project node.

        Each run is a propagation cost, or a ``(propagation, module_count)`` pair when the
        test needs the coverage to move as well.
        """
        from code_atlas.server.architecture_history import Snapshot, encode

        pairs = [r if isinstance(r, tuple) else (r, 100) for r in runs]
        self._projects[0]["architecture_snapshots"] = encode(
            [
                Snapshot(
                    at=f"2026-08-0{i + 1}T00:00:00+00:00",
                    commit=f"c{i}",
                    modules=modules,
                    edges=modules,
                    propagation_cost=cost,
                    core_size=0.1,
                    largest_cycle=1,
                    fan_in_gini=0.2,
                )
                for i, (cost, modules) in enumerate(pairs)
            ]
        )
        return self

    async def get_module_import_edges(self, project: str, path: str) -> dict[str, list[dict[str, Any]]]:
        return {
            "direct": [{"from_mod": a, "to_mod": b} for a, b in self._imports] + self.extra_import_rows,
            "indirect": [],
        }

    async def close(self) -> None: ...


@pytest.fixture(autouse=True)
def _route_architecture_pairs(monkeypatch):
    """The architecture pipeline reads ``fetch_architecture_pairs`` now; route it
    through the fake's import edges (all structural), which is exactly the old
    edge source these tests were written against — their dependency semantics
    (layers, cycles, hand-worked propagation) carry over unchanged."""
    from code_atlas.server import analysis

    real = analysis.fetch_architecture_pairs

    async def fake(graph, project, *, include_tests=False, include_guessed=False, test_patterns=()):
        if isinstance(graph, FakeGraph):
            # The real pipeline first — tests that patch build_module_graph get it
            # honoured. A bare FakeGraph cannot serve the module-graph reads and
            # falls back to its import edges.
            try:
                return await real(
                    graph,
                    project,
                    include_tests=include_tests,
                    include_guessed=include_guessed,
                    test_patterns=test_patterns,
                )
            except Exception:
                raw = await graph.get_module_import_edges(project, "")
                pairs = {
                    (str(r.get("from_mod")), str(r.get("to_mod"))): "structural"
                    for r in raw.get("direct", [])
                    if r.get("from_mod") and r.get("to_mod")
                }
                return analysis.ArchitecturePairs(
                    pairs=pairs,
                    all_pairs=dict(pairs),
                    module_paths={},
                    excluded_test_modules=0,
                    excluded_guessed_pairs=0,
                )
        return await real(
            graph, project, include_tests=include_tests, include_guessed=include_guessed, test_patterns=test_patterns
        )

    monkeypatch.setattr(analysis, "fetch_architecture_pairs", fake)


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


class TestWebTelemetry:
    """The UI was the one entry point producing no signals at all."""

    @staticmethod
    def _tracer(monkeypatch):
        import code_atlas.server.web.app as app_mod

        class _Span:
            def __init__(self, name, attributes=None):
                self.name = name
                self.attributes = dict(attributes or {})

            def set_attribute(self, key, value):
                self.attributes[key] = value

            def set_status(self, *_a, **_kw):
                pass

            def record_exception(self, *_a, **_kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                pass

        class _Tracer:
            def __init__(self):
                self.spans = []

            def start_as_current_span(self, name, **kwargs):
                span = _Span(name, kwargs.get("attributes"))
                self.spans.append(span)
                return span

        tracer = _Tracer()
        monkeypatch.setattr(app_mod, "_tracer", tracer)
        return tracer

    def test_a_request_is_spanned_by_route_not_by_path(self, monkeypatch):
        """`/entity/abc` and `/entity/def` are one route. Naming spans and metric series
        after the raw path gives every entity its own series -- unbounded cardinality is
        how a metrics database is taken down by its own instrumentation."""
        tracer = self._tracer(monkeypatch)
        client = _client(FakeGraph(), "demo")

        client.get("/")

        assert tracer.spans, "no span for a served request"
        span = tracer.spans[0]
        assert span.name == "web GET /"
        assert span.attributes["http.route"] == "/"
        assert span.attributes["http.response.status_code"] == 200

    def test_an_unrouted_path_is_not_measured_at_all(self, monkeypatch):
        """Litestar applies middleware *inside* routing, so a 404 never reaches this one.

        Pinned because it cuts both ways and the reasoning is easy to lose: a 404 sweep
        cannot mint one metric series per probed URL, but the request counter therefore
        means "requests that matched a route", not "requests received". If a later
        Litestar version moves middleware outside routing, this test says so.
        """
        tracer = self._tracer(monkeypatch)
        client = _client(FakeGraph(), "demo")

        assert client.get("/no-such-page-9f3a").status_code == 404
        assert client.get("/no-such-page-b21c").status_code == 404

        assert tracer.spans == []

    def test_the_metric_records_method_route_and_status(self, monkeypatch):
        import code_atlas.telemetry as tel

        self._tracer(monkeypatch)
        recorded: list[tuple] = []
        monkeypatch.setattr(
            tel._metrics,
            "web_requests",
            type("C", (), {"add": lambda _s, n, a=None: recorded.append((n, a))})(),
        )
        client = _client(FakeGraph(), "demo")

        client.get("/")

        assert recorded == [(1, {"method": "GET", "route": "/", "status": "200"})]


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
        graph = FakeGraph(projects=[{"name": "demo", "entity_count": 0}], counts=[])
        overview = await _service(graph, "demo").overview()

        assert overview.entity_count == 0
        assert not overview.caveat.is_complete, "an empty index must say so, not render as a clean zero"

    async def test_the_index_time_is_rendered_for_a_human(self):
        """`last_indexed_at` is written as time.time(), so it arrives as a float.

        str() put `1786176798.014237` on the landing page — technically the data, and no
        answer at all to "when was this indexed". Only running the real server showed it.
        """
        graph = FakeGraph(projects=[{"name": "demo", "entity_count": 1, "last_indexed_at": 1786176798.014237}])

        overview = await _service(graph, "demo").overview()

        assert overview.indexed_at is not None
        assert overview.indexed_at.startswith("2026-")
        assert "1786176798" not in overview.indexed_at

    async def test_an_iso_string_index_time_passes_through(self):
        """Older rows and the SQLite path may already hold a string."""
        graph = FakeGraph(projects=[{"name": "demo", "entity_count": 1, "last_indexed_at": "2026-08-08T00:00:00Z"}])

        overview = await _service(graph, "demo").overview()

        assert overview.indexed_at == "2026-08-08T00:00:00Z"

    async def test_the_real_backend_row_shape_is_understood(self):
        """Both backends return `[{"n": <node>}]` with the node's own property names.

        The service read `project`/`entities`/`indexed_at`, which no Project node carries,
        so it matched nothing and 404'd every real install as "not indexed" — while every
        test here passed against a fake that invented the flattened shape.
        """
        overview = await _service(FakeGraph(), "demo").overview()

        assert overview.project == "demo"
        assert overview.entity_count == 42

    async def test_absent_metadata_stays_absent(self):
        """Empty string and None both mean "not recorded" and must not render as data."""
        graph = FakeGraph(projects=[{"name": "demo", "entity_count": 1, "last_indexed_at": "", "git_hash": None}])
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

    async def test_hits_are_monotonic_in_the_score_they_report(self, monkeypatch):
        """A list not sorted by the number beside it reads as a broken ranker.

        The CLI and the MCP tools were fixed for this; the web layer was missed and kept
        showing the pre-boost rrf_score. Caught by reading a real /api/search response.
        """
        from code_atlas.search.engine import SearchResult

        results = [
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
                rrf_score=0.01 * (i + 1),
                ranked_score=1.0 - i,  # deliberately disagrees with rrf_score
            )
            for i in range(3)
        ]
        self._patch_search(monkeypatch, results)

        page = await _search_service(FakeGraph()).search("e", limit=10)

        scores = [h.score for h in page.hits]
        assert scores == sorted(scores, reverse=True), "hits must be ordered by the score shown"
        assert scores[0] == 1.0, "the reported score is the ranked one, not the raw fusion score"

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


def _architecture_service(graph: FakeGraph, project: str = "demo") -> ArchitectureViewService:
    return ArchitectureViewService(cast("GraphBackend", graph), project)


class TestArchitecturePopulation:
    """fetch_architecture_pairs states its exclusions instead of hiding them.

    Tests dilute propagation (nothing imports a test, so every test module adds an
    unreachable target to every denominator) and one guessed pair can bridge two
    subsystems in the closure — measured on this repo: 19% hard evidence vs 78%
    with guesses, shipped as a meaningless 41% blend of both effects.
    """

    @staticmethod
    def _patch_graph(monkeypatch) -> None:
        from code_atlas.server.analysis import ModuleGraph

        mg = ModuleGraph(
            modules={
                "app.core": {"file_path": "src/app/core.py"},
                "app.web": {"file_path": "src/app/web.py"},
                "app.util": {"file_path": "src/app/util.py"},
                "tests.test_core": {"file_path": "tests/test_core.py"},
            },
            edges={},
            partition=[],
            directed={
                ("app.web", "app.core"): 3.0,
                ("app.core", "app.util"): 1.0,
                ("tests.test_core", "app.core"): 5.0,
            },
            evidence={
                ("app.web", "app.core"): "structural",
                ("app.core", "app.util"): "guessed",
                ("tests.test_core", "app.core"): "structural",
            },
        )

        async def _fake(graph, project, path, *, test_patterns=()):
            return mg

        monkeypatch.setattr("code_atlas.server.analysis.build_module_graph", _fake)

    async def test_defaults_exclude_tests_and_guesses_and_count_them(self, monkeypatch):
        from code_atlas.server.analysis import fetch_architecture_pairs

        self._patch_graph(monkeypatch)
        source = await fetch_architecture_pairs(cast("GraphBackend", object()), "demo")

        assert set(source.pairs) == {("app.web", "app.core")}
        assert source.excluded_test_modules == 1
        assert source.excluded_guessed_pairs == 1
        assert set(source.all_pairs) == {("app.web", "app.core"), ("app.core", "app.util")}

    async def test_the_flags_restore_each_population(self, monkeypatch):
        from code_atlas.server.analysis import fetch_architecture_pairs

        self._patch_graph(monkeypatch)
        with_tests = await fetch_architecture_pairs(cast("GraphBackend", object()), "demo", include_tests=True)
        assert ("tests.test_core", "app.core") in with_tests.pairs

        with_guessed = await fetch_architecture_pairs(cast("GraphBackend", object()), "demo", include_guessed=True)
        assert ("app.core", "app.util") in with_guessed.pairs
        assert with_guessed.excluded_guessed_pairs == 0


class TestArchitectureView:
    """The mud view (ATL-119).

    The matrix earns its place over a node-link graph only if a healthy architecture and
    a rotten one produce visibly different pictures — so that is what these assert,
    rather than that the numbers merely came back.
    """

    async def test_a_layered_project_puts_every_mark_below_the_diagonal(self):
        """app -> service -> repo, ordered repo, service, app: rows exceed columns."""
        health = await _architecture_service(FakeGraph()).health()

        assert health.dsm_order == ("repo", "service", "app")
        assert all(row > col for row, col in health.dsm_marks), "a DAG must be fully lower-triangular"
        assert health.largest_cycle == 1
        assert health.cycles == ()

    async def test_a_cycle_shows_up_above_the_diagonal(self):
        """The one thing the ordering cannot hide, and the reason to draw it at all."""
        graph = FakeGraph(imports=[("a", "b"), ("b", "a")])

        health = await _architecture_service(graph).health()

        assert any(row < col for row, col in health.dsm_marks), "a cycle must break the triangle"
        assert [c.members for c in health.cycles] == [("a", "b")]
        assert health.core_size == 1.0

    async def test_a_cycle_names_the_edges_that_close_it(self):
        """Members alone say a subsystem is tangled; edges say which import to cut."""
        graph = FakeGraph(imports=[("a", "b"), ("b", "a"), ("a", "outside")])

        [cycle] = (await _architecture_service(graph).health()).cycles

        assert cycle.members == ("a", "b")
        assert cycle.edges == (("a", "b"), ("b", "a"))
        assert ("a", "outside") not in cycle.edges, "an edge leaving the cycle does not close it"

    async def test_propagation_cost_matches_the_hand_worked_value(self):
        """app reaches 2, service reaches 1, repo reaches 0 — 3/(3*2) = 0.5."""
        health = await _architecture_service(FakeGraph()).health()

        assert health.propagation_cost == pytest.approx(0.5)
        assert health.propagation_pct == "50.0%"

    async def test_a_truncated_matrix_says_so(self):
        """An N x N grid is quadratic in the page, so it is capped — but never silently."""
        chain = [(f"m{i}", f"m{i + 1}") for i in range(20)]
        health = await _architecture_service(FakeGraph(imports=chain)).health(dsm_limit=5)

        assert len(health.dsm_order) == 5
        assert health.dsm_truncated is True
        assert health.module_count == 21, "the metrics still cover the whole graph, not just the visible corner"

    async def test_marks_outside_the_visible_window_are_dropped_not_misplaced(self):
        """Truncation must not fold hidden modules onto visible coordinates."""
        chain = [(f"m{i}", f"m{i + 1}") for i in range(20)]
        health = await _architecture_service(FakeGraph(imports=chain)).health(dsm_limit=5)

        assert all(0 <= row < 5 and 0 <= col < 5 for row, col in health.dsm_marks)

    async def test_incomplete_rows_are_ignored(self):
        """A module with no qualified_name is not an edge to nowhere."""
        graph = FakeGraph(imports=[("a", "b")])
        graph.extra_import_rows = [{"from_mod": "c", "to_mod": ""}]

        health = await _architecture_service(graph).health()

        assert health.module_count == 2
        assert health.edge_count == 1

    async def test_a_project_with_no_imports_says_there_is_nothing_to_measure(self):
        """A propagation cost of 0.0 over an empty graph would read as excellent."""
        health = await _architecture_service(FakeGraph(imports=[])).health()

        assert health.module_count == 0
        assert not health.caveat.is_complete

    async def test_the_caveat_never_claims_completeness(self):
        """Extraction coverage varies by language (ATL-096), so this is a lower bound."""
        health = await _architecture_service(FakeGraph()).health()

        assert not health.caveat.is_complete


class TestArchitectureEndpoint:
    @staticmethod
    def _patch_graph(monkeypatch, *, imports: list[tuple[str, str]]) -> None:
        """The page reads the same CALLS+IMPORTS graph as the map; the API keeps the
        import-only contract, so only the page tests patch this."""
        from code_atlas.server.analysis import ModuleGraph

        names = sorted({n for e in imports for n in e})
        module_graph = ModuleGraph(
            modules={n: {"uid": f"u:{n}", "name": n, "qn": n, "file_path": f"src/{n}.py"} for n in names},
            edges={(min(e), max(e)): 1.0 for e in imports},
            directed=dict.fromkeys(imports, 1.0),
            partition=[names],
            evidence=dict.fromkeys(imports, "structural"),
        )

        async def _fake(graph, project, path, *, test_patterns=()):
            return module_graph

        monkeypatch.setattr("code_atlas.server.analysis.build_module_graph", _fake)

    def test_the_page_renders_the_matrix(self, monkeypatch):
        self._patch_graph(monkeypatch, imports=[("app", "service"), ("service", "repo")])

        with _client(FakeGraph(), "demo") as client:
            response = client.get("/architecture/")

        assert response.status_code == 200
        assert "repo.py" in response.text
        assert "Design structure matrix" in response.text
        assert "Propagation cost" in response.text

    def test_the_api_returns_the_same_numbers(self):
        with _client(FakeGraph(), "demo") as client:
            payload = client.get("/architecture/api").json()

        assert payload["propagation_cost"] == pytest.approx(0.5)
        assert payload["dsm_order"] == ["repo", "service", "app"]
        assert payload["largest_cycle"] == 1

    def test_the_page_names_the_edges_closing_each_cycle(self, monkeypatch):
        """The drill-down has to reach the page, not just the view model."""
        self._patch_graph(monkeypatch, imports=[("billing", "orders"), ("orders", "billing")])

        with _client(FakeGraph(), "demo") as client:
            body = client.get("/architecture/").text

        assert "billing.py" in body
        assert "orders.py" in body
        assert "Cut either edge" in body, "the specific edges must render, not only the member list"


class TestArchitectureTrend:
    """Trajectory is what the mud view is for (ATL-121)."""

    async def test_no_history_means_no_trend_rather_than_a_flat_line(self):
        health = await _architecture_service(FakeGraph()).health()

        assert health.trend is None

    async def test_a_single_run_is_not_a_trend(self):
        health = await _architecture_service(FakeGraph().with_snapshots([0.08])).health()

        assert health.trend is None

    async def test_a_rising_cost_is_reported_as_worse(self):
        graph = FakeGraph().with_snapshots([0.06, 0.07, 0.084])

        health = await _architecture_service(graph).health()

        assert health.trend is not None
        assert health.trend.has_trend
        assert health.trend.direction == "worse"
        assert health.trend.propagation_delta_pct == "+2.4%"
        assert len(health.trend.points) == 3

    async def test_a_coverage_change_refuses_to_call_it_decay(self):
        graph = FakeGraph().with_snapshots([(0.06, 100), (0.12, 200)])

        health = await _architecture_service(graph).health()

        assert health.trend is not None
        assert health.trend.coverage_changed
        assert health.trend.direction == "unclear"

    async def test_the_retention_bound_is_stated(self):
        """A window silently capped at fifty runs reads as the whole history."""
        graph = FakeGraph().with_snapshots([0.05, 0.06])

        health = await _architecture_service(graph).health()

        assert health.trend is not None
        assert "50" in health.trend.note

    def test_the_trend_renders_on_the_page(self, monkeypatch):
        TestArchitectureEndpoint._patch_graph(monkeypatch, imports=[("app", "service")])
        graph = FakeGraph().with_snapshots([0.06, 0.084])

        with _client(graph, "demo") as client:
            body = client.get("/architecture/").text

        assert "Across index runs" in body
        assert "worse +2.4 pts" in body


class TestHomeIsTheMap:
    """`/` is the map, not a dashboard (ATL-122)."""

    def test_the_landing_page_is_the_map(self):
        with _client(FakeGraph(), "demo") as client:
            body = client.get("/").text

        assert 'id="map-main"' in body, "the front door is the map island"
        assert "demo" in body, "the header chip names the project"

    def test_a_map_failure_degrades_rather_than_500ing(self):
        """The map became the landing page, so its failure is now the front door.

        FakeGraph has no `execute`, which is exactly what a backend that cannot serve
        the clustering reads looks like. Before this, the homepage returned 500.
        """
        with _client(FakeGraph(), "demo") as client:
            page = client.get("/")
            payload = client.get("/map/api").json()

        assert page.status_code == 200
        assert "could not be built" in payload["unavailable"]

    def test_an_unindexed_project_still_says_how_to_fix_it(self):
        """Distinct from a map that failed — the remedy is different."""
        with _client(FakeGraph(), "nope") as client:
            response = client.get("/")

        assert response.status_code == 404
        assert "atlas index" in response.text

    def test_the_dialog_cookie_switches_the_served_project(self):
        """The projects dialog writes a cookie; every service scopes to it."""
        graph = FakeGraph(
            projects=[
                {"name": "demo", "entity_count": 42},
                {"name": "second", "entity_count": 7},
            ]
        )
        with _client(graph, "demo") as client:
            client.cookies.set("atlas_projects", "second")
            body = client.get("/").text

        assert "second" in body
        assert "7 entities" in body


class TestProjectPicker:
    """Multi-select with monorepo children nested (ATL-122)."""

    @staticmethod
    def _multi() -> FakeGraph:
        return FakeGraph(
            projects=[
                {"name": "demo", "entity_count": 6879, "last_indexed_at": "2026-08-06T00:00:00Z"},
                {"name": "mono", "entity_count": 497, "last_indexed_at": "2026-07-17T00:00:00Z"},
                {"name": "mono/core", "entity_count": 2574, "last_indexed_at": "2026-07-17T00:00:00Z"},
                {"name": "mono/pipeline", "entity_count": 663, "last_indexed_at": "2026-07-17T00:00:00Z"},
            ]
        )

    async def test_monorepo_children_nest_under_their_parent(self):
        picker = await _picker(self._multi()).picker()

        roots = {p.name: p for p in picker.projects}
        assert set(roots) == {"demo", "mono"}
        assert {c.name for c in roots["mono"].children} == {"mono/core", "mono/pipeline"}

    async def test_a_slash_name_without_an_indexed_parent_stays_a_root(self):
        """Otherwise the picker invents a parent that is not in the graph."""
        graph = FakeGraph(projects=[{"name": "orphan/child", "entity_count": 10}])

        picker = await _picker(graph).picker()

        assert [p.name for p in picker.projects] == ["orphan/child"]

    async def test_selecting_several_reports_the_combined_cost(self):
        """Combining projects is the one choice here with a real performance cliff."""
        picker = await _picker(self._multi()).picker(("demo", "mono/core"))

        assert picker.selected_modules == 6879 + 2574
        assert picker.cost_note

    async def test_a_single_selection_needs_no_warning(self):
        picker = await _picker(self._multi()).picker(("demo",))

        assert picker.cost_note == ""

    async def test_the_current_project_is_preselected(self):
        picker = await _picker(self._multi(), "mono").picker()

        assert picker.selected == ("mono",)
        assert next(p for p in picker.projects if p.name == "mono").is_current

    def test_the_api_lists_every_project_with_its_state(self):
        """The dialog renders from this payload — names, nesting and staleness."""
        with _client(self._multi(), "demo") as client:
            payload = client.get("/api/projects").json()

        roots = {p["name"]: p for p in payload["projects"]}
        assert set(roots) == {"demo", "mono"}
        assert {c["name"] for c in roots["mono"]["children"]} == {"mono/core", "mono/pipeline"}
        assert all("state" in p and "indexed_ago" in p for p in payload["projects"])


def _picker(graph: FakeGraph, project: str = "demo"):
    from code_atlas.server.web.services import ProjectPickerService

    return ProjectPickerService(cast("GraphBackend", graph), project)


# The one place ResourceWarning is not a defect signal. `claim_port` exists to hand a
# still-bound socket to uvicorn -- releasing it first reopens the race it closes -- so
# these tests deliberately hold raw sockets and pass ownership around, and a warning
# about an unclosed one says nothing about correctness here.
#
# It is kept narrow deliberately: a global ignore cannot tell this from an abandoned
# client, which is how three real leaks stayed hidden. It may also no longer be needed --
# the CI failure that prompted it turned out to be the SO_REUSEADDR bug in `_bind`, not
# the warning -- but that was only ever reproducible on Linux, so it stays until someone
# has evidence rather than a guess.
@pytest.mark.filterwarnings("ignore::ResourceWarning")
class TestUiInstances:
    """`atlas ui` is run by hand, per checkout, so several are live at once as a matter
    of course. All of them defaulted to 8420 and the second one died on "address already
    in use"."""

    @staticmethod
    def _free_base() -> int:
        """A port number nothing is using, as a starting point for these tests."""
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            return probe.getsockname()[1]

    def test_a_second_claim_does_not_reuse_the_first_port(self):
        """Asserts the contract, not an exact number.

        The first version demanded `base` then `base + 1`, and it failed on a busy
        machine when something else already held `base + 1` -- the scan correctly moved
        on to `base + 2`, which is the behaviour being tested. Any port the OS happens to
        have free is outside this code's control; that the two claims differ and the scan
        moves upward is not.
        """
        from code_atlas.server.web.instances import claim_port

        base = self._free_base()
        first, port_a = claim_port("127.0.0.1", base)
        second, port_b = claim_port("127.0.0.1", base)
        try:
            assert port_a != port_b, "the second invocation took the first one's port"
            assert base <= port_a < port_b, "the scan should move upward from the preferred port"
        finally:
            first.close()
            second.close()

    def test_the_socket_is_returned_still_bound(self):
        """Returning the bound socket rather than closing it is the whole point: a
        check-then-bind leaves two simultaneous invocations able to pick the same port
        and both believe they won."""
        from code_atlas.server.web.instances import claim_port, port_is_free

        base = self._free_base()
        sock, port = claim_port("127.0.0.1", base)
        try:
            assert not port_is_free("127.0.0.1", port), "claim_port released the port it claimed"
        finally:
            sock.close()

    def test_running_out_of_ports_is_an_error_not_a_silent_reuse(self):
        """Exhaustion must raise rather than hand back a port someone already holds.

        Uses span=1 so the only candidate is the one port this test is itself holding.
        The first version claimed a span of three and assumed the OS had left three
        *contiguous* ports free; when something else held one of them, the setup loop
        raised the very OSError the assertion was waiting for, outside pytest.raises,
        and the test failed claiming the code was broken. Second time that assumption
        bit -- a free port is the OS's to give, a held port is ours to guarantee.
        """
        from code_atlas.server.web.instances import claim_port

        base = self._free_base()
        held, _port = claim_port("127.0.0.1", base, span=1)
        try:
            with pytest.raises(OSError, match="No free port"):
                claim_port("127.0.0.1", base, span=1)
        finally:
            held.close()

    def test_a_record_whose_port_is_free_is_pruned(self, tmp_path, monkeypatch):
        """The record is a report, not evidence. A hard kill leaves one behind, and the
        same probe that decides availability is what decides staleness -- which is why
        no pid liveness check is needed. (os.kill(pid, 0) would be the usual one and
        terminates the process on Windows.)"""
        import code_atlas.server.web.instances as mod

        monkeypatch.setattr(mod, "RUNTIME_DIR", tmp_path)
        base = self._free_base()
        with mod.registered("127.0.0.1", base, "ghost", "/tmp/ghost"):
            assert list(tmp_path.glob("*.json"))
            # nothing is listening on that port — the record outlived its process
            assert mod.live_instances() == []
        assert list(tmp_path.glob("*.json")) == []

    def test_a_record_whose_port_is_held_survives(self, tmp_path, monkeypatch):
        import code_atlas.server.web.instances as mod

        monkeypatch.setattr(mod, "RUNTIME_DIR", tmp_path)
        base = self._free_base()
        sock, port = mod.claim_port("127.0.0.1", base)
        try:
            with mod.registered("127.0.0.1", port, "alive", "/tmp/alive"):
                live = mod.live_instances()
                assert [i.project for i in live] == ["alive"]
                assert live[0].url == f"http://127.0.0.1:{port}"
        finally:
            sock.close()

    def test_an_unreadable_record_is_pruned_not_raised(self, tmp_path, monkeypatch):
        """A record written by an older shape must not break the CLI that reads it."""
        import code_atlas.server.web.instances as mod

        monkeypatch.setattr(mod, "RUNTIME_DIR", tmp_path)
        (tmp_path / "127.0.0.1-9999.json").write_text("{not json", encoding="utf-8")

        assert mod.live_instances() == []
        assert list(tmp_path.glob("*.json")) == []

    def test_registration_failure_does_not_stop_the_server(self, tmp_path, monkeypatch):
        """A UI that cannot write its record still serves perfectly well. Refusing to
        start over a bookkeeping detail trades a working tool for a nicety."""
        import code_atlas.server.web.instances as mod

        monkeypatch.setattr(mod, "RUNTIME_DIR", tmp_path / "nested")
        monkeypatch.setattr(mod.Path, "mkdir", _raise_oserror)

        with mod.registered("127.0.0.1", 8420, "proj", "/tmp/proj"):
            pass  # must not raise


def _raise_oserror(*_args, **_kwargs):
    raise OSError("read-only filesystem")
