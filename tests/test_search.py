"""Tests for the hybrid search module."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from code_atlas.search import (
    CompactNode,
    ExpandedContext,
    SearchResult,
    SearchType,
    _prioritize_callers,
    analyze_query,
    hybrid_search,
    rrf_fuse,
)

# ---------------------------------------------------------------------------
# SearchType enum
# ---------------------------------------------------------------------------


class TestSearchType:
    def test_values(self):
        assert SearchType.GRAPH == "graph"
        assert SearchType.VECTOR == "vector"
        assert SearchType.BM25 == "bm25"

    def test_list_all(self):
        assert set(SearchType) == {"graph", "vector", "bm25"}


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------


class TestRRFFuse:
    def test_single_channel(self):
        ranked = {"bm25": ["a", "b", "c"]}
        scores = rrf_fuse(ranked, k=60)
        assert list(scores.keys()) == ["a", "b", "c"]
        # Score for rank 0: 1/(60+1) ≈ 0.01639
        assert scores["a"] == pytest.approx(1 / 61, rel=1e-6)
        assert scores["b"] == pytest.approx(1 / 62, rel=1e-6)

    def test_multi_channel_overlap_boost(self):
        ranked = {"graph": ["a", "b"], "bm25": ["b", "c"]}
        scores = rrf_fuse(ranked, k=60)
        # "b" appears in both channels: 1/62 + 1/61
        assert scores["b"] > scores["a"]
        assert scores["b"] > scores["c"]

    def test_weights(self):
        ranked = {"graph": ["a"], "bm25": ["a"]}
        scores_equal = rrf_fuse(ranked, k=60, weights={"graph": 1.0, "bm25": 1.0})
        scores_weighted = rrf_fuse(ranked, k=60, weights={"graph": 3.0, "bm25": 1.0})
        # With higher graph weight, score should be higher
        assert scores_weighted["a"] > scores_equal["a"]

    def test_empty_input(self):
        assert rrf_fuse({}, k=60) == {}
        assert rrf_fuse({"graph": []}, k=60) == {}

    def test_k_parameter_effect(self):
        ranked = {"bm25": ["a", "b"]}
        scores_low_k = rrf_fuse(ranked, k=1)
        scores_high_k = rrf_fuse(ranked, k=100)
        # Lower k gives higher scores overall and bigger spread
        assert scores_low_k["a"] > scores_high_k["a"]
        spread_low = scores_low_k["a"] - scores_low_k["b"]
        spread_high = scores_high_k["a"] - scores_high_k["b"]
        assert spread_low > spread_high


# ---------------------------------------------------------------------------
# Query analysis
# ---------------------------------------------------------------------------


class TestAnalyzeQuery:
    def test_pascal_case(self):
        weights = analyze_query("UserService")
        assert weights["graph"] > weights["vector"]

    def test_snake_case(self):
        weights = analyze_query("get_user_by_id")
        assert weights["graph"] > weights["vector"]

    def test_dotted_path(self):
        weights = analyze_query("code_atlas.graph.GraphClient")
        assert weights["graph"] > weights["vector"]

    def test_natural_language(self):
        weights = analyze_query("find all functions that handle authentication")
        assert weights["vector"] > weights["graph"]

    def test_short_generic(self):
        # Short queries (≤2 words) bias toward identifier matching
        weights = analyze_query("search")
        assert weights["graph"] >= weights["vector"]

    def test_two_words(self):
        weights = analyze_query("user login")
        assert weights["graph"] > weights["vector"]


# ---------------------------------------------------------------------------
# SearchResult dataclass
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_frozen(self):
        r = SearchResult(
            uid="proj:mod.Foo",
            name="Foo",
            qualified_name="mod.Foo",
            kind="class",
            file_path="mod.py",
            line_start=1,
            line_end=10,
            signature="class Foo:",
            docstring="A class.",
            labels=["TypeDef"],
            rrf_score=0.05,
            sources={"graph": 1, "bm25": 2},
        )
        assert r.uid == "proj:mod.Foo"
        assert r.rrf_score == 0.05
        assert r.sources == {"graph": 1, "bm25": 2}
        with pytest.raises(AttributeError):
            r.name = "Bar"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CompactNode dataclass
# ---------------------------------------------------------------------------


class TestCompactNode:
    def test_frozen(self):
        n = CompactNode(uid="p:m.Foo", name="Foo", qualified_name="m.Foo", kind="class", file_path="m.py")
        assert n.uid == "p:m.Foo"
        with pytest.raises(AttributeError):
            n.name = "Bar"  # type: ignore[misc]

    def test_defaults(self):
        n = CompactNode(uid="p:x", name="x", qualified_name="x", kind="function", file_path="x.py")
        assert n.line_start is None
        assert n.line_end is None
        assert n.signature == ""
        assert n.docstring == ""
        assert n.labels == []

    def test_all_fields(self):
        n = CompactNode(
            uid="p:m.f",
            name="f",
            qualified_name="m.f",
            kind="function",
            file_path="m.py",
            line_start=10,
            line_end=20,
            signature="def f():",
            docstring="A function.",
            labels=["Callable"],
        )
        assert n.line_start == 10
        assert n.labels == ["Callable"]


# ---------------------------------------------------------------------------
# ExpandedContext dataclass
# ---------------------------------------------------------------------------


class TestExpandedContext:
    def test_frozen(self):
        target = CompactNode(uid="p:x", name="x", qualified_name="x", kind="function", file_path="x.py")
        ec = ExpandedContext(target=target)
        assert ec.target is target
        with pytest.raises(AttributeError):
            ec.target = target  # type: ignore[misc]

    def test_defaults(self):
        target = CompactNode(uid="p:x", name="x", qualified_name="x", kind="function", file_path="x.py")
        ec = ExpandedContext(target=target)
        assert ec.parent is None
        assert ec.siblings == []
        assert ec.callees == []
        assert ec.callers == []
        assert ec.docs == []
        assert ec.package_context == ""


# ---------------------------------------------------------------------------
# _prioritize_callers
# ---------------------------------------------------------------------------


class TestPrioritizeCallers:
    def _make(self, qn: str, file_path: str = "src/mod.py") -> CompactNode:
        name = qn.rsplit(".", 1)[-1]
        return CompactNode(uid=f"p:{qn}", name=name, qualified_name=qn, kind="function", file_path=file_path)

    def test_same_package_first(self):
        target_qn = "pkg.mod.target_func"
        callers = [
            self._make("other.lib.caller_a"),
            self._make("pkg.mod.caller_b"),
        ]
        ranked = _prioritize_callers(callers, target_qn)
        assert ranked[0].qualified_name == "pkg.mod.caller_b"
        assert ranked[1].qualified_name == "other.lib.caller_a"

    def test_non_test_first(self):
        target_qn = "pkg.mod.func"
        callers = [
            self._make("tests.test_mod.test_func", file_path="tests/test_mod.py"),
            self._make("pkg.other.caller", file_path="pkg/other.py"),
        ]
        ranked = _prioritize_callers(callers, target_qn)
        assert ranked[0].qualified_name == "pkg.other.caller"
        assert ranked[1].qualified_name == "tests.test_mod.test_func"

    def test_combined_ranking(self):
        """Same-package + non-test > different-package + non-test > test."""
        target_qn = "pkg.mod.func"
        callers = [
            self._make("tests.test_mod.test_func", file_path="tests/test_mod.py"),
            self._make("other.lib.helper", file_path="other/lib.py"),
            self._make("pkg.mod.nearby", file_path="pkg/mod.py"),
        ]
        ranked = _prioritize_callers(callers, target_qn)
        assert ranked[0].qualified_name == "pkg.mod.nearby"
        assert ranked[1].qualified_name == "other.lib.helper"
        assert ranked[2].qualified_name == "tests.test_mod.test_func"

    def test_shorter_qn_tiebreak(self):
        target_qn = "pkg.mod.func"
        callers = [
            self._make("pkg.mod.very_long_name_caller"),
            self._make("pkg.mod.short"),
        ]
        ranked = _prioritize_callers(callers, target_qn)
        assert ranked[0].qualified_name == "pkg.mod.short"

    def test_empty_list(self):
        assert _prioritize_callers([], "pkg.mod.func") == []


# ---------------------------------------------------------------------------
# Integration tests — hybrid_search
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestHybridSearchIntegration:
    """Integration tests requiring a live Memgraph instance."""

    @pytest.fixture
    async def seeded_graph(self, graph_client, settings):
        """Seed a few entities for search testing."""
        await graph_client.ensure_schema()

        # Create entities of different labels
        dim = graph_client._dimension
        dummy_vec = [0.1] * dim

        await graph_client.execute_write(
            "CREATE (n:Callable {uid: 'proj:mod.get_user', project_name: 'proj', "
            "name: 'get_user', qualified_name: 'mod.get_user', file_path: 'mod.py', "
            "kind: 'function', line_start: 10, line_end: 20, "
            "visibility: 'public', docstring: 'Get a user by ID', signature: 'def get_user(uid):', "
            "tags: [], embedding: $vec})",
            {"vec": dummy_vec},
        )
        await graph_client.execute_write(
            "CREATE (n:TypeDef {uid: 'proj:mod.UserService', project_name: 'proj', "
            "name: 'UserService', qualified_name: 'mod.UserService', file_path: 'mod.py', "
            "kind: 'class', line_start: 30, line_end: 50, "
            "visibility: 'public', docstring: 'Handles user operations', signature: 'class UserService:', "
            "tags: [], embedding: $vec})",
            {"vec": dummy_vec},
        )
        await graph_client.execute_write(
            "CREATE (n:Callable {uid: 'other:lib.get_user', project_name: 'other', "
            "name: 'get_user', qualified_name: 'lib.get_user', file_path: 'lib.py', "
            "kind: 'function', line_start: 1, line_end: 5, "
            "visibility: 'public', docstring: 'Another get_user', signature: 'def get_user():', "
            "tags: [], embedding: $vec})",
            {"vec": dummy_vec},
        )

        return graph_client

    async def test_graph_only(self, seeded_graph, settings):
        """Graph-only search finds entities by name matching."""
        results = await hybrid_search(
            graph=seeded_graph,
            embed=None,
            settings=settings.search,
            query="get_user",
            search_types=[SearchType.GRAPH],
            limit=10,
        )
        assert len(results) >= 2
        uids = [r.uid for r in results]
        assert "proj:mod.get_user" in uids
        assert "other:lib.get_user" in uids

    async def test_scope_filtering(self, seeded_graph, settings):
        """Scope filters results to a single project."""
        results = await hybrid_search(
            graph=seeded_graph,
            embed=None,
            settings=settings.search,
            query="get_user",
            search_types=[SearchType.GRAPH],
            limit=10,
            scope="proj",
        )
        uids = [r.uid for r in results]
        assert "proj:mod.get_user" in uids
        assert "other:lib.get_user" not in uids

    async def test_sources_provenance(self, seeded_graph, settings):
        """Each result tracks which channels found it."""
        results = await hybrid_search(
            graph=seeded_graph,
            embed=None,
            settings=settings.search,
            query="UserService",
            search_types=[SearchType.GRAPH],
            limit=10,
        )
        assert len(results) >= 1
        svc = next(r for r in results if r.uid == "proj:mod.UserService")
        assert "graph" in svc.sources
        assert svc.sources["graph"] >= 1

    async def test_hybrid_with_mock_embed(self, seeded_graph, settings):
        """Hybrid search with mocked embedding client."""
        dim = seeded_graph._dimension
        mock_embed = AsyncMock()
        mock_embed.embed_one = AsyncMock(return_value=[0.1] * dim)

        results = await hybrid_search(
            graph=seeded_graph,
            embed=mock_embed,
            settings=settings.search,
            query="get_user",
            limit=10,
        )
        # Should have results from multiple channels
        assert len(results) >= 1
        # At least one result should have graph source
        any_graph = any("graph" in r.sources for r in results)
        assert any_graph

    async def test_embed_failure_graceful(self, seeded_graph, settings):
        """If embedding fails, vector channel is skipped but others work."""
        mock_embed = AsyncMock()
        mock_embed.embed_one = AsyncMock(side_effect=RuntimeError("TEI down"))

        results = await hybrid_search(
            graph=seeded_graph,
            embed=mock_embed,
            settings=settings.search,
            query="get_user",
            limit=10,
        )
        # Should still get results from graph and/or bm25
        assert len(results) >= 1
        # No result should have vector source
        assert not any("vector" in r.sources for r in results)
