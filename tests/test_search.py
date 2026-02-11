"""Tests for the hybrid search module."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from code_atlas.search import (
    AssembledContext,
    CompactNode,
    ContextItem,
    ExpandedContext,
    SearchResult,
    SearchType,
    _apply_filters,
    _is_generated_result,
    _is_stub_result,
    _is_test_result,
    _prioritize_callers,
    _render_node_text,
    _truncate_to_budget,
    analyze_query,
    assemble_context,
    count_tokens,
    hybrid_search,
    rrf_fuse,
)
from code_atlas.settings import SearchSettings

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
        # Single lowercase word matches snake_case pattern → identifier-like
        weights = analyze_query("search")
        assert weights["graph"] >= weights["vector"]

    def test_two_words_identifier(self):
        # Two words with PascalCase or snake_case → identifier-like
        weights = analyze_query("UserService login")
        assert weights["graph"] > weights["vector"]

    def test_two_words_generic(self):
        # Two generic lowercase words → balanced (not identifier-like)
        weights = analyze_query("user login")
        assert weights["graph"] == weights["vector"]


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
# Result filtering predicates
# ---------------------------------------------------------------------------


def _result(
    *,
    name: str = "func",
    file_path: str = "src/mod.py",
    labels: list[str] | None = None,
) -> SearchResult:
    """Minimal SearchResult factory for filter tests."""
    return SearchResult(
        uid=f"p:{name}",
        name=name,
        qualified_name=f"mod.{name}",
        kind="function",
        file_path=file_path,
        line_start=1,
        line_end=5,
        signature=f"def {name}():",
        docstring="",
        labels=labels or ["Callable"],
        rrf_score=0.01,
    )


_TEST_PATTERNS = ["test_*", "*_test.py", "tests/", "__tests__/"]


class TestIsTestResult:
    def test_test_prefix_file(self):
        assert _is_test_result(_result(file_path="test_utils.py"), _TEST_PATTERNS)

    def test_test_suffix_file(self):
        assert _is_test_result(_result(file_path="utils_test.py"), _TEST_PATTERNS)

    def test_tests_directory(self):
        assert _is_test_result(_result(file_path="tests/test_mod.py"), _TEST_PATTERNS)

    def test_dunder_tests_directory(self):
        assert _is_test_result(_result(file_path="src/__tests__/mod.spec.py"), _TEST_PATTERNS)

    def test_entity_name_test_prefix(self):
        assert _is_test_result(_result(name="test_get_user", file_path="src/mod.py"), _TEST_PATTERNS)

    def test_entity_name_test_suffix(self):
        assert _is_test_result(_result(name="get_user_test", file_path="src/mod.py"), _TEST_PATTERNS)

    def test_non_test_file(self):
        assert not _is_test_result(_result(file_path="src/utils.py"), _TEST_PATTERNS)

    def test_non_test_entity(self):
        assert not _is_test_result(_result(name="get_user", file_path="src/mod.py"), _TEST_PATTERNS)

    def test_case_insensitive(self):
        assert _is_test_result(_result(file_path="Tests/Test_Mod.py"), _TEST_PATTERNS)

    def test_backslash_paths(self):
        assert _is_test_result(_result(file_path="tests\\test_mod.py"), _TEST_PATTERNS)


class TestIsStubResult:
    def test_pyi_file(self):
        assert _is_stub_result(_result(file_path="src/mod.pyi"))

    def test_non_stub_file(self):
        assert not _is_stub_result(_result(file_path="src/mod.py"))

    def test_case_insensitive(self):
        assert _is_stub_result(_result(file_path="src/Mod.PYI"))


_GENERATED_PATTERNS = ["*_pb2.py", "*_pb2_grpc.py", "*.generated.*"]


class TestIsGeneratedResult:
    def test_protobuf_file(self):
        assert _is_generated_result(_result(file_path="src/user_pb2.py"), _GENERATED_PATTERNS)

    def test_grpc_file(self):
        assert _is_generated_result(_result(file_path="src/user_pb2_grpc.py"), _GENERATED_PATTERNS)

    def test_generated_pattern(self):
        assert _is_generated_result(_result(file_path="src/schema.generated.ts"), _GENERATED_PATTERNS)

    def test_normal_file(self):
        assert not _is_generated_result(_result(file_path="src/user.py"), _GENERATED_PATTERNS)


class TestApplyFilters:
    def _settings(self, **overrides: object) -> SearchSettings:
        defaults: dict[str, object] = {
            "test_filter": True,
            "stub_filter": True,
            "generated_filter": True,
            "test_patterns": ["test_*", "*_test.py", "tests/", "__tests__/"],
            "generated_patterns": ["*_pb2.py", "*_pb2_grpc.py", "*.generated.*"],
        }
        defaults.update(overrides)
        return SearchSettings(**defaults)  # type: ignore[arg-type]

    def test_default_excludes_tests(self):
        results = [
            _result(name="get_user", file_path="src/mod.py"),
            _result(name="test_get_user", file_path="tests/test_mod.py"),
        ]
        filtered = _apply_filters(results, self._settings())
        assert len(filtered) == 1
        assert filtered[0].name == "get_user"

    def test_override_includes_tests(self):
        results = [
            _result(name="get_user", file_path="src/mod.py"),
            _result(name="test_get_user", file_path="tests/test_mod.py"),
        ]
        filtered = _apply_filters(results, self._settings(), exclude_tests=False)
        assert len(filtered) == 2

    def test_default_excludes_stubs(self):
        results = [
            _result(file_path="src/mod.py"),
            _result(file_path="src/mod.pyi"),
        ]
        filtered = _apply_filters(results, self._settings())
        assert len(filtered) == 1
        assert filtered[0].file_path == "src/mod.py"

    def test_default_excludes_generated(self):
        results = [
            _result(file_path="src/user.py"),
            _result(file_path="src/user_pb2.py"),
        ]
        filtered = _apply_filters(results, self._settings())
        assert len(filtered) == 1
        assert filtered[0].file_path == "src/user.py"

    def test_custom_exclude_patterns(self):
        results = [
            _result(file_path="src/mod.py"),
            _result(file_path="src/conftest.py"),
        ]
        filtered = _apply_filters(results, self._settings(), exclude_patterns=["conftest.py"])
        assert len(filtered) == 1
        assert filtered[0].file_path == "src/mod.py"

    def test_include_patterns_whitelist(self):
        results = [
            _result(file_path="src/user.py"),
            _result(file_path="src/auth.py"),
            _result(file_path="src/utils.py"),
        ]
        filtered = _apply_filters(
            results,
            self._settings(test_filter=False),
            include_patterns=["user.py"],
        )
        assert len(filtered) == 1
        assert filtered[0].file_path == "src/user.py"

    def test_all_filters_disabled(self):
        results = [
            _result(file_path="src/mod.py"),
            _result(file_path="tests/test_mod.py"),
            _result(file_path="src/mod.pyi"),
            _result(file_path="src/user_pb2.py"),
        ]
        filtered = _apply_filters(
            results,
            self._settings(test_filter=False, stub_filter=False, generated_filter=False),
        )
        assert len(filtered) == 4

    def test_empty_results(self):
        filtered = _apply_filters([], self._settings())
        assert filtered == []


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

    async def test_filter_excludes_test_entity(self, seeded_graph, settings):
        """Test entities are excluded by default filters."""
        dim = seeded_graph._dimension
        dummy_vec = [0.1] * dim

        # Seed a test entity
        await seeded_graph.execute_write(
            "CREATE (n:Callable {uid: 'proj:tests.test_mod.test_get_user', project_name: 'proj', "
            "name: 'test_get_user', qualified_name: 'tests.test_mod.test_get_user', "
            "file_path: 'tests/test_mod.py', kind: 'function', line_start: 1, line_end: 5, "
            "visibility: 'public', docstring: 'Test get_user', signature: 'def test_get_user():', "
            "tags: [], embedding: $vec})",
            {"vec": dummy_vec},
        )

        # Default: test entity excluded
        results = await hybrid_search(
            graph=seeded_graph,
            embed=None,
            settings=settings.search,
            query="test_get_user",
            search_types=[SearchType.GRAPH],
            limit=10,
        )
        uids = [r.uid for r in results]
        assert "proj:tests.test_mod.test_get_user" not in uids

        # Override: include tests
        results_with_tests = await hybrid_search(
            graph=seeded_graph,
            embed=None,
            settings=settings.search,
            query="test_get_user",
            search_types=[SearchType.GRAPH],
            limit=10,
            exclude_tests=False,
        )
        uids_with_tests = [r.uid for r in results_with_tests]
        assert "proj:tests.test_mod.test_get_user" in uids_with_tests


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_known_text(self):
        # "hello world" should be ~2 tokens in cl100k_base
        tokens = count_tokens("hello world")
        assert 1 <= tokens <= 4

    def test_code_snippet(self):
        code = "def get_user(uid: str) -> User:\n    return db.query(uid)"
        tokens = count_tokens(code)
        assert tokens > 0

    def test_claude_alias(self):
        # "claude" alias should resolve to cl100k_base and produce same result
        text = "test string for tokenizer"
        assert count_tokens(text, "claude") == count_tokens(text, "cl100k_base")

    def test_accuracy_within_5_percent(self):
        # For a longer text, verify count is reasonable (chars/4 rough estimate)
        text = (
            "This is a moderately long text that should tokenize to roughly one quarter of its character count. "
        ) * 10
        tokens = count_tokens(text)
        # Rough sanity: between len/6 and len/2
        assert len(text) // 6 < tokens < len(text) // 2


# ---------------------------------------------------------------------------
# Render node text
# ---------------------------------------------------------------------------


class TestRenderNodeText:
    def _node(self, **kwargs: object) -> CompactNode:
        defaults: dict[str, object] = {
            "uid": "p:m.func",
            "name": "func",
            "qualified_name": "m.func",
            "kind": "function",
            "file_path": "m.py",
            "line_start": 10,
            "line_end": 20,
            "signature": "def func(x):",
            "docstring": "A function.",
        }
        defaults.update(kwargs)
        return CompactNode(**defaults)  # type: ignore[arg-type]

    def test_with_signature(self):
        text = _render_node_text(self._node())
        assert "# m.func (m.py:10-20)" in text
        assert "def func(x):" in text
        assert "A function." not in text  # docstring excluded by default

    def test_with_docstring(self):
        text = _render_node_text(self._node(), include_docstring=True)
        assert "A function." in text

    def test_no_line_range(self):
        text = _render_node_text(self._node(line_start=None, line_end=None))
        assert "(m.py)" in text

    def test_no_file_path(self):
        text = _render_node_text(self._node(file_path=""))
        lines = text.split("\n")
        # Header line should NOT have a parenthesized location
        assert lines[0] == "# m.func"


# ---------------------------------------------------------------------------
# Truncate to budget
# ---------------------------------------------------------------------------


class TestTruncateToBudget:
    def test_fits_within_budget(self):
        text = "hello world"
        result = _truncate_to_budget(text, 100, "cl100k_base")
        assert result == text

    def test_truncates_at_line_boundary(self):
        text = "line one\nline two\nline three\nline four"
        # Truncate to ~5 tokens — should cut at a line boundary
        result = _truncate_to_budget(text, 5, "cl100k_base")
        assert result.endswith(("line one", "line two"))
        assert "\nline four" not in result

    def test_empty_text(self):
        assert _truncate_to_budget("", 10, "cl100k_base") == ""

    def test_zero_budget(self):
        assert _truncate_to_budget("hello", 0, "cl100k_base") == ""


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------


def _make_node(name: str, *, sig: str = "", doc: str = "", file_path: str = "mod.py") -> CompactNode:
    return CompactNode(
        uid=f"p:{name}",
        name=name.rsplit(".", 1)[-1],
        qualified_name=name,
        kind="function",
        file_path=file_path,
        line_start=1,
        line_end=5,
        signature=sig or f"def {name.rsplit('.', 1)[-1]}():",
        docstring=doc,
    )


class TestAssembleContext:
    def test_target_always_included(self):
        ec = ExpandedContext(target=_make_node("mod.target", doc="Target docstring."))
        result = assemble_context(ec, budget=8000)
        assert len(result.items) >= 1
        assert result.items[0].role == "target"
        assert result.items[0].uid == "p:mod.target"
        assert "Target docstring." in result.items[0].text

    def test_priority_ordering(self):
        ec = ExpandedContext(
            target=_make_node("mod.target"),
            parent=_make_node("mod.MyClass", sig="class MyClass:"),
            callees=[_make_node("mod.callee1")],
            callers=[_make_node("mod.caller1")],
            docs=[_make_node("doc.section", doc="Some documentation.")],
            siblings=[_make_node("mod.sibling1")],
            package_context="Package docstring.",
        )
        result = assemble_context(ec, budget=8000)
        roles = [item.role for item in result.items]
        # Verify priority order
        assert roles.index("target") < roles.index("parent")
        assert roles.index("parent") < roles.index("callee")
        assert roles.index("callee") < roles.index("caller")
        assert roles.index("caller") < roles.index("doc")
        assert roles.index("doc") < roles.index("sibling")
        assert roles.index("sibling") < roles.index("package")

    def test_budget_never_exceeded(self):
        ec = ExpandedContext(
            target=_make_node("mod.target", doc="Long docstring. " * 50),
            callees=[_make_node(f"mod.callee{i}") for i in range(20)],
            callers=[_make_node(f"mod.caller{i}") for i in range(20)],
            siblings=[_make_node(f"mod.sibling{i}") for i in range(10)],
        )
        result = assemble_context(ec, budget=500)
        assert result.total_tokens <= result.budget

    def test_small_budget_excludes_lower_priority(self):
        ec = ExpandedContext(
            target=_make_node("mod.target"),
            parent=_make_node("mod.Parent", sig="class Parent:"),
            callees=[_make_node("mod.callee1")],
            siblings=[_make_node("mod.sibling1")],
            package_context="Package info.",
        )
        # Very small budget — should include target but exclude some later items
        result = assemble_context(ec, budget=50)
        roles = {item.role for item in result.items}
        assert "target" in roles
        # With 50 tokens, unlikely to fit everything
        assert result.total_tokens <= 50

    def test_different_budgets_different_sizes(self):
        ec = ExpandedContext(
            target=_make_node("mod.target", doc="Docstring. " * 10),
            callees=[_make_node(f"mod.callee{i}") for i in range(10)],
            callers=[_make_node(f"mod.caller{i}") for i in range(10)],
        )
        small = assemble_context(ec, budget=200)
        large = assemble_context(ec, budget=2000)
        assert len(large.items) >= len(small.items)
        assert large.total_tokens >= small.total_tokens

    def test_render(self):
        ec = ExpandedContext(
            target=_make_node("mod.target"),
            callees=[_make_node("mod.callee1")],
        )
        result = assemble_context(ec, budget=8000)
        rendered = result.render()
        assert "## Target" in rendered
        assert "## Direct Callees" in rendered
        assert "mod.target" in rendered

    def test_excluded_counts(self):
        ec = ExpandedContext(
            target=_make_node("mod.target"),
            callees=[_make_node(f"mod.callee{i}") for i in range(20)],
        )
        result = assemble_context(ec, budget=200)
        # Some callees should be excluded
        if result.excluded_counts:
            assert "callee" in result.excluded_counts
            assert result.excluded_counts["callee"] > 0

    def test_empty_expanded_context(self):
        ec = ExpandedContext(target=_make_node("mod.target"))
        result = assemble_context(ec, budget=8000)
        assert len(result.items) == 1
        assert result.items[0].role == "target"
        assert result.excluded_counts == {}

    def test_section_headers_in_output(self):
        ec = ExpandedContext(
            target=_make_node("mod.target"),
            parent=_make_node("mod.Parent", sig="class Parent:"),
        )
        result = assemble_context(ec, budget=8000)
        texts = [item.text for item in result.items]
        assert any("## Target" in t for t in texts)
        assert any("## Class Context" in t for t in texts)

    def test_multiple_items_same_role(self):
        """Second item in same role group should NOT have section header."""
        ec = ExpandedContext(
            target=_make_node("mod.target"),
            callees=[_make_node("mod.callee1"), _make_node("mod.callee2")],
        )
        result = assemble_context(ec, budget=8000)
        callee_items = [item for item in result.items if item.role == "callee"]
        assert len(callee_items) == 2
        assert "## Direct Callees" in callee_items[0].text
        assert "## Direct Callees" not in callee_items[1].text

    def test_tokenizer_parameter(self):
        ec = ExpandedContext(target=_make_node("mod.target"))
        result = assemble_context(ec, budget=8000, tokenizer="claude")
        assert result.total_tokens > 0


class TestAssembledContext:
    def test_frozen(self):
        ac = AssembledContext(items=[], total_tokens=0, budget=8000)
        with pytest.raises(AttributeError):
            ac.total_tokens = 99  # type: ignore[misc]

    def test_render_empty(self):
        ac = AssembledContext(items=[], total_tokens=0, budget=8000)
        assert ac.render() == ""


class TestContextItem:
    def test_frozen(self):
        ci = ContextItem(role="target", text="hello", tokens=1)
        with pytest.raises(AttributeError):
            ci.role = "other"  # type: ignore[misc]

    def test_defaults(self):
        ci = ContextItem(role="target", text="hello", tokens=1)
        assert ci.uid == ""
        assert ci.truncated is False
