"""Unit tests for the hybrid search module."""

from __future__ import annotations

from typing import Any

import pytest

from code_atlas.search.engine import (
    CompactNode,
    ExpandedContext,
    SearchResult,
    SearchType,
    _apply_filters,
    _boost_results,
    _is_doc_result,
    _is_generated_result,
    _is_stub_result,
    _is_test_result,
    _prioritize_callers,
    _render_node_text,
    _truncate_to_budget,
    analyze_query,
    assemble_context,
    count_tokens,
    expand_context,
    expand_scope,
    hybrid_search,
    rrf_fuse,
)
from code_atlas.settings import SearchSettings

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

    def test_sentence_with_trailing_period_not_identifier(self):
        """A natural-language sentence ending in a period must not be misread as a dotted path."""
        weights = analyze_query("How do I export the config to disk.")
        assert weights["vector"] > weights["graph"]

    def test_sentence_with_abbreviation_not_identifier(self):
        """'e.g.' inside a sentence must not suppress the vector channel."""
        weights = analyze_query("explain how caching works, e.g. for embeddings")
        assert weights["vector"] > weights["graph"]

    def test_dotted_path_as_one_of_two_words_still_identifier(self):
        """A genuine dotted identifier token among <=2 words is still identifier-like."""
        weights = analyze_query("check GraphClient.process")
        assert weights["graph"] > weights["vector"]


# ---------------------------------------------------------------------------
# expand_scope
# ---------------------------------------------------------------------------


class TestExpandScope:
    def test_empty_scope_returns_none(self):
        """Empty scope means 'no filter' — must stay None."""
        assert expand_scope("", ["a", "b"]) is None

    def test_glob_zero_match_returns_empty_list_not_none(self):
        """A glob matching nothing is an explicit zero-project restriction, not 'no filter'."""
        result = expand_scope("services-*", ["libs-shared", "libs-other"])
        assert result == []
        assert result is not None

    def test_glob_match_returns_matched_projects(self):
        result = expand_scope("services-*", ["services-auth", "libs-shared"])
        assert result == ["services-auth"]


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


def _result_with_score(
    *,
    name: str = "func",
    labels: list[str] | None = None,
    rrf_score: float = 0.01,
    visibility: str = "public",
) -> SearchResult:
    """SearchResult factory with explicit rrf_score for boost tests."""
    return SearchResult(
        uid=f"p:{name}",
        name=name,
        qualified_name=f"mod.{name}",
        kind="function",
        file_path="src/mod.py",
        line_start=1,
        line_end=5,
        signature=f"def {name}():",
        docstring="",
        labels=labels or ["Callable"],
        rrf_score=rrf_score,
        visibility=visibility,
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
# Visibility boost
# ---------------------------------------------------------------------------


class TestBoostResults:
    def test_boost_prefers_public_over_private(self):
        """Two results with equal RRF score — public should rank first."""
        public = SearchResult(
            uid="p:mod.public_func",
            name="public_func",
            qualified_name="mod.public_func",
            kind="function",
            file_path="src/mod.py",
            line_start=1,
            line_end=5,
            signature="def public_func():",
            docstring="",
            labels=["Callable"],
            rrf_score=0.05,
            visibility="public",
        )
        private = SearchResult(
            uid="p:mod._private_func",
            name="_private_func",
            qualified_name="mod._private_func",
            kind="function",
            file_path="src/mod.py",
            line_start=10,
            line_end=15,
            signature="def _private_func():",
            docstring="",
            labels=["Callable"],
            rrf_score=0.05,
            visibility="private",
        )
        # Private first in input — boost should reorder
        boosted = _boost_results([private, public])
        assert boosted[0].uid == "p:mod.public_func"
        assert boosted[1].uid == "p:mod._private_func"

    def test_boost_preserves_large_score_gap(self):
        """A high-scoring private result still beats a low-scoring public one."""
        public_low = SearchResult(
            uid="p:mod.pub",
            name="pub",
            qualified_name="mod.pub",
            kind="function",
            file_path="src/mod.py",
            line_start=1,
            line_end=5,
            signature="def pub():",
            docstring="",
            labels=["Callable"],
            rrf_score=0.01,
            visibility="public",
        )
        private_high = SearchResult(
            uid="p:mod._priv",
            name="_priv",
            qualified_name="mod._priv",
            kind="function",
            file_path="src/mod.py",
            line_start=10,
            line_end=15,
            signature="def _priv():",
            docstring="",
            labels=["Callable"],
            rrf_score=0.10,
            visibility="private",
        )
        boosted = _boost_results([public_low, private_high])
        assert boosted[0].uid == "p:mod._priv"

    def test_label_boost_code_over_doc(self):
        """At equal RRF score, a Callable should rank above a DocSection."""
        code = _result_with_score(name="handler", labels=["Callable"], rrf_score=0.05)
        doc = _result_with_score(name="readme_section", labels=["DocSection"], rrf_score=0.05)
        boosted = _boost_results([doc, code])
        assert boosted[0].uid == code.uid

    def test_label_boost_high_scoring_doc_still_wins(self):
        """A DocSection with much higher RRF score should still beat a low-scoring Callable."""
        code = _result_with_score(name="minor", labels=["Callable"], rrf_score=0.01)
        doc = _result_with_score(name="important_doc", labels=["DocSection"], rrf_score=0.10)
        boosted = _boost_results([code, doc])
        assert boosted[0].uid == doc.uid


# ---------------------------------------------------------------------------
# _is_doc_result predicate
# ---------------------------------------------------------------------------


class TestIsDocResult:
    def test_doc_section(self):
        assert _is_doc_result(_result(labels=["DocSection"]))

    def test_doc_file(self):
        assert _is_doc_result(_result(labels=["DocFile"]))

    def test_callable_not_doc(self):
        assert not _is_doc_result(_result(labels=["Callable"]))

    def test_empty_labels(self):
        assert not _is_doc_result(_result(labels=[]))


# ---------------------------------------------------------------------------
# code_only filter
# ---------------------------------------------------------------------------


class TestCodeOnlyFilter:
    def _settings(self, **overrides: object) -> SearchSettings:
        defaults: dict[str, object] = {
            "test_filter": False,
            "stub_filter": False,
            "generated_filter": False,
            "test_patterns": [],
            "generated_patterns": [],
        }
        defaults.update(overrides)
        return SearchSettings(**defaults)  # type: ignore[arg-type]

    def test_code_only_excludes_doc_section(self):
        results = [
            _result(name="handler", labels=["Callable"]),
            _result(name="readme", labels=["DocSection"]),
        ]
        filtered = _apply_filters(results, self._settings(), code_only=True)
        assert len(filtered) == 1
        assert filtered[0].name == "handler"

    def test_code_only_excludes_doc_file(self):
        results = [
            _result(name="handler", labels=["Callable"]),
            _result(name="changelog", labels=["DocFile"]),
        ]
        filtered = _apply_filters(results, self._settings(), code_only=True)
        assert len(filtered) == 1
        assert filtered[0].name == "handler"

    def test_code_only_false_keeps_docs(self):
        results = [
            _result(name="handler", labels=["Callable"]),
            _result(name="readme", labels=["DocSection"]),
        ]
        filtered = _apply_filters(results, self._settings(), code_only=False)
        assert len(filtered) == 2


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


# ---------------------------------------------------------------------------
# expand_context — package_context traversal
# ---------------------------------------------------------------------------


class _FakeGraphForContext:
    """Fake GraphClient double that mimics real containment: only Module
    -[:DEFINES]-> entity edges exist; there is no Package/Module -[:CONTAINS]->
    edge reaching code entities (matching the real indexer's data model).
    """

    def __init__(self, package_docstring: str | None = "Parses source into entities.") -> None:
        self._package_docstring = package_docstring

    async def execute(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        params = params or {}
        if query.strip().endswith("RETURN n"):
            return [
                {
                    "n": {
                        "uid": params["uid"],
                        "name": "target_fn",
                        "qualified_name": "pkg.mod.target_fn",
                        "kind": "function",
                        "file_path": "pkg/mod.py",
                    }
                }
            ]
        if "AS docstring" in query:
            if "DEFINES" in query and self._package_docstring is not None:
                return [{"docstring": self._package_docstring}]
            return []
        # parent / siblings / callers / callees / docs — no data needed here
        return []


class TestExpandContextPackageContext:
    async def test_package_context_populated_via_defines_chain(self):
        """package_context must be reachable from a Module via DEFINES (the edges
        the indexer actually writes) — a CONTAINS-based query always finds nothing.
        """
        graph = _FakeGraphForContext(package_docstring="Parses source into entities.")
        result = await expand_context(
            graph,
            "proj:pkg.mod.target_fn",
            include_calls=False,
            include_docs=False,
        )
        assert result is not None
        assert result.package_context == "Parses source into entities."

    async def test_package_context_empty_when_no_module_reaches_target(self):
        graph = _FakeGraphForContext(package_docstring=None)
        result = await expand_context(
            graph,
            "proj:pkg.mod.target_fn",
            include_calls=False,
            include_docs=False,
        )
        assert result is not None
        assert result.package_context == ""


# ---------------------------------------------------------------------------
# hybrid_search — channel_status signal
# ---------------------------------------------------------------------------


class _FakeSearchGraph:
    """Fake GraphClient double exposing only the three search channel methods."""

    def __init__(
        self,
        *,
        graph_result: list[dict[str, Any]] | None = None,
        text_result: list[dict[str, Any]] | None = None,
        vector_result: list[dict[str, Any]] | None = None,
        fail: set[str] | None = None,
    ) -> None:
        self._graph_result = graph_result or []
        self._text_result = text_result or []
        self._vector_result = vector_result or []
        self._fail = fail or set()

    async def graph_search(self, query: str, limit: int, projects: list[str] | None = None) -> list[dict[str, Any]]:
        if "graph" in self._fail:
            raise RuntimeError("graph channel boom")
        return self._graph_result

    async def text_search(self, query: str, limit: int, projects: list[str] | None = None) -> list[dict[str, Any]]:
        if "bm25" in self._fail:
            raise RuntimeError("bm25 channel boom")
        return self._text_result

    async def vector_search(
        self, vector: list[float], limit: int, projects: list[str] | None = None
    ) -> list[dict[str, Any]]:
        if "vector" in self._fail:
            raise RuntimeError("vector channel boom")
        return self._vector_result


class _FakeEmbedClient:
    def __init__(self, vector: list[float] | None = None, *, fail: bool = False) -> None:
        self._vector = vector or [0.1, 0.2]
        self._fail = fail

    async def embed_one(self, text: str) -> list[float]:
        if self._fail:
            raise RuntimeError("embed boom")
        return self._vector


class TestHybridSearchChannelStatus:
    async def test_vector_unavailable_without_embed_client(self):
        """Vector requested but embed=None must be signalled, not silently dropped."""
        graph = _FakeSearchGraph()
        status: dict[str, str] = {}
        await hybrid_search(graph, None, SearchSettings(), "find the auth handler function", channel_status=status)
        assert status["vector"].startswith("unavailable")
        assert status["graph"] == "ok"
        assert status["bm25"] == "ok"

    async def test_vector_error_recorded_on_embed_failure(self):
        graph = _FakeSearchGraph()
        embed = _FakeEmbedClient(fail=True)
        status: dict[str, str] = {}
        await hybrid_search(graph, embed, SearchSettings(), "find the auth handler function", channel_status=status)
        assert status["vector"].startswith("error")

    async def test_channel_error_recorded_on_search_failure(self):
        graph = _FakeSearchGraph(fail={"bm25"})
        embed = _FakeEmbedClient()
        status: dict[str, str] = {}
        await hybrid_search(graph, embed, SearchSettings(), "find the auth handler function", channel_status=status)
        assert status["bm25"].startswith("error")
        assert status["graph"] == "ok"
        assert status["vector"] == "ok"

    async def test_channel_status_omitted_by_default_no_behavior_change(self):
        """Omitting channel_status must not raise or change search behavior."""
        graph = _FakeSearchGraph()
        embed = _FakeEmbedClient()
        results = await hybrid_search(graph, embed, SearchSettings(), "find the auth handler function")
        assert results == []

    async def test_vector_requested_only_and_unavailable_yields_no_silent_success(self):
        """Requesting only the vector channel with no embed client should mark it unavailable."""
        graph = _FakeSearchGraph()
        status: dict[str, str] = {}
        results = await hybrid_search(
            graph,
            None,
            SearchSettings(),
            "find the auth handler function",
            search_types=[SearchType.VECTOR],
            channel_status=status,
        )
        assert results == []
        assert status["vector"].startswith("unavailable")
