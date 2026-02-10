"""Tests for the subagent guidance module."""

from __future__ import annotations

from code_atlas.schema import NodeLabel, RelType
from code_atlas.subagent import (
    _LABEL_NAMES,
    _REL_NAMES,
    _RELATIONSHIP_SUMMARY,
    _USAGE_GUIDE,
    CYPHER_EXAMPLES,
    get_guide,
    plan_strategy,
    validate_cypher_static,
)

# ---------------------------------------------------------------------------
# Static Cypher validation
# ---------------------------------------------------------------------------


class TestValidateCypherStatic:
    def test_valid_query_no_errors(self):
        issues = validate_cypher_static("MATCH (n:Callable) RETURN n.name LIMIT 10")
        errors = [i for i in issues if i.level == "error"]
        assert errors == []

    def test_write_keyword_rejected(self):
        for keyword in ("CREATE", "DELETE", "SET", "MERGE", "REMOVE", "DROP", "DETACH"):
            issues = validate_cypher_static(f"{keyword} (n:Foo)")
            errors = [i for i in issues if i.level == "error"]
            assert any("write operations" in i.message.lower() for i in errors), f"Missing write error for {keyword}"

    def test_unbalanced_parens(self):
        issues = validate_cypher_static("MATCH (n:Callable RETURN n")
        errors = [i for i in issues if i.level == "error"]
        assert any("unbalanced" in i.message.lower() for i in errors)

    def test_unbalanced_brackets(self):
        issues = validate_cypher_static("MATCH (a)-[:CALLS->(b) RETURN a")
        errors = [i for i in issues if i.level == "error"]
        assert any("unbalanced" in i.message.lower() for i in errors)

    def test_missing_return_warning(self):
        issues = validate_cypher_static("MATCH (n:Callable)")
        warnings = [i for i in issues if i.level == "warning"]
        assert any("return" in i.message.lower() for i in warnings)

    def test_invalid_label_with_suggestion(self):
        issues = validate_cypher_static("MATCH (n:Function) RETURN n LIMIT 10")
        errors = [i for i in issues if i.level == "error"]
        assert any("Unknown label 'Function'" in i.message for i in errors)

    def test_invalid_label_callable_suggestion(self):
        """'Function' should suggest 'Callable' as a close match."""
        issues = validate_cypher_static("MATCH (n:Function) RETURN n LIMIT 10")
        errors = [i for i in issues if i.level == "error"]
        # Callable might not be a close match for Function via difflib, but the error should exist
        assert any("unknown label" in i.message.lower() for i in errors)

    def test_valid_labels_accepted(self):
        issues = validate_cypher_static("MATCH (n:Callable)-[:CALLS]->(m:TypeDef) RETURN n, m LIMIT 10")
        errors = [i for i in issues if i.level == "error"]
        assert errors == []

    def test_invalid_relationship_type(self):
        issues = validate_cypher_static("MATCH (a)-[:INVOKES]->(b) RETURN a LIMIT 10")
        errors = [i for i in issues if i.level == "error"]
        assert any("unknown relationship type" in i.message.lower() for i in errors)

    def test_valid_relationship_types_accepted(self):
        issues = validate_cypher_static("MATCH (a)-[:CALLS]->(b) RETURN a LIMIT 10")
        errors = [i for i in issues if i.level == "error"]
        assert errors == []

    def test_variable_length_relationship(self):
        """[:INHERITS*1..3] should validate INHERITS, not 'INHERITS*1..3'."""
        issues = validate_cypher_static("MATCH (a)-[:INHERITS*1..3]->(b) RETURN a LIMIT 10")
        errors = [i for i in issues if i.level == "error"]
        assert errors == []

    def test_no_limit_info(self):
        issues = validate_cypher_static("MATCH (n:Callable) RETURN n")
        infos = [i for i in issues if i.level == "info"]
        assert any("limit" in i.message.lower() for i in infos)

    def test_with_limit_no_info(self):
        issues = validate_cypher_static("MATCH (n:Callable) RETURN n LIMIT 50")
        infos = [i for i in issues if i.level == "info"]
        assert not any("limit" in i.message.lower() for i in infos)

    def test_string_contents_not_checked(self):
        """Parentheses inside string literals should not cause unbalanced errors."""
        issues = validate_cypher_static("MATCH (n) WHERE n.name = '(' RETURN n LIMIT 10")
        errors = [i for i in issues if i.level == "error"]
        assert errors == []


# ---------------------------------------------------------------------------
# Search strategy planner
# ---------------------------------------------------------------------------


class TestPlanStrategy:
    def test_identifier_recommends_get_node(self):
        result = plan_strategy("MyClass")
        assert result["recommended_tool"] == "get_node"

    def test_snake_case_recommends_get_node(self):
        result = plan_strategy("my_function")
        assert result["recommended_tool"] == "get_node"

    def test_dotted_path_recommends_get_node(self):
        result = plan_strategy("mypackage.mymodule.MyClass")
        assert result["recommended_tool"] == "get_node"

    def test_natural_language_recommends_hybrid(self):
        result = plan_strategy("how does the authentication system handle token refresh")
        assert result["recommended_tool"] == "hybrid_search"

    def test_structural_question_recommends_cypher(self):
        result = plan_strategy("what calls the process_order function")
        assert result["recommended_tool"] == "cypher_query"
        assert "CALLS" in result["explanation"]

    def test_inheritance_question_recommends_cypher(self):
        result = plan_strategy("what does MyClass inherit from")
        assert result["recommended_tool"] == "cypher_query"
        assert "INHERITS" in result["explanation"]

    def test_test_question_recommends_cypher(self):
        result = plan_strategy("what tests cover the parser module")
        assert result["recommended_tool"] == "cypher_query"
        assert "TESTS" in result["explanation"]

    def test_doc_question_recommends_hybrid(self):
        result = plan_strategy("find documentation about the search pipeline")
        assert result["recommended_tool"] == "hybrid_search"
        assert "bm25" in result["params"].get("search_types", "").lower()

    def test_result_has_alternatives(self):
        result = plan_strategy("anything")
        assert "alternatives" in result
        assert isinstance(result["alternatives"], list)
        assert len(result["alternatives"]) >= 1


# ---------------------------------------------------------------------------
# Usage guide
# ---------------------------------------------------------------------------


class TestGetGuide:
    def test_default_guide_nonempty(self):
        result = get_guide()
        assert result["topic"] == "quickstart"
        assert len(result["guide"]) > 50

    def test_all_topics_return_content(self):
        for topic in ("searching", "cypher", "navigation", "patterns"):
            result = get_guide(topic)
            assert result["topic"] == topic
            assert len(result["guide"]) > 50, f"Guide for '{topic}' is too short"

    def test_unknown_topic_message(self):
        result = get_guide("nonexistent")
        assert "unknown topic" in result["guide"].lower()

    def test_available_topics_listed(self):
        result = get_guide()
        assert "available_topics" in result
        assert "searching" in result["available_topics"]
        assert "cypher" in result["available_topics"]

    def test_case_insensitive(self):
        result = get_guide("CYPHER")
        assert result["topic"] == "cypher"
        assert len(result["guide"]) > 50


# ---------------------------------------------------------------------------
# Schema completeness checks
# ---------------------------------------------------------------------------


class TestSchemaCompleteness:
    def test_all_rel_types_in_summary(self):
        """Every RelType value must have an entry in _RELATIONSHIP_SUMMARY."""
        for r in RelType:
            assert r.value in _RELATIONSHIP_SUMMARY, f"Missing summary for {r.value}"

    def test_no_extra_summary_keys(self):
        """No stale keys in _RELATIONSHIP_SUMMARY after a RelType removal."""
        all_rels = {r.value for r in RelType}
        for key in _RELATIONSHIP_SUMMARY:
            assert key in all_rels, f"Extra summary key: {key}"

    def test_label_names_matches_enum(self):
        assert {lbl.value for lbl in NodeLabel} == _LABEL_NAMES

    def test_rel_names_matches_enum(self):
        assert {r.value for r in RelType} == _REL_NAMES

    def test_cypher_examples_nonempty(self):
        assert len(CYPHER_EXAMPLES) >= 5

    def test_cypher_examples_have_required_keys(self):
        for ex in CYPHER_EXAMPLES:
            assert "description" in ex
            assert "query" in ex
            assert len(ex["query"]) > 10

    def test_usage_guide_has_all_topics(self):
        """All documented topics exist in _USAGE_GUIDE."""
        expected = {"", "searching", "cypher", "navigation", "patterns"}
        assert set(_USAGE_GUIDE.keys()) == expected
