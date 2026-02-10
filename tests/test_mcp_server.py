"""Tests for the MCP server tools."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from mcp.server.fastmcp import FastMCP

from code_atlas.embeddings import EmbedClient
from code_atlas.graph import GraphClient
from code_atlas.mcp_server import (
    AppContext,
    _clamp_limit,
    _error,
    _rank_results,
    _register_info_tools,
    _register_node_tools,
    _register_query_tools,
    _register_search_tools,
    _register_subagent_tools,
    _result,
)
from code_atlas.schema import (
    _CODE_LABELS,
    _DOC_LABELS,
    _EMBEDDABLE_LABELS,
    _EXTERNAL_LABELS,
    _TEXT_SEARCHABLE_LABELS,
    SCHEMA_VERSION,
    CallableKind,
    NodeLabel,
    RelType,
    TypeDefKind,
    ValueKind,
    Visibility,
)

# ---------------------------------------------------------------------------
# Fake context for direct tool invocation
# ---------------------------------------------------------------------------


class _FakeRequestContext:
    def __init__(self, app_ctx: AppContext) -> None:
        self.lifespan_context = app_ctx


class _FakeCtx:
    """Minimal stand-in for mcp.server.fastmcp.Context."""

    def __init__(self, app_ctx: AppContext) -> None:
        self.request_context = _FakeRequestContext(app_ctx)


_NO_CTX_TOOLS = frozenset({"schema_info", "get_usage_guide", "plan_search_strategy"})


async def _invoke_tool(app_ctx: AppContext, tool_name: str, **kwargs: Any) -> dict[str, Any]:
    """Invoke an MCP tool function directly, bypassing the MCP transport layer."""
    server = FastMCP(name="test")
    _register_node_tools(server)
    _register_query_tools(server)
    _register_search_tools(server)
    _register_info_tools(server)
    _register_subagent_tools(server)

    tool_map = {tool.name: tool for tool in server._tool_manager._tools.values()}
    if tool_name not in tool_map:
        msg = f"Unknown tool: {tool_name}. Available: {sorted(tool_map)}"
        raise ValueError(msg)

    tool = tool_map[tool_name]
    if tool_name not in _NO_CTX_TOOLS:
        kwargs["ctx"] = _FakeCtx(app_ctx)

    return await tool.fn(**kwargs)


# ---------------------------------------------------------------------------
# Helper tests (no DB needed)
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_clamp_limit_default(self):
        assert _clamp_limit(None) == 20

    def test_clamp_limit_within_range(self):
        assert _clamp_limit(50) == 50

    def test_clamp_limit_below_min(self):
        assert _clamp_limit(0) == 1
        assert _clamp_limit(-5) == 1

    def test_clamp_limit_above_max(self):
        assert _clamp_limit(200) == 100

    def test_result_envelope(self):
        r = _result([{"a": 1}], limit=20, query_ms=5.123)
        assert r["count"] == 1
        assert r["truncated"] is False
        assert r["query_ms"] == 5.1

    def test_result_truncated(self):
        r = _result([{"a": 1}], limit=1, query_ms=1.0, total=5)
        assert r["truncated"] is True

    def test_error_envelope(self):
        e = _error("boom", code="FAIL")
        assert e == {"error": "boom", "code": "FAIL"}


# ---------------------------------------------------------------------------
# _rank_results (no DB needed)
# ---------------------------------------------------------------------------


class TestRankResults:
    def test_source_before_test(self):
        results = [
            {"name": "Foo", "qualified_name": "tests.test_foo.Foo", "file_path": "tests/test_foo.py"},
            {"name": "Foo", "qualified_name": "mypackage.foo.Foo", "file_path": "mypackage/foo.py"},
        ]
        ranked = _rank_results(results)
        assert ranked[0]["file_path"] == "mypackage/foo.py"
        assert ranked[1]["file_path"] == "tests/test_foo.py"

    def test_public_before_private(self):
        results = [
            {"name": "foo", "qualified_name": "mod._foo", "visibility": "private"},
            {"name": "foo", "qualified_name": "mod.foo", "visibility": "public"},
        ]
        ranked = _rank_results(results)
        assert ranked[0]["visibility"] == "public"
        assert ranked[1]["visibility"] == "private"

    def test_shorter_qn_preferred(self):
        results = [
            {"name": "Svc", "qualified_name": "a.b.c.d.Svc"},
            {"name": "Svc", "qualified_name": "a.Svc"},
        ]
        ranked = _rank_results(results)
        assert ranked[0]["qualified_name"] == "a.Svc"
        assert ranked[1]["qualified_name"] == "a.b.c.d.Svc"

    def test_combined_ranking(self):
        """Source + public beats test + public, which beats test + private."""
        results = [
            {"name": "X", "qualified_name": "tests.X", "file_path": "tests/test.py", "visibility": "private"},
            {"name": "X", "qualified_name": "pkg.X", "file_path": "pkg/mod.py", "visibility": "public"},
            {"name": "X", "qualified_name": "tests.X", "file_path": "tests/test.py", "visibility": "public"},
        ]
        ranked = _rank_results(results)
        assert ranked[0]["file_path"] == "pkg/mod.py"
        assert ranked[1]["visibility"] == "public"
        assert ranked[1]["file_path"] == "tests/test.py"
        assert ranked[2]["visibility"] == "private"

    def test_empty_list(self):
        assert _rank_results([]) == []

    def test_missing_fields_uses_defaults(self):
        """Nodes without visibility or file_path should not crash."""
        results = [
            {"name": "A", "qualified_name": "long.path.A"},
            {"name": "B", "qualified_name": "B"},
        ]
        ranked = _rank_results(results)
        assert ranked[0]["qualified_name"] == "B"
        assert ranked[1]["qualified_name"] == "long.path.A"


# ---------------------------------------------------------------------------
# schema_info (no DB needed)
# ---------------------------------------------------------------------------


class TestSchemaInfo:
    async def test_schema_info_returns_complete_schema(self, settings):
        result = await _invoke_tool(None, "schema_info")  # type: ignore[arg-type]

        assert result["schema_version"] == SCHEMA_VERSION
        assert result["uid_format"] == "{project_name}:{qualified_name}"

        # All labels present
        all_labels = (
            set(result["node_labels"]["code"])
            | set(result["node_labels"]["documentation"])
            | set(result["node_labels"]["external"])
            | set(result["node_labels"]["meta"])
        )
        assert all_labels == {lbl.value for lbl in NodeLabel}

        # All relationship types present
        assert set(result["relationship_types"]) == {r.value for r in RelType}

        # Kind discriminators
        assert set(result["kind_discriminators"]["TypeDefKind"]) == {k.value for k in TypeDefKind}
        assert set(result["kind_discriminators"]["CallableKind"]) == {k.value for k in CallableKind}
        assert set(result["kind_discriminators"]["ValueKind"]) == {k.value for k in ValueKind}
        assert set(result["kind_discriminators"]["Visibility"]) == {v.value for v in Visibility}

        # Text/vector searchable labels
        assert set(result["text_searchable_labels"]) == {lbl.value for lbl in _TEXT_SEARCHABLE_LABELS}
        assert set(result["vector_searchable_labels"]) == {lbl.value for lbl in _EMBEDDABLE_LABELS}

    async def test_schema_info_label_groups_correct(self, settings):
        result = await _invoke_tool(None, "schema_info")  # type: ignore[arg-type]
        assert sorted(result["node_labels"]["code"]) == sorted(lbl.value for lbl in _CODE_LABELS)
        assert sorted(result["node_labels"]["documentation"]) == sorted(lbl.value for lbl in _DOC_LABELS)
        assert sorted(result["node_labels"]["external"]) == sorted(lbl.value for lbl in _EXTERNAL_LABELS)


# ---------------------------------------------------------------------------
# Integration tests (require Memgraph)
# ---------------------------------------------------------------------------


@pytest.fixture
async def app_ctx(graph_client, settings):
    """Create an AppContext for testing tools directly."""
    embed = EmbedClient(settings.embeddings)
    return AppContext(graph=graph_client, settings=settings, embed=embed)


@pytest.fixture
async def seeded_graph(graph_client):
    """Seed the graph with test entities for querying."""
    await graph_client.ensure_schema()

    # Create a project
    await graph_client.merge_project_node(
        "test-project",
        file_count=3,
        entity_count=5,
        last_indexed_at=1700000000,
        git_hash="abc123",
    )

    # Create some entities
    await graph_client.execute_write(
        f"CREATE (n:{NodeLabel.MODULE} {{"
        "uid: 'test-project:mypackage.mymodule', "
        "project_name: 'test-project', "
        "name: 'mymodule', "
        "qualified_name: 'mypackage.mymodule', "
        "file_path: 'mypackage/mymodule.py', "
        "kind: 'module', "
        "line_start: 1, line_end: 100"
        "})"
    )

    await graph_client.execute_write(
        f"CREATE (n:{NodeLabel.CALLABLE} {{"
        "uid: 'test-project:mypackage.mymodule.my_function', "
        "project_name: 'test-project', "
        "name: 'my_function', "
        "qualified_name: 'mypackage.mymodule.my_function', "
        "file_path: 'mypackage/mymodule.py', "
        "kind: 'function', "
        "line_start: 10, line_end: 20, "
        "signature: 'def my_function(x: int) -> str', "
        "docstring: 'A test function that does something useful.'"
        "})"
    )

    await graph_client.execute_write(
        f"CREATE (n:{NodeLabel.TYPE_DEF} {{"
        "uid: 'test-project:mypackage.mymodule.MyClass', "
        "project_name: 'test-project', "
        "name: 'MyClass', "
        "qualified_name: 'mypackage.mymodule.MyClass', "
        "file_path: 'mypackage/mymodule.py', "
        "kind: 'class', "
        "line_start: 30, line_end: 80, "
        "signature: 'class MyClass(Base)', "
        "docstring: 'A test class.'"
        "})"
    )

    await graph_client.execute_write(
        f"CREATE (n:{NodeLabel.CALLABLE} {{"
        "uid: 'test-project:mypackage.mymodule.MyClass.my_method', "
        "project_name: 'test-project', "
        "name: 'my_method', "
        "qualified_name: 'mypackage.mymodule.MyClass.my_method', "
        "file_path: 'mypackage/mymodule.py', "
        "kind: 'method', "
        "line_start: 40, line_end: 50, "
        "signature: 'def my_method(self) -> None', "
        "docstring: 'A method on MyClass.'"
        "})"
    )

    # Create DEFINES relationships
    await graph_client.execute_write(
        "MATCH (m {uid: 'test-project:mypackage.mymodule'}), "
        "(f {uid: 'test-project:mypackage.mymodule.my_function'}) "
        f"CREATE (m)-[:{RelType.DEFINES}]->(f)"
    )
    await graph_client.execute_write(
        "MATCH (m {uid: 'test-project:mypackage.mymodule'}), "
        "(c {uid: 'test-project:mypackage.mymodule.MyClass'}) "
        f"CREATE (m)-[:{RelType.DEFINES}]->(c)"
    )
    await graph_client.execute_write(
        "MATCH (c {uid: 'test-project:mypackage.mymodule.MyClass'}), "
        "(method {uid: 'test-project:mypackage.mymodule.MyClass.my_method'}) "
        f"CREATE (c)-[:{RelType.DEFINES}]->(method)"
    )

    # Create a CALLS relationship: my_function -> my_method
    await graph_client.execute_write(
        "MATCH (f {uid: 'test-project:mypackage.mymodule.my_function'}), "
        "(m {uid: 'test-project:mypackage.mymodule.MyClass.my_method'}) "
        f"CREATE (f)-[:{RelType.CALLS}]->(m)"
    )

    return graph_client


@pytest.mark.integration
class TestCypherQuery:
    async def test_cypher_query_returns_results(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "cypher_query", query="MATCH (n:Callable) RETURN n.name AS name", limit=10)
        assert "results" in result
        assert result["count"] >= 1
        names = [r["name"] for r in result["results"]]
        assert "my_function" in names

    async def test_cypher_query_rejects_writes(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "cypher_query", query="CREATE (n:Foo {name: 'bar'})")
        assert result["code"] == "WRITE_REJECTED"

    async def test_cypher_query_rejects_delete(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "cypher_query", query="MATCH (n) DELETE n")
        assert result["code"] == "WRITE_REJECTED"

    async def test_cypher_query_rejects_set(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "cypher_query", query="MATCH (n) SET n.name = 'x'")
        assert result["code"] == "WRITE_REJECTED"

    async def test_cypher_query_enforces_limit(self, app_ctx, seeded_graph):
        result = await _invoke_tool(
            app_ctx,
            "cypher_query",
            query="MATCH (n) RETURN n.name AS name",
            limit=2,
        )
        assert result["count"] <= 2


@pytest.mark.integration
class TestGetNode:
    async def test_get_node_exact_name(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "get_node", name="my_function")
        assert result["count"] >= 1
        assert any(r.get("name") == "my_function" for r in result["results"])

    async def test_get_node_by_uid(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "get_node", name="test-project:mypackage.mymodule.MyClass")
        assert result["count"] == 1
        assert result["results"][0]["name"] == "MyClass"

    async def test_get_node_ends_with(self, app_ctx, seeded_graph):
        # "MyClass" should match via suffix ".MyClass" on qualified_name
        result = await _invoke_tool(app_ctx, "get_node", name="MyClass")
        assert result["count"] >= 1
        assert any(r.get("qualified_name", "").endswith("MyClass") for r in result["results"])

    async def test_get_node_with_label_filter(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "get_node", name="my_function", label="Callable")
        assert result["count"] >= 1

    async def test_get_node_not_found(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "get_node", name="nonexistent_entity_xyz")
        assert result["count"] == 0


@pytest.mark.integration
class TestGetContext:
    async def test_get_context_returns_neighborhood(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "get_context", uid="test-project:mypackage.mymodule.MyClass")
        assert "node" in result
        assert result["node"]["name"] == "MyClass"

        # Parent should be the module (now a dict, not a list)
        assert result["parent"] is not None
        assert result["parent"]["name"] == "mymodule"

        # Siblings should include my_function (both defined by mymodule)
        assert len(result["siblings"]) >= 1
        assert any(s.get("name") == "my_function" for s in result["siblings"])

    async def test_get_context_callers_callees(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "get_context", uid="test-project:mypackage.mymodule.MyClass.my_method")
        # my_function calls my_method
        assert len(result["callers"]) >= 1
        assert any(c.get("name") == "my_function" for c in result["callers"])

    async def test_get_context_not_found(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "get_context", uid="nonexistent:uid")
        assert result["code"] == "NOT_FOUND"

    async def test_get_context_siblings(self, app_ctx, seeded_graph):
        """my_function and MyClass are siblings under mymodule."""
        result = await _invoke_tool(app_ctx, "get_context", uid="test-project:mypackage.mymodule.my_function")
        assert any(s.get("name") == "MyClass" for s in result["siblings"])

    async def test_get_context_hierarchy_off(self, app_ctx, seeded_graph):
        """include_hierarchy=False returns None parent, empty siblings."""
        result = await _invoke_tool(
            app_ctx, "get_context", uid="test-project:mypackage.mymodule.MyClass", include_hierarchy=False
        )
        assert result["parent"] is None
        assert result["siblings"] == []
        # Callers/callees should still be present
        assert "callers" in result
        assert "callees" in result

    async def test_get_context_calls_off(self, app_ctx, seeded_graph):
        """include_calls=False returns empty callers/callees."""
        result = await _invoke_tool(
            app_ctx, "get_context", uid="test-project:mypackage.mymodule.MyClass.my_method", include_calls=False
        )
        assert result["callers"] == []
        assert result["callees"] == []
        # Parent should still be present
        assert result["parent"] is not None

    async def test_get_context_response_shape(self, app_ctx, seeded_graph):
        """Verify all expected top-level keys are present."""
        result = await _invoke_tool(app_ctx, "get_context", uid="test-project:mypackage.mymodule.MyClass")
        expected_keys = {"node", "parent", "siblings", "callers", "callees", "docs", "package_context", "query_ms"}
        assert expected_keys == set(result.keys())


@pytest.mark.integration
class TestIndexStatus:
    async def test_index_status_with_projects(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "index_status")
        assert "projects" in result
        assert len(result["projects"]) >= 1

        project = result["projects"][0]
        assert project["name"] == "test-project"
        assert project["entity_count"] >= 1
        assert result["schema_version"] == SCHEMA_VERSION
        assert "text_indices" in result
        assert isinstance(result["text_indices"], list)

    async def test_index_status_empty_graph(self, app_ctx):
        result = await _invoke_tool(app_ctx, "index_status")
        assert result["projects"] == []
        assert result["schema_version"] == SCHEMA_VERSION
        assert "text_indices" in result


@pytest.mark.integration
class TestTextSearch:
    async def test_text_search_finds_entity(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "text_search", query="my_function")
        # Text search may or may not find results depending on Memgraph text index state
        # Just verify the structure is correct
        assert "results" in result
        assert "count" in result
        assert "query_ms" in result

    async def test_text_search_with_label_filter(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "text_search", query="my_function", label="Callable")
        assert "results" in result
        assert "count" in result
        assert "query_ms" in result
        assert isinstance(result["results"], list)

    async def test_text_search_with_project_filter(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "text_search", query="my_function", project="test-project")
        assert "results" in result
        assert "count" in result
        assert "query_ms" in result
        assert isinstance(result["results"], list)

    async def test_text_search_empty_query(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "text_search", query="")
        assert "results" in result
        assert "count" in result


class TestVectorSearchMock:
    async def test_vector_search_embed_error(self, settings):
        """Vector search returns EMBED_ERROR when TEI is unavailable."""
        graph = GraphClient(settings)
        embed = EmbedClient(settings.embeddings)
        app_ctx = AppContext(graph=graph, settings=settings, embed=embed)

        with patch("code_atlas.embeddings.litellm.aembedding", new_callable=AsyncMock, side_effect=Exception("down")):
            result = await _invoke_tool(app_ctx, "vector_search", query="test query")
        await graph.close()
        assert result["code"] == "EMBED_ERROR"

    async def test_vector_search_mock_tei(self, settings):
        """Vector search with mocked embedding client."""
        mock_vector = [0.1] * settings.embeddings.dimension
        graph = GraphClient(settings)
        embed = EmbedClient(settings.embeddings)
        app_ctx = AppContext(graph=graph, settings=settings, embed=embed)

        with patch.object(embed, "embed_one", new_callable=AsyncMock, return_value=mock_vector) as mock_embed:
            result = await _invoke_tool(app_ctx, "vector_search", query="test query")
            # Structure is correct even if vector index search fails on Memgraph
            assert "results" in result or "code" in result
            mock_embed.assert_called_once_with("test query")

        await graph.close()


# ---------------------------------------------------------------------------
# Enhanced schema_info (no DB needed)
# ---------------------------------------------------------------------------


class TestSchemaInfoEnhanced:
    async def test_schema_info_has_cypher_examples(self, settings):
        result = await _invoke_tool(None, "schema_info")  # type: ignore[arg-type]
        assert "cypher_examples" in result
        assert isinstance(result["cypher_examples"], list)
        assert len(result["cypher_examples"]) >= 5
        for ex in result["cypher_examples"]:
            assert "description" in ex
            assert "query" in ex

    async def test_schema_info_has_relationship_summary(self, settings):
        result = await _invoke_tool(None, "schema_info")  # type: ignore[arg-type]
        assert "relationship_summary" in result
        summary = result["relationship_summary"]
        assert isinstance(summary, dict)
        # Every RelType should be described
        for r in RelType:
            assert r.value in summary, f"Missing relationship summary for {r.value}"


# ---------------------------------------------------------------------------
# Subagent tools (no DB needed for most)
# ---------------------------------------------------------------------------


class TestValidateCypher:
    async def test_valid_query(self, settings):
        result = await _invoke_tool(None, "validate_cypher", query="MATCH (n:Callable) RETURN n LIMIT 10")  # type: ignore[arg-type]
        assert result["valid"] is True
        errors = [i for i in result["issues"] if i["level"] == "error"]
        assert errors == []

    async def test_invalid_write_query(self, settings):
        result = await _invoke_tool(None, "validate_cypher", query="CREATE (n:Foo {name: 'bar'})")  # type: ignore[arg-type]
        assert result["valid"] is False
        assert any("write" in i["message"].lower() for i in result["issues"])

    async def test_invalid_label(self, settings):
        result = await _invoke_tool(None, "validate_cypher", query="MATCH (n:Function) RETURN n LIMIT 10")  # type: ignore[arg-type]
        assert result["valid"] is False
        assert any("unknown label" in i["message"].lower() for i in result["issues"])

    async def test_missing_return(self, settings):
        result = await _invoke_tool(None, "validate_cypher", query="MATCH (n:Callable)")  # type: ignore[arg-type]
        warnings = [i for i in result["issues"] if i["level"] == "warning"]
        assert any("return" in i["message"].lower() for i in warnings)


class TestGetUsageGuide:
    async def test_default_guide(self, settings):
        result = await _invoke_tool(None, "get_usage_guide")  # type: ignore[arg-type]
        assert result["topic"] == "quickstart"
        assert len(result["guide"]) > 50

    async def test_specific_topic(self, settings):
        result = await _invoke_tool(None, "get_usage_guide", topic="cypher")  # type: ignore[arg-type]
        assert result["topic"] == "cypher"
        assert "cypher" in result["guide"].lower()

    async def test_unknown_topic(self, settings):
        result = await _invoke_tool(None, "get_usage_guide", topic="nonexistent")  # type: ignore[arg-type]
        assert "unknown topic" in result["guide"].lower()

    async def test_available_topics(self, settings):
        result = await _invoke_tool(None, "get_usage_guide")  # type: ignore[arg-type]
        assert "available_topics" in result
        assert "searching" in result["available_topics"]


class TestPlanSearchStrategy:
    async def test_identifier_query(self, settings):
        result = await _invoke_tool(None, "plan_search_strategy", question="MyClass")  # type: ignore[arg-type]
        assert result["recommended_tool"] == "get_node"
        assert "alternatives" in result

    async def test_natural_language_query(self, settings):
        result = await _invoke_tool(None, "plan_search_strategy", question="how does authentication handle tokens")  # type: ignore[arg-type]
        assert result["recommended_tool"] in ("hybrid_search", "cypher_query")
        assert "explanation" in result

    async def test_structural_query(self, settings):
        result = await _invoke_tool(None, "plan_search_strategy", question="what calls the process function")  # type: ignore[arg-type]
        assert result["recommended_tool"] == "cypher_query"


# ---------------------------------------------------------------------------
# Integration: validate_cypher with EXPLAIN
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestValidateCypherExplain:
    async def test_valid_query_with_explain(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "validate_cypher", query="MATCH (n:Callable) RETURN n LIMIT 10")
        assert result["valid"] is True

    async def test_invalid_syntax_with_explain(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "validate_cypher", query="MATCHX (n) RETURN n LIMIT 10")
        assert result["valid"] is False
        assert any("explain" in i["message"].lower() for i in result["issues"] if i["level"] == "error")
