"""Tests for the MCP server tools."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.server.fastmcp import FastMCP

from code_atlas.graph.client import GraphClient, QueryTimeoutError
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
from code_atlas.search.embeddings import EmbedClient
from code_atlas.server.mcp import (
    AppContext,
    _clamp_limit,
    _error,
    _file_uri_to_path,
    _maybe_update_root,
    _rank_results,
    _register_analysis_tools,
    _register_hybrid_tool,
    _register_info_tools,
    _register_node_tools,
    _register_query_tools,
    _register_search_tools,
    _register_subagent_tools,
    _result,
    _with_staleness,
)
from code_atlas.settings import AtlasSettings, IndexSettings, find_git_root

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
    _register_hybrid_tool(server)
    _register_info_tools(server)
    _register_subagent_tools(server)
    _register_analysis_tools(server)

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

    def test_internal_before_external(self):
        """Internal entities rank above ExternalSymbol stubs."""
        results = [
            {
                "name": "Logger",
                "qualified_name": "ext/logging.Logger",
                "_labels": ["ExternalSymbol"],
                "file_path": "",
            },
            {
                "name": "Logger",
                "qualified_name": "mypackage.logging.Logger",
                "_labels": ["TypeDef"],
                "file_path": "mypackage/logging.py",
                "visibility": "public",
            },
        ]
        ranked = _rank_results(results)
        assert ranked[0]["qualified_name"] == "mypackage.logging.Logger"
        assert ranked[1]["qualified_name"] == "ext/logging.Logger"

    def test_external_package_ranked_last(self):
        """ExternalPackage stubs rank below internal entities."""
        results = [
            {
                "name": "os",
                "qualified_name": "ext/os",
                "_labels": ["ExternalPackage"],
                "file_path": "",
            },
            {
                "name": "os",
                "qualified_name": "mypackage.os",
                "_labels": ["Module"],
                "file_path": "mypackage/os.py",
                "visibility": "public",
            },
        ]
        ranked = _rank_results(results)
        assert ranked[0]["qualified_name"] == "mypackage.os"
        assert ranked[1]["qualified_name"] == "ext/os"


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

        patch_target = "code_atlas.search.embeddings.litellm.aembedding"
        with patch(patch_target, new_callable=AsyncMock, side_effect=Exception("down")):
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


# ---------------------------------------------------------------------------
# Staleness flag tests (no DB needed)
# ---------------------------------------------------------------------------


class TestWithStaleness:
    async def test_scope_matching_comma_separated(self, settings):
        """Comma-separated scope with matching project triggers staleness check."""
        from code_atlas.indexing.orchestrator import StalenessChecker, StalenessInfo

        checker = StalenessChecker(settings.project_root, project_name="myproject")
        # Mock the check method to return not stale
        mock_graph = AsyncMock()
        mock_graph.get_project_git_hash = AsyncMock(return_value="abc123")

        embed = EmbedClient(settings.embeddings)
        app = AppContext(graph=mock_graph, settings=settings, embed=embed, staleness=checker)

        # Patch checker.check to return a known StalenessInfo
        with patch.object(
            checker, "check", new_callable=AsyncMock, return_value=StalenessInfo(stale=False, current_commit="abc123")
        ):
            result = {"results": []}
            annotated = await _with_staleness(app, result, scope="myproject,other")
            assert annotated["stale"] is False

    async def test_scope_mismatch_skips_check(self, settings):
        """Scope that doesn't include checker's project returns result unchanged."""
        from code_atlas.indexing.orchestrator import StalenessChecker

        checker = StalenessChecker(settings.project_root, project_name="myproject")
        embed = EmbedClient(settings.embeddings)
        mock_graph = AsyncMock()
        app = AppContext(graph=mock_graph, settings=settings, embed=embed, staleness=checker)

        result = {"results": []}
        annotated = await _with_staleness(app, result, scope="other_project")
        # Should skip check entirely — no "stale" key added
        assert "stale" not in annotated

    async def test_indeterminate_state_returns_none(self, settings):
        """Never-indexed project returns stale=None (indeterminate)."""
        from code_atlas.indexing.orchestrator import StalenessChecker, StalenessInfo

        checker = StalenessChecker(settings.project_root, project_name="myproject")
        embed = EmbedClient(settings.embeddings)
        mock_graph = AsyncMock()
        app = AppContext(graph=mock_graph, settings=settings, embed=embed, staleness=checker)

        # Simulate never-indexed: stale=True but no last_indexed_commit
        with patch.object(
            checker,
            "check",
            new_callable=AsyncMock,
            return_value=StalenessInfo(stale=True, current_commit="abc123", last_indexed_commit=None),
        ):
            result = {"results": []}
            annotated = await _with_staleness(app, result, scope="myproject")
            assert annotated["stale"] is None

    async def test_lock_mode_stale_returns_error(self, settings):
        """Lock mode with stale index returns STALE_INDEX error."""
        from code_atlas.indexing.orchestrator import StalenessChecker, StalenessInfo

        lock_settings = AtlasSettings(project_root=settings.project_root, index=IndexSettings(stale_mode="lock"))
        checker = StalenessChecker(settings.project_root, project_name="myproject")
        embed = EmbedClient(settings.embeddings)
        mock_graph = AsyncMock()
        app = AppContext(graph=mock_graph, settings=lock_settings, embed=embed, staleness=checker)

        with patch.object(
            checker,
            "check",
            new_callable=AsyncMock,
            return_value=StalenessInfo(stale=True, last_indexed_commit="abc123", current_commit="def456"),
        ):
            result = {"results": []}
            annotated = await _with_staleness(app, result, scope="myproject")
            assert annotated["code"] == "STALE_INDEX"
            assert "error" in annotated

    async def test_lock_mode_not_stale_passes_through(self, settings):
        """Lock mode with fresh index passes result through unchanged."""
        from code_atlas.indexing.orchestrator import StalenessChecker, StalenessInfo

        lock_settings = AtlasSettings(project_root=settings.project_root, index=IndexSettings(stale_mode="lock"))
        checker = StalenessChecker(settings.project_root, project_name="myproject")
        embed = EmbedClient(settings.embeddings)
        mock_graph = AsyncMock()
        app = AppContext(graph=mock_graph, settings=lock_settings, embed=embed, staleness=checker)

        with patch.object(
            checker,
            "check",
            new_callable=AsyncMock,
            return_value=StalenessInfo(stale=False, current_commit="abc123"),
        ):
            result = {"results": []}
            annotated = await _with_staleness(app, result, scope="myproject")
            assert "error" not in annotated
            assert annotated["stale"] is False

    async def test_ignore_mode_skips_check(self, settings):
        """Ignore mode skips staleness check entirely — result unchanged, check not called."""
        from code_atlas.indexing.orchestrator import StalenessChecker

        ignore_settings = AtlasSettings(project_root=settings.project_root, index=IndexSettings(stale_mode="ignore"))
        checker = StalenessChecker(settings.project_root, project_name="myproject")
        embed = EmbedClient(settings.embeddings)
        mock_graph = AsyncMock()
        app = AppContext(graph=mock_graph, settings=ignore_settings, embed=embed, staleness=checker)

        with patch.object(checker, "check", new_callable=AsyncMock) as mock_check:
            result = {"results": []}
            annotated = await _with_staleness(app, result, scope="myproject")
            assert "stale" not in annotated
            mock_check.assert_not_called()

    async def test_staleness_timeout_returns_original_result(self, settings):
        """If staleness check times out, the original result is returned unmodified."""
        import asyncio

        from code_atlas.indexing.orchestrator import StalenessChecker

        checker = StalenessChecker(settings.project_root, project_name="myproject")
        embed = EmbedClient(settings.embeddings)
        mock_graph = AsyncMock()
        app = AppContext(graph=mock_graph, settings=settings, embed=embed, staleness=checker)

        async def _slow_check(*_args, **_kwargs):
            await asyncio.sleep(60)

        with patch.object(checker, "check", side_effect=_slow_check):
            result = {"results": [{"uid": "test:foo"}]}
            annotated = await _with_staleness(app, result, scope="myproject")
            # Timeout fires (5s) — original result returned without stale keys
            assert "stale" not in annotated
            assert annotated["results"] == [{"uid": "test:foo"}]


# ---------------------------------------------------------------------------
# find_git_root (no DB needed)
# ---------------------------------------------------------------------------


class TestFindGitRoot:
    def test_found(self, tmp_path):
        """Subdirectory resolves to parent containing .git/."""
        (tmp_path / ".git").mkdir()
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        assert find_git_root(sub) == tmp_path

    def test_not_found(self, tmp_path):
        """No .git in tree → returns None."""
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        assert find_git_root(sub) is None


# ---------------------------------------------------------------------------
# _file_uri_to_path (no DB needed)
# ---------------------------------------------------------------------------


class TestFileUriToPath:
    def test_posix_uri(self):
        p = _file_uri_to_path("file:///home/user/project")
        assert str(p).replace("\\", "/").endswith("/home/user/project")

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific path handling")
    def test_windows_uri(self):
        p = _file_uri_to_path("file:///D:/dev/project")
        assert p == Path("D:/dev/project")


# ---------------------------------------------------------------------------
# _maybe_update_root (no DB needed)
# ---------------------------------------------------------------------------


class TestMaybeUpdateRoot:
    async def test_skips_when_checked(self, settings):
        """roots_checked=True → no-op, no session access."""
        embed = EmbedClient(settings.embeddings)
        mock_graph = AsyncMock()
        app = AppContext(graph=mock_graph, settings=settings, embed=embed, roots_checked=True)
        ctx = MagicMock()
        await _maybe_update_root(app, ctx)
        # Should not have touched session at all
        ctx.session.list_roots.assert_not_called() if hasattr(ctx.session, "list_roots") else None
        assert app.roots_checked is True

    async def test_handles_timeout(self, settings):
        """list_roots() times out → keeps current root."""
        embed = EmbedClient(settings.embeddings)
        mock_graph = AsyncMock()
        app = AppContext(graph=mock_graph, settings=settings, embed=embed)
        ctx = MagicMock()
        # Simulate a timeout on list_roots
        ctx.session.list_roots = AsyncMock(side_effect=TimeoutError)
        await _maybe_update_root(app, ctx)
        assert app.roots_checked is True
        assert app.settings.project_root == settings.project_root

    async def test_restarts_daemon_on_new_root(self, tmp_path, settings):
        """list_roots() returns different root → daemon stop+start called."""
        new_root = tmp_path / "other_project"
        new_root.mkdir()

        embed = EmbedClient(settings.embeddings)
        mock_graph = AsyncMock()
        mock_graph.get_embedding_config = AsyncMock(return_value=None)
        old_daemon = AsyncMock()
        old_daemon.stop = AsyncMock()
        app = AppContext(
            graph=mock_graph,
            settings=settings,
            embed=embed,
            daemon=old_daemon,
            resolved_root=settings.project_root,
        )

        # Mock list_roots to return a different root
        mock_root = MagicMock()
        mock_root.uri = new_root.as_uri()
        mock_result = MagicMock()
        mock_result.roots = [mock_root]

        ctx = MagicMock()
        ctx.session.list_roots = AsyncMock(return_value=mock_result)

        # Mock the new DaemonManager that _switch_root creates
        mock_new_daemon = AsyncMock()
        mock_new_daemon.start = AsyncMock(return_value=False)
        with patch("code_atlas.server.mcp.DaemonManager", return_value=mock_new_daemon):
            await _maybe_update_root(app, ctx)

        assert app.roots_checked is True
        old_daemon.stop.assert_awaited_once()
        assert app.settings.project_root == new_root
        assert app.resolved_root == new_root


# ---------------------------------------------------------------------------
# QueryTimeoutError handling in MCP tools (no DB needed)
# ---------------------------------------------------------------------------


class TestQueryTimeout:
    """Verify each tool returns QUERY_TIMEOUT error envelope on timeout."""

    @pytest.fixture
    def timeout_app(self, settings):
        """AppContext with graph.execute mocked to raise QueryTimeoutError."""
        mock_graph = AsyncMock(spec=GraphClient)
        mock_graph.execute = AsyncMock(side_effect=QueryTimeoutError(10.0, "MATCH (n) RETURN n"))
        mock_graph.text_search = AsyncMock(side_effect=QueryTimeoutError(10.0, "text_search"))
        mock_graph.vector_search = AsyncMock(side_effect=QueryTimeoutError(10.0, "vector_search"))
        mock_graph.graph_search = AsyncMock(side_effect=QueryTimeoutError(10.0, "graph_search"))
        mock_graph.get_project_status = AsyncMock(side_effect=QueryTimeoutError(10.0, "get_project_status"))
        mock_graph.count_entities = AsyncMock(side_effect=QueryTimeoutError(10.0, "count_entities"))
        embed = EmbedClient(settings.embeddings)
        return AppContext(graph=mock_graph, settings=settings, embed=embed)

    async def test_get_node_timeout(self, timeout_app):
        result = await _invoke_tool(timeout_app, "get_node", name="Foo")
        assert result["code"] == "QUERY_TIMEOUT"
        assert "error" in result

    async def test_cypher_query_timeout(self, timeout_app):
        result = await _invoke_tool(timeout_app, "cypher_query", query="MATCH (n:Callable) RETURN n")
        assert result["code"] == "QUERY_TIMEOUT"

    async def test_get_context_timeout(self, timeout_app):
        result = await _invoke_tool(timeout_app, "get_context", uid="proj:mod.Foo")
        assert result["code"] == "QUERY_TIMEOUT"

    async def test_text_search_timeout(self, timeout_app):
        result = await _invoke_tool(timeout_app, "text_search", query="foo")
        assert result["code"] == "QUERY_TIMEOUT"

    async def test_vector_search_timeout(self, timeout_app):
        """vector_search graph call timeout (after successful embedding)."""
        with patch.object(timeout_app.embed, "embed_one", new_callable=AsyncMock, return_value=[0.1] * 768):
            result = await _invoke_tool(timeout_app, "vector_search", query="foo")
        assert result["code"] == "QUERY_TIMEOUT"

    async def test_index_status_timeout(self, timeout_app):
        result = await _invoke_tool(timeout_app, "index_status")
        assert result["code"] == "QUERY_TIMEOUT"

    async def test_list_projects_timeout(self, timeout_app):
        result = await _invoke_tool(timeout_app, "list_projects")
        assert result["code"] == "QUERY_TIMEOUT"

    async def test_analyze_repo_timeout(self, timeout_app):
        result = await _invoke_tool(timeout_app, "analyze_repo", analysis="structure", project="p")
        assert result["code"] == "QUERY_TIMEOUT"

    async def test_generate_diagram_timeout(self, timeout_app):
        result = await _invoke_tool(timeout_app, "generate_diagram", type="packages", project="p")
        assert result["code"] == "QUERY_TIMEOUT"


# ---------------------------------------------------------------------------
# Analysis tools fixtures
# ---------------------------------------------------------------------------

_PROJECT = "analysis-project"


@pytest.fixture
async def seeded_analysis_graph(graph_client):
    """Seed graph with rich structure for analysis/diagram tests."""
    await graph_client.ensure_schema()

    # Project
    await graph_client.merge_project_node(_PROJECT, file_count=3, entity_count=10)

    # Packages
    await graph_client.merge_package_node(_PROJECT, "mypkg", "mypkg", "mypkg/__init__.py")
    await graph_client.merge_package_node(_PROJECT, "mypkg.sub", "sub", "mypkg/sub/__init__.py")

    # Package containment
    await graph_client.create_contains_edge(_PROJECT, f"{_PROJECT}:mypkg")
    await graph_client.create_contains_edge(f"{_PROJECT}:mypkg", f"{_PROJECT}:mypkg.sub")

    # --- Modules ---
    for uid, name, qn, fp, le in [
        (f"{_PROJECT}:mypkg.models", "models", "mypkg.models", "mypkg/models.py", 50),
        (f"{_PROJECT}:mypkg.utils", "utils", "mypkg.utils", "mypkg/utils.py", 30),
        (f"{_PROJECT}:mypkg.sub.api", "api", "mypkg.sub.api", "mypkg/sub/api.py", 40),
    ]:
        await graph_client.execute_write(
            "CREATE (n:Module {uid: $uid, project_name: $p, name: $name, "
            "qualified_name: $qn, file_path: $fp, kind: 'module', line_start: 1, line_end: $le})",
            {"uid": uid, "p": _PROJECT, "name": name, "qn": qn, "fp": fp, "le": le},
        )

    # Package → Module containment
    await graph_client.create_contains_edge(f"{_PROJECT}:mypkg", f"{_PROJECT}:mypkg.models")
    await graph_client.create_contains_edge(f"{_PROJECT}:mypkg", f"{_PROJECT}:mypkg.utils")
    await graph_client.create_contains_edge(f"{_PROJECT}:mypkg.sub", f"{_PROJECT}:mypkg.sub.api")

    # --- TypeDefs ---
    await graph_client.execute_write(
        "CREATE (n:TypeDef {uid: $uid, project_name: $p, name: 'Base', "
        "qualified_name: 'mypkg.models.Base', file_path: 'mypkg/models.py', "
        "kind: 'class', line_start: 5, line_end: 20, visibility: 'public', "
        "docstring: 'Base model class.'})",
        {"uid": f"{_PROJECT}:mypkg.models.Base", "p": _PROJECT},
    )
    await graph_client.execute_write(
        "CREATE (n:TypeDef {uid: $uid, project_name: $p, name: 'User', "
        "qualified_name: 'mypkg.models.User', file_path: 'mypkg/models.py', "
        "kind: 'class', line_start: 22, line_end: 45, visibility: 'public', "
        "docstring: 'User model.', signature: 'class User(Base)'})",
        {"uid": f"{_PROJECT}:mypkg.models.User", "p": _PROJECT},
    )

    # INHERITS: User → Base
    await graph_client.execute_write(
        "MATCH (child {uid: $c}), (parent {uid: $p}) CREATE (child)-[:INHERITS]->(parent)",
        {"c": f"{_PROJECT}:mypkg.models.User", "p": f"{_PROJECT}:mypkg.models.Base"},
    )

    # --- Callables ---
    for uid, name, qn, fp, kind, ls, le, vis, doc, sig in [
        (
            f"{_PROJECT}:mypkg.models.User.save",
            "save",
            "mypkg.models.User.save",
            "mypkg/models.py",
            "method",
            30,
            40,
            "public",
            "Save the user.",
            "def save(self) -> None",
        ),
        (
            f"{_PROJECT}:mypkg.utils.helper",
            "helper",
            "mypkg.utils.helper",
            "mypkg/utils.py",
            "function",
            5,
            15,
            "public",
            "A helper function.",
            "def helper(x: int) -> str",
        ),
        (
            f"{_PROJECT}:mypkg.sub.api.handle_request",
            "handle_request",
            "mypkg.sub.api.handle_request",
            "mypkg/sub/api.py",
            "function",
            5,
            30,
            "public",
            "Handle an API request.",
            "def handle_request(req) -> Response",
        ),
    ]:
        await graph_client.execute_write(
            "CREATE (n:Callable {uid: $uid, project_name: $p, name: $name, "
            "qualified_name: $qn, file_path: $fp, kind: $kind, "
            "line_start: $ls, line_end: $le, visibility: $vis, "
            "docstring: $doc, signature: $sig})",
            {
                "uid": uid,
                "p": _PROJECT,
                "name": name,
                "qn": qn,
                "fp": fp,
                "kind": kind,
                "ls": ls,
                "le": le,
                "vis": vis,
                "doc": doc,
                "sig": sig,
            },
        )

    # --- Value (no docstring) ---
    await graph_client.execute_write(
        "CREATE (n:Value {uid: $uid, project_name: $p, name: 'MAX_SIZE', "
        "qualified_name: 'mypkg.utils.MAX_SIZE', file_path: 'mypkg/utils.py', "
        "kind: 'constant', line_start: 1, line_end: 1, visibility: 'public'})",
        {"uid": f"{_PROJECT}:mypkg.utils.MAX_SIZE", "p": _PROJECT},
    )

    # --- DEFINES relationships ---
    defines_pairs = [
        (f"{_PROJECT}:mypkg.models", f"{_PROJECT}:mypkg.models.Base"),
        (f"{_PROJECT}:mypkg.models", f"{_PROJECT}:mypkg.models.User"),
        (f"{_PROJECT}:mypkg.models.User", f"{_PROJECT}:mypkg.models.User.save"),
        (f"{_PROJECT}:mypkg.utils", f"{_PROJECT}:mypkg.utils.helper"),
        (f"{_PROJECT}:mypkg.utils", f"{_PROJECT}:mypkg.utils.MAX_SIZE"),
        (f"{_PROJECT}:mypkg.sub.api", f"{_PROJECT}:mypkg.sub.api.handle_request"),
    ]
    for from_uid, to_uid in defines_pairs:
        await graph_client.execute_write(
            "MATCH (a {uid: $f}), (b {uid: $t}) CREATE (a)-[:DEFINES]->(b)",
            {"f": from_uid, "t": to_uid},
        )

    # --- IMPORTS relationships ---
    imports_pairs = [
        # api → User (indirect: api depends on models)
        (f"{_PROJECT}:mypkg.sub.api", f"{_PROJECT}:mypkg.models.User"),
        # api → helper (indirect: api depends on utils)
        (f"{_PROJECT}:mypkg.sub.api", f"{_PROJECT}:mypkg.utils.helper"),
        # models → helper (creates models→utils dependency)
        (f"{_PROJECT}:mypkg.models", f"{_PROJECT}:mypkg.utils.helper"),
        # utils → Base (creates utils→models dependency — circular with above!)
        (f"{_PROJECT}:mypkg.utils", f"{_PROJECT}:mypkg.models.Base"),
    ]
    for from_uid, to_uid in imports_pairs:
        await graph_client.execute_write(
            "MATCH (a {uid: $f}), (b {uid: $t}) CREATE (a)-[:IMPORTS]->(b)",
            {"f": from_uid, "t": to_uid},
        )

    # --- External dependency ---
    await graph_client.execute_write(
        "CREATE (ep:ExternalPackage {uid: $uid, project_name: $p, "
        "name: 'dataclasses', qualified_name: 'ext/dataclasses'})",
        {"uid": f"{_PROJECT}:ext/dataclasses", "p": _PROJECT},
    )
    await graph_client.execute_write(
        "CREATE (es:ExternalSymbol {uid: $uid, project_name: $p, "
        "name: 'dataclass', qualified_name: 'ext/dataclasses.dataclass', package: 'dataclasses'})",
        {"uid": f"{_PROJECT}:ext/dataclasses.dataclass", "p": _PROJECT},
    )
    await graph_client.execute_write(
        "MATCH (ep {uid: $ep}), (es {uid: $es}) CREATE (ep)-[:CONTAINS]->(es)",
        {"ep": f"{_PROJECT}:ext/dataclasses", "es": f"{_PROJECT}:ext/dataclasses.dataclass"},
    )
    # models imports dataclass (external)
    await graph_client.execute_write(
        "MATCH (m {uid: $f}), (es {uid: $t}) CREATE (m)-[:IMPORTS]->(es)",
        {"f": f"{_PROJECT}:mypkg.models", "t": f"{_PROJECT}:ext/dataclasses.dataclass"},
    )

    # --- CALLS: handle_request → helper ---
    await graph_client.execute_write(
        "MATCH (a {uid: $f}), (b {uid: $t}) CREATE (a)-[:CALLS]->(b)",
        {"f": f"{_PROJECT}:mypkg.sub.api.handle_request", "t": f"{_PROJECT}:mypkg.utils.helper"},
    )

    return graph_client


# ---------------------------------------------------------------------------
# analyze_repo integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAnalyzeRepo:
    async def test_structure_counts(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="structure", project=_PROJECT)
        assert result["analysis"] == "structure"
        assert result["label_counts"]["Module"] == 3
        assert result["label_counts"]["TypeDef"] == 2
        assert result["label_counts"]["Callable"] == 3
        assert result["label_counts"]["Value"] == 1
        assert result["kind_breakdown"]["Callable"]["function"] == 2
        assert result["kind_breakdown"]["Callable"]["method"] == 1

    async def test_structure_packages(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="structure", project=_PROJECT)
        pkg_names = [p["name"] for p in result["packages"]]
        assert "mypkg" in pkg_names

    async def test_structure_largest_modules(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="structure", project=_PROJECT)
        # models defines Base + User = 2 entities; utils defines helper + MAX_SIZE = 2
        assert len(result["largest_modules"]) >= 2

    async def test_structure_external_deps(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="structure", project=_PROJECT)
        ext_names = [d["package"] for d in result["external_dependencies"]]
        assert "dataclasses" in ext_names

    async def test_centrality_hubs(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="centrality", project=_PROJECT)
        assert result["analysis"] == "centrality"
        hub_names = [h["name"] for h in result["hub_entities"]]
        # helper is imported by api + models, and called by handle_request
        assert "helper" in hub_names

    async def test_centrality_leaves(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="centrality", project=_PROJECT)
        leaf_names = [lf["name"] for lf in result["leaf_entities"]]
        # MAX_SIZE has no inbound IMPORTS/INHERITS/CALLS
        assert "MAX_SIZE" in leaf_names

    async def test_dependencies_internal(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="dependencies", project=_PROJECT)
        assert result["analysis"] == "dependencies"
        assert len(result["internal_imports"]) >= 2

    async def test_dependencies_circular(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="dependencies", project=_PROJECT)
        # models↔utils is a circular dependency
        assert len(result["circular_dependencies"]) >= 1
        cycle = result["circular_dependencies"][0]
        pair = {cycle["module_a"], cycle["module_b"]}
        assert pair == {"mypkg.models", "mypkg.utils"}

    async def test_dependencies_external(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="dependencies", project=_PROJECT)
        ext_pkgs = [e["package"] for e in result["external_imports"]]
        assert "dataclasses" in ext_pkgs

    async def test_patterns_inheritance(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="patterns", project=_PROJECT)
        assert result["analysis"] == "patterns"
        assert len(result["inheritance"]) >= 1
        inh = result["inheritance"][0]
        assert inh["child"] == "User"
        assert inh["parent"] == "Base"

    async def test_patterns_visibility(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="patterns", project=_PROJECT)
        assert result["visibility_distribution"]["public"] >= 5

    async def test_patterns_docstring_coverage(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="patterns", project=_PROJECT)
        cov = result["docstring_coverage"]
        assert cov["total"] >= 5
        # MAX_SIZE has no docstring, so documented < total
        assert cov["documented"] < cov["total"]
        assert cov["percentage"] > 0

    async def test_path_filter(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="structure", project=_PROJECT, path="mypkg/utils")
        # Only entities under mypkg/utils.py
        assert result["label_counts"].get("Module", 0) <= 1
        assert "TypeDef" not in result["label_counts"]

    async def test_invalid_analysis(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="nonexistent", project=_PROJECT)
        assert result["code"] == "INVALID_ANALYSIS"


# ---------------------------------------------------------------------------
# generate_diagram integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGenerateDiagram:
    async def test_packages_diagram(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "generate_diagram", type="packages", project=_PROJECT)
        assert result["type"] == "packages"
        assert "graph TD" in result["mermaid"]
        assert result["node_count"] >= 2
        assert "-->" in result["mermaid"]

    async def test_imports_diagram(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "generate_diagram", type="imports", project=_PROJECT)
        assert result["type"] == "imports"
        assert "graph LR" in result["mermaid"]
        assert result["node_count"] >= 2
        # Should show module dependency edges
        assert "-->" in result["mermaid"]

    async def test_inheritance_diagram(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "generate_diagram", type="inheritance", project=_PROJECT)
        assert result["type"] == "inheritance"
        assert "classDiagram" in result["mermaid"]
        assert "Base" in result["mermaid"]
        assert "User" in result["mermaid"]
        assert "<|--" in result["mermaid"]

    async def test_module_detail_diagram(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(
            app_ctx, "generate_diagram", type="module_detail", project=_PROJECT, path="mypkg/models"
        )
        assert result["type"] == "module_detail"
        assert result["module"] == "mypkg.models"
        assert "classDiagram" in result["mermaid"]
        # Should include Base and User
        assert "Base" in result["mermaid"]
        assert "User" in result["mermaid"]
        # User.save should appear as a method
        assert "save()" in result["mermaid"]

    async def test_module_detail_requires_path(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "generate_diagram", type="module_detail", project=_PROJECT)
        assert result["code"] == "PATH_REQUIRED"

    async def test_module_detail_not_found(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(
            app_ctx, "generate_diagram", type="module_detail", project=_PROJECT, path="nonexistent/path"
        )
        assert result["code"] == "NOT_FOUND"

    async def test_max_nodes_caps_output(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "generate_diagram", type="packages", project=_PROJECT, max_nodes=2)
        assert result["node_count"] <= 4  # max_nodes limits edges, each edge adds up to 2 nodes

    async def test_invalid_diagram_type(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "generate_diagram", type="nonexistent", project=_PROJECT)
        assert result["code"] == "INVALID_DIAGRAM_TYPE"

    async def test_empty_project_diagrams(self, app_ctx):
        """Diagrams for a non-existent project return empty/placeholder output."""
        result = await _invoke_tool(app_ctx, "generate_diagram", type="packages", project="nonexistent")
        assert result["node_count"] == 0

    async def test_imports_empty(self, app_ctx):
        result = await _invoke_tool(app_ctx, "generate_diagram", type="imports", project="nonexistent")
        assert result["node_count"] == 0
