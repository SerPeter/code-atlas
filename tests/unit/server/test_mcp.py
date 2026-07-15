"""Unit tests for the MCP server tools (no infrastructure required)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, ClassVar
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
    _compact_node,
    _default_scope_projects,
    _file_uri_to_path,
    _maybe_update_root,
    _rank_results,
    _register_analysis_tools,
    _register_hybrid_tool,
    _register_info_tools,
    _register_knowledge_tools,
    _register_node_tools,
    _register_query_tools,
    _register_search_tools,
    _register_subagent_tools,
    _resolve_hybrid_scope,
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
    _register_knowledge_tools(server)
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
# TestVectorSearchMock (no DB needed)
# ---------------------------------------------------------------------------


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
        mock_vector = [0.1] * (settings.embeddings.dimension or 768)
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
# Label validation on search tools — Cypher injection guard (no DB needed)
# ---------------------------------------------------------------------------

_MALICIOUS_LABEL = "callable', $query, 60) YIELD node WITH node LIMIT 1 MATCH (m) DETACH DELETE m //"


class TestSearchLabelValidation:
    async def test_text_search_rejects_malicious_label(self, settings):
        """An unwhitelisted label must be refused before any graph call (injection guard)."""
        graph = AsyncMock(spec=GraphClient)
        graph.text_search = AsyncMock()
        embed = EmbedClient(settings.embeddings)
        app = AppContext(graph=graph, settings=settings, embed=embed)

        result = await _invoke_tool(app, "text_search", query="x", label=_MALICIOUS_LABEL)
        assert "error" in result
        assert "Invalid label" in result["error"]
        graph.text_search.assert_not_awaited()

    async def test_vector_search_rejects_malicious_label(self, settings):
        graph = AsyncMock(spec=GraphClient)
        graph.vector_search = AsyncMock()
        embed = EmbedClient(settings.embeddings)
        app = AppContext(graph=graph, settings=settings, embed=embed, vector_enabled=True)

        result = await _invoke_tool(app, "vector_search", query="x", label=_MALICIOUS_LABEL)
        assert "error" in result
        assert "Invalid label" in result["error"]
        graph.vector_search.assert_not_awaited()

    async def test_text_search_accepts_valid_label(self, settings):
        graph = AsyncMock(spec=GraphClient)
        graph.text_search = AsyncMock(return_value=[])
        graph.batch_call_stats = AsyncMock(return_value={})
        embed = EmbedClient(settings.embeddings)
        app = AppContext(graph=graph, settings=settings, embed=embed)

        result = await _invoke_tool(app, "text_search", query="x", label="Callable")
        assert "error" not in result
        graph.text_search.assert_awaited_once()


# ---------------------------------------------------------------------------
# _default_scope_projects — default monorepo scope resolution (no DB needed)
# ---------------------------------------------------------------------------


class TestDefaultScopeProjects:
    async def test_falls_back_to_root_when_get_project_status_fails(self, settings):
        """DB unreachable/erroring must gracefully degrade to the root project name,
        not propagate and break search tools that call this helper."""
        from code_atlas.settings import derive_project_name

        graph = AsyncMock(spec=GraphClient)
        graph.get_project_status = AsyncMock(side_effect=RuntimeError("db down"))
        embed = EmbedClient(settings.embeddings)
        app = AppContext(graph=graph, settings=settings, embed=embed)

        root_name = derive_project_name(settings.project_root)
        result = await _default_scope_projects(app)
        assert result == [root_name]

    async def test_includes_sub_projects_and_excludes_unrelated(self, settings):
        """Sub-projects stored as '{root}/{sub}' are included; an unrelated project whose
        name merely shares the root as a substring (no '/' separator) must not match."""
        from code_atlas.settings import derive_project_name

        root_name = derive_project_name(settings.project_root)
        rows = [
            {"n": {"name": root_name}},
            {"n": {"name": f"{root_name}/sub"}},
            {"n": {"name": f"{root_name}-unrelated"}},
        ]
        graph = AsyncMock(spec=GraphClient)
        graph.get_project_status = AsyncMock(return_value=rows)
        embed = EmbedClient(settings.embeddings)
        app = AppContext(graph=graph, settings=settings, embed=embed)

        result = await _default_scope_projects(app)
        assert set(result) == {root_name, f"{root_name}/sub"}

    async def test_no_sub_projects_returns_root_only(self, settings):
        from code_atlas.settings import derive_project_name

        root_name = derive_project_name(settings.project_root)
        graph = AsyncMock(spec=GraphClient)
        graph.get_project_status = AsyncMock(return_value=[{"n": {"name": root_name}}])
        embed = EmbedClient(settings.embeddings)
        app = AppContext(graph=graph, settings=settings, embed=embed)

        result = await _default_scope_projects(app)
        assert result == [root_name]

    async def test_includes_extra_vaults_deduped(self, settings, tmp_path):
        """Configured extra_vaults (global vault, harness memory dir) must be appended to the
        default scope — otherwise a user's configured vaults are invisible to no-scope searches.
        A vault name that coincides with an existing root/sibling name must not be duplicated."""
        from code_atlas.settings import ExtraVaultSettings, derive_project_name

        root_name = derive_project_name(settings.project_root)
        settings.knowledge.extra_vaults = [
            ExtraVaultSettings(path=str(tmp_path / "vault"), project_name="global-vault"),
            ExtraVaultSettings(path=str(tmp_path / "vault2"), project_name=root_name),
        ]
        rows = [
            {"n": {"name": root_name}},
            {"n": {"name": f"{root_name}/sub"}},
        ]
        graph = AsyncMock(spec=GraphClient)
        graph.get_project_status = AsyncMock(return_value=rows)
        embed = EmbedClient(settings.embeddings)
        app = AppContext(graph=graph, settings=settings, embed=embed)

        result = await _default_scope_projects(app)
        assert result == [root_name, f"{root_name}/sub", "global-vault"]

    async def test_falls_back_to_root_and_extra_vaults_when_get_project_status_fails(self, settings, tmp_path):
        """The DB-unreachable fallback must also include extra_vaults, for consistency with the
        successful-lookup path."""
        from code_atlas.settings import ExtraVaultSettings, derive_project_name

        root_name = derive_project_name(settings.project_root)
        settings.knowledge.extra_vaults = [
            ExtraVaultSettings(path=str(tmp_path / "vault"), project_name="global-vault")
        ]
        graph = AsyncMock(spec=GraphClient)
        graph.get_project_status = AsyncMock(side_effect=RuntimeError("db down"))
        embed = EmbedClient(settings.embeddings)
        app = AppContext(graph=graph, settings=settings, embed=embed)

        result = await _default_scope_projects(app)
        assert result == [root_name, "global-vault"]


# ---------------------------------------------------------------------------
# hybrid_search input validation (no DB needed)
# ---------------------------------------------------------------------------


class TestHybridSearchValidation:
    async def test_invalid_search_types_returns_error(self, settings):
        """An unknown channel name must return a clean error envelope, not raise ValueError."""
        graph = AsyncMock(spec=GraphClient)
        embed = EmbedClient(settings.embeddings)
        app = AppContext(graph=graph, settings=settings, embed=embed)

        result = await _invoke_tool(app, "hybrid_search", query="foo", search_types="bogus_channel")
        assert "error" in result
        assert result["code"] == "INVALID_SEARCH_TYPES"

    async def test_non_object_weights_returns_error(self, settings):
        """Valid JSON that isn't an object (e.g. a list) must be rejected cleanly."""
        graph = AsyncMock(spec=GraphClient)
        embed = EmbedClient(settings.embeddings)
        app = AppContext(graph=graph, settings=settings, embed=embed)

        result = await _invoke_tool(app, "hybrid_search", query="foo", weights="[1, 2, 3]")
        assert "error" in result
        assert result["code"] == "INVALID_WEIGHTS"


# ---------------------------------------------------------------------------
# _resolve_hybrid_scope / hybrid_search — a scope matching zero projects must
# be treated as "search nothing", not silently collapse into "no filter"
# ---------------------------------------------------------------------------


class TestResolveHybridScopeZeroMatch:
    async def test_zero_match_glob_returns_none_not_empty_string(self, settings):
        """expand_scope's explicit "match nothing" ([]) must not collapse to
        "" — hybrid_search treats "" exactly like an unset scope (no filter)."""
        graph = AsyncMock(spec=GraphClient)
        graph.get_project_status = AsyncMock(return_value=[{"n": {"name": "libs-shared"}}])
        embed = EmbedClient(settings.embeddings)
        app = AppContext(graph=graph, settings=settings, embed=embed)

        resolved = await _resolve_hybrid_scope(app, "totally-nonexistent-*")
        assert resolved is None

    async def test_matching_glob_resolves_normally(self, settings):
        graph = AsyncMock(spec=GraphClient)
        graph.get_project_status = AsyncMock(
            return_value=[{"n": {"name": "libs-shared"}}, {"n": {"name": "libs-other"}}]
        )
        embed = EmbedClient(settings.embeddings)
        app = AppContext(graph=graph, settings=settings, embed=embed)

        resolved = await _resolve_hybrid_scope(app, "libs-*")
        assert resolved == "libs-shared,libs-other"


class TestHybridSearchZeroMatchScope:
    async def test_zero_match_scope_returns_empty_without_unfiltered_search(self, settings):
        """A scope glob matching zero indexed projects must return zero results
        and must NOT fall through to an unrestricted, unfiltered search."""
        graph = AsyncMock(spec=GraphClient)
        graph.get_project_status = AsyncMock(return_value=[{"n": {"name": "libs-shared"}}])
        embed = EmbedClient(settings.embeddings)
        app = AppContext(graph=graph, settings=settings, embed=embed)

        with patch("code_atlas.server.mcp._hybrid_search", new_callable=AsyncMock) as fake_search:
            result = await _invoke_tool(app, "hybrid_search", query="foo", scope="totally-nonexistent-*")

        fake_search.assert_not_awaited()
        assert result["results"] == []
        assert result["count"] == 0


# ---------------------------------------------------------------------------
# `truncated` field correctness (no DB needed) — was always False before the fix
# ---------------------------------------------------------------------------


class TestTruncatedField:
    async def test_text_search_truncated_true_when_more_results_than_limit(self, settings):
        available = [{"node": {"uid": f"p:e{i}", "name": "e"}, "score": 1.0} for i in range(30)]

        async def _fake_text_search(query, label="", limit=20, project="", projects=None):
            return available[:limit]

        graph = AsyncMock(spec=GraphClient)
        graph.text_search = AsyncMock(side_effect=_fake_text_search)
        embed = EmbedClient(settings.embeddings)
        app = AppContext(graph=graph, settings=settings, embed=embed)

        result = await _invoke_tool(app, "text_search", query="e", limit=20)
        assert result["count"] == 20
        assert result["truncated"] is True

    async def test_text_search_truncated_false_when_results_fit(self, settings):
        available = [{"node": {"uid": f"p:e{i}", "name": "e"}, "score": 1.0} for i in range(5)]

        async def _fake_text_search(query, label="", limit=20, project="", projects=None):
            return available[:limit]

        graph = AsyncMock(spec=GraphClient)
        graph.text_search = AsyncMock(side_effect=_fake_text_search)
        embed = EmbedClient(settings.embeddings)
        app = AppContext(graph=graph, settings=settings, embed=embed)

        result = await _invoke_tool(app, "text_search", query="e", limit=20)
        assert result["count"] == 5
        assert result["truncated"] is False

    async def test_vector_search_truncated_true_when_more_results_than_limit(self, settings):
        available = [{"node": {"uid": f"p:e{i}", "name": "e"}, "similarity": 0.9} for i in range(30)]

        async def _fake_vector_search(vector, label="", limit=20, project="", threshold=0.0, projects=None):
            return available[:limit]

        graph = AsyncMock(spec=GraphClient)
        graph.vector_search = AsyncMock(side_effect=_fake_vector_search)
        embed = EmbedClient(settings.embeddings)
        app = AppContext(graph=graph, settings=settings, embed=embed, vector_enabled=True)

        with patch.object(embed, "embed_one", new_callable=AsyncMock, return_value=[0.1] * 768):
            result = await _invoke_tool(app, "vector_search", query="e", limit=20)
        assert result["count"] == 20
        assert result["truncated"] is True

    async def test_hybrid_search_truncated_true_when_more_results_than_limit(self, settings):
        from code_atlas.search.engine import SearchResult

        available = [
            SearchResult(
                uid=f"p:e{i}",
                name="e",
                qualified_name=f"mod.e{i}",
                kind="function",
                file_path="mod.py",
                line_start=1,
                line_end=2,
                signature="",
                docstring="",
                labels=["Callable"],
                rrf_score=1.0,
            )
            for i in range(30)
        ]

        async def _fake_hybrid_search(*, limit, **_kwargs):
            return available[:limit]

        graph = AsyncMock(spec=GraphClient)
        embed = EmbedClient(settings.embeddings)
        app = AppContext(graph=graph, settings=settings, embed=embed)

        with patch("code_atlas.server.mcp._hybrid_search", side_effect=_fake_hybrid_search):
            result = await _invoke_tool(app, "hybrid_search", query="e", limit=20)
        assert result["count"] == 20
        assert result["truncated"] is True

    async def test_hybrid_search_not_truncated_when_results_fit(self, settings):
        from code_atlas.search.engine import SearchResult

        available = [
            SearchResult(
                uid=f"p:e{i}",
                name="e",
                qualified_name=f"mod.e{i}",
                kind="function",
                file_path="mod.py",
                line_start=1,
                line_end=2,
                signature="",
                docstring="",
                labels=["Callable"],
                rrf_score=1.0,
            )
            for i in range(5)
        ]

        async def _fake_hybrid_search(*, limit, **_kwargs):
            return available[:limit]

        graph = AsyncMock(spec=GraphClient)
        embed = EmbedClient(settings.embeddings)
        app = AppContext(graph=graph, settings=settings, embed=embed)

        with patch("code_atlas.server.mcp._hybrid_search", side_effect=_fake_hybrid_search):
            result = await _invoke_tool(app, "hybrid_search", query="e", limit=20)
        assert result["count"] == 5
        assert result["truncated"] is False


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


# ---------------------------------------------------------------------------
# cypher_query write-keyword guard vs string literals (no DB needed)
# ---------------------------------------------------------------------------


class TestCypherQueryWriteKeywordGuard:
    async def test_allows_string_literal_matching_write_keyword(self, settings):
        """A literal value equal to a write keyword (e.g. 'set') must not trigger rejection."""
        graph = AsyncMock(spec=GraphClient)
        graph.execute = AsyncMock(return_value=[{"name": "set"}])
        embed = EmbedClient(settings.embeddings)
        app = AppContext(graph=graph, settings=settings, embed=embed)

        result = await _invoke_tool(app, "cypher_query", query="MATCH (n) WHERE n.name = 'set' RETURN n.name AS name")
        assert "error" not in result
        graph.execute.assert_awaited_once()

    async def test_still_rejects_unquoted_write_keyword(self, settings):
        graph = AsyncMock(spec=GraphClient)
        embed = EmbedClient(settings.embeddings)
        app = AppContext(graph=graph, settings=settings, embed=embed)

        result = await _invoke_tool(app, "cypher_query", query="MATCH (n) WHERE n.name = 'x' SET n.name = 'y'")
        assert result["code"] == "WRITE_REJECTED"
        graph.execute.assert_not_awaited()


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

    async def test_staleness_query_timeout_returns_original_result(self, settings):
        """QueryTimeoutError (raised by checker.check -> graph.execute on a slow DB query)
        must be caught alongside plain TimeoutError — not propagate and discard results."""
        from code_atlas.indexing.orchestrator import StalenessChecker

        checker = StalenessChecker(settings.project_root, project_name="myproject")
        embed = EmbedClient(settings.embeddings)
        mock_graph = AsyncMock()
        app = AppContext(graph=mock_graph, settings=settings, embed=embed, staleness=checker)

        with patch.object(checker, "check", side_effect=QueryTimeoutError(5.0, "get_project_git_hash")):
            result = {"results": [{"uid": "test:foo"}]}
            annotated = await _with_staleness(app, result, scope="myproject")
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
        """list_roots() returns a different *project* root → daemon stop+start called."""
        new_root = tmp_path / "other_project"
        new_root.mkdir()
        (new_root / ".git").mkdir()  # a real project root (git repo) — eligible to switch

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

    async def test_ignores_non_project_root(self, tmp_path, settings):
        """A probed root that is not an Atlas project (no atlas.toml, not a git root)
        must NOT hijack the served project namespace — identity stays stable."""
        bare_root = tmp_path / "not_a_project"
        bare_root.mkdir()  # no .git, no atlas.toml

        embed = EmbedClient(settings.embeddings)
        mock_graph = AsyncMock()
        old_daemon = AsyncMock()
        old_daemon.stop = AsyncMock()
        app = AppContext(
            graph=mock_graph,
            settings=settings,
            embed=embed,
            daemon=old_daemon,
            resolved_root=settings.project_root,
        )

        mock_root = MagicMock()
        mock_root.uri = bare_root.as_uri()
        mock_result = MagicMock()
        mock_result.roots = [mock_root]

        ctx = MagicMock()
        ctx.session.list_roots = AsyncMock(return_value=mock_result)

        await _maybe_update_root(app, ctx)

        assert app.roots_checked is True
        old_daemon.stop.assert_not_awaited()
        assert app.settings.project_root == settings.project_root
        assert app.resolved_root == settings.project_root


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
# _compact_node detail modes (no DB needed)
# ---------------------------------------------------------------------------


class TestCompactNodeDetail:
    """Verify _compact_node respects the detail parameter."""

    def _make_record(self) -> dict[str, Any]:
        """Build a fake node record with source and a long docstring."""

        class FakeNode(dict):
            labels: ClassVar[list[str]] = ["Callable"]

            def items(self):
                return super().items()

        node = FakeNode(
            uid="proj:mod.func",
            name="func",
            qualified_name="mod.func",
            kind="function",
            file_path="mod.py",
            line_start=1,
            line_end=10,
            signature="def func(x: int) -> str",
            docstring="A" * 300,
            visibility="public",
            source="def func(x: int) -> str:\n    return str(x)",
        )
        return {"node": node, "score": 1.5}

    def test_summary_truncates_docstring(self):
        record = self._make_record()
        result = _compact_node(record, detail="summary")
        assert result["docstring"].endswith("...")
        assert len(result["docstring"]) < 300
        assert "source" not in result

    def test_full_includes_source_and_full_docstring(self):
        record = self._make_record()
        result = _compact_node(record, detail="full")
        assert result["docstring"] == "A" * 300
        assert result["source"] == "def func(x: int) -> str:\n    return str(x)"

    def test_default_is_summary(self):
        record = self._make_record()
        result = _compact_node(record)
        assert "source" not in result
        assert result["docstring"].endswith("...")
