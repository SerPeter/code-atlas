"""Unit tests for the MCP server tools (no infrastructure required)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.server.fastmcp import FastMCP

from code_atlas.backends.sqlite_graph import SqliteGraphClient
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
    IndexNotReadyError,
    _clamp_depth,
    _compact_node,
    _default_scope_projects,
    _ensure_root,
    _file_uri_to_path,
    _maybe_update_root,
    _parse_rel_types,
    _rank_results,
    _register_analysis_tools,
    _register_hybrid_tool,
    _register_info_tools,
    _register_knowledge_tools,
    _register_node_tools,
    _register_query_tools,
    _register_search_tools,
    _register_subagent_tools,
    _register_traversal_tools,
    _resolve_hybrid_scope,
    _resolve_test_patterns,
    _with_staleness,
)
from code_atlas.settings import AtlasSettings, IndexSettings, SearchSettings, find_git_root

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
    _register_traversal_tools(server)

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


class TestResolveTestPatterns:
    """_resolve_test_patterns backs analyze_repo/find_dead_code/find_complexity_hotspots/
    find_communities/find_hotspots's exclude_tests param — same override semantics as
    hybrid_search's own exclude_tests."""

    def test_none_defers_to_settings_default_true(self):
        settings = SearchSettings(test_filter=True)
        assert _resolve_test_patterns(settings, None) == tuple(settings.test_patterns)

    def test_none_defers_to_settings_default_false(self):
        settings = SearchSettings(test_filter=False)
        assert _resolve_test_patterns(settings, None) == ()

    def test_explicit_true_overrides_settings_default_false(self):
        settings = SearchSettings(test_filter=False)
        assert _resolve_test_patterns(settings, True) == tuple(settings.test_patterns)

    def test_explicit_false_overrides_settings_default_true(self):
        settings = SearchSettings(test_filter=True)
        assert _resolve_test_patterns(settings, False) == ()


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
            | set(result["node_labels"]["marker"])
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
# `truncated` field correctness (no DB needed) — was always False before ATL-104,
# and reported a fabricated total between then and ATL-111
# ---------------------------------------------------------------------------


class TestTruncatedField:
    """The contract: `truncated` is False only when nothing was withheld.

    These assertions used to read ``shown + cut == total``, which the fabricated
    numbers satisfied — a search fetching ``limit + 1`` reported ``total = 21`` for
    ``limit = 20``, so 20 + 1 == 21 held while the repo had 5,000 matches. The test
    checked arithmetic consistency, not truth, and passed throughout (ATL-111).

    A search cannot afford a real count, so it now says so: ``total`` and ``cut`` are
    ``None`` and ``has_more`` is True. "Unknown" is a fact; 1 was not.
    """

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
        assert result["truncated"]["has_more"] is True
        assert result["truncated"]["shown"] == 20
        # The load-bearing pair: a count that was never computed must read as unknown,
        # not as a number bounded by the fetch size.
        assert result["truncated"]["total"] is None
        assert result["truncated"]["cut"] is None

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
        assert result["truncated"]["has_more"] is True
        assert result["truncated"]["shown"] == 20
        # The load-bearing pair: a count that was never computed must read as unknown,
        # not as a number bounded by the fetch size.
        assert result["truncated"]["total"] is None
        assert result["truncated"]["cut"] is None

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
        assert result["truncated"]["has_more"] is True
        assert result["truncated"]["shown"] == 20
        # The load-bearing pair: a count that was never computed must read as unknown,
        # not as a number bounded by the fetch size.
        assert result["truncated"]["total"] is None
        assert result["truncated"]["cut"] is None

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

    async def test_cut_is_never_a_number_bounded_by_the_fetch_size(self, settings):
        """The regression this exists to catch, stated as a negative.

        With 5,000 matches and limit 20 the old code answered
        ``{"shown": 20, "total": 21, "cut": 1}`` — while the MCP server instruction
        told agents to "read `cut` before concluding a short list is a complete one".
        Any small integer here is a lie; only None is honest.
        """
        available = [{"node": {"uid": f"p:e{i}", "name": "e"}, "score": 1.0} for i in range(5000)]

        async def _fake_text_search(query, label="", limit=20, project="", projects=None):
            return available[:limit]

        graph = AsyncMock(spec=GraphClient)
        graph.text_search = AsyncMock(side_effect=_fake_text_search)
        embed = EmbedClient(settings.embeddings)
        app = AppContext(graph=graph, settings=settings, embed=embed)

        result = await _invoke_tool(app, "text_search", query="e", limit=20)
        assert result["truncated"]["cut"] is None, "a fetch-bounded count must not be reported as `cut`"
        assert result["truncated"]["total"] is None

    async def test_a_result_that_knows_nothing_was_withheld_still_reports_false(self, settings):
        """`truncated: False` must keep meaning "complete" — agents depend on it."""
        from code_atlas.server.mcp import _result

        envelope = _result([{"a": 1}], limit=20, query_ms=1.0)
        assert envelope["truncated"] is False


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


# ---------------------------------------------------------------------------
# cypher_query/validate_cypher on the sqlite backend — deliberate exception
# (see graph/protocol.py, ADR-0015): arbitrary agent-authored Cypher has no
# SQL translation, so these must return a clean structured error, not crash.
# ---------------------------------------------------------------------------


class TestCypherToolsSqliteBackendGuard:
    async def test_cypher_query_returns_unsupported_backend_error(self, settings, tmp_path):
        graph = SqliteGraphClient(tmp_path / "graph.sqlite3")
        embed = EmbedClient(settings.embeddings)
        app = AppContext(graph=graph, settings=settings, embed=embed)  # type: ignore[invalid-argument-type]

        result = await _invoke_tool(app, "cypher_query", query="MATCH (n:Callable) RETURN n LIMIT 10")

        assert result["code"] == "UNSUPPORTED_BACKEND"
        assert "error" in result
        await graph.close()

    async def test_validate_cypher_skips_explain_with_info_issue_not_crash(self, settings, tmp_path):
        graph = SqliteGraphClient(tmp_path / "graph.sqlite3")
        embed = EmbedClient(settings.embeddings)
        app = AppContext(graph=graph, settings=settings, embed=embed)  # type: ignore[invalid-argument-type]

        result = await _invoke_tool(app, "validate_cypher", query="MATCH (n:Callable) RETURN n LIMIT 10")

        assert result["valid"] is True  # static checks alone still pass
        assert any("sqlite" in i["message"].lower() for i in result["issues"])
        await graph.close()


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
    @pytest.fixture(autouse=True)
    def _fast_staleness_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Shrink the staleness budget so the timeout tests do not sit through it.

        Two tests here drive the timeout branch by making the check block, and with the
        real 5s budget each cost 5.01s -- measured, a third of this suite's runtime for
        two tests. What is under test is what the code *does* when the check does not
        finish in time, never how long "in time" is, so a smaller budget cannot make them
        vacuous. It was an inline literal until now, which left them no way to shrink it.
        """
        monkeypatch.setattr("code_atlas.server.mcp._STALENESS_TIMEOUT_S", 0.05)

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

    async def test_lock_mode_refuses_when_freshness_cannot_be_verified(self, settings):
        """`lock` must fail CLOSED (ATL-111).

        Its whole purpose is refusing answers from a stale index. Serving one because the
        check timed out defeats the setting the user deliberately chose — the one place
        where carrying on is the wrong default.
        """
        import asyncio

        from code_atlas.indexing.orchestrator import StalenessChecker

        lock_settings = AtlasSettings(project_root=settings.project_root, index=IndexSettings(stale_mode="lock"))
        checker = StalenessChecker(settings.project_root, project_name="myproject")
        embed = EmbedClient(settings.embeddings)
        app = AppContext(graph=AsyncMock(), settings=lock_settings, embed=embed, staleness=checker)

        async def _slow_check(*_args, **_kwargs):
            await asyncio.sleep(60)

        with patch.object(checker, "check", side_effect=_slow_check):
            annotated = await _with_staleness(app, {"results": [{"uid": "t:f"}]}, scope="myproject")

        assert annotated["code"] == "STALE_UNKNOWN"
        assert "results" not in annotated, "a locked query must not return rows it could not vouch for"

    async def test_stale_mode_rejects_an_unknown_value(self):
        """A typo used to fall through to warn, silently disabling `lock`."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            # Deliberately invalid. The Literal makes ty reject it statically too,
            # which is the fix working rather than a problem with the test.
            IndexSettings(stale_mode="lcok")  # ty: ignore[invalid-argument-type]

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
            # `ignore` returns before the check runs at all, so the envelope is
            # genuinely untouched — unlike the timeout path, which must say so.
            assert annotated == result
            assert "stale" not in annotated
            mock_check.assert_not_called()

    async def test_staleness_timeout_says_the_check_did_not_run(self, settings):
        """A check that could not run must not read as "verified fresh" (ATL-111).

        This previously asserted the envelope came back untouched. An absent `stale` key
        is indistinguishable from a confirmed-fresh one, so "unmodified" was the bug.
        """
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
            # Explicitly "unknown", not absent — absent is what read as fresh.
            assert annotated["stale"] is None
            assert annotated["stale_check"] == "timed_out"
            assert annotated["results"] == [{"uid": "test:foo"}]

    async def test_staleness_query_timeout_says_the_check_did_not_run(self, settings):
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
            # Explicitly "unknown", not absent — absent is what read as fresh.
            assert annotated["stale"] is None
            assert annotated["stale_check"] == "timed_out"
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
        # analyze_repo/generate_diagram/trace_path/find_dead_code/find_complexity_hotspots/
        # blast_radius now call named GraphBackend methods (query construction moved into
        # GraphClient — see graph/protocol.py) instead of graph.execute() directly, so the
        # mock needs those entry points configured too, not just .execute().
        mock_graph.get_structure_overview = AsyncMock(side_effect=QueryTimeoutError(10.0, "get_structure_overview"))
        mock_graph.get_diagram_packages = AsyncMock(side_effect=QueryTimeoutError(10.0, "get_diagram_packages"))
        mock_graph.trace_path_between = AsyncMock(side_effect=QueryTimeoutError(10.0, "trace_path_between"))
        mock_graph.get_dead_code_candidates = AsyncMock(side_effect=QueryTimeoutError(10.0, "get_dead_code_candidates"))
        mock_graph.get_complexity_hotspots = AsyncMock(side_effect=QueryTimeoutError(10.0, "get_complexity_hotspots"))
        mock_graph.get_module_summary = AsyncMock(side_effect=QueryTimeoutError(10.0, "get_module_summary"))
        mock_graph.node_exists = AsyncMock(side_effect=QueryTimeoutError(10.0, "node_exists"))
        # get_node/get_context/index_status/list_projects now call named GraphBackend methods
        # (query construction moved into GraphClient — see graph/protocol.py) instead of
        # graph.execute() directly, so the mock needs those entry points configured too.
        mock_graph.get_node_exact_matches = AsyncMock(side_effect=QueryTimeoutError(10.0, "get_node_exact_matches"))
        mock_graph.get_node_partial_matches = AsyncMock(side_effect=QueryTimeoutError(10.0, "get_node_partial_matches"))
        mock_graph.get_entity_by_uid = AsyncMock(side_effect=QueryTimeoutError(10.0, "get_entity_by_uid"))
        mock_graph.get_label_counts = AsyncMock(side_effect=QueryTimeoutError(10.0, "get_label_counts"))
        mock_graph.get_project_dependency_edges = AsyncMock(
            side_effect=QueryTimeoutError(10.0, "get_project_dependency_edges")
        )
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

    async def test_trace_path_timeout(self, timeout_app):
        result = await _invoke_tool(timeout_app, "trace_path", from_uid="p:a", to_uid="p:b")
        assert result["code"] == "QUERY_TIMEOUT"

    async def test_find_dead_code_timeout(self, timeout_app):
        result = await _invoke_tool(timeout_app, "find_dead_code", project="p")
        assert result["code"] == "QUERY_TIMEOUT"

    async def test_find_complexity_hotspots_timeout(self, timeout_app):
        result = await _invoke_tool(timeout_app, "find_complexity_hotspots", project="p")
        assert result["code"] == "QUERY_TIMEOUT"

    async def test_blast_radius_timeout(self, timeout_app):
        result = await _invoke_tool(timeout_app, "blast_radius", uid="p:a")
        assert result["code"] == "QUERY_TIMEOUT"

    async def test_summarize_module_timeout(self, timeout_app):
        result = await _invoke_tool(timeout_app, "summarize_module", path="pkg", project="p")
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


# ---------------------------------------------------------------------------
# _parse_rel_types / _clamp_depth (trace_path / blast_radius helpers)
# ---------------------------------------------------------------------------


class TestParseRelTypes:
    def test_empty_returns_default(self):
        types, error = _parse_rel_types("", ("CALLS",))
        assert types == ("CALLS",)
        assert error is None

    def test_valid_csv_parsed(self):
        types, error = _parse_rel_types("CALLS, IMPORTS", ("CALLS",))
        assert types == ("CALLS", "IMPORTS")
        assert error is None

    def test_invalid_rel_type_errors(self):
        types, error = _parse_rel_types("NOT_A_REL", ("CALLS",))
        assert types == ("CALLS",)
        assert error is not None
        assert error["code"] == "INVALID_EDGE_TYPES"


class TestClampDepth:
    def test_clamps_above_max(self):
        assert _clamp_depth(999) == 10

    def test_clamps_below_min(self):
        assert _clamp_depth(0) == 1

    def test_passes_through_in_range(self):
        assert _clamp_depth(5) == 5


# ---------------------------------------------------------------------------
# First-index readiness gate (Phase 4) — _ensure_root() wait/timeout, and
# app_lifespan's needs_first_index/first_index_ready computation
# ---------------------------------------------------------------------------


class _FakeSchemaGraph:
    """Stateful graph.get_schema_version()/ensure_schema() double.

    ensure_schema() flips the stored version from *initial_version* to the
    current SCHEMA_VERSION, mirroring GraphClient's real behavior — a
    regression guard for computing needs_first_index from the version
    BEFORE calling ensure_schema() (calling ensure_schema() first would
    always observe a non-None version afterward).
    """

    def __init__(self, initial_version: int | None) -> None:
        self._version = initial_version
        self.ping = AsyncMock(return_value=True)
        self.close = AsyncMock()
        self.get_embedding_config = AsyncMock(return_value=None)

    async def get_schema_version(self) -> int | None:
        return self._version

    async def ensure_schema(self) -> None:
        self._version = SCHEMA_VERSION


class _FakeDaemonManager:
    """DaemonManager stand-in for app_lifespan tests — never runs real catch-up,
    so first_index_ready is left exactly as app_lifespan itself set it."""

    def __init__(self) -> None:
        self.start = AsyncMock(return_value=True)
        self.stop = AsyncMock()


class TestAppLifespanNeedsFirstIndex:
    """needs_first_index/first_index_ready computed in app_lifespan (mcp.py)."""

    async def test_already_provisioned_backend_never_waits(self, settings, monkeypatch):
        from code_atlas.server.mcp import create_mcp_server

        settings.embeddings.enabled = False
        graph = _FakeSchemaGraph(initial_version=SCHEMA_VERSION)
        monkeypatch.setattr("code_atlas.server.mcp.create_graph_client", AsyncMock(return_value=graph))
        monkeypatch.setattr("code_atlas.server.mcp.DaemonManager", _FakeDaemonManager)

        mcp = create_mcp_server(settings, catchup=False)
        lifespan = mcp.settings.lifespan
        assert lifespan is not None
        async with lifespan(mcp) as app_ctx:
            assert app_ctx.needs_first_index is False
            assert app_ctx.first_index_ready.is_set() is True

    async def test_fresh_backend_needs_first_index_and_starts_unready(self, settings, monkeypatch):
        from code_atlas.server.mcp import create_mcp_server

        settings.embeddings.enabled = False
        graph = _FakeSchemaGraph(initial_version=None)
        monkeypatch.setattr("code_atlas.server.mcp.create_graph_client", AsyncMock(return_value=graph))
        monkeypatch.setattr("code_atlas.server.mcp.DaemonManager", _FakeDaemonManager)

        mcp = create_mcp_server(settings, catchup=False)
        lifespan = mcp.settings.lifespan
        assert lifespan is not None
        async with lifespan(mcp) as app_ctx:
            assert app_ctx.needs_first_index is True
            assert app_ctx.first_index_ready.is_set() is False


class TestFindCommunitiesBackendVisibility:
    """find_communities is Memgraph-only — its clustering is pure Python (no MAGE), but
    the module inventory / module-pair CALLS reads it clusters are still raw Cypher with
    no GraphBackend method. Hidden from tools/list entirely on the embedded SQLite
    backend, present on Memgraph."""

    async def test_hidden_on_sqlite_backend(self, settings, tmp_path, monkeypatch):
        from code_atlas.server.mcp import create_mcp_server

        settings.embeddings.enabled = False
        graph = SqliteGraphClient(tmp_path / "graph.sqlite3")
        monkeypatch.setattr("code_atlas.server.mcp.create_graph_client", AsyncMock(return_value=graph))
        monkeypatch.setattr("code_atlas.server.mcp.DaemonManager", _FakeDaemonManager)

        mcp = create_mcp_server(settings, catchup=False)
        lifespan = mcp.settings.lifespan
        assert lifespan is not None
        async with lifespan(mcp):
            tool_names = {t.name for t in await mcp.list_tools()}
            assert "find_communities" not in tool_names
            assert "find_dead_code" in tool_names  # sanity: other shortcut tools stay

    async def test_present_on_memgraph_backend(self, settings, monkeypatch):
        from code_atlas.server.mcp import create_mcp_server

        settings.embeddings.enabled = False
        graph = _FakeSchemaGraph(initial_version=SCHEMA_VERSION)
        monkeypatch.setattr("code_atlas.server.mcp.create_graph_client", AsyncMock(return_value=graph))
        monkeypatch.setattr("code_atlas.server.mcp.DaemonManager", _FakeDaemonManager)

        mcp = create_mcp_server(settings, catchup=False)
        lifespan = mcp.settings.lifespan
        assert lifespan is not None
        async with lifespan(mcp):
            tool_names = {t.name for t in await mcp.list_tools()}
            assert "find_communities" in tool_names


class TestEnsureRootGate:
    """_ensure_root()'s bounded wait/timeout enforcement (mcp.py)."""

    async def test_blocks_then_proceeds_once_ready(self, settings):
        """A tool call against a fresh backend blocks until first_index_ready fires,
        then proceeds normally — mirrors what every gated @mcp.tool call does."""
        graph = AsyncMock(spec=GraphClient)
        embed = EmbedClient(settings.embeddings)
        ready = asyncio.Event()
        app = AppContext(graph=graph, settings=settings, embed=embed, needs_first_index=True, first_index_ready=ready)
        ctx = _FakeCtx(app)

        async def _unblock_shortly():
            await asyncio.sleep(0.05)
            ready.set()

        asyncio.get_running_loop().create_task(_unblock_shortly())

        result = await asyncio.wait_for(_ensure_root(ctx), timeout=2.0)  # type: ignore[arg-type]
        assert result is app

    async def test_never_unblocked_raises_index_not_ready_within_bounded_time(self, settings, monkeypatch):
        """Simulates daemon.start() returning False (queue unreachable, catch-up
        never runs) — the gate must fail fast with a bounded wait, never hang."""
        monkeypatch.setattr("code_atlas.server.mcp._INDEX_READY_TIMEOUT_S", 0.05)
        graph = AsyncMock(spec=GraphClient)
        embed = EmbedClient(settings.embeddings)
        app = AppContext(
            graph=graph, settings=settings, embed=embed, needs_first_index=True, first_index_ready=asyncio.Event()
        )
        ctx = _FakeCtx(app)

        with pytest.raises(IndexNotReadyError):
            await asyncio.wait_for(_ensure_root(ctx), timeout=2.0)  # type: ignore[arg-type]

    async def test_require_index_false_bypasses_wait(self, settings):
        """health_check/index_status pass require_index=False — must return
        immediately even though first_index_ready is never set."""
        graph = AsyncMock(spec=GraphClient)
        embed = EmbedClient(settings.embeddings)
        app = AppContext(
            graph=graph, settings=settings, embed=embed, needs_first_index=True, first_index_ready=asyncio.Event()
        )
        ctx = _FakeCtx(app)

        result = await asyncio.wait_for(_ensure_root(ctx, require_index=False), timeout=1.0)  # type: ignore[arg-type]
        assert result is app


class TestGatedToolsSurfaceIndexRequired:
    """End-to-end (via _invoke_tool): gated tools return INDEX_REQUIRED;
    health_check bypasses the gate entirely."""

    async def test_get_node_surfaces_index_required(self, settings, monkeypatch):
        monkeypatch.setattr("code_atlas.server.mcp._INDEX_READY_TIMEOUT_S", 0.05)
        graph = AsyncMock(spec=GraphClient)
        embed = EmbedClient(settings.embeddings)
        app = AppContext(
            graph=graph, settings=settings, embed=embed, needs_first_index=True, first_index_ready=asyncio.Event()
        )

        result = await asyncio.wait_for(_invoke_tool(app, "get_node", name="Foo"), timeout=2.0)
        assert result["code"] == "INDEX_REQUIRED"
        graph.execute.assert_not_awaited()

    async def test_health_check_bypasses_gate(self, settings):
        from code_atlas.server.health import CheckResult, CheckStatus, HealthReport

        report = HealthReport(
            checks=[CheckResult(name="memgraph", status=CheckStatus.OK, message="Connected")],
            elapsed_ms=1.0,
        )
        graph = AsyncMock(spec=GraphClient)
        embed = EmbedClient(settings.embeddings)
        daemon = AsyncMock()
        app = AppContext(
            graph=graph,
            settings=settings,
            embed=embed,
            daemon=daemon,
            needs_first_index=True,
            first_index_ready=asyncio.Event(),  # never set
        )

        with patch("code_atlas.server.mcp.run_health_checks", new_callable=AsyncMock, return_value=report):
            result = await asyncio.wait_for(_invoke_tool(app, "health_check"), timeout=1.0)

        assert result["ok"] is True


# ---------------------------------------------------------------------------
# summarize_module — ADR-0013 shortcut tool over analyze_repo("module_summary")
# ---------------------------------------------------------------------------


class TestSummarizeModule:
    """summarize_module is a thin wrapper (ADR-0013), so what matters is that it
    delegates with the analysis pre-set and forwards path/limit/test patterns
    exactly like find_dead_code/find_complexity_hotspots do."""

    @pytest.fixture
    def summary_app(self, settings):
        mock_graph = AsyncMock(spec=GraphClient)
        mock_graph.get_module_summary = AsyncMock(
            return_value={
                "modules": [{"qn": "pkg.mod", "name": "mod", "file_path": "pkg/mod.py", "docstring": None}],
                "entities": [],
                "internal_edges": [],
                "fan_in": [],
                "fan_out": [],
                "docs": [],
            }
        )
        return AppContext(graph=mock_graph, settings=settings, embed=EmbedClient(settings.embeddings))

    async def test_shortcut_delegates_with_analysis_preset(self, summary_app):
        result = await _invoke_tool(summary_app, "summarize_module", path="pkg", project="proj")

        assert result["analysis"] == "module_summary"
        assert result["path"] == "pkg"
        summary_app.graph.get_module_summary.assert_awaited_once()

    async def test_shortcut_scales_limit_before_hitting_the_backend(self, summary_app):
        await _invoke_tool(summary_app, "summarize_module", path="pkg", project="proj", limit=5)

        assert summary_app.graph.get_module_summary.call_args[0] == ("proj", "pkg", 50, 150)

    async def test_shortcut_clamps_limit_to_max(self, summary_app):
        await _invoke_tool(summary_app, "summarize_module", path="pkg", project="proj", limit=10_000)

        assert summary_app.graph.get_module_summary.call_args[0][2] == 100 * 10

    async def test_exclude_tests_false_disables_boundary_filtering(self, summary_app):
        """exclude_tests=False must reach _resolve_test_patterns the same way the
        other analyze_repo shortcuts wire it."""
        summary_app.graph.get_module_summary = AsyncMock(
            return_value={
                "modules": [{"qn": "pkg.mod", "name": "mod", "file_path": "pkg/mod.py", "docstring": None}],
                "entities": [],
                "internal_edges": [],
                "fan_in": [
                    {
                        "from_qn": "tests.unit.test_mod.test_x",
                        "from_name": "test_x",
                        "from_path": "tests/unit/test_mod.py",
                        "from_label": "Callable",
                        "to_qn": "pkg.mod.f",
                        "rel_type": "CALLS",
                        "props": {},
                    }
                ],
                "fan_out": [],
                "docs": [],
            }
        )

        filtered = await _invoke_tool(summary_app, "summarize_module", path="pkg", project="proj")
        unfiltered = await _invoke_tool(
            summary_app, "summarize_module", path="pkg", project="proj", exclude_tests=False
        )

        assert filtered["fan_in_count"] == 0
        assert unfiltered["fan_in_count"] == 1

    async def test_missing_path_is_a_clean_error_not_a_query(self, summary_app):
        result = await _invoke_tool(summary_app, "summarize_module", path="", project="proj")

        assert result["code"] == "PATH_REQUIRED"
        summary_app.graph.get_module_summary.assert_not_awaited()

    async def test_analyze_repo_exposes_the_same_analysis(self, summary_app):
        result = await _invoke_tool(summary_app, "analyze_repo", analysis="module_summary", project="proj", path="pkg")

        assert result["analysis"] == "module_summary"


# ---------------------------------------------------------------------------
# Backend identity on every tool result (ATL-133)
# ---------------------------------------------------------------------------


class _RecordingSpan:
    """Captures what a tool span was told, without needing the OTel SDK."""

    def __init__(self, name: str, attributes: dict | None = None) -> None:
        self.name = name
        self.attributes = dict(attributes or {})
        self.exceptions: list[BaseException] = []
        self.status: object = None

    def set_attribute(self, key, value) -> None:
        self.attributes[key] = value

    def set_status(self, status, description=None) -> None:
        self.status = status if description is None else (status, description)

    def record_exception(self, exception, **_kwargs) -> None:
        self.exceptions.append(exception)

    def end(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        pass


class _RecordingTracer:
    def __init__(self) -> None:
        self.spans: list[_RecordingSpan] = []

    def start_as_current_span(self, name, **kwargs):
        span = _RecordingSpan(name, kwargs.get("attributes"))
        self.spans.append(span)
        return span


class TestNoIndexMode:
    """Indexing is per-worktree, not per-session.

    When several agent sessions share one checkout, each MCP server starts its own
    watcher over the same files and its own catch-up against the same lease. The extra
    ones contribute nothing but contention -- and one such collision is what left a
    real index idle with 30k embeddings outstanding.
    """

    @staticmethod
    def _args(auto_index: bool):
        from code_atlas.indexing.daemon import DaemonManager
        from code_atlas.settings import AtlasSettings

        return DaemonManager(), AtlasSettings(), asyncio.Event(), auto_index

    async def test_disabled_starts_nothing_and_says_why(self):
        from code_atlas.server.mcp import _spawn_indexing

        daemon, settings, ready, auto_index = self._args(False)
        started: list[object] = []
        daemon.start = lambda *a, **kw: started.append(a)

        # graph is only forwarded to daemon.start, which is stubbed here
        task = _spawn_indexing(daemon, settings, None, catchup=True, auto_index=auto_index, first_index_ready=ready)  # type: ignore[invalid-argument-type]

        assert task is None
        assert started == []
        assert "--no-index" in daemon.disabled_reason

    async def test_disabled_still_releases_the_readiness_gate(self):
        """Nothing else will ever set this event. Left clear, every tool call that needs
        an index blocks on it for the full readiness timeout before answering -- a
        query-only server that is slow for no reason."""
        from code_atlas.server.mcp import _spawn_indexing

        daemon, settings, ready, auto_index = self._args(False)
        assert not ready.is_set()

        _spawn_indexing(
            daemon,
            settings,
            None,  # type: ignore[invalid-argument-type]
            catchup=True,
            auto_index=auto_index,
            first_index_ready=ready,
        )

        assert ready.is_set()

    async def test_enabled_starts_the_daemon(self):
        from code_atlas.server.mcp import _spawn_indexing

        daemon, settings, ready, auto_index = self._args(True)
        calls: list[dict] = []

        async def fake_start(*_a, **kw):
            calls.append(kw)
            return True

        daemon.start = fake_start

        # graph is only forwarded to daemon.start, which is stubbed here
        task = _spawn_indexing(daemon, settings, None, catchup=True, auto_index=auto_index, first_index_ready=ready)  # type: ignore[invalid-argument-type]
        assert task is not None
        await task

        assert calls, "daemon.start was never called"
        assert calls[0]["catchup"] is True
        assert daemon.disabled_reason == ""


class TestToolTracing:
    """`_tracer` existed in mcp.py from the day telemetry was added and was never used.

    Every trace the system produced started in the middle of the stack -- graph.execute,
    hybrid_search, embed.embed_one -- with nothing above it naming the tool an agent had
    called. On an agent-facing server that is the one span you cannot do without.
    """

    @staticmethod
    def _server():
        from code_atlas.server.mcp import create_mcp_server
        from code_atlas.settings import AtlasSettings

        return create_mcp_server(AtlasSettings())

    async def test_a_tool_call_opens_a_named_span(self, monkeypatch):
        import code_atlas.server.mcp as mcp_mod

        tracer = _RecordingTracer()
        monkeypatch.setattr(mcp_mod, "_tracer", tracer)
        server = self._server()
        tool = next(t for t in server._tool_manager._tools.values() if t.name == "schema_info")

        await tool.fn()

        assert [sp.name for sp in tracer.spans] == ["mcp.tool.schema_info"]
        assert tracer.spans[0].attributes["mcp.tool"] == "schema_info"
        assert tracer.spans[0].attributes["mcp.status"] == "ok"

    async def test_every_registered_tool_is_traced(self, monkeypatch):
        """Derived from the registry, not a hand-written list -- same reason as the
        backend stamp: a 24th tool must not ship silently untraced."""
        import code_atlas.server.mcp as mcp_mod

        tracer = _RecordingTracer()
        monkeypatch.setattr(mcp_mod, "_tracer", tracer)
        server = self._server()
        server._tool_manager._tools.clear()

        @server.tool(description="probe")
        async def probe() -> dict:
            return {"ok": True}

        tool = next(iter(server._tool_manager._tools.values()))
        await tool.fn()
        assert tracer.spans[0].name == "mcp.tool.probe"

    async def test_an_error_payload_marks_the_span_failed(self, monkeypatch):
        """Most tools here report failure by *returning* {"error": ...}. Without this
        the span looks clean and the failure is invisible to every error filter."""
        import code_atlas.server.mcp as mcp_mod

        tracer = _RecordingTracer()
        monkeypatch.setattr(mcp_mod, "_tracer", tracer)
        server = self._server()
        server._tool_manager._tools.clear()

        @server.tool(description="probe")
        async def probe() -> dict:
            return {"error": "boom", "code": "QUERY_ERROR"}

        tool = next(iter(server._tool_manager._tools.values()))
        await tool.fn()

        span = tracer.spans[0]
        assert span.attributes["mcp.status"] == "error"
        assert span.attributes["mcp.code"] == "QUERY_ERROR"
        assert span.status is not None, "an error payload must set span status"

    async def test_a_raised_exception_is_recorded_and_re_raised(self, monkeypatch):
        import code_atlas.server.mcp as mcp_mod

        tracer = _RecordingTracer()
        monkeypatch.setattr(mcp_mod, "_tracer", tracer)
        server = self._server()
        server._tool_manager._tools.clear()

        @server.tool(description="probe")
        async def probe() -> dict:
            raise RuntimeError("kaboom")

        tool = next(iter(server._tool_manager._tools.values()))
        with pytest.raises(RuntimeError):
            await tool.fn()

        span = tracer.spans[0]
        assert [type(e) for e in span.exceptions] == [RuntimeError]
        assert span.attributes["mcp.status"] == "exception"

    async def test_a_tool_call_records_its_metrics(self, monkeypatch):
        import code_atlas.server.mcp as mcp_mod
        import code_atlas.telemetry as tel

        calls: list[tuple] = []
        latencies: list[tuple] = []
        monkeypatch.setattr(
            tel._metrics, "tool_calls", type("C", (), {"add": lambda _s, n, a=None: calls.append((n, a))})()
        )
        monkeypatch.setattr(
            tel._metrics, "tool_latency", type("H", (), {"record": lambda _s, n, a=None: latencies.append((n, a))})()
        )
        monkeypatch.setattr(mcp_mod, "_tracer", _RecordingTracer())

        server = self._server()
        tool = next(t for t in server._tool_manager._tools.values() if t.name == "schema_info")
        await tool.fn()

        assert calls == [(1, {"tool": "schema_info", "status": "ok"})]
        assert latencies[0][1] == {"tool": "schema_info"}
        assert latencies[0][0] >= 0

    async def test_tracing_and_stamping_compose_without_eating_the_schema(self):
        """Two wrappers now sit between FastMCP and each tool. FastMCP builds schemas by
        inspecting the function, so a wrapper that dropped __wrapped__ would publish
        every tool as taking no arguments -- silently, and only visible to a client."""
        server = self._server()
        tools = {t.name: t for t in await server.list_tools()}
        params = set((tools["get_node"].inputSchema or {}).get("properties", {}))
        assert {"name", "label", "limit", "offset", "detail"} <= params


class TestBackendStamp:
    """`backend.graph` defaults to "auto" and falls back to SQLite whenever Memgraph
    is unreachable, which on a machine without Docker is the ordinary outcome. Before
    ATL-133 only index_status said so, and an agent reading find_dead_code or
    blast_radius could not tell which engine answered it.
    """

    @staticmethod
    def _server():
        from code_atlas.server.mcp import create_mcp_server
        from code_atlas.settings import AtlasSettings

        return create_mcp_server(AtlasSettings())

    async def test_every_registered_tool_is_stamped(self):
        """Derived from the registry, never a hand-written list.

        A hand-written list is how the 24th tool ships unstamped: nothing fails, the
        payload is simply silent about a degraded backend, which is the exact failure
        mode this story exists to remove.
        """
        server = self._server()
        tools = await server.list_tools()
        assert len(tools) >= 23

        unwrapped = [t.name for t in server._tool_manager._tools.values() if not hasattr(t.fn, "__wrapped__")]
        assert unwrapped == [], f"tools registered without the backend stamp: {unwrapped}"

    async def test_the_stamp_is_transparent_to_the_schema(self):
        """The wrapper must not eat the parameters — FastMCP builds each tool's schema
        by inspecting the function, and a bare *args/**kwargs wrapper would publish
        every tool as taking nothing."""
        server = self._server()
        tools = {t.name: t for t in await server.list_tools()}
        params = set((tools["get_node"].inputSchema or {}).get("properties", {}))
        assert {"name", "label", "limit"} <= params

    async def test_a_healthy_backend_says_nothing(self, monkeypatch):
        """Silence means the supported configuration. Adding a field to every healthy
        result would cost tokens on every call to say nothing."""
        import code_atlas.server.mcp as mcp_mod

        monkeypatch.setattr(mcp_mod, "_BACKEND_NOTE", {})
        server = self._server()
        tool = next(t for t in server._tool_manager._tools.values() if t.name == "schema_info")

        result = await tool.fn()

        assert "backend" not in result
        assert "backend_warning" not in result

    async def test_a_degraded_backend_names_itself(self, monkeypatch):
        import code_atlas.server.mcp as mcp_mod

        note = {"backend": "sqlite-embedded", "backend_warning": "Answered by the embedded backend."}
        monkeypatch.setattr(mcp_mod, "_BACKEND_NOTE", note)
        server = self._server()
        tool = next(t for t in server._tool_manager._tools.values() if t.name == "schema_info")

        result = await tool.fn()

        assert result["backend"] == "sqlite-embedded"
        assert result["backend_warning"] == note["backend_warning"]

    async def test_a_tool_that_already_named_the_backend_is_not_overwritten(self, monkeypatch):
        """index_status carries the backend in its own payload shape. The stamp must
        defer to it rather than double-writing a possibly different string."""
        import code_atlas.server.mcp as mcp_mod

        monkeypatch.setattr(mcp_mod, "_BACKEND_NOTE", {"backend": "sqlite-embedded"})
        server = self._server()
        server._tool_manager._tools.clear()

        @server.tool(description="probe")
        async def probe() -> dict:
            return {"backend": "already-said-it", "ok": True}

        tool = next(iter(server._tool_manager._tools.values()))
        assert (await tool.fn())["backend"] == "already-said-it"

    def test_the_warning_text_has_one_source(self):
        """One string, one source. Two tools disagreeing about what "degraded" means is
        worse than neither saying it, and a copied literal is how they drift apart."""
        import inspect

        import code_atlas.server.mcp as mcp_mod

        source = inspect.getsource(mcp_mod)
        assert source.count("Answered by the embedded SQLite fallback") == 1

    def test_a_non_sqlite_backend_produces_no_note_at_all(self):
        import code_atlas.server.mcp as mcp_mod

        assert mcp_mod._backend_note(object()) == {}
