"""Unit tests for health check module (mocked clients — no infrastructure needed)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from code_atlas.health import (
    CheckResult,
    CheckStatus,
    HealthReport,
    check_config,
    check_embeddings,
    check_index,
    check_memgraph,
    check_schema,
    check_valkey,
    run_health_checks,
)
from code_atlas.settings import AtlasSettings, EmbeddingSettings, MemgraphSettings, RedisSettings

# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


def test_report_ok_all_pass():
    report = HealthReport(
        checks=[
            CheckResult("a", CheckStatus.OK, "fine"),
            CheckResult("b", CheckStatus.OK, "fine"),
        ],
        elapsed_ms=10.0,
    )
    assert report.ok is True


def test_report_ok_with_warns():
    report = HealthReport(
        checks=[
            CheckResult("a", CheckStatus.OK, "fine"),
            CheckResult("b", CheckStatus.WARN, "degraded"),
        ],
        elapsed_ms=10.0,
    )
    assert report.ok is True


def test_report_fail():
    report = HealthReport(
        checks=[
            CheckResult("a", CheckStatus.OK, "fine"),
            CheckResult("b", CheckStatus.FAIL, "down"),
        ],
        elapsed_ms=10.0,
    )
    assert report.ok is False


# ---------------------------------------------------------------------------
# check_memgraph
# ---------------------------------------------------------------------------


async def test_check_memgraph_success():
    graph = AsyncMock()
    graph.ping = AsyncMock(return_value=True)
    mg_settings = MemgraphSettings()

    result = await check_memgraph(graph, mg_settings)
    assert result.status == CheckStatus.OK
    assert "Connected" in result.message


async def test_check_memgraph_failure():
    graph = AsyncMock()
    graph.ping = AsyncMock(side_effect=ConnectionRefusedError("refused"))
    mg_settings = MemgraphSettings()

    result = await check_memgraph(graph, mg_settings)
    assert result.status == CheckStatus.FAIL
    assert "Unreachable" in result.message


async def test_check_memgraph_none():
    mg_settings = MemgraphSettings()
    result = await check_memgraph(None, mg_settings)
    assert result.status == CheckStatus.FAIL
    assert "No client" in result.message


# ---------------------------------------------------------------------------
# check_embeddings
# ---------------------------------------------------------------------------


async def test_check_embeddings_success():
    embed = AsyncMock()
    embed.health_check = AsyncMock(return_value=True)
    embed_settings = EmbeddingSettings()

    result = await check_embeddings(embed, embed_settings)
    assert result.status == CheckStatus.OK
    assert "Responding" in result.message


async def test_check_embeddings_failure():
    embed = AsyncMock()
    embed.health_check = AsyncMock(return_value=False)
    embed_settings = EmbeddingSettings()

    result = await check_embeddings(embed, embed_settings)
    assert result.status == CheckStatus.WARN
    assert "Unreachable" in result.message


async def test_check_embeddings_none():
    embed_settings = EmbeddingSettings()
    result = await check_embeddings(None, embed_settings)
    assert result.status == CheckStatus.WARN
    assert "No client" in result.message


# ---------------------------------------------------------------------------
# check_valkey
# ---------------------------------------------------------------------------


async def test_check_valkey_success():
    redis_settings = RedisSettings()
    with patch("code_atlas.health.EventBus") as mock_bus_cls:
        bus_instance = AsyncMock()
        bus_instance.ping = AsyncMock(return_value=True)
        bus_instance.close = AsyncMock()
        mock_bus_cls.return_value = bus_instance

        result = await check_valkey(redis_settings)
        assert result.status == CheckStatus.OK
        assert "Connected" in result.message
        bus_instance.close.assert_awaited_once()


async def test_check_valkey_failure():
    redis_settings = RedisSettings()
    with patch("code_atlas.health.EventBus") as mock_bus_cls:
        bus_instance = AsyncMock()
        bus_instance.ping = AsyncMock(side_effect=ConnectionRefusedError("refused"))
        bus_instance.close = AsyncMock()
        mock_bus_cls.return_value = bus_instance

        result = await check_valkey(redis_settings)
        assert result.status == CheckStatus.WARN
        assert "Unreachable" in result.message
        bus_instance.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# check_config
# ---------------------------------------------------------------------------


async def test_check_config_valid(tmp_path):
    (tmp_path / ".git").mkdir()
    settings = AtlasSettings(project_root=tmp_path)
    result = await check_config(settings)
    assert result.status == CheckStatus.OK
    assert "Valid root" in result.message


async def test_check_config_no_git(tmp_path):
    settings = AtlasSettings(project_root=tmp_path)
    result = await check_config(settings)
    assert result.status == CheckStatus.WARN
    assert "No git repo" in result.message


# ---------------------------------------------------------------------------
# check_schema
# ---------------------------------------------------------------------------


async def test_check_schema_matches():
    from code_atlas.schema import SCHEMA_VERSION

    graph = AsyncMock()
    graph.get_schema_version = AsyncMock(return_value=SCHEMA_VERSION)
    result = await check_schema(graph)
    assert result.status == CheckStatus.OK
    assert "current" in result.message


async def test_check_schema_missing():
    graph = AsyncMock()
    graph.get_schema_version = AsyncMock(return_value=None)
    result = await check_schema(graph)
    assert result.status == CheckStatus.WARN
    assert "No schema" in result.message


async def test_check_schema_newer():
    from code_atlas.schema import SCHEMA_VERSION

    graph = AsyncMock()
    graph.get_schema_version = AsyncMock(return_value=SCHEMA_VERSION + 1)
    result = await check_schema(graph)
    assert result.status == CheckStatus.FAIL
    assert "newer" in result.detail


# ---------------------------------------------------------------------------
# check_index
# ---------------------------------------------------------------------------


async def test_check_index_no_projects(tmp_path):
    graph = AsyncMock()
    graph.get_project_status = AsyncMock(return_value=[])
    settings = AtlasSettings(project_root=tmp_path)

    result = await check_index(graph, settings)
    assert result.status == CheckStatus.WARN
    assert "No indexed projects" in result.message


async def test_check_index_stale(tmp_path):
    (tmp_path / ".git").mkdir()
    node = MagicMock()
    node.items.return_value = [("name", "myproject")]
    node.get = lambda k, d=None: {"name": "myproject"}.get(k, d)
    graph = AsyncMock()
    graph.get_project_status = AsyncMock(return_value=[{"n": node}])
    graph.get_project_git_hash = AsyncMock(return_value="aabbccdd")
    settings = AtlasSettings(project_root=tmp_path)

    with patch("code_atlas.health.StalenessChecker") as mock_checker_cls:
        from code_atlas.indexer import StalenessInfo

        checker = MagicMock()
        checker.check = AsyncMock(return_value=StalenessInfo(stale=True, last_indexed_commit="aabbccdd"))
        mock_checker_cls.return_value = checker

        result = await check_index(graph, settings)
        assert result.status == CheckStatus.WARN
        assert "stale" in result.message


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------


async def test_skips_db_checks_when_memgraph_down(tmp_path):
    (tmp_path / ".git").mkdir()
    settings = AtlasSettings(project_root=tmp_path)

    graph = AsyncMock()
    graph.ping = AsyncMock(side_effect=ConnectionRefusedError("refused"))
    graph.close = AsyncMock()

    embed = AsyncMock()
    embed.health_check = AsyncMock(return_value=True)

    with patch("code_atlas.health.EventBus") as mock_bus_cls:
        bus_instance = AsyncMock()
        bus_instance.ping = AsyncMock(return_value=True)
        bus_instance.close = AsyncMock()
        mock_bus_cls.return_value = bus_instance

        report = await run_health_checks(settings, graph=graph, embed=embed)

    # Should have 7 checks total (config, memgraph, embeddings, valkey, schema, embedding_model, index)
    assert len(report.checks) == 7
    assert report.ok is False

    # Schema, embedding_model, and index should be marked as FAIL/skipped
    by_name = {c.name: c for c in report.checks}
    assert by_name["schema"].status == CheckStatus.FAIL
    assert "Skipped" in by_name["schema"].message
    assert by_name["embedding_model"].status == CheckStatus.FAIL
    assert "Skipped" in by_name["embedding_model"].message
    assert by_name["index"].status == CheckStatus.FAIL
    assert "Skipped" in by_name["index"].message


async def test_all_pass_when_healthy(tmp_path):
    from code_atlas.indexer import StalenessInfo
    from code_atlas.schema import SCHEMA_VERSION

    (tmp_path / ".git").mkdir()
    settings = AtlasSettings(project_root=tmp_path)

    # Mock graph
    node = MagicMock()
    node.items.return_value = [("name", "test-project")]
    node.get = lambda k, d=None: {"name": "test-project"}.get(k, d)

    graph = AsyncMock()
    graph.ping = AsyncMock(return_value=True)
    graph.close = AsyncMock()
    graph.get_schema_version = AsyncMock(return_value=SCHEMA_VERSION)
    graph.get_project_status = AsyncMock(return_value=[{"n": node}])
    graph.get_project_git_hash = AsyncMock(return_value=None)
    graph.get_embedding_config = AsyncMock(return_value=None)

    # Mock embed
    embed = AsyncMock()
    embed.health_check = AsyncMock(return_value=True)

    with (
        patch("code_atlas.health.EventBus") as mock_bus_cls,
        patch("code_atlas.health.StalenessChecker") as mock_checker_cls,
    ):
        bus_instance = AsyncMock()
        bus_instance.ping = AsyncMock(return_value=True)
        bus_instance.close = AsyncMock()
        mock_bus_cls.return_value = bus_instance

        checker = MagicMock()
        checker.check = AsyncMock(return_value=StalenessInfo(stale=False))
        mock_checker_cls.return_value = checker

        report = await run_health_checks(settings, graph=graph, embed=embed)

    assert report.ok is True
    assert len(report.checks) == 7
    for c in report.checks:
        assert c.status in (CheckStatus.OK, CheckStatus.WARN), f"{c.name} unexpectedly {c.status}: {c.message}"
