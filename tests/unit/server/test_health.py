"""Unit tests for health check module (mocked clients — no infrastructure needed)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from code_atlas.server.health import (
    CheckResult,
    CheckStatus,
    HealthReport,
    check_config,
    check_embeddings,
    check_index,
    check_memgraph,
    check_pipeline,
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


def test_report_degraded_when_warn():
    """A WARN keeps ok=True but must surface as degraded so it isn't silent."""
    report = HealthReport(
        checks=[
            CheckResult("a", CheckStatus.OK, "fine"),
            CheckResult("b", CheckStatus.WARN, "degraded"),
        ],
        elapsed_ms=10.0,
    )
    assert report.ok is True
    assert report.degraded is True


def test_report_not_degraded_when_all_ok():
    report = HealthReport(checks=[CheckResult("a", CheckStatus.OK, "fine")], elapsed_ms=10.0)
    assert report.degraded is False


def test_report_degraded_when_fail():
    report = HealthReport(checks=[CheckResult("a", CheckStatus.FAIL, "down")], elapsed_ms=10.0)
    assert report.degraded is True


# ---------------------------------------------------------------------------
# check_pipeline (fake DaemonManager.status())
# ---------------------------------------------------------------------------


class _FakeDaemon:
    def __init__(self, status: dict, bus: object | None = None) -> None:
        self._status = status
        self.bus = bus

    def status(self) -> dict:
        return self._status


def test_check_pipeline_ok():
    daemon = _FakeDaemon({"tasks_running": 2, "tasks_total": 2, "crash_counts": {}, "last_crash": {}})
    result = check_pipeline(daemon)  # type: ignore[arg-type]
    assert result.status == CheckStatus.OK
    assert "2" in result.message


def test_check_pipeline_warn_on_crash():
    daemon = _FakeDaemon(
        {"tasks_running": 2, "tasks_total": 2, "crash_counts": {"ast-0": 3}, "last_crash": {"ast-0": "ValueError()"}}
    )
    result = check_pipeline(daemon)  # type: ignore[arg-type]
    assert result.status == CheckStatus.WARN
    assert "ast-0" in result.message


def test_check_pipeline_fail_on_dead_task():
    daemon = _FakeDaemon({"tasks_running": 1, "tasks_total": 2, "crash_counts": {}, "last_crash": {}})
    result = check_pipeline(daemon)  # type: ignore[arg-type]
    assert result.status == CheckStatus.FAIL
    assert "dead" in result.message.lower()


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
    assert "Memgraph" in result.message


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


async def test_check_memgraph_names_sqlite_backend_when_active(tmp_path):
    """Health-check honesty: a SqliteGraphClient must be reported as SQLite, never Memgraph."""
    from code_atlas.backends.sqlite_graph import SqliteGraphClient

    graph = SqliteGraphClient(tmp_path / "graph.sqlite3")
    mg_settings = MemgraphSettings()

    result = await check_memgraph(graph, mg_settings)
    assert result.status == CheckStatus.OK
    assert "SQLite" in result.message
    assert "Memgraph" not in result.message
    await graph.close()


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


async def test_check_embeddings_unreachable_names_provider():
    """Unreachable embeddings must name the provider/endpoint so the failure is actionable."""
    embed = AsyncMock()
    embed.health_check = AsyncMock(return_value=False)
    embed_settings = EmbeddingSettings()

    result = await check_embeddings(embed, embed_settings)
    assert result.status == CheckStatus.WARN
    assert embed_settings.provider in result.message
    assert embed_settings.base_url in result.message


# ---------------------------------------------------------------------------
# check_valkey
# ---------------------------------------------------------------------------


async def test_check_valkey_success():
    redis_settings = RedisSettings()
    bus = AsyncMock()
    bus.ping = AsyncMock(return_value=True)

    result = await check_valkey(bus, redis_settings)
    assert result.status == CheckStatus.OK
    assert "Connected" in result.message
    assert "Valkey" in result.message


async def test_check_valkey_failure():
    redis_settings = RedisSettings()
    bus = AsyncMock()
    bus.ping = AsyncMock(side_effect=ConnectionRefusedError("refused"))

    result = await check_valkey(bus, redis_settings)
    assert result.status == CheckStatus.WARN
    assert "Unreachable" in result.message


async def test_check_valkey_down_names_indexing_disabled():
    """Valkey down must loudly state that indexing is disabled (not a silent WARN)."""
    redis_settings = RedisSettings()
    bus = AsyncMock()
    bus.ping = AsyncMock(side_effect=ConnectionRefusedError("refused"))

    result = await check_valkey(bus, redis_settings)
    assert result.status == CheckStatus.WARN
    assert "indexing" in (result.message + " " + result.detail).lower()


async def test_check_valkey_none():
    redis_settings = RedisSettings()
    result = await check_valkey(None, redis_settings)
    assert result.status == CheckStatus.WARN
    assert "No client" in result.message


async def test_check_valkey_names_sqlite_backend_when_active(tmp_path):
    """Health-check honesty: a SqliteEventBus must be reported as SQLite, never Valkey."""
    from code_atlas.backends.sqlite_queue import SqliteEventBus

    bus = SqliteEventBus(tmp_path / "queue.sqlite3")
    redis_settings = RedisSettings()

    result = await check_valkey(bus, redis_settings)
    assert result.status == CheckStatus.OK
    assert "SQLite" in result.message
    assert "Valkey" not in result.message
    await bus.close()


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


async def test_check_config_resolves_dotenv_when_caller_passes_none(tmp_path):
    """The MCP server has no dotenv handle to pass — cli.py loads the file before
    handing off. Reporting 'not found' for a loaded .env sent a past debugging
    session chasing a phantom stale process.
    """
    (tmp_path / ".git").mkdir()
    settings = AtlasSettings(project_root=tmp_path)
    with patch("code_atlas.server.health.find_dotenv", return_value="/somewhere/.env"):
        result = await check_config(settings)
    assert ".env: /somewhere/.env" in result.detail


async def test_check_config_reports_missing_dotenv(tmp_path):
    (tmp_path / ".git").mkdir()
    settings = AtlasSettings(project_root=tmp_path)
    with patch("code_atlas.server.health.find_dotenv", return_value=""):
        result = await check_config(settings)
    assert ".env: not found" in result.detail


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

    with patch("code_atlas.server.health.StalenessChecker") as mock_checker_cls:
        from code_atlas.indexing.orchestrator import StalenessInfo

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

    bus = AsyncMock()
    bus.ping = AsyncMock(return_value=True)

    report = await run_health_checks(settings, graph=graph, embed=embed, bus=bus)

    # Should have 8 checks total (mode, config, memgraph, embeddings, valkey, schema, embedding_model, index)
    assert len(report.checks) == 8
    assert report.ok is False

    # Schema, embedding_model, and index should be marked as FAIL/skipped
    by_name = {c.name: c for c in report.checks}
    assert by_name["mode"].status == CheckStatus.OK
    assert by_name["schema"].status == CheckStatus.FAIL
    assert "Skipped" in by_name["schema"].message
    assert by_name["embedding_model"].status == CheckStatus.FAIL
    assert "Skipped" in by_name["embedding_model"].message
    assert by_name["index"].status == CheckStatus.FAIL
    assert "Skipped" in by_name["index"].message


async def test_all_pass_when_healthy(tmp_path):
    from code_atlas.indexing.orchestrator import StalenessInfo
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

    bus = AsyncMock()
    bus.ping = AsyncMock(return_value=True)

    with patch("code_atlas.server.health.StalenessChecker") as mock_checker_cls:
        checker = MagicMock()
        checker.check = AsyncMock(return_value=StalenessInfo(stale=False))
        mock_checker_cls.return_value = checker

        report = await run_health_checks(settings, graph=graph, embed=embed, bus=bus)

    assert report.ok is True
    assert len(report.checks) == 8
    for c in report.checks:
        assert c.status in (CheckStatus.OK, CheckStatus.WARN), f"{c.name} unexpectedly {c.status}: {c.message}"


async def test_pipeline_check_appended_when_daemon_passed(tmp_path):
    from code_atlas.indexing.orchestrator import StalenessInfo
    from code_atlas.schema import SCHEMA_VERSION

    (tmp_path / ".git").mkdir()
    settings = AtlasSettings(project_root=tmp_path)

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

    embed = AsyncMock()
    embed.health_check = AsyncMock(return_value=True)

    daemon = _FakeDaemon({"tasks_running": 2, "tasks_total": 2, "crash_counts": {}, "last_crash": {}})

    bus = AsyncMock()
    bus.ping = AsyncMock(return_value=True)

    with patch("code_atlas.server.health.StalenessChecker") as mock_checker_cls:
        checker = MagicMock()
        checker.check = AsyncMock(return_value=StalenessInfo(stale=False))
        mock_checker_cls.return_value = checker

        report = await run_health_checks(
            settings,
            graph=graph,
            embed=embed,
            daemon=daemon,  # type: ignore[arg-type]
            bus=bus,
        )

    by_name = {c.name: c for c in report.checks}
    assert "pipeline" in by_name
    assert by_name["pipeline"].status == CheckStatus.OK
    assert len(report.checks) == 9


async def test_no_pipeline_check_without_daemon(tmp_path):
    (tmp_path / ".git").mkdir()
    settings = AtlasSettings(project_root=tmp_path)

    graph = AsyncMock()
    graph.ping = AsyncMock(side_effect=ConnectionRefusedError("refused"))
    graph.close = AsyncMock()
    embed = AsyncMock()
    embed.health_check = AsyncMock(return_value=True)

    bus = AsyncMock()
    bus.ping = AsyncMock(return_value=True)

    report = await run_health_checks(settings, graph=graph, embed=embed, bus=bus)

    assert "pipeline" not in {c.name for c in report.checks}


async def test_run_health_checks_uses_daemon_bus_when_not_explicitly_passed(tmp_path):
    """No explicit *bus* is passed, but *daemon* has a live one — that live bus must be
    probed (and left open) instead of opening a redundant new connection.
    """
    (tmp_path / ".git").mkdir()
    settings = AtlasSettings(project_root=tmp_path)

    graph = AsyncMock()
    graph.ping = AsyncMock(side_effect=ConnectionRefusedError("refused"))
    graph.close = AsyncMock()
    embed = AsyncMock()
    embed.health_check = AsyncMock(return_value=True)

    daemon_bus = AsyncMock()
    daemon_bus.ping = AsyncMock(return_value=True)
    daemon = _FakeDaemon({"tasks_running": 1, "tasks_total": 1, "crash_counts": {}, "last_crash": {}}, bus=daemon_bus)

    report = await run_health_checks(settings, graph=graph, embed=embed, daemon=daemon)  # type: ignore[arg-type]

    by_name = {c.name: c for c in report.checks}
    assert by_name["valkey"].status == CheckStatus.OK
    daemon_bus.ping.assert_awaited_once()
    daemon_bus.close.assert_not_called()  # not owned by run_health_checks — must not be closed


async def test_run_health_checks_builds_own_bus_when_none_available(tmp_path):
    """Neither *bus* nor a daemon-provided one is available — run_health_checks falls
    back to the same ``create_event_bus`` factory used everywhere else, and closes
    what it opened.
    """
    (tmp_path / ".git").mkdir()
    settings = AtlasSettings(project_root=tmp_path)

    graph = AsyncMock()
    graph.ping = AsyncMock(side_effect=ConnectionRefusedError("refused"))
    graph.close = AsyncMock()
    embed = AsyncMock()
    embed.health_check = AsyncMock(return_value=True)

    own_bus = AsyncMock()
    own_bus.ping = AsyncMock(return_value=True)

    with patch("code_atlas.server.health.create_event_bus", AsyncMock(return_value=own_bus)) as mock_factory:
        report = await run_health_checks(settings, graph=graph, embed=embed)

    mock_factory.assert_awaited_once_with(settings)
    by_name = {c.name: c for c in report.checks}
    assert by_name["valkey"].status == CheckStatus.OK
    own_bus.close.assert_awaited_once()  # owned by run_health_checks — must be closed
