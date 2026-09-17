"""Unit tests for health check module (mocked clients — no infrastructure needed)."""

from __future__ import annotations

from typing import ClassVar
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
    """Defaults filled in so each test states only the keys it is about.

    A literal status dict per test means every new field DaemonManager reports breaks
    five unrelated tests with a KeyError, which is noise, not signal.
    """

    _DEFAULTS: ClassVar[dict] = {
        "tasks_running": 0,
        "tasks_total": 0,
        "crash_counts": {},
        "last_crash": {},
        "disabled_reason": "",
    }

    def __init__(self, status: dict, bus: object | None = None) -> None:
        self._status = {**self._DEFAULTS, **status}
        self.bus = bus

    def status(self) -> dict:
        return self._status


def test_check_pipeline_reports_a_deliberately_disabled_pipeline():
    """`--no-index` and a pipeline that died on startup both leave zero tasks running.
    Reporting the first as "0 task(s) running -- OK" sends someone hunting a bug
    instead of reading their own configuration."""
    daemon = _FakeDaemon({"disabled_reason": "indexing disabled (--no-index)"})
    result = check_pipeline(daemon)  # ty: ignore[invalid-argument-type]
    assert result.status == CheckStatus.OK
    assert "--no-index" in result.message


def test_check_pipeline_ok():
    daemon = _FakeDaemon({"tasks_running": 2, "tasks_total": 2, "crash_counts": {}, "last_crash": {}})
    result = check_pipeline(daemon)  # ty: ignore[invalid-argument-type]
    assert result.status == CheckStatus.OK
    assert "2" in result.message


def test_check_pipeline_warn_on_crash():
    daemon = _FakeDaemon(
        {"tasks_running": 2, "tasks_total": 2, "crash_counts": {"ast-0": 3}, "last_crash": {"ast-0": "ValueError()"}}
    )
    result = check_pipeline(daemon)  # ty: ignore[invalid-argument-type]
    assert result.status == CheckStatus.WARN
    assert "ast-0" in result.message


def test_check_pipeline_fail_on_dead_task():
    daemon = _FakeDaemon({"tasks_running": 1, "tasks_total": 2, "crash_counts": {}, "last_crash": {}})
    result = check_pipeline(daemon)  # ty: ignore[invalid-argument-type]
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


async def test_connect_timeout_comes_from_health_settings(tmp_path):
    """A backend slower than `[health] connect_timeout_s` is down; raising the setting makes it up."""
    import asyncio

    from code_atlas.settings import HealthSettings

    async def slow_ping() -> bool:
        await asyncio.sleep(0.2)
        return True

    (tmp_path / ".git").mkdir()
    settings = AtlasSettings(project_root=tmp_path, health=HealthSettings(connect_timeout_s=0.05))
    graph, bus, embed = AsyncMock(), AsyncMock(), AsyncMock()
    graph.ping, bus.ping = slow_ping, slow_ping
    bus.read_indexer_lease.return_value = None
    report = await run_health_checks(settings, graph=graph, bus=bus, embed=embed)
    by_name = {c.name: c.status for c in report.checks}
    assert (by_name["memgraph"], by_name["valkey"]) == (CheckStatus.FAIL, CheckStatus.WARN)

    # The same slow backend, given the time a remote one needs.
    assert (await check_memgraph(graph, MemgraphSettings(), timeout_s=2.0)).status == CheckStatus.OK
    assert (await check_valkey(bus, RedisSettings(), timeout_s=2.0)).status == CheckStatus.OK


async def test_a_hung_valkey_costs_one_connect_timeout_not_two(tmp_path):
    """Valkey reads run beside the other connect checks, bounded -- not queued after them, unbounded."""
    import asyncio
    import time

    from code_atlas.settings import HealthSettings

    async def hang(*_args, **_kwargs):
        await asyncio.sleep(30)

    (tmp_path / ".git").mkdir()
    settings = AtlasSettings(project_root=tmp_path, health=HealthSettings(connect_timeout_s=0.3))
    graph, bus, embed = AsyncMock(), AsyncMock(), AsyncMock()
    graph.ping = AsyncMock(side_effect=ConnectionRefusedError("refused"))
    bus.ping, bus.read_indexer_lease = hang, hang

    t0 = time.perf_counter()
    report = await run_health_checks(settings, graph=graph, bus=bus, embed=embed)
    elapsed = time.perf_counter() - t0

    assert {c.name: c.status for c in report.checks}["valkey"] == CheckStatus.WARN
    assert elapsed < 0.55, f"{elapsed:.2f}s: the lease read waited behind the ping, or was never bounded"


async def test_a_slow_embedding_provider_is_not_called_broken():
    import asyncio

    async def slow() -> bool:
        await asyncio.sleep(1)
        return True

    embed = AsyncMock()
    embed.health_check = slow
    settings = EmbeddingSettings(enabled=True, provider="litellm", model="m")
    result = await check_embeddings(embed, settings, timeout_s=0.05)
    assert result.status == CheckStatus.WARN
    assert "timeout" in result.message
    assert "API key" not in result.suggestion


async def test_check_memgraph_none():
    mg_settings = MemgraphSettings()
    result = await check_memgraph(None, mg_settings)
    assert result.status == CheckStatus.FAIL
    assert "No client" in result.message


async def test_check_memgraph_warns_when_the_sqlite_fallback_is_active(tmp_path):
    """A fully-degraded install must not report an unqualified OK (ATL-112).

    This assertion was inverted until ATL-112: it required `status == OK`, which pinned
    the defect as the contract. An undeclared graph backend falls back to SQLite whenever
    Memgraph is unreachable, so on a machine without Docker running this is the *default*
    outcome — and ADR-0015 calls SQLite explicitly not a parity replacement. WARN keeps
    `report.ok` True (the tool still works) while flipping `report.degraded`, which is
    exactly what the distinction is for.
    """
    from code_atlas.backends.sqlite_graph import SqliteGraphClient

    async with SqliteGraphClient(tmp_path / "graph.sqlite3") as graph:
        mg_settings = MemgraphSettings()

        result = await check_memgraph(graph, mg_settings)
        assert result.status == CheckStatus.WARN
        assert "SQLite" in result.message
        assert "NOT Memgraph" in result.message, "it must say which engine it is *not*"
        assert result.suggestion, "a warning without a remedy is just noise"


async def test_a_healthy_memgraph_check_stays_a_plain_ok(tmp_path):
    """The warning must not fire for the supported configuration."""

    class _Memgraph:
        async def ping(self) -> bool:
            return True

    result = await check_memgraph(_Memgraph(), MemgraphSettings())  # ty: ignore[invalid-argument-type]

    assert result.status == CheckStatus.OK
    assert "Memgraph" in result.message


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
    bus.read_indexer_lease.return_value = None  # no foreign indexer
    bus.ping = AsyncMock(return_value=True)

    result = await check_valkey(bus, redis_settings)
    assert result.status == CheckStatus.OK
    assert "Connected" in result.message
    assert "Valkey" in result.message


async def test_check_valkey_failure():
    redis_settings = RedisSettings()
    bus = AsyncMock()
    bus.read_indexer_lease.return_value = None  # no foreign indexer
    bus.ping = AsyncMock(side_effect=ConnectionRefusedError("refused"))

    result = await check_valkey(bus, redis_settings)
    assert result.status == CheckStatus.WARN
    assert "Unreachable" in result.message


async def test_check_valkey_down_names_indexing_disabled():
    """Valkey down must loudly state that indexing is disabled (not a silent WARN)."""
    redis_settings = RedisSettings()
    bus = AsyncMock()
    bus.read_indexer_lease.return_value = None  # no foreign indexer
    bus.ping = AsyncMock(side_effect=ConnectionRefusedError("refused"))

    result = await check_valkey(bus, redis_settings)
    assert result.status == CheckStatus.WARN
    assert "indexing" in (result.message + " " + result.detail).lower()


async def test_check_valkey_none():
    redis_settings = RedisSettings()
    result = await check_valkey(None, redis_settings)
    assert result.status == CheckStatus.WARN
    assert "No client" in result.message


async def test_check_valkey_names_postgres_when_that_is_the_queue():
    """A Postgres queue reports itself, its address, and never its password."""
    from pydantic import SecretStr

    from code_atlas.backends.postgres_queue import PostgresConnections, PostgresEventBus
    from code_atlas.settings import PostgresSettings

    bus = PostgresEventBus(PostgresConnections(PostgresSettings(host="pg.local", password=SecretStr("s3cret"))))
    bus.ping = AsyncMock(return_value=True)

    ok = await check_valkey(bus, RedisSettings())
    assert ok.status == CheckStatus.OK
    assert "Postgres (pg.local:5432/atlas)" in ok.message
    assert "Valkey" not in ok.message

    bus.ping = AsyncMock(side_effect=ConnectionRefusedError("refused"))
    down = await check_valkey(bus, RedisSettings())
    assert down.status == CheckStatus.WARN
    assert "Postgres" in down.message
    assert "valkey" not in down.suggestion.lower()
    assert "s3cret" not in down.message + down.detail + down.suggestion


async def test_check_valkey_warns_when_the_sqlite_fallback_is_active(tmp_path):
    """The embedded queue must not report an unqualified OK either (ATL-112).

    Inverted for the same reason as the memgraph check: ADR-0015 calls the embedded path
    "not a parity replacement for the Memgraph+Valkey path", and names a concrete gap —
    blocking reads are emulated by short polling, because SQLite has no server-side
    blocking. `atlas health` printed a green tick for a fallback nobody chose.
    """
    from code_atlas.backends.sqlite_queue import SqliteEventBus

    async with SqliteEventBus(tmp_path / "queue.sqlite3") as bus:
        redis_settings = RedisSettings()

        result = await check_valkey(bus, redis_settings)
        assert result.status == CheckStatus.WARN
        assert "SQLite" in result.message
        assert "NOT Valkey" in result.message
        assert result.suggestion


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
    bus.read_indexer_lease.return_value = None  # no foreign indexer
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
    bus.read_indexer_lease.return_value = None  # no foreign indexer
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
    bus.read_indexer_lease.return_value = None  # no foreign indexer
    bus.ping = AsyncMock(return_value=True)

    with patch("code_atlas.server.health.StalenessChecker") as mock_checker_cls:
        checker = MagicMock()
        checker.check = AsyncMock(return_value=StalenessInfo(stale=False))
        mock_checker_cls.return_value = checker

        report = await run_health_checks(
            settings,
            graph=graph,
            embed=embed,
            daemon=daemon,  # ty: ignore[invalid-argument-type]
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
    bus.read_indexer_lease.return_value = None  # no foreign indexer
    bus.ping = AsyncMock(return_value=True)

    report = await run_health_checks(settings, graph=graph, embed=embed, bus=bus)

    assert "pipeline" not in {c.name for c in report.checks}


async def test_the_daemon_is_only_consulted_for_pipeline_liveness(tmp_path):
    """The daemon no longer doubles as a source of connections.

    run_health_checks used to fall back to `daemon.bus` when no bus was passed, which
    was the service-locator half of its design: a function reaching for a dependency
    through an unrelated collaborator. It also could not work when indexing was off,
    since `daemon.bus` is None then -- exactly when someone is most likely to be asking
    what is wrong.

    The caller passes the bus it holds; the daemon is consulted for one thing, which is
    whether its pipeline tasks are alive.
    """
    (tmp_path / ".git").mkdir()
    settings = AtlasSettings(project_root=tmp_path)

    graph = AsyncMock()
    graph.ping = AsyncMock(side_effect=ConnectionRefusedError("refused"))
    graph.close = AsyncMock()
    embed = AsyncMock()
    embed.health_check = AsyncMock(return_value=True)

    bus = AsyncMock()
    bus.read_indexer_lease.return_value = None  # no foreign indexer
    bus.ping = AsyncMock(return_value=True)

    daemon_bus = AsyncMock()
    daemon = _FakeDaemon({"tasks_running": 1, "tasks_total": 1}, bus=daemon_bus)

    report = await run_health_checks(settings, graph=graph, bus=bus, embed=embed, daemon=daemon)  # ty: ignore[invalid-argument-type]

    by_name = {c.name: c for c in report.checks}
    assert by_name["valkey"].status == CheckStatus.OK
    bus.ping.assert_awaited_once()  # the passed bus is the one probed
    daemon_bus.ping.assert_not_awaited()
    assert by_name["pipeline"].status == CheckStatus.OK
    daemon_bus.close.assert_not_called()  # not owned by run_health_checks — must not be closed


async def test_run_health_checks_closes_nothing_it_was_given(tmp_path):
    """The inverse of what this test used to assert.

    It previously checked that run_health_checks built its own bus and closed it. That
    behaviour is gone: creating connections to report on connections answers a subtly
    different question -- "can a fresh connection reach these services" rather than "are
    the connections this process is using healthy" -- and it needed a pair of ownership
    booleans and a finally to avoid closing a caller's client by accident.

    Now the caller passes both and keeps them. A health check that closed the graph the
    MCP session is serving from would end the session that asked how it was doing.
    """
    (tmp_path / ".git").mkdir()
    settings = AtlasSettings(project_root=tmp_path)

    graph = AsyncMock()
    graph.ping = AsyncMock(side_effect=ConnectionRefusedError("refused"))
    graph.close = AsyncMock()
    embed = AsyncMock()
    embed.health_check = AsyncMock(return_value=True)

    bus = AsyncMock()
    bus.ping = AsyncMock(return_value=True)

    report = await run_health_checks(settings, graph=graph, bus=bus, embed=embed)

    by_name = {c.name: c for c in report.checks}
    assert by_name["valkey"].status == CheckStatus.OK
    graph.close.assert_not_awaited()
    bus.close.assert_not_awaited()
