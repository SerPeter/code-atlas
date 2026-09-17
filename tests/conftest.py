"""Shared test constants and infrastructure fixtures for Code Atlas.

The infra fixtures live here — not in ``tests/integration/conftest.py`` — because
``tests/bench/`` tests are also marked ``integration`` and request them. A conftest
is only visible to its own directory subtree, so keeping ``graph_client`` under
``tests/integration/`` made ``pytest -m integration`` from the repo root error on
the bench tests while ``pytest tests/integration -m integration`` passed. See
ADR-0010 for the isolation design these fixtures implement.

Everything here is lazy: the session-scoped ``_infra_endpoints`` fixture is only
instantiated when a test actually requests it, so a unit-only run never starts a
container. ``GraphClient`` is imported inside the fixtures rather than at module
level for the same reason — it pulls in the neo4j driver (~4s), which a unit run
that touches no graph code should not pay for.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from pydantic import SecretStr

from code_atlas.schema import GLOBAL_PROJECT, generate_drop_text_index_ddl, generate_drop_vector_index_ddl
from code_atlas.settings import (
    AtlasSettings,
    BackendSettings,
    EmbeddingSettings,
    MemgraphSettings,
    PostgresSettings,
    RedisSettings,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from code_atlas.graph.client import GraphClient

# ---------------------------------------------------------------------------
# Constants re-exported to test modules
# ---------------------------------------------------------------------------

TEST_DRAIN_TIMEOUT_S: float = 60.0
"""Shortened drain timeout for integration tests (default 600s is too long)."""

NO_EMBED = EmbeddingSettings(enabled=False)
"""Embedding settings that disable the embed stage entirely — use for pipeline tests
that don't need real or mocked embeddings."""


# ---------------------------------------------------------------------------
# Infrastructure endpoint discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InfraEndpoints:
    """Resolved host/port pairs for test infrastructure."""

    memgraph_host: str
    memgraph_port: int
    valkey_host: str
    valkey_port: int


def _export_atlas_env(ep: InfraEndpoints) -> InfraEndpoints:
    """Point any AtlasSettings constructed inside tests at the test instances.

    Defense-in-depth: fixtures pass explicit Memgraph/Redis settings, but the
    repo-root atlas.toml hardcodes the production ports (7687/6379), so a bare
    ``AtlasSettings(project_root=tmp_path)`` would otherwise resolve to the
    production infrastructure (init > env > toml). Env beats toml.
    """
    os.environ["ATLAS_BACKEND__GRAPH__MEMGRAPH__HOST"] = ep.memgraph_host
    os.environ["ATLAS_BACKEND__GRAPH__MEMGRAPH__PORT"] = str(ep.memgraph_port)
    os.environ["ATLAS_BACKEND__QUEUE__VALKEY__HOST"] = ep.valkey_host
    os.environ["ATLAS_BACKEND__QUEUE__VALKEY__PORT"] = str(ep.valkey_port)
    return ep


@pytest.fixture(scope="session")
def _infra_endpoints() -> Iterator[InfraEndpoints]:
    """Provision ISOLATED test Memgraph + Valkey for the whole session.

    1. Explicit env override — ``ATLAS_TEST_MEMGRAPH_PORT`` / ``ATLAS_TEST_VALKEY_PORT``
       point at existing isolated instances (CI service containers, or the
       compose fast path: ``docker compose --profile test up -d`` →
       memgraph-test :7688, valkey-test :6380).
    2. Default — session-scoped testcontainers on random ports.
    3. Skip with instructions if Docker is unavailable.

    The production ports (7687/6379) are NEVER used or probed. The
    production-data guard (_assert_disposable_db) applies on every path.
    """
    mg_env = os.environ.get("ATLAS_TEST_MEMGRAPH_PORT")
    vk_env = os.environ.get("ATLAS_TEST_VALKEY_PORT")
    if mg_env or vk_env:
        # Unset var falls back to its compose test-profile port
        yield _export_atlas_env(
            InfraEndpoints(
                memgraph_host="localhost",
                memgraph_port=int(mg_env or "7688"),
                valkey_host="localhost",
                valkey_port=int(vk_env or "6380"),
            )
        )
        return

    # --- default: testcontainers on random ports ---
    _fast_path_hint = (
        "docker compose --profile test up -d && "
        "ATLAS_TEST_MEMGRAPH_PORT=7688 ATLAS_TEST_VALKEY_PORT=6380 uv run pytest -m integration"
    )
    try:
        from testcontainers.core.container import DockerContainer
        from testcontainers.core.wait_strategies import ExecWaitStrategy
    except ImportError:
        pytest.skip(
            "testcontainers not installed (uv sync --group dev). "
            f"Alternatively point tests at an existing isolated stack: {_fast_path_hint}"
        )

    # Windows: testcontainers may resolve host as named pipe
    os.environ.setdefault("TC_HOST", "localhost")

    # Readiness probes mirror the compose healthchecks — Memgraph accepts TCP
    # before Bolt is queryable, so port/log waits are not enough.
    mg = (
        # MAGE-enabled image (matches docker-compose.yml) — required for
        # leiden_community_detection.get() (analyze_repo(analysis="communities")).
        DockerContainer("memgraph/memgraph-mage:3.12.0")
        .with_exposed_ports(7687)
        .with_command("--log-level=WARNING --memory-limit=2048 --storage-wal-enabled=false")
        .waiting_for(ExecWaitStrategy(["bash", "-c", "echo 'RETURN 1;' | mgconsole"]).with_startup_timeout(60))
    )
    vk = (
        DockerContainer("valkey/valkey:8-alpine")
        .with_exposed_ports(6379)
        .with_command('valkey-server --appendonly no --save "" --maxmemory 64mb --maxmemory-policy noeviction')
        .waiting_for(ExecWaitStrategy(["valkey-cli", "ping"]).with_startup_timeout(30))
    )

    try:
        mg.start()
        vk.start()
    except Exception:
        with contextlib.suppress(Exception):
            mg.stop()
        pytest.skip(
            "Docker unavailable — integration tests start testcontainers by default. "
            f"Start Docker, or use an existing isolated stack: {_fast_path_hint}"
        )

    mg_host = mg.get_container_host_ip()
    mg_port = int(mg.get_exposed_port(7687))
    vk_host = vk.get_container_host_ip()
    vk_port = int(vk.get_exposed_port(6379))

    yield _export_atlas_env(
        InfraEndpoints(memgraph_host=mg_host, memgraph_port=mg_port, valkey_host=vk_host, valkey_port=vk_port)
    )

    mg.stop()
    vk.stop()


# ---------------------------------------------------------------------------
# Postgres (queue backend) — separate and lazy
# ---------------------------------------------------------------------------
#
# Not part of `_infra_endpoints`: that fixture backs every integration test, and folding
# Postgres into it would start a third container for all of them. Only a test that asks
# for `pg_settings` pays for this one.

_PG_TEST_DATABASE = "atlas_test"


@pytest.fixture(scope="session")
def _postgres_endpoint() -> Iterator[PostgresSettings]:
    """Provision an ISOLATED Postgres for the queue-backend tests.

    1. ``ATLAS_TEST_POSTGRES_PORT`` points at an existing isolated instance (the compose
       fast path: ``docker compose --profile test up -d postgres-test`` → :5434).
    2. Default — a session-scoped ``postgres:18-alpine`` testcontainer on a random port.
    3. Skip with instructions if Docker is unavailable.
    """
    port_env = os.environ.get("ATLAS_TEST_POSTGRES_PORT")
    if port_env:
        yield PostgresSettings(
            host="localhost", port=int(port_env), user="atlas", password=SecretStr("atlas"), database=_PG_TEST_DATABASE
        )
        return

    hint = "docker compose --profile test up -d postgres-test && ATLAS_TEST_POSTGRES_PORT=5434 uv run pytest ..."
    try:
        from testcontainers.core.container import DockerContainer
        from testcontainers.core.wait_strategies import ExecWaitStrategy
    except ImportError:
        pytest.skip(f"testcontainers not installed (uv sync --group dev). Alternatively: {hint}")

    os.environ.setdefault("TC_HOST", "localhost")
    pg = (
        DockerContainer("postgres:18-alpine")
        .with_env("POSTGRES_USER", "atlas")
        .with_env("POSTGRES_PASSWORD", "atlas")
        .with_env("POSTGRES_DB", _PG_TEST_DATABASE)
        .with_exposed_ports(5432)
        # -h 127.0.0.1, not the socket: the image's init phase runs a socket-only server
        # that answers pg_isready and then restarts, so a socket probe reports ready early.
        .waiting_for(
            ExecWaitStrategy(
                ["pg_isready", "-U", "atlas", "-d", _PG_TEST_DATABASE, "-h", "127.0.0.1"]
            ).with_startup_timeout(60)
        )
    )
    try:
        pg.start()
    except Exception:
        with contextlib.suppress(Exception):
            pg.stop()
        pytest.skip(f"Docker unavailable — Postgres tests start a testcontainer by default. Or: {hint}")

    yield PostgresSettings(
        host=pg.get_container_host_ip(),
        port=int(pg.get_exposed_port(5432)),
        user="atlas",
        password=SecretStr("atlas"),
        database=_PG_TEST_DATABASE,
    )
    pg.stop()


async def _empty_postgres_queue(pg: PostgresSettings) -> None:
    import asyncpg

    from code_atlas.backends.postgres_queue import SCHEMA, connect_kwargs, ensure_queue_schema

    conn = await asyncpg.connect(**connect_kwargs(pg))
    try:
        await ensure_queue_schema(conn)
        await conn.execute(
            f"TRUNCATE {SCHEMA}.messages, {SCHEMA}.groups, {SCHEMA}.deliveries, {SCHEMA}.consumers, "
            f"{SCHEMA}.leases, {SCHEMA}.rate_bucket, {SCHEMA}.rate_scale"
        )
    finally:
        await conn.close()


@pytest.fixture
def pg_settings(_postgres_endpoint: PostgresSettings) -> PostgresSettings:
    """Postgres settings with an empty ``atlas_queue`` schema.

    The wipe guard for this backend is the database name: tables are only truncated in a
    database whose name starts with ``atlas_test``. A production queue database named
    otherwise aborts the session instead of being emptied.

    Synchronous on purpose, with the cleanup on its own loop in a worker thread: a
    parametrized fixture reaches this one through ``request.getfixturevalue``, and an async
    fixture cannot be set up from inside another one's running loop.
    """
    if not _postgres_endpoint.database.startswith(_PG_TEST_DATABASE):
        pytest.exit(
            f"REFUSING to truncate atlas_queue in Postgres database {_postgres_endpoint.database!r} at "
            f"{_postgres_endpoint.host}:{_postgres_endpoint.port} — only databases named "
            f"{_PG_TEST_DATABASE}* are treated as disposable.",
            returncode=1,
        )
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            pool.submit(asyncio.run, _empty_postgres_queue(_postgres_endpoint)).result()
        except OSError as exc:
            pytest.skip(f"Postgres not available: {exc}")
    return _postgres_endpoint


# ---------------------------------------------------------------------------
# Production-data guard
# ---------------------------------------------------------------------------

_GUARD_OK: set[tuple[str, int]] = set()
"""(host, port) pairs already verified safe to wipe this session."""


async def _assert_disposable_db(client: GraphClient, host: str, port: int) -> None:
    """Refuse to run destructive fixtures against a DB that looks like production.

    All integration/bench test data derives project names from pytest tmp
    directories (``test_...`` / ``bench_...``); anything else means this is a
    real index. Aborts the whole session via pytest.exit — never wipes.
    Override with ATLAS_TEST_DB=1 only for known-disposable instances (CI).

    ``GLOBAL_PROJECT`` is allowlisted alongside those prefixes: it is the
    sentinel ``project_name`` on globally-shared nodes (EnvVar), so a test that
    indexes a single ``os.getenv`` call would otherwise abort the entire
    session with a message that reads exactly like a production-safety
    incident. This does not weaken the guard — a real index still trips on its
    own project names, which is what identifies it as real.
    """
    if os.environ.get("ATLAS_TEST_DB") == "1" or (host, port) in _GUARD_OK:
        return
    rows = await client.execute(
        "MATCH (n) "
        "WHERE (n:Project AND NOT (n.name STARTS WITH 'test' OR n.name STARTS WITH 'bench')) "
        "   OR (n.project_name IS NOT NULL AND n.project_name <> $global_project "
        "       AND NOT (n.project_name STARTS WITH 'test' OR n.project_name STARTS WITH 'bench')) "
        "RETURN DISTINCT coalesce(n.project_name, n.name) AS name LIMIT 5",
        {"global_project": GLOBAL_PROJECT},
    )
    if rows:
        names = sorted({r["name"] for r in rows})
        pytest.exit(
            f"REFUSING to wipe Memgraph at {host}:{port} — it contains non-test data (projects: {names}). "
            "This looks like a production index; integration fixtures would DESTROY it. "
            "Unset ATLAS_TEST_MEMGRAPH_PORT/ATLAS_TEST_VALKEY_PORT to use disposable testcontainers, "
            "or point them at the isolated compose stack (docker compose --profile test up -d → "
            "memgraph-test :7688, valkey-test :6380). If this instance really is disposable "
            "(e.g. residue on the test instance from an aborted run — clear it with "
            "`docker compose restart memgraph-test`), set ATLAS_TEST_DB=1 to override.",
            returncode=1,
        )
    _GUARD_OK.add((host, port))


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path, _infra_endpoints: InfraEndpoints):
    """Create test settings with dynamic infrastructure ports.

    Passes ``embeddings=EmbeddingSettings()`` explicitly to prevent the
    user's ``.env`` (e.g. ``ATLAS_EMBEDDINGS__DIMENSION=1536``) from
    leaking into tests via pydantic-settings env var resolution.

    ``tests/unit/conftest.py`` shadows this with an infra-free variant —
    nearest conftest wins, so unit tests never touch a container.
    """
    return AtlasSettings(
        project_root=tmp_path,
        backend=BackendSettings(
            graph={
                "memgraph": MemgraphSettings(
                    host=_infra_endpoints.memgraph_host,
                    port=_infra_endpoints.memgraph_port,
                )
            },
            queue={
                "valkey": RedisSettings(
                    host=_infra_endpoints.valkey_host,
                    port=_infra_endpoints.valkey_port,
                    stream_prefix=f"test-{uuid.uuid4().hex[:8]}",
                )
            },
        ),
        embeddings=EmbeddingSettings(),
    )


@pytest.fixture
async def graph_client(settings) -> AsyncIterator[GraphClient]:
    """Async GraphClient fixture.

    Wipes all data and search indices before each test for isolation.
    Vector/text indices are dropped so ensure_schema can recreate them
    at the dimension specified by the test settings.
    """
    from code_atlas.graph.client import GraphClient

    # Scoped to a block: an abandoned AsyncBoltDriver makes neo4j warn whenever the GC
    # eventually notices, blaming whichever unrelated test is running by then -- which
    # is how five integration tests went red once warnings became errors. The skip and
    # the disposable-db guard below can both raise, and the block closes either way.
    async with GraphClient(settings) as client:
        try:
            await client.ping()
        except Exception:
            pytest.skip("Memgraph not available")

        await _assert_disposable_db(client, settings.memgraph.host, settings.memgraph.port)
        # Clean slate: wipe nodes and drop search indices (dimension may differ)
        await client.execute_write("MATCH (n) DETACH DELETE n")
        for stmt in generate_drop_vector_index_ddl():
            with contextlib.suppress(Exception):
                await client.execute_write(stmt)
        for stmt in generate_drop_text_index_ddl():
            with contextlib.suppress(Exception):
                await client.execute_write(stmt)

        yield client

        # Clean up after test
        await client.execute_write("MATCH (n) DETACH DELETE n")


@pytest.fixture
async def event_bus(settings) -> AsyncIterator:
    """Centralised EventBus fixture — replaces per-module duplicates.

    Flushes all pipeline streams before each test to prevent cross-test
    contamination (same pattern as graph_client wiping nodes).
    """
    from code_atlas.events import EventBus, Topic, redis_client

    # Scoped to a block -- see the graph fixture; an abandoned redis Connection produces
    # the same misdirected ResourceWarning. The client is the thing with a lifecycle; the
    # bus only borrows it, as in production.
    async with redis_client(settings.redis) as redis:
        bus = EventBus(redis, settings.redis)
        try:
            await bus.ping()
        except Exception:
            pytest.skip("Valkey not available")

        # Clean slate: flush all pipeline streams
        for topic in Topic:
            key = f"{bus._prefix}:{topic.value}"
            await bus._redis.delete(key)

        yield bus
