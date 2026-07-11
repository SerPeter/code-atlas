"""Integration test fixtures — provision ISOLATED test Memgraph + Valkey.

Default: session-scoped testcontainers on random ports (Docker required).
Fast path: ``docker compose --profile test up -d`` then export
``ATLAS_TEST_MEMGRAPH_PORT=7688 ATLAS_TEST_VALKEY_PORT=6380``. Never
connects to the production instances on 7687/6379.
"""

from __future__ import annotations

import contextlib
import os
import socket
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from code_atlas.graph.client import GraphClient
from code_atlas.schema import generate_drop_text_index_ddl, generate_drop_vector_index_ddl
from code_atlas.settings import AtlasSettings, EmbeddingSettings, MemgraphSettings, RedisSettings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

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


def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a TCP port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _export_atlas_env(ep: InfraEndpoints) -> InfraEndpoints:
    """Point any AtlasSettings constructed inside tests at the test instances.

    Defense-in-depth: fixtures pass explicit Memgraph/Redis settings, but the
    repo-root atlas.toml hardcodes the production ports (7687/6379), so a bare
    ``AtlasSettings(project_root=tmp_path)`` would otherwise resolve to the
    production infrastructure (init > env > toml). Env beats toml.
    """
    os.environ["ATLAS_MEMGRAPH__HOST"] = ep.memgraph_host
    os.environ["ATLAS_MEMGRAPH__PORT"] = str(ep.memgraph_port)
    os.environ["ATLAS_REDIS__HOST"] = ep.valkey_host
    os.environ["ATLAS_REDIS__PORT"] = str(ep.valkey_port)
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
        DockerContainer("memgraph/memgraph:3.7.2")
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
    """
    if os.environ.get("ATLAS_TEST_DB") == "1" or (host, port) in _GUARD_OK:
        return
    rows = await client.execute(
        "MATCH (n) "
        "WHERE (n:Project AND NOT (n.name STARTS WITH 'test' OR n.name STARTS WITH 'bench')) "
        "   OR (n.project_name IS NOT NULL "
        "       AND NOT (n.project_name STARTS WITH 'test' OR n.project_name STARTS WITH 'bench')) "
        "RETURN DISTINCT coalesce(n.project_name, n.name) AS name LIMIT 5"
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
    """
    return AtlasSettings(
        project_root=tmp_path,
        memgraph=MemgraphSettings(
            host=_infra_endpoints.memgraph_host,
            port=_infra_endpoints.memgraph_port,
        ),
        redis=RedisSettings(
            host=_infra_endpoints.valkey_host,
            port=_infra_endpoints.valkey_port,
            stream_prefix=f"test-{uuid.uuid4().hex[:8]}",
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
    client = GraphClient(settings)
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
    await client.close()


@pytest.fixture
async def event_bus(settings) -> AsyncIterator:
    """Centralised EventBus fixture — replaces per-module duplicates.

    Flushes all pipeline streams before each test to prevent cross-test
    contamination (same pattern as graph_client wiping nodes).
    """
    from code_atlas.events import EventBus, Topic

    bus = EventBus(settings.redis)
    try:
        await bus.ping()
    except Exception:
        pytest.skip("Valkey not available")

    # Clean slate: flush all pipeline streams
    for topic in Topic:
        key = f"{bus._prefix}:{topic.value}"
        await bus._redis.delete(key)

    yield bus
    await bus.close()


# ---------------------------------------------------------------------------
# TEI (embedding service) fixtures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TEIEndpoint:
    """Resolved TEI host/port."""

    host: str
    port: int


@pytest.fixture(scope="session")
def _tei_endpoint() -> Iterator[TEIEndpoint]:
    """Discover or start a TEI container.

    Tries default port 8080 first, falls back to testcontainers with
    ``TaylorAI/gte-tiny`` (384-dim, ~45 MB download).
    """
    default_host, default_port = "localhost", 8080

    if _is_port_open(default_host, default_port):
        yield TEIEndpoint(host=default_host, port=default_port)
        return

    try:
        from testcontainers.core.container import DockerContainer
        from testcontainers.core.wait_strategies import LogMessageWaitStrategy
    except ImportError:
        pytest.skip("TEI not running and testcontainers not installed")

    os.environ.setdefault("TC_HOST", "localhost")

    tei = (
        DockerContainer("ghcr.io/huggingface/text-embeddings-inference:cpu-1.8")
        .with_exposed_ports(80)
        .with_command("--model-id TaylorAI/gte-tiny --port 80")
        .waiting_for(LogMessageWaitStrategy("Ready").with_startup_timeout(120))
    )

    try:
        tei.start()
    except Exception:
        pytest.skip("Docker not available for TEI")

    host = tei.get_container_host_ip()
    port = int(tei.get_exposed_port(80))

    yield TEIEndpoint(host=host, port=port)

    tei.stop()


@pytest.fixture
async def tei_settings(tmp_path, _infra_endpoints: InfraEndpoints, _tei_endpoint: TEIEndpoint):
    """Settings configured to use a real TEI embedding service.

    Auto-detects vector dimension from the running TEI instance so that
    GraphClient creates vector indices at the correct size.
    """
    from code_atlas.search.embeddings import EmbedClient

    tei_url = f"http://{_tei_endpoint.host}:{_tei_endpoint.port}"
    probe_settings = EmbeddingSettings(enabled=True, base_url=tei_url)
    dimension = await EmbedClient(probe_settings).detect_dimension()

    return AtlasSettings(
        project_root=tmp_path,
        memgraph=MemgraphSettings(
            host=_infra_endpoints.memgraph_host,
            port=_infra_endpoints.memgraph_port,
        ),
        redis=RedisSettings(
            host=_infra_endpoints.valkey_host,
            port=_infra_endpoints.valkey_port,
            stream_prefix=f"test-{uuid.uuid4().hex[:8]}",
        ),
        embeddings=EmbeddingSettings(
            provider="tei",
            model="TaylorAI/gte-tiny",
            base_url=tei_url,
            dimension=dimension,
        ),
    )


@pytest.fixture
async def tei_graph_client(tei_settings) -> AsyncIterator[GraphClient]:
    """GraphClient wired to TEI-configured settings (384-dim vectors)."""
    client = GraphClient(tei_settings)
    try:
        await client.ping()
    except Exception:
        pytest.skip("Memgraph not available")

    await _assert_disposable_db(client, tei_settings.memgraph.host, tei_settings.memgraph.port)
    await client.execute_write("MATCH (n) DETACH DELETE n")
    for stmt in generate_drop_vector_index_ddl():
        with contextlib.suppress(Exception):
            await client.execute_write(stmt)
    for stmt in generate_drop_text_index_ddl():
        with contextlib.suppress(Exception):
            await client.execute_write(stmt)

    yield client

    await client.execute_write("MATCH (n) DETACH DELETE n")
    await client.close()


@pytest.fixture
async def tei_event_bus(tei_settings) -> AsyncIterator:
    """EventBus wired to TEI-configured settings."""
    from code_atlas.events import EventBus

    bus = EventBus(tei_settings.redis)
    try:
        await bus.ping()
    except Exception:
        pytest.skip("Valkey not available")
    yield bus
    await bus.close()
