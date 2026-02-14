"""Shared test fixtures for Code Atlas.

Session-scoped container management: tries default ports first (for
``docker compose up`` users), falls back to testcontainers if ports are
closed, skips if no Docker.
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
# Constants re-exported to test modules
# ---------------------------------------------------------------------------

TEST_DRAIN_TIMEOUT_S: float = 60.0
"""Shortened drain timeout for integration tests (default 600s is too long)."""

NO_EMBED = EmbeddingSettings(enabled=False)
"""Embedding settings that disable Tier3 entirely — use for pipeline tests
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


def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a TCP port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def _infra_endpoints() -> Iterator[InfraEndpoints]:
    """Discover or start Memgraph + Valkey.

    1. Try default ports (``docker compose up`` workflow) — zero overhead.
    2. Fall back to testcontainers with random ports.
    3. Skip if no Docker and services aren't running.
    """
    default_mg_host, default_mg_port = "localhost", 7687
    default_vk_host, default_vk_port = "localhost", 6379

    if _is_port_open(default_mg_host, default_mg_port) and _is_port_open(default_vk_host, default_vk_port):
        yield InfraEndpoints(
            memgraph_host=default_mg_host,
            memgraph_port=default_mg_port,
            valkey_host=default_vk_host,
            valkey_port=default_vk_port,
        )
        return

    # --- testcontainers fallback ---
    try:
        from testcontainers.core.container import DockerContainer
        from testcontainers.core.wait_strategies import LogMessageWaitStrategy
    except ImportError:
        pytest.skip("Services not running and testcontainers not installed")

    # Windows: testcontainers may resolve host as named pipe
    os.environ.setdefault("TC_HOST", "localhost")

    mg = (
        DockerContainer("memgraph/memgraph:3.7.2")
        .with_exposed_ports(7687)
        .with_command("--log-level=WARNING --memory-limit=2048 --storage-wal-enabled=false")
        .waiting_for(LogMessageWaitStrategy("is ready").with_startup_timeout(60))
    )
    vk = (
        DockerContainer("valkey/valkey:8-alpine")
        .with_exposed_ports(6379)
        .with_command('valkey-server --appendonly no --save "" --maxmemory 64mb --maxmemory-policy noeviction')
    )

    try:
        mg.start()
        vk.start()
    except Exception:
        pytest.skip("Docker not available and services not running")

    mg_host = mg.get_container_host_ip()
    mg_port = int(mg.get_exposed_port(7687))
    vk_host = vk.get_container_host_ip()
    vk_port = int(vk.get_exposed_port(6379))

    yield InfraEndpoints(
        memgraph_host=mg_host,
        memgraph_port=mg_port,
        valkey_host=vk_host,
        valkey_port=vk_port,
    )

    mg.stop()
    vk.stop()


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
