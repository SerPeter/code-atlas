"""Integration-only fixtures — the TEI (embedding service) tier.

The shared infrastructure fixtures (``_infra_endpoints``, ``settings``,
``graph_client``, ``event_bus``) and the production-data guard live in
``tests/conftest.py`` so that ``tests/bench/`` — whose tests also carry the
``integration`` marker — can see them. Only the TEI fixtures, which nothing
outside this directory requests, stay here.

``_GUARD_OK`` is re-exported: ``test_infra_isolation.py`` and
``graph/test_client.py`` import both guard symbols from this module.
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
from tests.conftest import _GUARD_OK, _assert_disposable_db  # noqa: F401 — _GUARD_OK re-exported, see docstring

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from tests.conftest import InfraEndpoints


@pytest.fixture(autouse=True)
def _fast_pipeline_timings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the pipeline's production pacing to test scale.

    These tests assert on what ends up in the graph, never on how long a drain settles
    or how wide a batch window is, so shrinking the pacing cannot make any of them
    vacuous. Autouse because every test that drives a real pipeline pays all of it.

    Deliberately in *this* conftest and not ``tests/conftest.py``: ``tests/bench/``
    inherits the root conftest, and bench exists to measure throughput in wall clock.
    An autouse timing patch visible from there would silently move the numbers the
    benchmarks are for. ``tests/bench/`` has its own conftest and does not see this file.

    ``_AST_WINDOW_S`` is 0.1 and not 0: ``is_reindex`` in consumers.py tests
    ``time_window_s == 0``, so zero would switch the resolution cadence rather than just
    speed the batch window up.
    """
    monkeypatch.setattr("code_atlas.indexing.orchestrator._DRAIN_SETTLE_S", 0.1)
    monkeypatch.setattr("code_atlas.indexing.orchestrator._DRAIN_POLL_S", 0.05)
    monkeypatch.setattr("code_atlas.indexing.orchestrator._DRAIN_POLL_MAX_S", 0.1)
    monkeypatch.setattr("code_atlas.indexing.consumers._AST_WINDOW_S", 0.1)
    monkeypatch.setattr("code_atlas.indexing.daemon._RESTART_BACKOFF_S", 0.01)


def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a TCP port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


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
