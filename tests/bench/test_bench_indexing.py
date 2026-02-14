"""Indexing benchmark.

Measures full and delta index throughput using mock embeddings.
Requires Memgraph + Valkey.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from code_atlas.events import EventBus
from code_atlas.indexer import index_project
from code_atlas.settings import AtlasSettings

if TYPE_CHECKING:
    from pathlib import Path

    from code_atlas.graph import GraphClient

pytestmark = [pytest.mark.bench, pytest.mark.integration]


@pytest.fixture
async def bench_bus():
    """Create and connect an EventBus for benchmarking."""
    settings = AtlasSettings()
    bus = EventBus(settings.redis)
    try:
        await bus.ping()
    except Exception:
        pytest.skip("Valkey not available")
    yield bus
    await bus.close()


async def test_full_index_throughput(
    graph_client: GraphClient, bench_bus: EventBus, bench_small: tuple[Path, list[str]]
):
    """Measure full indexing throughput (files/sec) with mock embeddings."""
    root, _rel_paths = bench_small
    settings = AtlasSettings(project_root=root)

    # Mock embedding client to return random vectors instantly
    dim = graph_client._dimension
    mock_embed = AsyncMock()
    mock_embed.embed_batch = AsyncMock(return_value=[[0.1] * dim])
    mock_embed.embed_one = AsyncMock(return_value=[0.1] * dim)

    with (
        patch("code_atlas.indexer.EmbedClient", return_value=mock_embed),
        patch("code_atlas.indexer.EmbedCache", return_value=None),
    ):
        start = time.perf_counter()
        result = await index_project(settings, graph_client, bench_bus, full_reindex=True, drain_timeout_s=120.0)
        elapsed = time.perf_counter() - start

    fps = result.files_scanned / elapsed if elapsed > 0 else 0
    report = {
        "benchmark": "full_index",
        "files_scanned": result.files_scanned,
        "files_published": result.files_published,
        "entities_total": result.entities_total,
        "elapsed_s": round(elapsed, 3),
        "files_per_sec": round(fps, 1),
    }
    print(f"\n{json.dumps(report, indent=2)}")


async def test_delta_index_throughput(
    graph_client: GraphClient, bench_bus: EventBus, bench_small: tuple[Path, list[str]]
):
    """Measure delta indexing throughput after modifying 10% of files."""
    root, rel_paths = bench_small
    settings = AtlasSettings(project_root=root)

    dim = graph_client._dimension
    mock_embed = AsyncMock()
    mock_embed.embed_batch = AsyncMock(return_value=[[0.1] * dim])
    mock_embed.embed_one = AsyncMock(return_value=[0.1] * dim)

    # First do a full index
    with (
        patch("code_atlas.indexer.EmbedClient", return_value=mock_embed),
        patch("code_atlas.indexer.EmbedCache", return_value=None),
    ):
        await index_project(settings, graph_client, bench_bus, full_reindex=True, drain_timeout_s=120.0)

    # Modify 10% of files
    py_paths = [p for p in rel_paths if p.endswith(".py") and "__init__" not in p]
    n_modify = max(1, len(py_paths) // 10)
    for rel_path in py_paths[:n_modify]:
        abs_path = root / rel_path.replace("/", "\\")
        content = abs_path.read_text(encoding="utf-8")
        abs_path.write_text(content + "\n# modified\n", encoding="utf-8")

    # Delta index
    with (
        patch("code_atlas.indexer.EmbedClient", return_value=mock_embed),
        patch("code_atlas.indexer.EmbedCache", return_value=None),
    ):
        start = time.perf_counter()
        result = await index_project(settings, graph_client, bench_bus, drain_timeout_s=120.0)
        elapsed = time.perf_counter() - start

    report = {
        "benchmark": "delta_index",
        "mode": result.mode,
        "files_published": result.files_published,
        "entities_total": result.entities_total,
        "elapsed_s": round(elapsed, 3),
    }
    print(f"\n{json.dumps(report, indent=2)}")
