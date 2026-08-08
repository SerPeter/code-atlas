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

from code_atlas.indexing.orchestrator import index_project
from code_atlas.settings import AtlasSettings
from tests.conftest import NO_EMBED

if TYPE_CHECKING:
    from pathlib import Path

    from code_atlas.events import EventBus
    from code_atlas.graph.client import GraphClient

pytestmark = [pytest.mark.bench, pytest.mark.integration]


async def test_full_index_throughput(
    graph_client: GraphClient, event_bus: EventBus, bench_small: tuple[Path, list[str]]
):
    """Measure full indexing throughput (files/sec) with mock embeddings."""
    root, _rel_paths = bench_small
    settings = AtlasSettings(project_root=root, embeddings=NO_EMBED)

    # Mock embedding client to return random vectors instantly
    dim = graph_client._dimension
    mock_embed = AsyncMock()
    mock_embed.embed_batch = AsyncMock(return_value=[[0.1] * dim])
    mock_embed.embed_one = AsyncMock(return_value=[0.1] * dim)

    with (
        patch("code_atlas.indexing.orchestrator.EmbedClient", return_value=mock_embed),
        patch("code_atlas.indexing.orchestrator.EmbedCache", return_value=None),
    ):
        start = time.perf_counter()
        result = await index_project(settings, graph_client, event_bus, full_reindex=True, drain_timeout_s=120.0)
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
    graph_client: GraphClient, event_bus: EventBus, bench_small: tuple[Path, list[str]]
):
    """Measure delta indexing throughput after modifying 10% of files."""
    root, rel_paths = bench_small
    settings = AtlasSettings(project_root=root, embeddings=NO_EMBED)

    dim = graph_client._dimension
    mock_embed = AsyncMock()
    mock_embed.embed_batch = AsyncMock(return_value=[[0.1] * dim])
    mock_embed.embed_one = AsyncMock(return_value=[0.1] * dim)

    # First do a full index
    with (
        patch("code_atlas.indexing.orchestrator.EmbedClient", return_value=mock_embed),
        patch("code_atlas.indexing.orchestrator.EmbedCache", return_value=None),
    ):
        await index_project(settings, graph_client, event_bus, full_reindex=True, drain_timeout_s=120.0)

    # Modify 10% of files
    py_paths = [p for p in rel_paths if p.endswith(".py") and "__init__" not in p]
    n_modify = max(1, len(py_paths) // 10)
    for rel_path in py_paths[:n_modify]:
        abs_path = root / rel_path
        content = abs_path.read_text(encoding="utf-8")
        abs_path.write_text(content + "\n# modified\n", encoding="utf-8")

    # Delta index
    with (
        patch("code_atlas.indexing.orchestrator.EmbedClient", return_value=mock_embed),
        patch("code_atlas.indexing.orchestrator.EmbedCache", return_value=None),
    ):
        start = time.perf_counter()
        result = await index_project(settings, graph_client, event_bus, drain_timeout_s=120.0)
        elapsed = time.perf_counter() - start

    report = {
        "benchmark": "delta_index",
        "mode": result.mode,
        "files_published": result.files_published,
        "entities_total": result.entities_total,
        "elapsed_s": round(elapsed, 3),
    }
    print(f"\n{json.dumps(report, indent=2)}")


async def test_indexing_a_medium_project_stays_inside_its_budget(
    graph_client: GraphClient, event_bus: EventBus, bench_medium: tuple[Path, list[str]]
):
    """The only bench with an assertion, and the only one above ~100 files.

    Every other benchmark prints JSON and returns, and every graph-touching one uses
    `bench_small` (100 files) — `bench_medium` and `bench_large` existed but were
    referenced by nothing. So no test built a graph large enough for an unindexed
    lookup to differ from an indexed one, which is why a CALLS write whose cost grew
    with graph size reached a user's first real C++ project before anything noticed
    (ATL-114).

    The budget is deliberately loose. This is a tripwire for a change in COMPLEXITY —
    a query that goes quadratic, an unlabelled uid match that scans every node per row
    — not a throughput regression detector. CI hardware varies; algorithmic blowup does
    not care.
    """
    root, _rel_paths = bench_medium
    settings = AtlasSettings(project_root=root, embeddings=NO_EMBED)

    dim = graph_client._dimension
    mock_embed = AsyncMock()
    mock_embed.embed_batch = AsyncMock(return_value=[[0.1] * dim])
    mock_embed.embed_one = AsyncMock(return_value=[0.1] * dim)

    budget_s = 900.0
    with (
        patch("code_atlas.indexing.orchestrator.EmbedClient", return_value=mock_embed),
        patch("code_atlas.indexing.orchestrator.EmbedCache", return_value=None),
    ):
        start = time.perf_counter()
        result = await index_project(settings, graph_client, event_bus, full_reindex=True, drain_timeout_s=budget_s)
        elapsed = time.perf_counter() - start

    report = {
        "benchmark": "medium_index_budget",
        "files": result.files_scanned,
        "entities": result.entities_total,
        "elapsed_s": round(elapsed, 1),
        "budget_s": budget_s,
    }
    print(f"\n{json.dumps(report, indent=2)}")

    # Ordered most-specific first: a wrong number is more informative than a timeout.
    assert result.entities_total > 1000, "medium corpus should produce thousands of entities"
    assert result.drained, "pipeline did not drain — a write likely exceeded its timeout"
    assert elapsed < budget_s, (
        f"indexing {result.files_scanned} files took {elapsed:.0f}s, over the {budget_s:.0f}s budget"
    )
