"""Indexing benchmark.

Measures full and delta index throughput using mock embeddings.
Requires Memgraph + Valkey.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from code_atlas.indexing.consumers import ASTConsumer
from code_atlas.indexing.orchestrator import index_project
from code_atlas.settings import AtlasSettings
from tests.conftest import NO_EMBED

if TYPE_CHECKING:
    from collections.abc import Iterator
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


# ---------------------------------------------------------------------------
# No-op re-check cost (ATL-151 / ADR-0042 decision 4)
# ---------------------------------------------------------------------------


@dataclass
class _RecheckCost:
    """What one index run asked of the two seams a no-op ``--full`` used to hammer.

    Read off the arguments the calls carry, never off their Cypher. A classifier that
    string-matches a query goes quietly to zero the day someone reformats it, and a
    benchmark reporting zero for the wrong reason is worse than no benchmark at all.
    """

    rel_batches: int = 0
    """TX2 relationship transactions that still had a file left to write."""
    files_rewritten: int = 0
    """Files reaching TX2's relationship phase. Never zero even when the skip is
    working perfectly: a file with no prior data is not a skip candidate at all, and
    the corpus's empty ``__init__.py`` files re-enter as new on every single run. They
    carry no relationships, so they cost nothing — which is why the two numbers below,
    not this one, are the ones asserted on."""
    files_swept: int = 0
    """Files whose existing edges were DELETEd first — the delete volume. New files
    are exempt from the sweep, so this counts only real edge destruction."""
    rel_rows: int = 0
    """Relationship rows pooled into the create phase — the write volume itself."""
    flushes: int = 0
    flush_s: float = 0.0
    """Wall time inside ``_flush_deferred_resolution``, summed over the run."""
    buffered_rels: int = 0
    """Rels handed to deferred resolution. ADR-0026's replay depends on this being
    untouched by the write skip, so it is measured rather than asserted in prose."""

    def report(self, arm: str) -> dict[str, object]:
        return {
            "arm": arm,
            "rel_batches": self.rel_batches,
            "files_rewritten": self.files_rewritten,
            "files_swept": self.files_swept,
            "rel_rows": self.rel_rows,
            "flushes": self.flushes,
            "flush_s": round(self.flush_s, 3),
            "buffered_rels": self.buffered_rels,
        }


@contextlib.contextmanager
def _measure_recheck(
    monkeypatch: pytest.MonkeyPatch, graph: GraphClient, *, stored_fingerprints: bool
) -> Iterator[_RecheckCost]:
    """Count relationship writes and time the resolution flush for one index run.

    ``stored_fingerprints=False`` is the "before" arm: the stored ``rels_hash`` is
    hidden, so nothing ever compares equal and TX2 rewrites every file in the batch —
    which is what v0.11.0 did permanently, having no fingerprint to compare against at
    all. Hiding the *read* rather than the fingerprint itself keeps the write side
    honest, so the arm that runs after this one still finds a correct stored value.
    """
    cost = _RecheckCost()
    real_recreate = graph._recreate_batch_relationships
    real_flush = ASTConsumer._flush_deferred_resolution

    async def _counting_recreate(project_name, file_rels, new_file_paths):
        cost.rel_batches += 1
        cost.files_rewritten += len(file_rels)
        cost.files_swept += sum(1 for fp in file_rels if fp not in new_file_paths)
        cost.rel_rows += sum(len(rels) for rels in file_rels.values())
        return await real_recreate(project_name, file_rels, new_file_paths)

    async def _timed_flush(self, *, final: bool = False):
        cost.flushes += 1
        cost.buffered_rels += self._pending_rel_count()
        start = time.perf_counter()
        try:
            return await real_flush(self, final=final)
        finally:
            cost.flush_s += time.perf_counter() - start

    async def _no_stored(project_name, file_paths):
        return {}

    with monkeypatch.context() as m:
        m.setattr(graph, "_recreate_batch_relationships", _counting_recreate)
        m.setattr(ASTConsumer, "_flush_deferred_resolution", _timed_flush)
        if not stored_fingerprints:
            m.setattr(graph, "get_batch_rels_hashes", _no_stored)
        yield cost


@pytest.mark.timeout(600)
async def test_a_noop_full_recheck_barely_writes_relationships(
    graph_client: GraphClient,
    event_bus: EventBus,
    bench_small: tuple[Path, list[str]],
    monkeypatch: pytest.MonkeyPatch,
):
    """ATL-151's verification, as a repeatable number rather than a one-off report.

    Three runs over identical bytes: one to build the graph and warm the relationship
    fingerprints, then the same no-op ``--full`` twice — once with the stored
    fingerprints hidden (v0.11.0's behaviour) and once for real. Both arms are printed,
    so the win is a ratio this corpus computes for itself rather than a constant copied
    off one machine.

    The assertions are tripwires, not budgets. Wall clock is reported and never
    asserted: this shares one Memgraph with the rest of the suite, and a timing gate
    that fires on co-tenancy teaches people to re-run red tests (see
    ``test_vector_search_latency``). What *is* asserted is countable and deterministic:

    - the before arm actually wrote something, or the after arm's number means nothing;
    - the after arm writes an order of magnitude less;
    - **both arms buffer the identical rel set for deferred resolution.** ATL-151 as
      written also skipped the buffer for an unchanged file; that is refused
      deliberately, because ADR-0026 added the buffer to fix a measured loss (file A
      unchanged, module B new, A->B never resolves — CALLS 9,058 -> 9,713 when it
      landed). This equality is what makes that refusal a fact instead of a comment.

    Expect ``flush_s`` to come out roughly EQUAL in the two arms — the resolution work
    is the same work either way, which is the whole point of the third assertion. If it
    ever collapses in the after arm, someone has taken the buffer skip: with it applied
    this bench measured 2.85s -> 0.17s, and that is the trade being declined, not a
    regression in the flush.
    """
    root, _rel_paths = bench_small
    settings = AtlasSettings(project_root=root, embeddings=NO_EMBED)

    await index_project(settings, graph_client, event_bus, full_reindex=True, drain_timeout_s=120.0)

    with _measure_recheck(monkeypatch, graph_client, stored_fingerprints=False) as before:
        start = time.perf_counter()
        await index_project(settings, graph_client, event_bus, full_reindex=True, drain_timeout_s=120.0)
        before_s = time.perf_counter() - start

    with _measure_recheck(monkeypatch, graph_client, stored_fingerprints=True) as after:
        start = time.perf_counter()
        result = await index_project(settings, graph_client, event_bus, full_reindex=True, drain_timeout_s=120.0)
        after_s = time.perf_counter() - start

    report = {
        "benchmark": "noop_full_recheck_rel_writes",
        "files_scanned": result.files_scanned,
        "before": {**before.report("no stored fingerprint"), "elapsed_s": round(before_s, 3)},
        "after": {**after.report("fingerprint honoured"), "elapsed_s": round(after_s, 3)},
    }
    print(f"\n{json.dumps(report, indent=2)}")

    assert result.mode == "full"
    # Non-vacuity first: every ratio below is meaningless if the before arm was already
    # free, and it would be — silently — if the corpus ever stopped emitting per-file
    # relationships or the spy stopped seeing them.
    assert before.rel_rows > 0, "the before arm wrote no relationships — the measurement is not measuring"
    assert before.files_rewritten > 0, "the before arm rewrote no file — the measurement is not measuring"

    assert after.rel_rows * 10 <= before.rel_rows, (
        f"no-op re-check still writes {after.rel_rows} relationship rows against "
        f"{before.rel_rows} unoptimised — the ATL-151 fingerprint is not holding"
    )
    assert after.files_swept * 10 <= before.files_swept, (
        f"no-op re-check still sweeps {after.files_swept} files' edges against {before.files_swept} unoptimised"
    )
    assert after.buffered_rels == before.buffered_rels, (
        f"deferred resolution saw {after.buffered_rels} rels with the skip on and "
        f"{before.buffered_rels} with it off — skipping the write must not skip the buffer (ADR-0026)"
    )
