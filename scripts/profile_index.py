"""Profile the indexing pipeline per-tier with wall-clock timing.

Usage:
    uv run python scripts/profile_index.py [--full] [--no-cache]

Monkey-patches key methods with timing wrappers, runs index_project()
on the current repo, and prints a breakdown of where time is spent.

Flags:
    --full       Force full reindex (wipe existing data)
    --no-cache   Flush Valkey embedding cache before indexing
"""

from __future__ import annotations

import asyncio
import functools
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Timing infrastructure
# ---------------------------------------------------------------------------

# Monotonic reference for trace offsets (set at index start)
_t0_ref: float = 0.0


@dataclass
class TraceEvent:
    name: str
    start_offset: float  # seconds since _t0_ref
    duration: float  # seconds


@dataclass
class TimingStat:
    calls: int = 0
    total_s: float = 0.0
    min_s: float = float("inf")
    max_s: float = 0.0

    def record(self, elapsed: float) -> None:
        self.calls += 1
        self.total_s += elapsed
        self.min_s = min(self.min_s, elapsed)
        self.max_s = max(self.max_s, elapsed)

    @property
    def avg_s(self) -> float:
        return self.total_s / self.calls if self.calls else 0.0


@dataclass
class TimingReport:
    stats: dict[str, TimingStat] = field(default_factory=lambda: defaultdict(TimingStat))
    trace: list[TraceEvent] = field(default_factory=list)

    def record(self, name: str, start: float, elapsed: float) -> None:
        self.stats[name].record(elapsed)
        self.trace.append(TraceEvent(name, start - _t0_ref, elapsed))


_report = TimingReport()


def timed_async(name: str):
    """Decorator for async methods — records wall-clock timing."""

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            t0 = time.monotonic()
            try:
                return await fn(*args, **kwargs)
            finally:
                elapsed = time.monotonic() - t0
                _report.record(name, t0, elapsed)

        return wrapper

    return decorator


def timed_sync(name: str):
    """Decorator for sync functions — records wall-clock timing."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.monotonic()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.monotonic() - t0
                _report.record(name, t0, elapsed)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Monkey-patching
# ---------------------------------------------------------------------------


def patch_all():
    """Wrap key methods with timing instrumentation."""
    from code_atlas.graph import client as gc
    from code_atlas.indexing import orchestrator as orch
    from code_atlas.parsing import ast as parser_mod
    from code_atlas.parsing import detectors as det_mod
    from code_atlas.search import embeddings as emb_mod

    # --- Orchestrator ---
    orch.scan_files = timed_sync("orch.scan_files")(orch.scan_files)
    orch._decide_delta_mode = timed_async("orch.delta_decision")(orch._decide_delta_mode)
    orch._create_package_hierarchy = timed_async("orch.create_packages")(orch._create_package_hierarchy)
    orch._publish_events = timed_async("orch.publish_events")(orch._publish_events)
    orch._wait_for_drain = timed_async("orch.wait_for_drain")(orch._wait_for_drain)

    # --- Parser ---
    parser_mod.parse_file = timed_sync("tier2.parse_file")(parser_mod.parse_file)

    # --- Detectors ---
    det_mod.run_detectors = timed_async("tier2.run_detectors")(det_mod.run_detectors)

    # --- Graph client methods ---
    gc.GraphClient.upsert_file_entities = timed_async("tier2.upsert_file_entities")(gc.GraphClient.upsert_file_entities)
    gc.GraphClient.upsert_batch_entities = timed_async("tier2.upsert_batch_entities")(
        gc.GraphClient.upsert_batch_entities
    )
    gc.GraphClient.resolve_imports = timed_async("tier2.resolve_imports")(gc.GraphClient.resolve_imports)
    gc.GraphClient.build_resolution_lookup = timed_async("tier2.build_resolution_lookup")(
        gc.GraphClient.build_resolution_lookup
    )
    gc.GraphClient.resolve_calls = timed_async("tier2.resolve_calls")(gc.GraphClient.resolve_calls)
    gc.GraphClient.resolve_type_refs = timed_async("tier2.resolve_type_refs")(gc.GraphClient.resolve_type_refs)
    gc.GraphClient.merge_package_batch = timed_async("orch.merge_package_batch")(gc.GraphClient.merge_package_batch)
    gc.GraphClient.delete_file_entities = timed_async("tier2.delete_file_entities")(gc.GraphClient.delete_file_entities)
    gc.GraphClient.apply_property_enrichments = timed_async("tier2.apply_enrichments")(
        gc.GraphClient.apply_property_enrichments
    )

    # Upsert inner breakdown (run inside managed transaction)
    gc.GraphClient.get_file_content_hashes = timed_async("upsert.get_hashes")(gc.GraphClient.get_file_content_hashes)
    gc.GraphClient.get_batch_file_content_hashes = timed_async("upsert.get_batch_hashes")(
        gc.GraphClient.get_batch_file_content_hashes
    )
    gc.GraphClient._batch_create_entities = timed_async("upsert.batch_create")(gc.GraphClient._batch_create_entities)
    gc.GraphClient._batch_update_entities = timed_async("upsert.batch_update")(gc.GraphClient._batch_update_entities)
    gc.GraphClient._batch_update_positions = timed_async("upsert.batch_positions")(
        gc.GraphClient._batch_update_positions
    )
    gc.GraphClient._batch_delete_entities = timed_async("upsert.batch_delete")(gc.GraphClient._batch_delete_entities)
    gc.GraphClient._recreate_file_relationships = timed_async("upsert.recreate_rels")(
        gc.GraphClient._recreate_file_relationships
    )
    gc.GraphClient._recreate_batch_relationships = timed_async("upsert.recreate_batch_rels")(
        gc.GraphClient._recreate_batch_relationships
    )

    # Low-level graph execute
    gc.GraphClient.execute = timed_async("graph.execute")(gc.GraphClient.execute)
    gc.GraphClient._execute_write_with_retry = timed_async("graph.execute_write")(
        gc.GraphClient._execute_write_with_retry
    )

    # --- Tier 3 ---
    gc.GraphClient.read_entity_texts = timed_async("tier3.read_entity_texts")(gc.GraphClient.read_entity_texts)
    gc.GraphClient.write_embeddings = timed_async("tier3.write_embeddings")(gc.GraphClient.write_embeddings)
    gc.GraphClient.write_embed_hashes = timed_async("tier3.write_embed_hashes")(gc.GraphClient.write_embed_hashes)
    gc.GraphClient.write_embeddings_and_hashes = timed_async("tier3.write_embeddings_and_hashes")(
        gc.GraphClient.write_embeddings_and_hashes
    )
    gc.GraphClient.run_in_write_transaction = timed_async("tier3.run_in_write_tx")(
        gc.GraphClient.run_in_write_transaction
    )
    emb_mod.EmbedClient.embed_batch = timed_async("tier3.embed_api")(emb_mod.EmbedClient.embed_batch)

    # Tier 3 cache
    emb_mod.EmbedCache.get_many = timed_async("tier3.cache_get_many")(emb_mod.EmbedCache.get_many)
    emb_mod.EmbedCache.put_many = timed_async("tier3.cache_put_many")(emb_mod.EmbedCache.put_many)

    # --- EventBus ---
    from code_atlas.events import EventBus

    EventBus.publish_many = timed_async("bus.publish_many")(EventBus.publish_many)
    # read_batch excluded — mostly idle XREADGROUP block time, not actual work.
    EventBus.read_pending = timed_async("bus.read_pending")(EventBus.read_pending)
    EventBus.ack = timed_async("bus.ack")(EventBus.ack)
    EventBus.stream_group_info_multi = timed_async("bus.stream_group_info_multi")(EventBus.stream_group_info_multi)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


_TRACE_NOISE = frozenset(
    {
        "graph.execute",
        "graph.execute_write",
        "bus.read_pending",
        "bus.ack",
        "bus.stream_group_info_multi",
    }
)


def print_trace(wall_time: float, *, min_duration: float = 0.05):
    """Print a chronological trace of significant operations.

    Filters out low-level graph/bus noise and events shorter than *min_duration*.
    """
    events = [e for e in _report.trace if e.name not in _TRACE_NOISE and e.duration >= min_duration]
    if not events:
        print("\n(No trace events above threshold)")
        return

    print("\n" + "=" * 100)
    print(f" TRACE VIEW — wall clock: {wall_time:.2f}s  (threshold: {min_duration * 1000:.0f}ms)")
    print("=" * 100)

    # Timeline header
    print(f" {'#':>4}  {'T+start':>9}  {'Duration':>10}  {'T+end':>9}  {'Operation'}")
    print(f" {'':>4}  {'':>9}  {'':>10}  {'':>9}  {'-' * 60}")

    for i, ev in enumerate(events, 1):
        end = ev.start_offset + ev.duration
        dur_str = f"{ev.duration * 1000:.0f}ms" if ev.duration < 1 else f"{ev.duration:.2f}s"
        start_str = f"{ev.start_offset:.3f}s"
        end_str = f"{end:.3f}s"

        # Visual bar -- scale to 40-char width
        bar_start = int(ev.start_offset / wall_time * 40) if wall_time > 0 else 0
        bar_len = max(1, int(ev.duration / wall_time * 40)) if wall_time > 0 else 1
        bar = "." * bar_start + "#" * bar_len

        print(f" {i:>4}  {start_str:>9}  {dur_str:>10}  {end_str:>9}  {ev.name}")
        print(f"       {'':>9}  {'':>10}  {'':>9}  {bar}")

    print()


def print_report(wall_time: float):
    """Print a formatted timing report."""
    stats = _report.stats

    print("\n" + "=" * 80)
    print(f" CUMULATIVE REPORT — wall clock: {wall_time:.2f}s")
    print("=" * 80)

    # Group by tier
    groups = {
        "Orchestrator": [
            "orch.scan_files",
            "orch.delta_decision",
            "orch.create_packages",
            "orch.publish_events",
            "orch.wait_for_drain",
            "orch.merge_package_batch",
        ],
        "Tier 1 (Graph metadata)": [],  # Tier 1 is mostly passthrough
        "Tier 2 (AST + Graph)": [
            "tier2.parse_file",
            "tier2.run_detectors",
            "tier2.upsert_file_entities",
            "tier2.upsert_batch_entities",
            "tier2.delete_file_entities",
            "tier2.apply_enrichments",
            "tier2.resolve_imports",
            "tier2.build_resolution_lookup",
            "tier2.resolve_calls",
            "tier2.resolve_type_refs",
        ],
        "Tier 3 (Embeddings)": [
            "tier3.read_entity_texts",
            "tier3.cache_get_many",
            "tier3.embed_api",
            "tier3.cache_put_many",
            "tier3.write_embeddings",
            "tier3.write_embed_hashes",
            "tier3.write_embeddings_and_hashes",
            "tier3.run_in_write_tx",
        ],
        "Graph I/O (low-level)": ["graph.execute", "graph.execute_write"],
        "Event Bus (Valkey)": [
            "bus.publish_many",
            "bus.read_pending",
            "bus.ack",
            "bus.stream_group_info_multi",
        ],
    }

    for group_name, keys in groups.items():
        group_total = sum(stats[k].total_s for k in keys if k in stats)
        if group_total == 0 and not any(k in stats for k in keys):
            continue

        pct = (group_total / wall_time * 100) if wall_time > 0 else 0
        print(f"\n{'-' * 80}")
        print(f" {group_name}  ({group_total:.3f}s total, {pct:.1f}% of wall)")
        print(f"{'-' * 80}")
        print(f" {'Operation':<40} {'Calls':>6} {'Total':>8} {'Avg':>8} {'Min':>8} {'Max':>8}")

        for key in keys:
            if key not in stats:
                continue
            s = stats[key]
            min_ms = s.min_s * 1000 if s.min_s != float("inf") else 0
            print(
                f" {key:<40} {s.calls:>6} {s.total_s:>7.3f}s {s.avg_s * 1000:>7.1f}ms "
                f"{min_ms:>7.1f}ms {s.max_s * 1000:>7.1f}ms"
            )

    # Uncategorized
    all_keys = {k for keys in groups.values() for k in keys}
    uncategorized = {k: v for k, v in stats.items() if k not in all_keys}
    if uncategorized:
        print(f"\n{'-' * 80}")
        print(" Uncategorized")
        print(f"{'-' * 80}")
        print(f" {'Operation':<40} {'Calls':>6} {'Total':>8} {'Avg':>8} {'Min':>8} {'Max':>8}")
        for key, s in sorted(uncategorized.items(), key=lambda x: -x[1].total_s):
            min_ms = s.min_s * 1000 if s.min_s != float("inf") else 0
            print(
                f" {key:<40} {s.calls:>6} {s.total_s:>7.3f}s {s.avg_s * 1000:>7.1f}ms "
                f"{min_ms:>7.1f}ms {s.max_s * 1000:>7.1f}ms"
            )

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    global _t0_ref

    full_reindex = "--full" in sys.argv
    no_cache = "--no-cache" in sys.argv

    patch_all()

    # Import after patching so the wrappers are in place
    from code_atlas.events import EventBus
    from code_atlas.graph.client import GraphClient
    from code_atlas.indexing.orchestrator import index_project
    from code_atlas.search.embeddings import EmbedCache
    from code_atlas.settings import AtlasSettings

    project_root = Path().resolve()
    settings = AtlasSettings(project_root=project_root)

    print(f"Profiling index on: {project_root}")
    print(f"Mode: {'full' if full_reindex else 'auto (delta/full)'}")
    print(f"Embeddings: {'enabled' if settings.embeddings.enabled else 'disabled'}")
    print(f"Cache: {'disabled (--no-cache)' if no_cache else 'enabled'}")
    print()

    graph = GraphClient(settings)
    bus = EventBus(settings.redis, project_name="")

    try:
        await graph.ensure_schema()

        if no_cache and settings.embeddings.enabled and settings.embeddings.cache_ttl_days > 0:
            cache = EmbedCache(settings.redis, settings.embeddings)
            await cache.clear_all_models()
            print("Flushed embedding cache")

        _t0_ref = time.monotonic()
        result = await index_project(
            settings,
            graph,
            bus,
            full_reindex=full_reindex,
            drain_timeout_s=120.0,
        )
        wall_time = time.monotonic() - _t0_ref

        print(
            f"\nIndex result: {result.files_scanned} files, {result.entities_total} entities, "
            f"{result.duration_s:.2f}s ({result.mode})"
        )

        print_trace(wall_time)
        print_report(wall_time)

    finally:
        await bus.close()
        await graph.close()


if __name__ == "__main__":
    asyncio.run(main())
