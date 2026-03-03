"""Profile the indexing pipeline per-tier with wall-clock and CPU timing.

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
    duration: float  # seconds (wall)
    cpu_time: float  # seconds (process CPU)


@dataclass
class TimingStat:
    calls: int = 0
    total_s: float = 0.0
    max_s: float = 0.0
    samples: list[float] = field(default_factory=list)
    # CPU time
    cpu_total_s: float = 0.0
    cpu_max_s: float = 0.0
    cpu_samples: list[float] = field(default_factory=list)

    def record(self, elapsed: float, cpu_elapsed: float) -> None:
        self.calls += 1
        self.total_s += elapsed
        self.max_s = max(self.max_s, elapsed)
        self.samples.append(elapsed)
        self.cpu_total_s += cpu_elapsed
        self.cpu_max_s = max(self.cpu_max_s, cpu_elapsed)
        self.cpu_samples.append(cpu_elapsed)

    @property
    def avg_s(self) -> float:
        return self.total_s / self.calls if self.calls else 0.0

    @property
    def cpu_avg_s(self) -> float:
        return self.cpu_total_s / self.calls if self.calls else 0.0

    @property
    def p95_s(self) -> float:
        if not self.samples:
            return 0.0
        s = sorted(self.samples)
        return s[int(len(s) * 0.95)]

    @property
    def cpu_p95_s(self) -> float:
        if not self.cpu_samples:
            return 0.0
        s = sorted(self.cpu_samples)
        return s[int(len(s) * 0.95)]


@dataclass
class TimingReport:
    stats: dict[str, TimingStat] = field(default_factory=lambda: defaultdict(TimingStat))
    trace: list[TraceEvent] = field(default_factory=list)

    def record(self, name: str, start: float, elapsed: float, cpu_elapsed: float) -> None:
        self.stats[name].record(elapsed, cpu_elapsed)
        self.trace.append(TraceEvent(name, start - _t0_ref, elapsed, cpu_elapsed))


_report = TimingReport()


def timed_async(name: str):
    """Decorator for async methods — records wall-clock and CPU timing."""

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            t0 = time.monotonic()
            c0 = time.process_time()
            try:
                return await fn(*args, **kwargs)
            finally:
                elapsed = time.monotonic() - t0
                cpu_elapsed = time.process_time() - c0
                _report.record(name, t0, elapsed, cpu_elapsed)

        return wrapper

    return decorator


def timed_sync(name: str):
    """Decorator for sync functions — records wall-clock and CPU timing."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.monotonic()
            c0 = time.process_time()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.monotonic() - t0
                cpu_elapsed = time.process_time() - c0
                _report.record(name, t0, elapsed, cpu_elapsed)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Per-task on-CPU time via Handle._run() patch
# ---------------------------------------------------------------------------

# Accumulates actual on-CPU time per task, excluding event loop contention.
# Key = id(task), Value = (task_name, cumulative_oncpu_seconds).
_task_oncpu: dict[int, tuple[str, float]] = {}


def _patch_handles():
    """Patch asyncio Handle to measure per-task on-CPU time.

    The event loop is single-threaded and non-preemptive: between a task's
    resume and its next yield, nothing else runs.  Timing Handle._run()
    therefore gives exact synchronous execution time per task step, free of
    contention noise from other coroutines.

    Must be called *before* ``asyncio.run()``.
    """
    _orig_handle = asyncio.events.Handle

    class _TimedHandle(_orig_handle):
        __slots__ = ()

        def _run(self):  # type: ignore[override]  # CPython internal
            t0 = time.perf_counter()
            try:
                return super()._run()
            finally:
                elapsed = time.perf_counter() - t0
                cb = getattr(self._callback, "__self__", None)
                if isinstance(cb, asyncio.Task):
                    tid = id(cb)
                    prev = _task_oncpu.get(tid)
                    _task_oncpu[tid] = (cb.get_name(), (prev[1] if prev else 0.0) + elapsed)

    asyncio.events.Handle = _TimedHandle  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Event Bus throughput metrics
# ---------------------------------------------------------------------------


@dataclass
class _BusMetrics:
    total_calls: int = 0
    empty_calls: int = 0  # returned 0 messages (idle polls)
    total_messages: int = 0


_bus_metrics = _BusMetrics()


# ---------------------------------------------------------------------------
# Monkey-patching
# ---------------------------------------------------------------------------


def patch_all():
    """Wrap key methods with timing instrumentation."""
    from code_atlas.events import EventBus
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
    parser_mod.parse_file = timed_sync("ast.parse_file")(parser_mod.parse_file)

    # --- Detectors ---
    det_mod.run_detectors = timed_async("ast.run_detectors")(det_mod.run_detectors)

    # --- Graph client methods ---
    gc.GraphClient.upsert_file_entities = timed_async("ast.upsert_file_entities")(gc.GraphClient.upsert_file_entities)
    gc.GraphClient.upsert_batch_entities = timed_async("ast.upsert_batch_entities")(
        gc.GraphClient.upsert_batch_entities
    )
    gc.GraphClient.resolve_imports = timed_async("ast.resolve_imports")(gc.GraphClient.resolve_imports)
    gc.GraphClient.build_resolution_lookup = timed_async("ast.build_resolution_lookup")(
        gc.GraphClient.build_resolution_lookup
    )
    gc.GraphClient.resolve_calls = timed_async("ast.resolve_calls")(gc.GraphClient.resolve_calls)
    gc.GraphClient.resolve_type_refs = timed_async("ast.resolve_type_refs")(gc.GraphClient.resolve_type_refs)
    gc.GraphClient.merge_package_batch = timed_async("orch.merge_package_batch")(gc.GraphClient.merge_package_batch)
    gc.GraphClient.delete_file_entities = timed_async("ast.delete_file_entities")(gc.GraphClient.delete_file_entities)
    gc.GraphClient.apply_property_enrichments = timed_async("ast.apply_enrichments")(
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

    # --- Embed stage ---
    gc.GraphClient.read_entity_texts = timed_async("embed.read_entity_texts")(gc.GraphClient.read_entity_texts)
    gc.GraphClient.write_embeddings_and_hashes = timed_async("embed.write_embeddings_and_hashes")(
        gc.GraphClient.write_embeddings_and_hashes
    )
    emb_mod.EmbedClient.embed_batch = timed_async("embed.embed_api")(emb_mod.EmbedClient.embed_batch)

    # Embed cache
    emb_mod.EmbedCache.get_many = timed_async("embed.cache_get_many")(emb_mod.EmbedCache.get_many)
    emb_mod.EmbedCache.put_many = timed_async("embed.cache_put_many")(emb_mod.EmbedCache.put_many)

    # --- EventBus ---
    EventBus.publish_many = timed_async("bus.publish_many")(EventBus.publish_many)
    EventBus.read_pending = timed_async("bus.read_pending")(EventBus.read_pending)
    EventBus.ack = timed_async("bus.ack")(EventBus.ack)
    EventBus.stream_group_info_multi = timed_async("bus.stream_group_info_multi")(EventBus.stream_group_info_multi)

    # --- EventBus throughput counter (not timed — read_batch is mostly idle) ---
    original_read_batch = EventBus.read_batch

    @functools.wraps(original_read_batch)
    async def _counting_read_batch(*args, **kwargs):
        result = await original_read_batch(*args, **kwargs)
        _bus_metrics.total_calls += 1
        if not result:
            _bus_metrics.empty_calls += 1
        else:
            _bus_metrics.total_messages += len(result)
        return result

    EventBus.read_batch = _counting_read_batch  # type: ignore[invalid-assignment]  # monkey-patch


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt_dur(seconds: float) -> str:
    """Format a duration as ms or s depending on magnitude."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def print_trace(wall_time: float, *, min_duration: float = 0.05):
    """Print a chronological trace of significant operations.

    Filters out low-level bus noise and events shorter than *min_duration*.
    """
    noise = frozenset({"bus.read_pending", "bus.ack", "bus.stream_group_info_multi"})
    events = [e for e in _report.trace if e.name not in noise and e.duration >= min_duration]
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
        dur_str = _fmt_dur(ev.duration)
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

    print("\n" + "=" * 100)
    print(f" CUMULATIVE REPORT — wall clock: {wall_time:.2f}s")
    print("=" * 100)

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
        "AST Stage (Parse + Graph)": [
            "ast.parse_file",
            "ast.run_detectors",
            "ast.upsert_file_entities",
            "ast.upsert_batch_entities",
            "ast.delete_file_entities",
            "ast.apply_enrichments",
            "ast.resolve_imports",
            "ast.build_resolution_lookup",
            "ast.resolve_calls",
            "ast.resolve_type_refs",
            # Upsert inner breakdown
            "upsert.get_hashes",
            "upsert.get_batch_hashes",
            "upsert.batch_create",
            "upsert.batch_update",
            "upsert.batch_positions",
            "upsert.batch_delete",
            "upsert.recreate_rels",
            "upsert.recreate_batch_rels",
        ],
        "Embed Stage (Embeddings)": [
            "embed.read_entity_texts",
            "embed.cache_get_many",
            "embed.embed_api",
            "embed.cache_put_many",
            "embed.write_embeddings_and_hashes",
        ],
        "Event Bus (Valkey)": [
            "bus.publish_many",
            "bus.read_pending",
            "bus.ack",
            "bus.stream_group_info_multi",
        ],
    }

    header = f" {'Operation':<40} {'Calls':>6} {'Total':>8} {'Avg':>8} {'p95':>8} {'Max':>8}"

    for group_name, keys in groups.items():
        group_total = sum(stats[k].total_s for k in keys if k in stats)
        if group_total == 0 and not any(k in stats for k in keys):
            continue

        pct = (group_total / wall_time * 100) if wall_time > 0 else 0
        print(f"\n{'-' * 100}")
        print(f" {group_name}  ({group_total:.3f}s total, {pct:.1f}% of wall)")
        print(f"{'-' * 100}")
        print(header)

        for key in keys:
            if key not in stats:
                continue
            s = stats[key]
            # Wall row
            print(
                f" {key:<40} {s.calls:>6} {s.total_s:>7.3f}s"
                f" {_fmt_dur(s.avg_s):>8} {_fmt_dur(s.p95_s):>8} {_fmt_dur(s.max_s):>8}"
            )
            # CPU row (omit if negligible)
            if s.cpu_total_s >= 0.01:
                print(
                    f" {'':40} {'cpu:':>6} {s.cpu_total_s:>7.3f}s"
                    f" {_fmt_dur(s.cpu_avg_s):>8} {_fmt_dur(s.cpu_p95_s):>8} {_fmt_dur(s.cpu_max_s):>8}"
                )

    # Uncategorized
    all_keys = {k for keys in groups.values() for k in keys}
    uncategorized = {k: v for k, v in stats.items() if k not in all_keys}
    if uncategorized:
        print(f"\n{'-' * 100}")
        print(" Uncategorized")
        print(f"{'-' * 100}")
        print(header)
        for key, s in sorted(uncategorized.items(), key=lambda x: -x[1].total_s):
            print(
                f" {key:<40} {s.calls:>6} {s.total_s:>7.3f}s"
                f" {_fmt_dur(s.avg_s):>8} {_fmt_dur(s.p95_s):>8} {_fmt_dur(s.max_s):>8}"
            )
            if s.cpu_total_s >= 0.01:
                print(
                    f" {'':40} {'cpu:':>6} {s.cpu_total_s:>7.3f}s"
                    f" {_fmt_dur(s.cpu_avg_s):>8} {_fmt_dur(s.cpu_p95_s):>8} {_fmt_dur(s.cpu_max_s):>8}"
                )

    # Event Bus throughput
    bm = _bus_metrics
    if bm.total_calls > 0:
        productive = bm.total_calls - bm.empty_calls
        avg_per = bm.total_messages / productive if productive > 0 else 0.0
        print(f"\n{'-' * 100}")
        print(" Event Bus Throughput")
        print(f"{'-' * 100}")
        print(f"   read_batch calls: {bm.total_calls}  ({bm.empty_calls} idle, {productive} with data)")
        print(f"   Messages delivered: {bm.total_messages}  (avg {avg_per:.1f}/productive call)")

    _print_task_oncpu(wall_time)
    print()


def _print_task_oncpu(wall_time: float) -> None:
    """Print per-task on-CPU time from the Handle._run() patch."""
    if not _task_oncpu:
        return
    tasks_sorted = sorted(_task_oncpu.values(), key=lambda x: -x[1])
    total_oncpu = sum(t[1] for t in tasks_sorted)
    # Only show tasks with >= 1% of wall time, up to 15
    significant = [(n, t) for n, t in tasks_sorted if t >= wall_time * 0.01]
    shown = significant[:15]
    rest_count = len(tasks_sorted) - len(shown)
    rest_oncpu = total_oncpu - sum(t for _, t in shown)

    print(f"\n{'-' * 100}")
    print(f" Task On-CPU Time  (total: {_fmt_dur(total_oncpu)}, {len(tasks_sorted)} tasks)")
    print(f"{'-' * 100}")
    print(f" {'Task':<50} {'On-CPU':>10}")
    for name, oncpu in shown:
        pct = (oncpu / wall_time * 100) if wall_time > 0 else 0
        print(f" {name:<50} {_fmt_dur(oncpu):>10}  ({pct:.1f}%)")
    if rest_count > 0:
        pct = (rest_oncpu / wall_time * 100) if wall_time > 0 else 0
        print(f" {'... ' + str(rest_count) + ' more tasks':<50} {_fmt_dur(rest_oncpu):>10}  ({pct:.1f}%)")


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
    _patch_handles()
    asyncio.run(main())
