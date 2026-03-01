"""Profile the query side — per-tool and per-operation breakdown.

Usage:
    uv run python scripts/profile_query.py [--iterations N] [--no-warmup] [--no-vector]

Collects OTel span timings from the production code and wraps each query
with wall-clock timing. Companion script to ``scripts/profile_index.py``.

Flags:
    --iterations N   Timed iterations per query (default 5)
    --no-warmup      Skip warmup pass
    --no-vector      Skip vector/hybrid queries (lightweight mode without TEI)
"""

# OTel must be configured BEFORE importing code_atlas modules (they call
# get_tracer() at module scope). This forces E402 on later imports — expected.
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from typing import Any


@dataclass
class TimingStat:
    calls: int = 0
    total_s: float = 0.0
    max_s: float = 0.0
    samples: list[float] = field(default_factory=list)

    def record(self, elapsed: float) -> None:
        self.calls += 1
        self.total_s += elapsed
        self.max_s = max(self.max_s, elapsed)
        self.samples.append(elapsed)

    @property
    def avg_s(self) -> float:
        return self.total_s / self.calls if self.calls else 0.0

    @property
    def p95_s(self) -> float:
        if not self.samples:
            return 0.0
        s = sorted(self.samples)
        return s[int(len(s) * 0.95)]


@dataclass
class TimingReport:
    stats: dict[str, TimingStat] = field(default_factory=lambda: defaultdict(TimingStat))


class _CollectorExporter(SpanExporter):
    """Collects completed spans into TimingStat records."""

    def __init__(self, report: TimingReport) -> None:
        self._report = report

    def export(self, spans):
        for span in spans:
            name = span.name
            start_ns = span.start_time or 0
            end_ns = span.end_time or 0
            elapsed = (end_ns - start_ns) / 1e9
            self._report.stats[name].record(elapsed)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


_span_report = TimingReport()

_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_CollectorExporter(_span_report)))
trace.set_tracer_provider(_provider)

# Enable telemetry flag so get_tracer() returns real tracers
import code_atlas.telemetry as _telem  # noqa: E402

_telem._enabled = True
_telem._initialized = True

# NOW import modules (they call get_tracer() at module level)
import asyncio  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from code_atlas.graph.client import GraphClient  # noqa: E402
from code_atlas.search.embeddings import EmbedClient  # noqa: E402
from code_atlas.search.engine import SearchType, expand_context, hybrid_search  # noqa: E402
from code_atlas.search.guidance import (  # noqa: E402
    get_guide,
    plan_strategy,
    validate_cypher_explain,
    validate_cypher_static,
)
from code_atlas.server.analysis import analyze_repo, generate_diagram  # noqa: E402
from code_atlas.server.health import run_health_checks  # noqa: E402
from code_atlas.settings import AtlasSettings, derive_project_name  # noqa: E402

# ---------------------------------------------------------------------------
# Query workload
# ---------------------------------------------------------------------------


@dataclass
class QuerySpec:
    name: str  # Display name — matches MCP tool name
    fn: Callable[[], Coroutine[Any, Any, Any]]  # async callable
    category: str  # Grouping key


# Tool-level timing (explicit wall-clock, separate from OTel spans)
_tool_report = TimingReport()


# ---------------------------------------------------------------------------
# get_node cascade (replicates mcp.py 2-stage lookup)
# ---------------------------------------------------------------------------


async def _get_node(graph: GraphClient, name: str, limit: int = 20) -> list[dict]:
    """Replicate the 2-stage cascade from mcp.py's get_node tool."""
    # Stage A: Exact matches (uid + exact name) — 1 RTT
    found = await graph.execute(
        f"MATCH (n {{uid: $name}}) RETURN n LIMIT {limit} "
        f"UNION ALL "
        f"MATCH (n) WHERE n.name = $name RETURN n LIMIT {limit}",
        {"name": name},
    )
    if found:
        return found

    # Stage B: Partial matches (suffix > prefix > contains) — 1 RTT
    return await graph.execute(
        f"MATCH (n) WHERE n.qualified_name ENDS WITH $suffix "
        f"RETURN n, 3 AS _match_score LIMIT {limit} "
        f"UNION ALL "
        f"MATCH (n) WHERE n.qualified_name STARTS WITH $prefix "
        f"RETURN n, 2 AS _match_score LIMIT {limit} "
        f"UNION ALL "
        f"MATCH (n) WHERE n.qualified_name CONTAINS $name OR n.name CONTAINS $name "
        f"RETURN n, 1 AS _match_score LIMIT {limit}",
        {"name": name, "suffix": f".{name}", "prefix": f"{name}."},
    )


# ---------------------------------------------------------------------------
# validate_cypher (replicates mcp.py logic)
# ---------------------------------------------------------------------------


async def _validate_cypher(graph: GraphClient, query: str) -> dict:
    """Replicate mcp.py validate_cypher tool."""
    issues = validate_cypher_static(query)
    explain_issue = await validate_cypher_explain(graph, query)
    if explain_issue is not None:
        issues.append(explain_issue)
    return {"valid": not any(i.level == "error" for i in issues), "issues": issues}


# ---------------------------------------------------------------------------
# index_status (replicates mcp.py logic)
# ---------------------------------------------------------------------------


async def _index_status(graph: GraphClient) -> dict:
    """Replicate mcp.py index_status tool."""
    projects_raw = await graph.get_project_status()
    projects = []
    for row in projects_raw:
        node = row.get("n")
        if node is None:
            continue
        props = dict(node.items()) if hasattr(node, "items") else node
        name = props.get("name", "?")
        entity_count = await graph.count_entities(name)
        projects.append({"name": name, "entity_count": entity_count})

    label_counts_raw = await graph.execute(
        "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY count DESC"
    )
    vec_info = await graph.get_vector_index_info()
    text_info = await graph.get_text_index_info()
    return {
        "projects": projects,
        "label_counts": {r["label"]: r["count"] for r in label_counts_raw},
        "vector_indices": vec_info,
        "text_indices": text_info,
    }


# ---------------------------------------------------------------------------
# list_projects (replicates mcp.py logic)
# ---------------------------------------------------------------------------


async def _list_projects(graph: GraphClient) -> list[dict]:
    """Replicate mcp.py list_projects tool."""
    projects_raw = await graph.get_project_status()
    depends = await graph.execute(
        "MATCH (a:Project)-[:DEPENDS_ON]->(b:Project) RETURN a.name AS from_proj, b.name AS to_proj"
    )
    deps_map: dict[str, list[str]] = {}
    for r in depends:
        deps_map.setdefault(r["from_proj"], []).append(r["to_proj"])

    result = []
    for row in projects_raw:
        node = row.get("n")
        if node is None:
            continue
        props = dict(node.items()) if hasattr(node, "items") else node
        name = props.get("name", "?")
        entity_count = await graph.count_entities(name)
        result.append({"name": name, "entity_count": entity_count, "depends_on": deps_map.get(name, [])})
    return result


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------


def _fmt_dur(seconds: float) -> str:
    """Format a duration as ms or s depending on magnitude."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def _print_query_benchmark(iterations: int) -> None:
    """Print per-query timing table."""
    stats = _tool_report.stats
    if not stats:
        return

    print("\n" + "=" * 70)
    print(f" QUERY BENCHMARK — {iterations} iterations")
    print("=" * 70)
    print(f" {'Query':<40} {'Avg':>8} {'p95':>8} {'Max':>8}")

    for name in sorted(stats, key=lambda k: -stats[k].avg_s):
        s = stats[name]
        print(f" {name:<40} {_fmt_dur(s.avg_s):>8} {_fmt_dur(s.p95_s):>8} {_fmt_dur(s.max_s):>8}")


def _print_operation_breakdown() -> None:
    """Print OTel span breakdown grouped by category."""
    stats = _span_report.stats
    if not stats:
        return

    groups: dict[str, list[str]] = {
        "Search Channels": [
            "graph.graph_search",
            "graph.text_search",
            "graph.vector_search",
        ],
        "Embedding": [
            "embed.embed_one",
            "embed.embed_batch",
        ],
        "Enrichment": [
            "graph.batch_call_stats",
        ],
        "Fusion & Filtering": [
            "hybrid_search",
            "embed_query",
            "rrf_fuse",
            "filter_and_boost",
        ],
        "Context Expansion": [
            "expand_context",
        ],
        "Memgraph RTT (all queries)": [
            "graph.execute",
            "graph.execute_write",
        ],
    }

    print("\n" + "=" * 80)
    print(" OPERATION BREAKDOWN (cumulative)")
    print("=" * 80)

    header = f" {'Operation':<35} {'Calls':>6} {'Total':>8} {'Avg':>8} {'p95':>8} {'Max':>8}"

    for group_name, keys in groups.items():
        present = [k for k in keys if k in stats]
        if not present:
            continue

        print(f"\n {group_name}")
        print(header)
        for key in present:
            s = stats[key]
            print(
                f"   {key:<33} {s.calls:>6} {s.total_s:>7.3f}s"
                f" {_fmt_dur(s.avg_s):>8} {_fmt_dur(s.p95_s):>8} {_fmt_dur(s.max_s):>8}"
            )

    # Uncategorized spans
    all_keys = {k for keys in groups.values() for k in keys}
    uncategorized = {k: v for k, v in stats.items() if k not in all_keys and v.calls > 0}
    if uncategorized:
        print(f"\n {'Other'}")
        print(header)
        for key, s in sorted(uncategorized.items(), key=lambda x: -x[1].total_s):
            print(
                f"   {key:<33} {s.calls:>6} {s.total_s:>7.3f}s"
                f" {_fmt_dur(s.avg_s):>8} {_fmt_dur(s.p95_s):>8} {_fmt_dur(s.max_s):>8}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> tuple[int, bool, bool]:
    """Parse CLI arguments. Returns (iterations, warmup, use_vector)."""
    iterations = 5
    warmup = True
    use_vector = True

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--iterations" and i + 1 < len(args):
            iterations = int(args[i + 1])
            i += 2
        elif args[i] == "--no-warmup":
            warmup = False
            i += 1
        elif args[i] == "--no-vector":
            use_vector = False
            i += 1
        else:
            print(f"Unknown argument: {args[i]}")
            sys.exit(1)
    return iterations, warmup, use_vector


def _build_queries(
    graph: GraphClient,
    embed: EmbedClient | None,
    settings: AtlasSettings,
    sample_uid: str,
    sample_vector: list[float] | None,
) -> list[QuerySpec]:
    """Build the query workload covering all 15 MCP tools."""
    queries: list[QuerySpec] = []
    search_settings = settings.search
    non_vector_types = [SearchType.GRAPH, SearchType.BM25]
    project_name = derive_project_name(settings.project_root)

    # --- Search tools ---
    if embed is not None:
        queries.append(
            QuerySpec(
                "hybrid_search",
                lambda: hybrid_search(graph, embed, search_settings, "parse_file", limit=10),
                "search",
            )
        )
    else:
        queries.append(
            QuerySpec(
                "hybrid_search",
                lambda: hybrid_search(
                    graph, None, search_settings, "parse_file", limit=10, search_types=non_vector_types
                ),
                "search",
            )
        )

    queries.append(QuerySpec("text_search", lambda: graph.text_search("EventBus", limit=10), "search"))

    if sample_vector is not None:
        _vec = sample_vector  # capture for lambda
        queries.append(QuerySpec("vector_search", lambda: graph.vector_search(_vec, limit=10), "search"))

    queries.append(QuerySpec("get_node", lambda: _get_node(graph, "GraphClient"), "search"))

    # --- Navigation tools ---
    queries.append(QuerySpec("get_context", lambda: expand_context(graph, sample_uid), "navigation"))
    queries.append(
        QuerySpec(
            "cypher_query",
            lambda: graph.execute("MATCH (n:Callable) RETURN n.name, n.kind LIMIT 20"),
            "navigation",
        )
    )

    # --- Analysis tools ---
    queries.append(
        QuerySpec(
            "analyze_repo",
            lambda: analyze_repo(graph, "structure", project_name, limit=20),
            "analysis",
        )
    )
    queries.append(
        QuerySpec(
            "generate_diagram",
            lambda: generate_diagram(graph, "packages", project_name, max_nodes=30),
            "analysis",
        )
    )

    # --- Guidance tools ---
    queries.append(QuerySpec("get_usage_guide", lambda: _async_wrap(get_guide, ""), "guidance"))
    queries.append(
        QuerySpec("plan_search_strategy", lambda: _async_wrap(plan_strategy, "find all test files"), "guidance")
    )
    queries.append(
        QuerySpec(
            "validate_cypher",
            lambda: _validate_cypher(graph, "MATCH (n:Callable) RETURN n.name LIMIT 10"),
            "guidance",
        )
    )
    queries.append(QuerySpec("schema_info", lambda: _async_wrap(_schema_info), "guidance"))

    # --- Status tools ---
    queries.append(QuerySpec("index_status", lambda: _index_status(graph), "status"))
    queries.append(QuerySpec("list_projects", lambda: _list_projects(graph), "status"))
    queries.append(QuerySpec("health_check", lambda: run_health_checks(settings, graph=graph, embed=embed), "status"))

    return queries


async def _async_wrap(fn: Callable, *args: object) -> object:
    """Wrap a sync function as a coroutine."""
    return fn(*args)


def _schema_info() -> dict:
    """Replicate mcp.py schema_info (pure data, no DB)."""
    from code_atlas.schema import (
        SCHEMA_VERSION,
        CallableKind,
        NodeLabel,
        RelType,
        TypeDefKind,
        Visibility,
    )

    return {
        "node_labels": sorted(lbl.value for lbl in NodeLabel),
        "relationship_types": sorted(r.value for r in RelType),
        "kind_discriminators": {
            "TypeDefKind": sorted(k.value for k in TypeDefKind),
            "CallableKind": sorted(k.value for k in CallableKind),
            "Visibility": sorted(v.value for v in Visibility),
        },
        "schema_version": SCHEMA_VERSION,
    }


async def _run_pass(
    queries: list[QuerySpec], iterations: int, *, record: bool, embed: EmbedClient | None = None
) -> None:
    """Run all queries for *iterations* rounds, optionally recording to _tool_report."""
    for i in range(iterations):
        # Clear embedding LRU cache so every iteration hits the embedding API
        if embed is not None:
            embed._query_cache.clear()
        for spec in queries:
            t0 = time.monotonic()
            try:
                await spec.fn()
            except Exception as exc:
                label = "WARN" if not record else "ERROR"
                print(f"  {label}: {spec.name} iter {i + 1}: {exc}")
                continue
            if record:
                elapsed = time.monotonic() - t0
                _tool_report.stats[spec.name].record(elapsed)


async def main() -> None:
    iterations, warmup, use_vector = _parse_args()

    project_root = Path().resolve()
    settings = AtlasSettings(project_root=project_root)

    print(f"Profiling queries on: {project_root}")
    print(f"Iterations: {iterations}, Warmup: {warmup}, Vector: {use_vector}")
    print()

    graph = GraphClient(settings)
    embed: EmbedClient | None = None
    if use_vector and settings.embeddings.enabled:
        embed = EmbedClient(settings.embeddings)

    try:
        if not await graph.ping():
            print("ERROR: Cannot connect to Memgraph")
            return

        # Auto-discover a UID for get_context
        uid_records = await graph.execute("MATCH (n:Callable) RETURN n.uid LIMIT 1")
        if not uid_records:
            print("ERROR: No Callable nodes found — index a project first")
            return
        sample_uid: str = uid_records[0]["n.uid"]
        print(f"Sample UID: {sample_uid}")

        # Get an embedding vector for vector_search (if enabled)
        sample_vector: list[float] | None = None
        if embed is not None:
            sample_vector = await embed.embed_one("parse file")

        queries = _build_queries(graph, embed, settings, sample_uid, sample_vector)

        if warmup:
            print("Warmup pass...")
            await _run_pass(queries, 1, record=False, embed=embed)
            _span_report.stats.clear()
            print("Warmup done.\n")

        print(f"Running {iterations} iterations per query...")
        await _run_pass(queries, iterations, record=True, embed=embed)

        _print_query_benchmark(iterations)
        _print_operation_breakdown()
        print()

    finally:
        await graph.close()


if __name__ == "__main__":
    asyncio.run(main())
