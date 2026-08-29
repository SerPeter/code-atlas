# ADR-0037: Telemetry is a product surface, not a debug aid

## Status

Accepted (2026-08-29)

## Context

Telemetry had been present since early on: an `[otel]` extra, `ObservabilitySettings`, `init_telemetry`, and roughly 25
span sites across events, graph, consumers, orchestrator, embeddings and search. None of it worked.

Every module binds its tracer once at import — `_tracer = get_tracer(__name__)` — and every entry point imports its
dependencies before it reads settings and calls `init_telemetry`. `cli.mcp` imports `server.mcp` on its first line and
initializes telemetry three lines later. Because `get_tracer` decided on the `_enabled` flag at call time and returned a
concrete object, every one of those module-level tracers was a permanent no-op, with the extra installed and telemetry
fully enabled. Metrics were unaffected, because `get_metrics()` is called at use time against a global that
`init_telemetry` rebinds — so whatever was looked at had numbers on it, and the missing traces were never noticed.

Beyond that, three things were absent rather than broken:

- **No MCP tool spans.** `_tracer` was imported in `server/mcp.py` and never used. Every trace began mid-stack —
  `graph.execute`, `hybrid_search`, `embed.embed_one` — with nothing above it naming the tool an agent had called.
- **No logs.** loguru wrote to stderr only. Under MCP that stderr belongs to the agent client's subprocess plumbing and
  reaches no file, so the decisions that actually explain this system ("skipping startup catch-up", "lease lost while
  still indexing") were unavailable after the fact. One investigation had to reconstruct pipeline state with `redis-cli`
  against Valkey.
- **No pipeline numbers.** Backlog depth, batch throughput and where each embedding vector came from were all computed
  in-process and none of them reported.

## Decision

Treat the observability layer as a shipped surface with the same evidentiary standard as the rest of the product: it
must be verified against a running backend, not asserted.

1. `get_tracer`/`get_meter` return handles that resolve on first use and cache the result. The regression test stubs the
   OTel boundary rather than skipping when the extra is absent — the defect was in our binding logic, so the test has to
   run everywhere.
2. Every MCP tool call opens a root span and records `atlas_mcp_tool_calls` / `atlas_mcp_tool_latency_seconds`,
   installed through the same `mcp.tool` seam as the backend stamp. Failure is recorded both ways it arrives: a raised
   exception, and the `{"error": ...}` payload most tools here return.
3. loguru is bridged to OTLP logs through OTel's `LoggingHandler`, which stamps each record with the active span's trace
   and span id. Correlation is therefore structural, not a convention. The OTLP sink has its own level, separate from
   the console: DEBUG is useful locally and ruinous as remote volume at one line per file across a 60k-file index.
4. The resource names the process well enough to separate every way atlas processes overlap. One collector serves them
   all, and the three overlaps need three different discriminators:

   | Overlap                         | Discriminator         | Why the others do not suffice                             |
   | ------------------------------- | --------------------- | --------------------------------------------------------- |
   | Several repos                   | `atlas.project`       | —                                                         |
   | Several worktrees of one repo   | `atlas.root`          | `atlas.project` is the directory name; basenames collide  |
   | Several processes, one worktree | `service.instance.id` | host:pid — shares its prefix with the indexer lease owner |

   `atlas.indexing` is deliberately separate from `atlas.role`. An MCP server started without `--no-index` runs the
   watcher and pipeline itself, so `role="mcp"` covered both a query-only server and the machine's only indexer. "Who is
   actually indexing this checkout" is the first question asked when nothing is being indexed, or when two things are,
   and role could not answer it.

5. Backlog, batch outcome, events consumed, embedding provenance and watcher activity are exported where the values
   already exist.
6. The web UI, which had never called `init_telemetry` at all, reports `role="web"` and carries a span and a latency
   sample per request — identified by Litestar's route template rather than the request path, since per-path series is
   how a metrics database is taken down by its own instrumentation.
7. The backends ship as `docker compose --profile telemetry`: VictoriaMetrics, VictoriaLogs, VictoriaTraces, an OTel
   Collector, and Grafana with datasources provisioned as code.

## Consequences

The Collector is not optional scaffolding. An OTLP SDK exports every signal to one endpoint and Victoria stores the
three signals in three services, so something must fan out; `configs/otel-collector.yaml` does only that.

Verified end to end against the running stack rather than reasoned about: a span emitted by `code_atlas` arrives in
VictoriaTraces as `mcp.tool.probe`, its metrics arrive in VictoriaMetrics as `atlas_mcp_tool_calls_total` and
`atlas_pipeline_backlog`, and the log line emitted inside that span arrives in VictoriaLogs carrying the _same_ trace id
(`23d3847942716da4a89d348a708b4c73`) that the span reports. Trace-to-logs click-through works because the ids genuinely
match, not because a field was named `trace_id`.

Cost: the OTel packages remain an optional extra, so nothing changes for a default install. With telemetry off, tracer
handles resolve to no-ops, no log sink is installed, and the daemon does not start its backlog sampler — the sampler is
an XINFO round-trip on a timer and would otherwise run forever for nobody.

One measurement is narrower than it looks, deliberately. Litestar applies middleware inside routing, so an unrouted path
never reaches the telemetry middleware: `atlas_web_requests` counts requests that matched a route, not requests
received. That was verified rather than assumed, and it cuts both ways — a 404 sweep cannot mint one metric series per
probed URL. A test pins the behaviour, so a future Litestar that moves middleware outside routing surfaces as a failure
rather than as a silent change of meaning.

VictoriaMetrics promotes resource attributes to metric labels, so concurrent processes do not overwrite each other's
counters — verified with two processes writing the same instrument under different roots. The price is that
`service_instance_id` is host:pid, so every process start mints a new series set. That is correct for the question it
answers ("which of the two servers in this worktree") and cheap for long-lived daemons, but a frequently-run
`atlas index` accumulates series over the retention window. If cardinality ever becomes the problem, the fix is to strip
`service.instance.id` and `process.pid` from the collector's **metrics** pipeline only — logs and traces need them and
cost nothing to keep.

Grafana (3000) and VictoriaMetrics (8428) use popular ports. A second local stack — the trading-engine monitoring
compose is the obvious one — will collide, and the host-side mapping is the place to fix that.
