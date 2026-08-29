"""OpenTelemetry integration for Code Atlas.

All OTel dependencies are **optional** (``[otel]`` extra).  When not installed,
every public function returns lightweight no-op stubs so the rest of the
codebase can instrument unconditionally with zero overhead.
"""

from __future__ import annotations

import contextlib
import os
import socket
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from code_atlas.settings import ObservabilitySettings

# ---------------------------------------------------------------------------
# Availability flag
# ---------------------------------------------------------------------------

try:
    from opentelemetry import metrics as otel_metrics
    from opentelemetry import trace as otel_trace

    _HAS_OTEL = True
except ModuleNotFoundError:
    _HAS_OTEL = False

# ---------------------------------------------------------------------------
# No-op stubs (used when OTel is not installed or disabled)
# ---------------------------------------------------------------------------


class _NoOpSpan:
    """Minimal span-like object that does nothing."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any, description: str | None = None) -> None:
        pass

    def record_exception(self, exception: BaseException, **kwargs: Any) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, *args: object) -> None:
        pass


class _NoOpTracer:
    """Tracer that always returns ``_NoOpSpan``."""

    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoOpSpan:  # noqa: ARG002
        return _NoOpSpan()

    @contextmanager
    def start_span(self, name: str, **kwargs: Any) -> Iterator[_NoOpSpan]:  # noqa: ARG002
        yield _NoOpSpan()


class _NoOpCounter:
    def add(self, amount: int | float, attributes: dict[str, Any] | None = None) -> None:
        pass


class _NoOpHistogram:
    def record(self, amount: int | float, attributes: dict[str, Any] | None = None) -> None:
        pass


class _NoOpMeter:
    def create_counter(self, name: str, **kwargs: Any) -> _NoOpCounter:  # noqa: ARG002
        return _NoOpCounter()

    def create_histogram(self, name: str, **kwargs: Any) -> _NoOpHistogram:  # noqa: ARG002
        return _NoOpHistogram()

    def create_observable_gauge(self, name: str, **kwargs: Any) -> None:  # noqa: ARG002
        return None


# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

_initialized: bool = False
_enabled: bool = False
_log_sink_id: int | None = None
_logger_provider: Any = None

# ---------------------------------------------------------------------------
# Lazy handles
#
# Every module in this package binds its tracer once, at import time
# (``_tracer = get_tracer(__name__)``), and every entry point imports its
# dependencies *before* it reads settings and calls ``init_telemetry`` --
# ``cli.mcp`` imports ``server.mcp`` (and transitively graph, engine, consumers,
# events) on its first line, then initializes telemetry three lines later.
#
# So a factory that decided on ``_enabled`` at call time froze all ~25 span sites
# in the codebase as no-ops *even with telemetry enabled and OTel installed*.
# Measured, not theorized: after ``init_telemetry``, ``graph.client._tracer`` was
# still a ``_NoOpTracer`` while a fresh ``get_tracer()`` returned a real
# ``Tracer``. Nothing errored and nothing was logged -- the traces were simply
# never emitted, which is the worst failure mode an observability layer has.
#
# Metrics were never affected: ``get_metrics()`` is called at use time and reads
# the module global that ``init_telemetry`` rebinds. That asymmetry is why the
# hole survived -- whatever dashboard existed had numbers on it.
# ---------------------------------------------------------------------------

_NOOP_TRACER = _NoOpTracer()
_NOOP_METER = _NoOpMeter()


class _LazyTracer:
    """Tracer handle that resolves on first use after ``init_telemetry``."""

    __slots__ = ("_name", "_real")

    def __init__(self, name: str) -> None:
        self._name = name
        self._real: Any = None

    def _resolve(self) -> Any:
        if not (_HAS_OTEL and _enabled):
            return _NOOP_TRACER
        if self._real is None:
            self._real = otel_trace.get_tracer(self._name)
        return self._real

    def start_as_current_span(self, name: str, **kwargs: Any) -> Any:
        return self._resolve().start_as_current_span(name, **kwargs)

    def start_span(self, name: str, **kwargs: Any) -> Any:
        return self._resolve().start_span(name, **kwargs)


class _LazyMeter:
    """Meter handle that resolves on first use after ``init_telemetry``."""

    __slots__ = ("_name", "_real")

    def __init__(self, name: str) -> None:
        self._name = name
        self._real: Any = None

    def _resolve(self) -> Any:
        if not (_HAS_OTEL and _enabled):
            return _NOOP_METER
        if self._real is None:
            self._real = otel_metrics.get_meter(self._name)
        return self._real

    def create_counter(self, name: str, **kwargs: Any) -> Any:
        return self._resolve().create_counter(name, **kwargs)

    def create_histogram(self, name: str, **kwargs: Any) -> Any:
        return self._resolve().create_histogram(name, **kwargs)

    def create_observable_gauge(self, name: str, **kwargs: Any) -> Any:
        return self._resolve().create_observable_gauge(name, **kwargs)


def get_tracer(name: str) -> Any:
    """Return a tracer handle. Safe to call at module import time."""
    return _LazyTracer(name)


def get_meter(name: str) -> Any:
    """Return a meter handle. Safe to call at module import time."""
    return _LazyMeter(name)


# ---------------------------------------------------------------------------
# Metric instruments (centralized, lazy-initialized)
# ---------------------------------------------------------------------------


@dataclass
class _Metrics:
    """Central registry of metric instruments."""

    query_count: Any = field(default_factory=_NoOpCounter)
    query_latency: Any = field(default_factory=_NoOpHistogram)
    search_results_count: Any = field(default_factory=_NoOpHistogram)
    index_files_total: Any = field(default_factory=_NoOpCounter)
    index_entities_total: Any = field(default_factory=_NoOpCounter)
    index_duration: Any = field(default_factory=_NoOpHistogram)
    embedding_latency: Any = field(default_factory=_NoOpHistogram)
    tool_calls: Any = field(default_factory=_NoOpCounter)
    tool_latency: Any = field(default_factory=_NoOpHistogram)
    batches_processed: Any = field(default_factory=_NoOpCounter)
    events_consumed: Any = field(default_factory=_NoOpCounter)
    watcher_events: Any = field(default_factory=_NoOpCounter)
    embeddings_total: Any = field(default_factory=_NoOpCounter)
    web_requests: Any = field(default_factory=_NoOpCounter)
    web_latency: Any = field(default_factory=_NoOpHistogram)


_metrics = _Metrics()


def get_metrics() -> _Metrics:
    """Return the centralized metrics namespace."""
    return _metrics


# ---------------------------------------------------------------------------
# Initialization / shutdown
# ---------------------------------------------------------------------------


def init_telemetry(
    settings: ObservabilitySettings,
    *,
    role: str = "",
    project: str = "",
    root: str = "",
    indexing: bool | None = None,
) -> None:
    """Configure OTel providers and instruments based on *settings*.

    Safe to call multiple times — only the first call has effect.
    When OTel packages are not installed this is a no-op.

    One collector serves every atlas process on the machine, and they overlap in three
    separate ways. Each needs its own discriminator, because none of the others is
    sufficient on its own:

    - **Several repos.** ``atlas.project`` separates them.
    - **Several worktrees of one repo.** ``atlas.project`` is derived from the directory
      name, so two worktrees whose basename matches collide. ``atlas.root`` carries the
      absolute path and cannot.
    - **Several processes in one worktree** — the shape this deployment is built around,
      an indexer plus one MCP server per agent session. ``service.instance.id`` is
      host:pid. It shares its prefix with the indexer lease owner (host:pid:nonce), so
      "which process holds the lease" resolves to a process you have signals for.

    *role* names the entry point (``mcp``, ``daemon``, ``index``, ``watch``, ``search``,
    ``web``). *indexing* is separate from it on purpose: an MCP server started without
    ``--no-index`` runs the watcher and pipeline itself, so role alone cannot answer
    "who is actually indexing this checkout" — the question you ask first when nothing
    is being indexed, or when two things are.
    """
    global _initialized, _enabled, _metrics  # noqa: PLW0603

    if _initialized:
        return
    _initialized = True

    if not settings.enabled or not _HAS_OTEL:
        logger.debug("Telemetry disabled (enabled={}, otel_installed={})", settings.enabled, _HAS_OTEL)
        return

    _enabled = True

    from opentelemetry import metrics as _otel_metrics  # noqa: PLC0415
    from opentelemetry import trace as _otel_trace  # noqa: PLC0415
    from opentelemetry.sdk.metrics import MeterProvider  # noqa: PLC0415
    from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
    from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased  # noqa: PLC0415

    attributes: dict[str, Any] = {
        "service.name": settings.service_name,
        "service.version": _get_version(),
        # Several atlas processes report at once by design -- a daemon plus one MCP
        # server per agent session per worktree. Without an instance id their signals
        # merge into one indistinguishable stream.
        "service.instance.id": f"{socket.gethostname()}:{os.getpid()}",
        "host.name": socket.gethostname(),
        "process.pid": os.getpid(),
    }
    if role:
        attributes["atlas.role"] = role
    if project:
        attributes["atlas.project"] = project
    if root:
        attributes["atlas.root"] = root
    if indexing is not None:
        attributes["atlas.indexing"] = "on" if indexing else "off"
    resource = Resource.create(attributes)

    # Tracer provider
    sampler = TraceIdRatioBased(settings.sample_rate)
    tracer_provider = TracerProvider(resource=resource, sampler=sampler)

    span_exporter = _build_span_exporter(settings)
    if span_exporter is not None:
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: PLC0415

        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))

    _otel_trace.set_tracer_provider(tracer_provider)

    # Meter provider
    metric_reader = _build_metric_reader(settings)
    readers = [metric_reader] if metric_reader is not None else []
    meter_provider = MeterProvider(resource=resource, metric_readers=readers)
    _otel_metrics.set_meter_provider(meter_provider)

    # Create metric instruments
    meter = _otel_metrics.get_meter("code_atlas")
    _metrics = _Metrics(
        query_count=meter.create_counter("atlas_query_count", description="Total search queries"),
        query_latency=meter.create_histogram("atlas_query_latency_seconds", description="Query latency", unit="s"),
        search_results_count=meter.create_histogram(
            "atlas_search_results_count", description="Search results per query"
        ),
        index_files_total=meter.create_counter("atlas_index_files_total", description="Total files indexed"),
        index_entities_total=meter.create_counter("atlas_index_entities_total", description="Total entities indexed"),
        index_duration=meter.create_histogram(
            "atlas_index_duration_seconds", description="Index operation duration", unit="s"
        ),
        embedding_latency=meter.create_histogram(
            "atlas_embedding_latency_seconds", description="Embedding API latency", unit="s"
        ),
        tool_calls=meter.create_counter("atlas_mcp_tool_calls", description="MCP tool invocations"),
        tool_latency=meter.create_histogram(
            "atlas_mcp_tool_latency_seconds", description="MCP tool wall time", unit="s"
        ),
        batches_processed=meter.create_counter(
            "atlas_pipeline_batches", description="Consumer batches, by topic and outcome"
        ),
        events_consumed=meter.create_counter("atlas_pipeline_events_consumed", description="Events pulled off a topic"),
        watcher_events=meter.create_counter(
            "atlas_watcher_events", description="File changes published by the watcher"
        ),
        embeddings_total=meter.create_counter(
            "atlas_embeddings_total", description="Entity embeddings, by where the vector came from"
        ),
        web_requests=meter.create_counter("atlas_web_requests", description="Web UI requests"),
        web_latency=meter.create_histogram(
            "atlas_web_latency_seconds", description="Web UI request wall time", unit="s"
        ),
    )

    # A gauge, because backlog is the one pipeline number that goes down as well as up.
    # Sourced from a cache rather than queried in the callback: the depth comes from an
    # async XINFO round-trip and observable-gauge callbacks are synchronous.
    meter.create_observable_gauge(
        "atlas_pipeline_backlog",
        callbacks=[_observe_backlog],
        description="Unprocessed events per topic (pending + lag)",
    )

    if settings.export_logs:
        _init_log_export(settings, resource)

    logger.info(
        "Telemetry initialized (exporter={}, sample_rate={}, logs={})",
        settings.exporter,
        settings.sample_rate,
        settings.log_level if settings.export_logs else "off",
    )


# Queue depth per topic, refreshed by the daemon's sampler. "How far behind is the
# pipeline" was previously answerable only by hand, with redis-cli against Valkey.
_backlog: dict[str, int] = {}


def is_enabled() -> bool:
    """Whether telemetry is initialized and exporting.

    Lets a caller skip work that only exists to feed telemetry — the daemon's backlog
    sampler is an XINFO round-trip on a timer, pointless when nothing is collecting.
    """
    return _enabled


def set_backlog(topic: str, pending: int) -> None:
    """Publish the current queue depth for *topic* to the backlog gauge."""
    _backlog[topic] = pending


def _observe_backlog(options: Any) -> list[Any]:  # noqa: ARG001
    from opentelemetry.metrics import Observation  # noqa: PLC0415

    return [Observation(value, {"topic": topic}) for topic, value in _backlog.items()]


def _init_log_export(settings: ObservabilitySettings, resource: Any) -> None:
    """Route loguru through OTLP so logs land beside the spans that produced them.

    Traces say a step ran and how long it took; the log line says *what it decided*
    -- "skipping startup catch-up", "lease lost while still indexing", "4 of 6 vector
    indices missing". Diagnosing this pipeline has repeatedly meant reading those
    lines, and until now they existed only on the stderr of whichever process emitted
    them. Under MCP that stderr belongs to the agent client's subprocess plumbing and
    reaches no file at all, so the record was simply unavailable after the fact.

    Bridged via OTel's ``LoggingHandler``, which loguru accepts directly as a sink.
    It stamps each record with the active span's trace and span id, so a log line and
    the span it happened inside are joined without any correlation id of our own.
    """
    global _log_sink_id, _logger_provider  # noqa: PLW0603

    exporter = _build_log_exporter(settings)
    if exporter is None:
        return

    from opentelemetry._logs import set_logger_provider  # noqa: PLC0415
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler  # noqa: PLC0415
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor  # noqa: PLC0415

    provider = LoggerProvider(resource=resource)
    provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    set_logger_provider(provider)
    _logger_provider = provider

    # A second sink, not a replacement: the terminal/stderr sink the CLI installs stays
    # exactly as it was. Level is separate from the console level on purpose -- DEBUG is
    # useful locally and ruinous as remote volume across a 60k-file index.
    _log_sink_id = logger.add(
        LoggingHandler(logger_provider=provider),
        level=settings.log_level,
        format="{message}",
    )


def _build_log_exporter(settings: ObservabilitySettings) -> Any:
    """Build a log-record exporter based on settings, or ``None``."""
    if settings.exporter == "none":
        return None
    if settings.exporter == "console":
        from opentelemetry.sdk._logs.export import ConsoleLogExporter  # noqa: PLC0415

        return ConsoleLogExporter()
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter  # noqa: PLC0415

    return OTLPLogExporter(endpoint=settings.endpoint)


def mark_span_error(span: Any, exc: BaseException | None = None, description: str = "") -> None:
    """Flag *span* as failed, recording *exc* when given.

    Exists so call sites can set a real OTel status without importing OTel. Every
    ``opentelemetry`` import in this package is contained in this module, and a span
    that merely carries an ``error=true`` attribute is invisible to the error filters
    in Grafana and the Jaeger UI -- they read span status, not attributes.
    """
    if exc is not None:
        span.record_exception(exc)
    if not (_HAS_OTEL and _enabled):
        span.set_status(None, description or None)
        return
    from opentelemetry.trace import Status, StatusCode  # noqa: PLC0415

    span.set_status(Status(StatusCode.ERROR, description or (str(exc) if exc else None)))


def shutdown_telemetry() -> None:
    """Flush and shut down OTel providers. Safe to call even when not initialized."""
    global _initialized, _enabled, _log_sink_id, _logger_provider  # noqa: PLW0603

    if not _initialized or not _enabled or not _HAS_OTEL:
        return

    from opentelemetry import metrics as _otel_metrics  # noqa: PLC0415
    from opentelemetry import trace as _otel_trace  # noqa: PLC0415

    # Detach the loguru sink before the provider dies, not after: shutdown itself logs,
    # and a sink pointing at a shut-down provider turns those last lines into handler
    # errors on stderr — the noisiest possible way to end a clean run.
    if _log_sink_id is not None:
        with contextlib.suppress(ValueError):
            logger.remove(_log_sink_id)
        _log_sink_id = None
    if _logger_provider is not None:
        _logger_provider.shutdown()
        _logger_provider = None

    tp = _otel_trace.get_tracer_provider()
    if hasattr(tp, "shutdown"):
        tp.shutdown()  # type: ignore[call-non-callable]  # runtime hasattr guard

    mp = _otel_metrics.get_meter_provider()
    if hasattr(mp, "shutdown"):
        mp.shutdown()  # type: ignore[call-non-callable]  # runtime hasattr guard

    _initialized = False
    _enabled = False
    logger.debug("Telemetry shut down")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_version() -> str:
    """Best-effort version string."""
    try:
        from importlib.metadata import version  # noqa: PLC0415

        return version("code-atlas")
    except Exception:
        return "0.0.0-dev"


def _build_span_exporter(settings: ObservabilitySettings) -> Any:
    """Build a span exporter based on settings, or ``None``."""
    if settings.exporter == "none":
        return None
    if settings.exporter == "console":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter  # noqa: PLC0415

        return ConsoleSpanExporter()
    # Default: OTLP gRPC
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # noqa: PLC0415

    return OTLPSpanExporter(endpoint=settings.endpoint)


def _build_metric_reader(settings: ObservabilitySettings) -> Any:
    """Build a metric reader based on settings, or ``None``."""
    if settings.exporter == "none":
        return None
    if settings.exporter == "console":
        from opentelemetry.sdk.metrics.export import (  # noqa: PLC0415
            ConsoleMetricExporter,
            PeriodicExportingMetricReader,
        )

        return PeriodicExportingMetricReader(ConsoleMetricExporter())
    # Default: OTLP gRPC
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter  # noqa: PLC0415
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader  # noqa: PLC0415

    return PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=settings.endpoint))
