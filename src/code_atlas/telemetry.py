"""OpenTelemetry integration for Code Atlas.

All OTel dependencies are **optional** (``[otel]`` extra).  When not installed,
every public function returns lightweight no-op stubs so the rest of the
codebase can instrument unconditionally with zero overhead.
"""

from __future__ import annotations

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


_metrics = _Metrics()


def get_metrics() -> _Metrics:
    """Return the centralized metrics namespace."""
    return _metrics


# ---------------------------------------------------------------------------
# Initialization / shutdown
# ---------------------------------------------------------------------------


def init_telemetry(settings: ObservabilitySettings) -> None:
    """Configure OTel providers and instruments based on *settings*.

    Safe to call multiple times — only the first call has effect.
    When OTel packages are not installed this is a no-op.
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

    resource = Resource.create(
        {
            "service.name": settings.service_name,
            "service.version": _get_version(),
        }
    )

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
    )

    logger.info("Telemetry initialized (exporter={}, sample_rate={})", settings.exporter, settings.sample_rate)


def shutdown_telemetry() -> None:
    """Flush and shut down OTel providers. Safe to call even when not initialized."""
    global _initialized, _enabled  # noqa: PLW0603

    if not _initialized or not _enabled or not _HAS_OTEL:
        return

    from opentelemetry import metrics as _otel_metrics  # noqa: PLC0415
    from opentelemetry import trace as _otel_trace  # noqa: PLC0415

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
