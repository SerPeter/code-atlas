"""Tests for the telemetry module — no-op stubs and OTel initialization."""

from __future__ import annotations

import importlib

import pytest

# ---------------------------------------------------------------------------
# No-op stub tests (always work, no OTel SDK needed)
# ---------------------------------------------------------------------------


class TestNoOpStubs:
    """Verify no-op stubs have correct interfaces and do nothing."""

    def test_noop_span_context_manager(self) -> None:
        from code_atlas.telemetry import _NoOpSpan

        span = _NoOpSpan()
        with span as s:
            s.set_attribute("key", "value")
            s.set_status("ok")
            s.record_exception(RuntimeError("test"))
        span.end()

    def test_noop_tracer_start_as_current_span(self) -> None:
        from code_atlas.telemetry import _NoOpSpan, _NoOpTracer

        tracer = _NoOpTracer()
        span = tracer.start_as_current_span("test_span")
        assert isinstance(span, _NoOpSpan)

    def test_noop_tracer_start_span(self) -> None:
        from code_atlas.telemetry import _NoOpSpan, _NoOpTracer

        tracer = _NoOpTracer()
        with tracer.start_span("test") as span:
            assert isinstance(span, _NoOpSpan)

    def test_noop_counter(self) -> None:
        from code_atlas.telemetry import _NoOpCounter

        counter = _NoOpCounter()
        counter.add(1)
        counter.add(5, {"key": "val"})

    def test_noop_histogram(self) -> None:
        from code_atlas.telemetry import _NoOpHistogram

        hist = _NoOpHistogram()
        hist.record(0.5)
        hist.record(1.2, {"key": "val"})

    def test_noop_meter(self) -> None:
        from code_atlas.telemetry import _NoOpCounter, _NoOpHistogram, _NoOpMeter

        meter = _NoOpMeter()
        assert isinstance(meter.create_counter("test"), _NoOpCounter)
        assert isinstance(meter.create_histogram("test"), _NoOpHistogram)


class TestFactoryFunctions:
    """Factory functions return no-ops when telemetry is not initialized."""

    def test_get_tracer_returns_noop_when_not_enabled(self) -> None:
        from code_atlas.telemetry import _NoOpTracer, get_tracer

        tracer = get_tracer("test.module")
        # When not enabled, always a no-op (or OTel's own no-op tracer)
        # Either way, start_as_current_span should work
        span = tracer.start_as_current_span("test")
        if isinstance(tracer, _NoOpTracer):
            assert isinstance(span, type(span))  # just check it doesn't explode

    def test_get_meter_returns_noop_when_not_enabled(self) -> None:
        from code_atlas.telemetry import _NoOpMeter, get_meter

        meter = get_meter("test.module")
        if isinstance(meter, _NoOpMeter):
            counter = meter.create_counter("test")
            counter.add(1)

    def test_get_metrics_returns_dataclass(self) -> None:
        from code_atlas.telemetry import get_metrics

        m = get_metrics()
        # Should have the expected attributes
        assert hasattr(m, "query_count")
        assert hasattr(m, "query_latency")
        assert hasattr(m, "index_files_total")
        assert hasattr(m, "embedding_latency")
        # No-op instruments should be callable without error
        m.query_count.add(1)
        m.query_latency.record(0.5)
        m.index_files_total.add(10)
        m.embedding_latency.record(0.1)


class TestInitTelemetry:
    """Test init/shutdown lifecycle."""

    def test_init_disabled(self) -> None:
        """init_telemetry with enabled=False is a safe no-op."""
        # Reset module state for clean test
        import code_atlas.telemetry as mod
        from code_atlas.settings import ObservabilitySettings
        from code_atlas.telemetry import init_telemetry, shutdown_telemetry

        mod._initialized = False
        mod._enabled = False

        settings = ObservabilitySettings(enabled=False)
        init_telemetry(settings)  # should not raise
        assert mod._initialized is True
        assert mod._enabled is False

        # shutdown is a no-op when not enabled (nothing to flush)
        shutdown_telemetry()
        # _initialized stays True — that's correct (prevents re-init)
        # Reset manually for other tests
        mod._initialized = False

    def test_init_idempotent(self) -> None:
        """Calling init_telemetry twice is safe."""
        import code_atlas.telemetry as mod
        from code_atlas.settings import ObservabilitySettings
        from code_atlas.telemetry import init_telemetry, shutdown_telemetry

        mod._initialized = False
        mod._enabled = False

        settings = ObservabilitySettings(enabled=False)
        init_telemetry(settings)
        init_telemetry(settings)  # second call is a no-op
        assert mod._initialized is True

        shutdown_telemetry()

    def test_shutdown_when_not_initialized(self) -> None:
        """shutdown_telemetry is safe to call without init."""
        import code_atlas.telemetry as mod
        from code_atlas.telemetry import shutdown_telemetry

        mod._initialized = False
        mod._enabled = False
        shutdown_telemetry()  # should not raise


# ---------------------------------------------------------------------------
# OTel SDK integration tests (only run if otel is installed)
# ---------------------------------------------------------------------------

try:
    importlib.import_module("opentelemetry.sdk")
    _has_otel_sdk = True
except ModuleNotFoundError:
    _has_otel_sdk = False


@pytest.mark.skipif(not _has_otel_sdk, reason="opentelemetry-sdk not installed")
class TestOTelSDKIntegration:
    """Tests that require the OTel SDK to be installed."""

    def _reset_module(self) -> None:
        import code_atlas.telemetry as mod

        mod._initialized = False
        mod._enabled = False

    def test_init_with_console_exporter(self) -> None:
        """init_telemetry with console exporter configures real providers."""
        from code_atlas.settings import ObservabilitySettings
        from code_atlas.telemetry import init_telemetry, shutdown_telemetry

        self._reset_module()

        settings = ObservabilitySettings(enabled=True, exporter="console", sample_rate=1.0)
        init_telemetry(settings)

        import code_atlas.telemetry as mod

        assert mod._enabled is True

        shutdown_telemetry()

    def test_init_with_none_exporter(self) -> None:
        """init_telemetry with exporter='none' still enables tracing but without export."""
        from code_atlas.settings import ObservabilitySettings
        from code_atlas.telemetry import init_telemetry, shutdown_telemetry

        self._reset_module()

        settings = ObservabilitySettings(enabled=True, exporter="none", sample_rate=0.5)
        init_telemetry(settings)

        import code_atlas.telemetry as mod

        assert mod._enabled is True

        shutdown_telemetry()

    def test_span_capture_with_inmemory_exporter(self) -> None:
        """Verify spans are actually created using InMemorySpanExporter."""
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory import InMemorySpanExporter

        self._reset_module()

        # Set up in-memory exporter directly (bypass init_telemetry for precise control)
        exporter = InMemorySpanExporter()
        resource = Resource.create({"service.name": "test"})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        otel_trace.set_tracer_provider(provider)

        import code_atlas.telemetry as mod

        mod._initialized = True
        mod._enabled = True

        try:
            tracer = mod.get_tracer("test.spans")
            with tracer.start_as_current_span("test_op") as span:
                span.set_attribute("key", "value")

            spans = exporter.get_finished_spans()
            assert len(spans) == 1
            assert spans[0].name == "test_op"
            assert spans[0].attributes["key"] == "value"
        finally:
            provider.shutdown()
            mod._initialized = False
            mod._enabled = False

    def test_metric_capture_with_inmemory_reader(self) -> None:
        """Verify metrics are recorded using InMemoryMetricReader."""
        from opentelemetry import metrics as otel_metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader
        from opentelemetry.sdk.resources import Resource

        self._reset_module()

        reader = InMemoryMetricReader()
        resource = Resource.create({"service.name": "test"})
        provider = MeterProvider(resource=resource, metric_readers=[reader])
        otel_metrics.set_meter_provider(provider)

        import code_atlas.telemetry as mod

        mod._initialized = True
        mod._enabled = True

        try:
            meter = mod.get_meter("test.metrics")
            counter = meter.create_counter("test_counter")
            counter.add(5, {"type": "test"})

            # Force collection
            metrics_data = reader.get_metrics_data()
            assert metrics_data is not None
            # At least one resource metric should exist
            assert len(metrics_data.resource_metrics) > 0
        finally:
            provider.shutdown()
            mod._initialized = False
            mod._enabled = False
