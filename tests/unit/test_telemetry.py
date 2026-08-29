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
        from code_atlas.telemetry import _NoOpSpan, _NoOpTracer, get_tracer

        tracer = get_tracer("test.module")
        assert isinstance(tracer._resolve(), _NoOpTracer)
        assert isinstance(tracer.start_as_current_span("test"), _NoOpSpan)

    def test_get_meter_returns_noop_when_not_enabled(self) -> None:
        from code_atlas.telemetry import _NoOpCounter, _NoOpHistogram, _NoOpMeter, get_meter

        meter = get_meter("test.module")
        assert isinstance(meter._resolve(), _NoOpMeter)
        assert isinstance(meter.create_counter("test"), _NoOpCounter)
        assert isinstance(meter.create_histogram("test"), _NoOpHistogram)

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


class _StubTracer:
    """Stands in for an OTel ``Tracer`` so this runs without the ``[otel]`` extra."""

    def __init__(self, name: str) -> None:
        self.name = name


class _StubApi:
    """Stands in for ``opentelemetry.trace`` / ``opentelemetry.metrics``."""

    def get_tracer(self, name: str) -> _StubTracer:
        return _StubTracer(name)

    def get_meter(self, name: str) -> _StubTracer:
        return _StubTracer(name)


class TestLazyResolution:
    """Handles bound at import time must still trace once telemetry is switched on.

    This is the shape of the real defect: every module does
    ``_tracer = get_tracer(__name__)`` at import, and every entry point imports its
    dependencies before it calls ``init_telemetry``. A factory that decided on
    ``_enabled`` at *call* time therefore handed out permanent no-ops, and all ~25
    span sites in the package silently emitted nothing with telemetry fully enabled.

    Stubbing the OTel boundary rather than skipping without it is deliberate: the
    defect was in this module's binding logic, not in OTel, so the test that pins it
    must run on every CI job -- including the ones without the extra installed.
    """

    def test_tracer_bound_before_init_resolves_after(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import code_atlas.telemetry as mod

        tracer = mod.get_tracer("bound.early")
        assert isinstance(tracer._resolve(), mod._NoOpTracer), "must be inert while disabled"

        monkeypatch.setattr(mod, "_HAS_OTEL", True)
        monkeypatch.setattr(mod, "otel_trace", _StubApi(), raising=False)
        monkeypatch.setattr(mod, "_enabled", True)

        resolved = tracer._resolve()
        assert isinstance(resolved, _StubTracer), "a handle taken before init stayed a no-op"
        assert resolved.name == "bound.early"
        assert tracer._resolve() is resolved, "resolution should be cached, not re-derived per span"

    def test_meter_bound_before_init_resolves_after(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import code_atlas.telemetry as mod

        meter = mod.get_meter("bound.early")
        assert isinstance(meter._resolve(), mod._NoOpMeter)

        monkeypatch.setattr(mod, "_HAS_OTEL", True)
        monkeypatch.setattr(mod, "otel_metrics", _StubApi(), raising=False)
        monkeypatch.setattr(mod, "_enabled", True)

        assert isinstance(meter._resolve(), _StubTracer)

    def test_module_level_tracers_are_lazy(self) -> None:
        """The regression is only prevented if the real modules use the lazy handle.

        A future refactor that resolves eagerly again would leave these as plain
        ``_NoOpTracer`` instances and reintroduce the silent hole.
        """
        import code_atlas.events
        import code_atlas.graph.client
        import code_atlas.indexing.consumers
        import code_atlas.search.engine
        import code_atlas.telemetry as mod

        for module in (
            code_atlas.events,
            code_atlas.graph.client,
            code_atlas.indexing.consumers,
            code_atlas.search.engine,
        ):
            assert isinstance(module._tracer, mod._LazyTracer), f"{module.__name__} binds a frozen tracer"


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
