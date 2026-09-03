"""The SQLite backend reports on the same instruments the Memgraph client does.

Until ATL-158 `backends/` had no telemetry at all: 4,332 lines, no spans, no metrics,
no counters. These tests pin the three properties that make the new instrumentation
worth having — correct attribution, honest separation of our SQL from SQLite's own, and
a scan detector that discriminates rather than alarms — plus the one that keeps it from
costing anything when telemetry is off.
"""

from __future__ import annotations

import importlib

import pytest

from code_atlas.backends.instrumentation import (
    IN_TX,
    StatementLog,
    TimedConnection,
    analyse_scans,
    capture_statements,
    in_transaction,
    statement_kind,
)
from code_atlas.backends.sqlite_graph import SqliteGraphClient

try:
    importlib.import_module("opentelemetry.sdk")
    _has_otel_sdk = True
except ModuleNotFoundError:  # pragma: no cover - exercised by the skip
    _has_otel_sdk = False


def _samples_by_op(reader) -> dict[tuple[str, str], int]:
    """Sample counts on ``atlas_graph_query_seconds``, keyed by (op, kind).

    The *count* is the point, not the sum: a histogram's sample count per label is the
    round-trip count, which is what makes this comparable across backends and stable
    across machines.
    """
    out: dict[tuple[str, str], int] = {}
    data = reader.get_metrics_data()
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name != "atlas_graph_query_seconds":
                    continue
                for point in metric.data.data_points:
                    key = (point.attributes.get("op"), point.attributes.get("kind"))
                    out[key] = out.get(key, 0) + point.count
    return out


@pytest.fixture
def metric_reader(monkeypatch: pytest.MonkeyPatch):
    """Install an in-memory meter and enable telemetry for one test.

    Deliberately does not call ``init_telemetry``: that would build its own meter
    provider and detach this reader, and because spans would keep working the failure
    would present as "metrics stopped, traces fine" rather than as an error.
    """
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    import code_atlas.telemetry as tel

    reader = InMemoryMetricReader()
    meter = MeterProvider(metric_readers=[reader]).get_meter("code_atlas")
    monkeypatch.setattr(
        tel,
        "_metrics",
        tel._Metrics(graph_query_seconds=meter.create_histogram("atlas_graph_query_seconds", unit="s")),
    )
    monkeypatch.setattr(tel, "_enabled", True)
    monkeypatch.setattr(tel, "_initialized", True)
    return reader


@pytest.mark.skipif(not _has_otel_sdk, reason="opentelemetry-sdk not installed")
class TestAttribution:
    async def test_a_query_is_attributed_to_the_method_that_issued_it(self, tmp_path, metric_reader):
        """`op` must name the SqliteGraphClient method, not a plumbing frame.

        The failure this guards against is not "no data" but *plausible* data: before
        the proxy's own pass-throughs were added to the skip set, every captured
        statement was attributed to `_record`, which reads like a real answer on a
        dashboard and tells you nothing.
        """
        async with SqliteGraphClient(tmp_path / "g.sqlite3", dimension=8) as client:
            await client.ensure_schema()
            await client.merge_project_node("testproj")
            await client.count_entities("testproj")

        ops = _samples_by_op(metric_reader)
        assert ("count_entities", "read") in ops, f"count_entities was not attributed: {sorted(ops)}"
        assert ("merge_project_node", "write") in ops
        assert not any(op in {"execute", "_record", "_timed_sql", "_get_conn", "unknown"} for op, _ in ops), (
            f"a statement was attributed to plumbing rather than to its caller: {sorted(ops)}"
        )

    async def test_reads_and_writes_carry_different_kinds(self, tmp_path, metric_reader):
        async with SqliteGraphClient(tmp_path / "g.sqlite3", dimension=8) as client:
            await client.ensure_schema()
            await client.merge_project_node("testproj")

        kinds = {kind for _, kind in _samples_by_op(metric_reader)}
        assert "read" in kinds, f"no read was recorded: {kinds}"
        assert "write" in kinds, f"no write was recorded: {kinds}"

    async def test_a_commit_is_its_own_line(self, tmp_path, metric_reader):
        """Commit is the statement that reaches the disk, so it is not blended in."""
        async with SqliteGraphClient(tmp_path / "g.sqlite3", dimension=8) as client:
            await client.ensure_schema()
            await client.merge_project_node("testproj")

        assert any(op == "commit" for op, _ in _samples_by_op(metric_reader))


@pytest.mark.skipif(not _has_otel_sdk, reason="opentelemetry-sdk not installed")
class TestCostWhenTelemetryIsOff:
    async def test_nothing_is_recorded_and_nothing_is_traced(self, tmp_path, monkeypatch):
        """The guard is the whole reason this is affordable at 213 call sites.

        Asserted by proving the stack walk never runs: `caller_name` is what makes an
        instrumented statement cost more than a boolean check, and with telemetry off it
        must not be reached at all.
        """
        import code_atlas.telemetry as tel

        monkeypatch.setattr(tel, "_enabled", False)
        calls: list[object] = []
        monkeypatch.setattr(
            "code_atlas.backends.instrumentation.caller_name",
            lambda *a, **k: calls.append(1) or "x",
        )

        async with SqliteGraphClient(tmp_path / "g.sqlite3", dimension=8) as client:
            await client.ensure_schema()
            await client.merge_project_node("testproj")

        assert calls == [], "the stack walk ran with telemetry disabled"


class TestTransactionKind:
    def test_kind_reflects_the_transaction_boundary(self):
        assert statement_kind("SELECT 1") == "read"
        assert statement_kind("  insert into t values (1)") == "write"

    async def test_in_transaction_marks_and_restores(self):
        assert IN_TX.get() is False
        async with in_transaction():
            assert statement_kind("SELECT 1") == "read_tx"
            assert statement_kind("UPDATE t SET a = 1") == "write_tx"
        assert IN_TX.get() is False, "the marker leaked past its block"


class TestOwnershipIsRespected:
    async def test_an_injected_connection_is_not_wrapped(self, tmp_path):
        """ADR-0038 applies to more than closing.

        A caller that hands over a connection gets that object back from `_get_conn`,
        not a decorated stand-in it never asked for. The first cut of this
        instrumentation wrapped injected connections too and broke
        `test_injected_fake_connection_is_used_directly`, which is exactly the contract
        this restates from the instrumentation side.
        """
        from unittest.mock import AsyncMock

        fake = AsyncMock()
        client = SqliteGraphClient(tmp_path / "unused.sqlite3", conn=fake)
        assert await client._get_conn() is fake

    async def test_a_self_opened_connection_is_wrapped(self, tmp_path):
        async with SqliteGraphClient(tmp_path / "g.sqlite3", dimension=8) as client:
            assert isinstance(await client._get_conn(), TimedConnection)


class TestStatementCapture:
    async def test_our_sql_is_separated_from_sqlites_own(self, tmp_path):
        """An entity write drags FTS5 shadow-table maintenance behind it.

        A single total cannot tell "our query regressed" from "the full-text index is
        merging segments", so the two are counted apart. The assertion is that the
        trace sees at least what we issued — SQLite never runs fewer statements than it
        was asked for.
        """
        async with SqliteGraphClient(tmp_path / "g.sqlite3", dimension=8) as client:
            await client.ensure_schema()
            await client.merge_project_node("testproj")
            conn = await client._get_conn()
            async with capture_statements(conn) as log:
                await client.count_entities("testproj")
                await client.get_batch_file_hashes("testproj", ["a.py"])

        summary = log.summary()
        assert summary["app_statements"] == 2, summary
        assert summary["traced_statements"] >= summary["app_statements"], summary
        assert summary["internal_statements"] == len(log.internal)

    async def test_the_capture_stops_at_the_block_boundary(self, tmp_path):
        async with SqliteGraphClient(tmp_path / "g.sqlite3", dimension=8) as client:
            await client.ensure_schema()
            conn = await client._get_conn()
            async with capture_statements(conn) as log:
                await client.count_entities("testproj")
            before = len(log.app)
            await client.count_entities("testproj")

        assert len(log.app) == before, "statements were recorded after the capture ended"


class TestScanDetection:
    async def test_it_discriminates_rather_than_flagging_everything(self, tmp_path):
        """Bidirectional on purpose.

        A detector that reports every statement as a scan would satisfy a one-sided
        assertion while being useless, so a known scanning query has to be caught in the
        same run that a known index-seeking one comes back clean.

        The scanning side is a **control statement written here**, not a product query.
        It used to be `_get_batch_file_prop` — the file-hash gate — because that genuinely
        scanned: every node index was partial (`WHERE labels = '<Label>'`) and its
        `labels IN (...)` predicate could use none of them. Adding the two unqualified
        indices fixed that, and this test failed, which is the suite working. But it
        showed the test had been resting on a product defect as its fixture, so the next
        fix would break it again. A `json_extract` on a key nobody indexes cannot be
        optimised away by any schema change, so it stays a scan on purpose.
        """
        async with SqliteGraphClient(tmp_path / "g.sqlite3", dimension=8) as client:
            await client.ensure_schema()
            await client.merge_project_node("testproj")
            conn = await client._get_conn()
            async with capture_statements(conn) as log:
                await client.get_batch_file_hashes("testproj", ["a.py", "b.py"])
                await client._nodes_by_uid(conn, ["testproj:module:a"])
                # Through the proxy, not `conn.raw`: capture happens in the wrapper, so a
                # raw execute is never recorded and the control would be silently absent.
                cur = await conn.execute(
                    "SELECT uid FROM nodes WHERE json_extract(props_json, '$.unindexed_control') = ?", ("x",)
                )
                await cur.close()
            report = await analyse_scans(conn, log)

        scanning_ops = {op for op, _, _ in report.scanning}
        assert scanning_ops, f"the control statement was not flagged — the detector is silent: {report.summary()}"
        assert report.non_scanning >= 1, (
            f"no statement came back clean — the detector is flagging everything: {report.summary()}"
        )
        assert "_nodes_by_uid" not in scanning_ops, "a uid lookup should seek its index, not scan"
        assert "_get_batch_file_prop" not in scanning_ops, (
            "the file-hash gate is scanning again — ix_nodes_labels_project_name is gone or no longer applies, "
            f"and every unchanged file now costs a walk of the whole nodes table: {report.summary()}"
        )

    async def test_amplification_is_executions_times_rows(self):
        from code_atlas.backends.instrumentation import ScanReport

        report = ScanReport(node_rows=1000, scanning=(("op_a", "SELECT 1", 3),), non_scanning=2)
        assert report.scanning_executions == 3
        assert report.amplification == 3000
        assert report.summary()["scan_amplification"] == 3000

    async def test_an_empty_log_reports_nothing_rather_than_failing(self, tmp_path):
        async with SqliteGraphClient(tmp_path / "g.sqlite3", dimension=8) as client:
            await client.ensure_schema()
            report = await analyse_scans(await client._get_conn(), StatementLog())

        assert report.scanning == ()
        assert report.amplification == 0
