"""Architecture snapshots and trend (ATL-121).

The trend's job is to answer "is this getting worse". Its most important behaviour is
therefore the one where it *refuses* to answer: a metric that moved because coverage
moved is not a codebase that decayed, and saying "worse" there would be exactly the
confident wrong answer this project keeps removing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import pytest

from code_atlas.server.architecture import ArchitectureMetrics
from code_atlas.server.architecture_history import (
    MAX_SNAPSHOTS,
    Snapshot,
    decode,
    encode,
    load,
    record,
    snapshot_from_metrics,
    trend,
)

if TYPE_CHECKING:
    from code_atlas.graph.protocol import GraphBackend


def _snap(propagation: float, *, modules: int = 100, at: str = "2026-08-08T00:00:00+00:00") -> Snapshot:
    return Snapshot(
        at=at,
        commit="abc123",
        modules=modules,
        edges=modules * 2,
        propagation_cost=propagation,
        core_size=0.1,
        largest_cycle=1,
        fan_in_gini=0.3,
    )


class _Graph:
    """Records what was written, so the round trip can be checked without a database."""

    def __init__(self, stored: Any = None):
        self.stored = stored
        self.writes: list[dict[str, Any]] = []

    async def get_project_status(self, project_name: str | None = None) -> list[dict[str, Any]]:
        return [{"n": {"uid": "demo", "architecture_snapshots": self.stored} if self.stored else {"uid": "demo"}}]

    async def update_project_metadata(self, project_name: str, **metadata: Any) -> None:
        self.writes.append(metadata)
        self.stored = metadata.get("architecture_snapshots", self.stored)

    async def close(self) -> None: ...


class TestSnapshotEncoding:
    def test_a_snapshot_survives_a_round_trip(self):
        original = _snap(0.084)

        assert decode(encode([original])) == [original]

    def test_coverage_travels_with_the_numbers(self):
        """Without it, "propagation rose" and "extraction improved" are indistinguishable."""
        original = Snapshot(
            at="2026-08-08T00:00:00+00:00",
            commit="c",
            modules=50,
            edges=100,
            propagation_cost=0.1,
            core_size=0.2,
            largest_cycle=3,
            fan_in_gini=0.4,
            skipped_languages=("cpp", "rust"),
        )

        [restored] = decode(encode([original]))

        assert restored.modules == 50
        assert restored.skipped_languages == ("cpp", "rust")

    def test_one_unreadable_entry_does_not_cost_the_rest(self):
        """A history is append-only across versions; tolerate what an older one wrote."""
        good = encode([_snap(0.1), _snap(0.2)])

        restored = decode([good[0], "not json at all", "{}", good[1]])

        assert [s.propagation_cost for s in restored] == [0.1, 0.2]

    def test_a_non_list_property_is_no_history_rather_than_a_crash(self):
        assert decode(None) == []
        assert decode("nonsense") == []

    def test_retention_is_bounded(self):
        """A daemon-indexed repo would otherwise grow this without limit."""
        many = [_snap(i / 1000) for i in range(MAX_SNAPSHOTS + 20)]

        stored = encode(many)

        assert len(stored) == MAX_SNAPSHOTS
        # The newest are kept, not the oldest.
        assert decode(stored)[-1].propagation_cost == many[-1].propagation_cost

    def test_metrics_become_a_snapshot(self):
        metrics = ArchitectureMetrics(
            module_count=12,
            edge_count=20,
            propagation_cost=0.0841234567,
            core_size=0.25,
            largest_cycle=3,
            fan_in_gini=0.5,
        )

        snapshot = snapshot_from_metrics(
            metrics, commit="deadbeef", skipped_languages=("go",), at=datetime(2026, 8, 8, tzinfo=UTC)
        )

        assert snapshot.modules == 12
        assert snapshot.propagation_cost == 0.084123
        assert snapshot.commit == "deadbeef"
        assert snapshot.at.startswith("2026-08-08")


class TestRecording:
    async def test_a_snapshot_is_appended_to_the_existing_history(self):
        graph = _Graph(stored=encode([_snap(0.1)]))

        written = await record(cast("GraphBackend", graph), "demo", _snap(0.2))

        assert written
        assert [s.propagation_cost for s in decode(graph.stored)] == [0.1, 0.2]

    async def test_the_first_snapshot_starts_a_history(self):
        graph = _Graph()

        await record(cast("GraphBackend", graph), "demo", _snap(0.1))

        assert len(decode(graph.stored)) == 1

    async def test_a_write_failure_never_propagates(self):
        """Telemetry about an index run must not be able to fail the run."""

        class _Broken(_Graph):
            async def update_project_metadata(self, project_name: str, **metadata: Any) -> None:
                raise RuntimeError("backend is down")

        written = await record(cast("GraphBackend", _Broken()), "demo", _snap(0.1))

        assert written is False, "the failure is reported as a return value, not raised"

    async def test_a_read_failure_never_propagates(self):
        class _Broken(_Graph):
            async def get_project_status(self, project_name: str | None = None) -> list[dict[str, Any]]:
                raise RuntimeError("backend is down")

        assert await record(cast("GraphBackend", _Broken()), "demo", _snap(0.1)) is False

    async def test_loading_a_project_with_no_history_is_empty_not_an_error(self):
        assert await load(cast("GraphBackend", _Graph()), "demo") == []


class TestTrend:
    def test_a_single_snapshot_is_not_a_trend(self):
        """A line through one point invents a direction the data does not contain."""
        assert trend([_snap(0.1)]) is None
        assert trend([]) is None

    def test_a_rising_propagation_cost_reads_as_worse(self):
        movement = trend([_snap(0.06), _snap(0.07), _snap(0.084)])

        assert movement is not None
        assert movement.direction == "worse"
        assert movement.propagation_delta == pytest.approx(0.024)

    def test_a_falling_propagation_cost_reads_as_better(self):
        movement = trend([_snap(0.17), _snap(0.02)])

        assert movement is not None
        assert movement.direction == "better"

    def test_a_negligible_change_reads_as_flat(self):
        movement = trend([_snap(0.100), _snap(0.1002)])

        assert movement is not None
        assert movement.direction == "flat"

    def test_a_coverage_change_makes_the_direction_unclear(self):
        """The refusal that matters.

        Propagation rose, but the graph grew by 60% — a language whose extraction improved
        adds real dependencies that were always there. Calling that decay would be wrong.
        """
        movement = trend([_snap(0.06, modules=100), _snap(0.12, modules=160)])

        assert movement is not None
        assert movement.coverage_changed
        assert movement.direction == "unclear"

    def test_a_small_coverage_drift_still_permits_a_verdict(self):
        """Every index run moves the module count slightly; that cannot mute the signal."""
        movement = trend([_snap(0.06, modules=100), _snap(0.12, modules=104)])

        assert movement is not None
        assert not movement.coverage_changed
        assert movement.direction == "worse"

    def test_the_window_bounds_what_is_compared(self):
        movement = trend([_snap(i / 100) for i in range(30)], window=5)

        assert movement is not None
        assert movement.count == 5
        assert movement.first.propagation_cost == 0.25
