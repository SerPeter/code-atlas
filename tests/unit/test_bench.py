"""The bench report keeps work, contention and pacing apart, and never sums overlaps.

The arithmetic here is the whole point of the module, so it is tested directly rather
than only through an indexing run: a report that adds two concurrent consumers together
produces a number larger than the run it describes, and the resulting negative idle time
is the kind of thing that reads as plausible on a table.
"""

from __future__ import annotations

import pytest

from code_atlas.bench import (
    BenchReport,
    ConsumerBreakdown,
    PhaseTiming,
    classify,
    render,
    stub_provider,
)


def _phase(stage: str, phase: str, seconds: float, units: int = 0, unit_name: str = "", calls: int = 1) -> PhaseTiming:
    return PhaseTiming(stage=stage, phase=phase, seconds=seconds, calls=calls, units=units, unit_name=unit_name)


class TestClassification:
    def test_the_three_lines_are_distinct(self):
        assert classify("ast", "parse") == "work"
        assert classify("embed", "write") == "work"
        assert classify("embed", "write_lock_wait") == "contention"
        assert classify("embed", "provider") == "external"

    def test_an_unknown_phase_counts_as_work(self):
        """Unknown means "someone added a phase and nobody classified it".

        Counting it as work is the conservative direction: it shows up in the number
        being compared rather than hiding in a line nobody reads.
        """
        assert classify("ast", "something_new") == "work"


class TestReportArithmetic:
    def test_concurrent_consumers_are_not_summed(self):
        """The guard that keeps the report from claiming more time than the run took.

        Two consumers, 6s and 4s, inside a 10s run. Summing gives 10s accounted and 0s
        pacing, which would be wrong twice over — they overlap, so the real floor is the
        slower one at 6s, leaving 4s unaccounted.
        """
        report = BenchReport(
            total_s=10.0,
            consumers=(
                ConsumerBreakdown("ast", (_phase("ast", "parse", 6.0),)),
                ConsumerBreakdown("embed", (_phase("embed", "write", 4.0),)),
            ),
        )
        assert report.accounted_s == 6.0, "accounted must be the slowest consumer, not the sum"
        assert report.unaccounted_s == 4.0

    def test_unaccounted_never_goes_negative(self):
        """Clock skew or a span that outlives its run must not print a negative wait."""
        report = BenchReport(
            total_s=1.0,
            consumers=(ConsumerBreakdown("ast", (_phase("ast", "parse", 5.0),)),),
        )
        assert report.unaccounted_s == 0.0

    def test_the_three_classes_are_summed_apart(self):
        consumer = ConsumerBreakdown(
            "embed",
            (
                _phase("embed", "write", 1.0),
                _phase("embed", "write_lock_wait", 2.0),
                _phase("embed", "provider", 4.0),
            ),
        )
        assert consumer.work_s == 1.0
        assert consumer.contention_s == 2.0
        assert consumer.external_s == 4.0
        assert consumer.accounted_s == 7.0

    def test_round_trips_aggregate_across_kinds(self):
        report = BenchReport(
            total_s=1.0,
            consumers=(),
            round_trips={("upsert", "write"): 3, ("upsert", "write_tx"): 4, ("read_x", "read"): 2},
        )
        assert report.round_trips_by_op() == {"upsert": 7, "read_x": 2}
        assert report.total_round_trips == 9

    def test_work_counters_are_the_reproducible_half(self):
        """What two runs are compared on. Times are excluded deliberately."""
        report = BenchReport(
            total_s=9.9,
            consumers=(ConsumerBreakdown("ast", (_phase("ast", "parse", 1.234, units=10, unit_name="files"),)),),
            round_trips={("op_a", "read"): 5},
        )
        counters = report.work_counters()
        assert counters["ast.parse.files"] == 10
        assert counters["rt.op_a"] == 5
        assert not any(isinstance(v, float) for v in counters.values()), "a duration leaked into the counters"


class TestPhaseRate:
    def test_a_phase_without_a_unit_has_no_rate(self):
        assert _phase("embed", "write_lock_wait", 1.0).rate is None

    def test_a_zero_duration_phase_has_no_rate(self):
        assert _phase("ast", "parse", 0.0, units=5, unit_name="files").rate is None

    def test_rate_is_units_per_second(self):
        assert _phase("ast", "parse", 2.0, units=10, unit_name="files").rate == 5.0


class TestRender:
    def test_the_table_names_all_three_lines(self):
        report = BenchReport(
            total_s=10.0,
            consumers=(
                ConsumerBreakdown(
                    "ast",
                    (
                        _phase("ast", "parse", 1.0, units=10, unit_name="files"),
                        _phase("ast", "resolve", 2.0, units=40, unit_name="rels"),
                    ),
                ),
            ),
            round_trips={("upsert_entities", "write_tx"): 12},
            corpus="tiny",
            backend="sqlite",
        )
        out = render(report)
        assert "tiny" in out
        assert "sqlite" in out
        assert "pacing + gaps" in out
        assert "contention" in out
        assert "10 files" in out
        assert "upsert_entities" in out
        assert "consumers overlap" in out, "the table must say why accounted is not a sum"

    def test_an_empty_run_renders_without_raising(self):
        assert "wall clock" in render(BenchReport(total_s=0.0, consumers=()))


class TestProviderStub:
    """The one place a benchmark substitutes rather than observes."""

    async def test_one_vector_per_text(self):
        """Arity is the property worth guarding.

        The `AsyncMock` this replaces returned a single vector for any batch size, so a
        change that sent the wrong number of texts would have looked fine.
        """
        from code_atlas.search.embeddings import EmbedClient

        with stub_provider(dimension=16) as stats:
            response = await EmbedClient._embed_call(None, {"input": ["a", "b", "c"]})  # ty: ignore[invalid-argument-type]  # the stub ignores self

        assert len(response.data) == 3
        assert all(len(item["embedding"]) == 16 for item in response.data)
        assert stats.calls == 1
        assert stats.texts == 3

    async def test_the_same_text_yields_the_same_vector(self):
        """Determinism is not cosmetic.

        The graph is the dedup layer (ADR-0036) and the freshness check compares hashes
        of the embedded text. A stub returning random vectors would make those paths
        behave differently under benchmark than in production, so the stage would stop
        measuring what it claims to.
        """
        from code_atlas.search.embeddings import EmbedClient

        with stub_provider(dimension=8):
            first = await EmbedClient._embed_call(None, {"input": ["same"]})  # ty: ignore[invalid-argument-type]  # the stub ignores self
            second = await EmbedClient._embed_call(None, {"input": ["same", "other"]})  # ty: ignore[invalid-argument-type]  # the stub ignores self

        assert first.data[0]["embedding"] == second.data[0]["embedding"]
        assert second.data[0]["embedding"] != second.data[1]["embedding"]

    async def test_the_real_method_is_restored(self):
        from code_atlas.search.embeddings import EmbedClient

        original = EmbedClient._embed_call
        with stub_provider(dimension=8):
            assert EmbedClient._embed_call is not original
        assert EmbedClient._embed_call is original

    async def test_it_is_restored_even_when_the_body_raises(self):
        from code_atlas.search.embeddings import EmbedClient

        original = EmbedClient._embed_call
        with pytest.raises(RuntimeError), stub_provider(dimension=8):
            raise RuntimeError("boom")
        assert EmbedClient._embed_call is original

    async def test_an_empty_batch_produces_no_vectors(self):
        from code_atlas.search.embeddings import EmbedClient

        with stub_provider(dimension=8) as stats:
            response = await EmbedClient._embed_call(None, {"input": []})  # ty: ignore[invalid-argument-type]  # the stub ignores self

        assert response.data == []
        assert stats.texts == 0
