"""Baselines: what makes a comparison valid, and what must never fail a run.

With no CI running this suite, the stored baseline *is* the signal — there is no
continuous series to spot a trend in and no fixed budget that survives a hardware
change. So the rules about when two results may be compared carry the whole story, and
they are tested here rather than only exercised through the CLI.
"""

from __future__ import annotations

import json

import pytest

from code_atlas.bench import (
    BenchReport,
    ConsumerBreakdown,
    PhaseTiming,
    baseline_path,
    compare_baseline,
    fingerprint,
    fingerprint_diff,
    save_baseline,
)


def _report(*, backend: str = "sqlite", files: int = 10, rels: int = 40) -> BenchReport:
    return BenchReport(
        total_s=12.5,
        consumers=(
            ConsumerBreakdown(
                "ast",
                (
                    PhaseTiming("ast", "parse", 1.0, calls=1, units=files, unit_name="files"),
                    PhaseTiming("ast", "resolve", 2.0, calls=1, units=rels, unit_name="rels"),
                ),
            ),
        ),
        round_trips={("upsert_entities", "write_tx"): 12},
        corpus="tiny",
        backend=backend,
    )


class TestFingerprint:
    def test_it_records_what_is_stable_not_a_speed_score(self):
        fp = fingerprint()
        assert set(fp) == {"cpu", "cores", "ram_gb", "os", "python"}
        assert not any("score" in k or "speed" in k for k in fp), (
            "a speed number measured at run time is itself load-dependent, which would make the "
            "fingerprint non-deterministic and defeat its only job"
        )

    def test_it_is_stable_within_a_process(self):
        assert fingerprint() == fingerprint()

    def test_diff_names_the_fields_that_moved(self):
        a = {"cpu": "x", "cores": 8, "os": "Linux"}
        b = {"cpu": "x", "cores": 16, "os": "Linux"}
        assert fingerprint_diff(a, b) == ["cores"]
        assert fingerprint_diff(a, a) == []


class TestBaselineFileLayout:
    def test_one_file_per_backend_and_corpus(self, tmp_path):
        """Separate files so a reader cannot accidentally diff Memgraph against SQLite."""
        a = baseline_path(tmp_path, backend="sqlite", corpus="repo@abc")
        b = baseline_path(tmp_path, backend="memgraph", corpus="repo@abc")
        c = baseline_path(tmp_path, backend="sqlite", corpus="other@abc")
        assert len({a, b, c}) == 3

    def test_the_written_file_is_diffable(self, tmp_path):
        """Sorted keys and indented, because the review diff is where 'is this expected?'
        actually gets asked."""
        path = baseline_path(tmp_path, backend="sqlite", corpus="tiny")
        save_baseline(path, report=_report(), corpus_commit="abc123")
        text = path.read_text(encoding="utf-8")

        payload = json.loads(text)
        assert payload["backend"] == "sqlite"
        assert payload["corpus_commit"] == "abc123"
        assert list(payload) == sorted(payload), "keys are not sorted, so diffs will churn"
        assert text.endswith("\n")


class TestComparisonValidity:
    def test_a_missing_baseline_is_reported_not_an_error(self, tmp_path):
        result = compare_baseline(
            baseline_path(tmp_path, backend="sqlite", corpus="tiny"), report=_report(), corpus_commit="abc"
        )
        assert not result.comparable
        assert "no baseline" in result.reason

    def test_a_different_backend_is_never_comparable(self, tmp_path):
        """Memgraph and SQLite are different systems; only each against its own history."""
        path = baseline_path(tmp_path, backend="sqlite", corpus="tiny")
        save_baseline(path, report=_report(backend="sqlite"), corpus_commit="abc")

        result = compare_baseline(path, report=_report(backend="memgraph"), corpus_commit="abc")
        assert not result.comparable
        assert "backend differs" in result.reason

    def test_a_moved_corpus_refuses_comparison(self, tmp_path):
        """Different bytes are a different measurement, not a different reading."""
        path = baseline_path(tmp_path, backend="sqlite", corpus="tiny")
        save_baseline(path, report=_report(), corpus_commit="aaaaaaaaaaaa")

        result = compare_baseline(path, report=_report(), corpus_commit="bbbbbbbbbbbb")
        assert not result.comparable
        assert "corpus moved" in result.reason

    def test_a_different_machine_is_reported_and_never_fails(self, tmp_path, monkeypatch):
        """The hard rule.

        A threshold that fires because somebody moved machines teaches people to ignore
        it, which costs more than the regression it was meant to catch. So a fingerprint
        mismatch produces a report and no comparison at all — not a lenient comparison.
        """
        path = baseline_path(tmp_path, backend="sqlite", corpus="tiny")
        save_baseline(path, report=_report(files=10), corpus_commit="abc")

        stored = json.loads(path.read_text(encoding="utf-8"))
        stored["fingerprint"]["cores"] = (stored["fingerprint"].get("cores") or 0) + 999
        path.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # A real regression is present as well, so this proves the mismatch suppresses the
        # verdict rather than the run simply having nothing to report.
        result = compare_baseline(path, report=_report(files=9999), corpus_commit="abc")
        assert not result.comparable
        assert "cores" in result.fingerprint_fields
        assert not result.regressed, "a cross-machine difference must never read as a regression"
        assert "not gated" in result.render()


class TestCounterComparison:
    def test_identical_runs_show_no_deltas(self, tmp_path):
        path = baseline_path(tmp_path, backend="sqlite", corpus="tiny")
        save_baseline(path, report=_report(), corpus_commit="abc")

        result = compare_baseline(path, report=_report(), corpus_commit="abc")
        assert result.comparable
        assert result.counter_deltas == {}
        assert not result.regressed
        assert "identical" in result.render()

    def test_a_grown_counter_is_a_regression(self, tmp_path):
        path = baseline_path(tmp_path, backend="sqlite", corpus="tiny")
        save_baseline(path, report=_report(rels=40), corpus_commit="abc")

        result = compare_baseline(path, report=_report(rels=400), corpus_commit="abc")
        assert result.comparable
        assert result.regressed
        assert result.counter_deltas["ast.resolve.rels"] == (40, 400)
        assert "+ ast.resolve.rels" in result.render()

    def test_a_shrunk_counter_is_not_a_regression(self, tmp_path):
        """Work removed is the outcome an optimisation is trying to produce.

        Reported so the diff shows it, never failed — a suite that fails on improvement
        gets its thresholds raised until it stops meaning anything.
        """
        path = baseline_path(tmp_path, backend="sqlite", corpus="tiny")
        save_baseline(path, report=_report(rels=400), corpus_commit="abc")

        result = compare_baseline(path, report=_report(rels=40), corpus_commit="abc")
        assert result.comparable
        assert not result.regressed
        assert result.counter_deltas["ast.resolve.rels"] == (400, 40)

    def test_timings_are_stored_but_never_compared(self, tmp_path):
        """Times are for a human reading the diff, not for a threshold.

        Two runs whose counters match exactly but whose durations differ must not report a
        regression — that difference is the machine, not the code.
        """
        path = baseline_path(tmp_path, backend="sqlite", corpus="tiny")
        save_baseline(path, report=_report(), corpus_commit="abc")
        assert "timings_s" in json.loads(path.read_text(encoding="utf-8"))

        slower = BenchReport(
            total_s=999.0,
            consumers=_report().consumers,
            round_trips=_report().round_trips,
            corpus="tiny",
            backend="sqlite",
        )
        result = compare_baseline(path, report=slower, corpus_commit="abc")
        assert result.comparable
        assert not result.regressed, "a slower run with identical counters must not read as a regression"


class TestRecordingIsDeliberate:
    def test_comparing_does_not_rewrite_the_baseline(self, tmp_path):
        """A suite that rewrites its own baseline on a passing run cannot detect drift:
        every run compares against the last one and every step looks like no change."""
        path = baseline_path(tmp_path, backend="sqlite", corpus="tiny")
        save_baseline(path, report=_report(rels=40), corpus_commit="abc")
        before = path.read_text(encoding="utf-8")

        compare_baseline(path, report=_report(rels=4000), corpus_commit="abc")

        assert path.read_text(encoding="utf-8") == before, "comparison mutated the stored baseline"


@pytest.mark.parametrize("commit", ["", "abc"])
def test_an_unpinned_corpus_still_compares_on_the_same_machine(tmp_path, commit):
    """An empty commit means 'not a pinned corpus', not 'a corpus called empty'.

    Refusing to compare would make every unpinned local run useless; comparing across two
    *different* commits is the case that must be refused, and it is tested above.
    """
    path = baseline_path(tmp_path, backend="sqlite", corpus="tiny")
    save_baseline(path, report=_report(), corpus_commit=commit)
    result = compare_baseline(path, report=_report(), corpus_commit=commit)
    assert result.comparable
