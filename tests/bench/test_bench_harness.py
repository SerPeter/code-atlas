"""The harness measures a real pipeline run, and the three lines mean what they say.

Runs entirely on the embedded backend, so it needs no Docker and no Memgraph — the
point being verified is the accounting, and SQLite exercises the same `timed_phase` and
`atlas_graph_query_seconds` instruments the network backend does.

Marked `bench` because a full index takes tens of seconds and this suite is invoked
deliberately, never by CI.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

from code_atlas.bench import capture, stub_provider

if TYPE_CHECKING:
    from pathlib import Path

    from code_atlas.bench import BenchReport

pytestmark = [pytest.mark.bench, pytest.mark.slow]


def _write_corpus(root: Path, n_files: int = 6) -> None:
    """A corpus with cross-file references and a non-code file.

    Non-code matters: a Python-only corpus never produces a DocSection, which is exactly
    how a whole-graph scan per doc file shipped unnoticed.
    """
    pkg = root / "src" / "pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    for i in range(n_files):
        prev = f"from src.pkg.mod{i - 1} import C{i - 1}\n" if i else ""
        (pkg / f"mod{i}.py").write_text(
            f'{prev}\nclass C{i}:\n    """Doc for C{i}."""\n\n'
            f"    def m(self, x: int) -> int:\n        return helper{i}(x)\n\n\n"
            f'def helper{i}(x: int) -> int:\n    """Helper {i}."""\n    return x + {i}\n',
            encoding="utf-8",
        )
    (root / "README.md").write_text("# Corpus\n\nMarkdown, so the doc path runs.\n", encoding="utf-8")


async def _run(root: Path, db_dir: Path, project: str) -> BenchReport:
    """One measured index over *root*, entirely in-process."""
    from code_atlas.backends import use_backends
    from code_atlas.indexing.orchestrator import index_project
    from code_atlas.settings import AtlasSettings

    settings = AtlasSettings(
        project_root=root,
        backend={"graph": {"sqlite": {}}, "queue": {"sqlite": {}}, "sqlite_data_dir": str(db_dir)},
    )
    cap = capture()
    cap.clear()
    with stub_provider(settings.embeddings.dimension or 768):
        async with use_backends(settings, with_bus=True) as backends:
            # `index_project` does not create the schema -- the CLI calls `ensure_schema`
            # separately. Without it every FTS5 and vec0 write is swallowed by
            # `_safe_exec` as "no such table", so the measured system has no text or
            # vector index at all and its write cost is understated.
            await backends.graph.ensure_schema()
            started = time.perf_counter()
            await index_project(
                settings,
                backends.graph,  # ty: ignore[invalid-argument-type]  # index_project accepts either backend
                backends.bus,  # ty: ignore[invalid-argument-type]  # index_project accepts either backend
                full_reindex=True,
                project_name=project,
                limiter=backends.limiter,
            )
            total_s = time.perf_counter() - started
    return cap.report(total_s=total_s, corpus="harness", backend="sqlite")


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    _write_corpus(root)
    return root


class TestTheReportDescribesTheRun:
    async def test_phases_arrive_with_their_work_units(self, corpus: Path, tmp_path: Path):
        """`ast.parse` must carry its `files` attribute, not just a duration.

        A duration with no unit cannot be compared between corpora, and the attribute is
        the only place the count exists — the `atlas_stage_seconds` histogram agrees on
        the time and cannot say how many files it covered.
        """
        report = await _run(corpus, tmp_path / "db1", "bench-test-units")

        ast = report.consumer("ast")
        assert ast is not None, "no ast phases were captured at all"
        parse = next((p for p in ast.phases if p.phase == "parse"), None)
        assert parse is not None, f"no parse phase: {[p.phase for p in ast.phases]}"
        assert parse.units > 0, "parse reported a duration with no file count"
        assert parse.unit_name == "files"

    async def test_round_trips_are_attributed_to_calling_methods(self, corpus: Path, tmp_path: Path):
        """The sample count per `op` is the round-trip count, and `op` is a real method."""
        report = await _run(corpus, tmp_path / "db2", "bench-test-rt")

        ops = report.round_trips_by_op()
        assert ops, "no database round-trips were recorded"
        assert report.total_round_trips > 10
        assert not any(op in {"execute", "_record", "unknown", "None"} for op in ops), (
            f"a round-trip was attributed to plumbing: {sorted(ops)}"
        )

    async def test_the_run_is_fully_accounted_or_says_it_is_not(self, corpus: Path, tmp_path: Path):
        report = await _run(corpus, tmp_path / "db3", "bench-test-acct")

        assert report.total_s > 0
        assert report.accounted_s <= report.total_s, "accounted more time than the run took"
        assert report.unaccounted_s >= 0


class TestReproducibility:
    async def test_two_runs_over_identical_bytes_agree_exactly(self, corpus: Path, tmp_path: Path):
        """Work counters must be *equal*, not close.

        This is the property the whole suite rests on: if a no-op change cannot be
        distinguished from a real one, no comparison downstream means anything. Times are
        deliberately excluded — they never reproduce and asserting on them would make
        this flaky for the wrong reason.

        Each run gets a fresh database, so this compares the work a full index does from
        empty, not the work a second index skips.
        """
        first = await _run(corpus, tmp_path / "run_a", "bench-repro-a")
        second = await _run(corpus, tmp_path / "run_b", "bench-repro-b")

        # Span-derived counters only. Round-trips accumulate across runs in one process
        # (the metric reader is cumulative), so comparing them here would compare a
        # number against itself-plus-more and always fail.
        def phase_units(report: BenchReport) -> dict[str, int]:
            return {f"{p.stage}.{p.phase}": p.units for c in report.consumers for p in c.phases if p.unit_name}

        a, b = phase_units(first), phase_units(second)
        assert a, "no work counters were captured, so equality is vacuous"
        assert a == b, f"identical corpora produced different work:\n  {a}\n  {b}"


class TestPacingIsNotWork:
    async def test_a_longer_drain_lands_in_pacing_and_not_in_any_phase(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The load-bearing separation, proved by moving one production constant.

        `_wait_for_drain` holds `lag == 0` for `_DRAIN_SETTLE_S` before declaring the
        pipeline idle. That wait is deliberate and correct, and it is not work. Raising
        it must grow the unaccounted line and leave every phase where it was — if it
        leaked into a stage's work instead, the number this suite compares between runs
        would move whenever someone retuned a constant.

        Phases are compared by their work *units*, not their durations: the same corpus
        does the same amount of work either way, and wall-clock per phase wobbles.
        """
        import code_atlas.indexing.orchestrator as orch

        baseline = await _run(corpus, tmp_path / "drain_base", "bench-drain-a")

        monkeypatch.setattr(orch, "_DRAIN_SETTLE_S", orch._DRAIN_SETTLE_S + 4.0)
        slowed = await _run(corpus, tmp_path / "drain_slow", "bench-drain-b")

        def units(report: BenchReport) -> dict[str, int]:
            return {f"{p.stage}.{p.phase}": p.units for c in report.consumers for p in c.phases if p.unit_name}

        base_units = units(baseline)
        # Non-vacuity first: two empty dicts compare equal, and this whole test would
        # then pass while measuring nothing.
        assert base_units, "no work units captured, so the equality below proves nothing"
        assert base_units == units(slowed), "raising the drain floor changed the work done"
        assert slowed.unaccounted_s > baseline.unaccounted_s, (
            f"a 4s longer drain did not show up in the pacing line: "
            f"{baseline.unaccounted_s:.2f}s -> {slowed.unaccounted_s:.2f}s"
        )
