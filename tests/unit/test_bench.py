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


class TestCorpusResolution:
    """A corpus that moves under a benchmark produces numbers nobody can compare."""

    @staticmethod
    def _source_repo(root):
        import subprocess

        root.mkdir(parents=True, exist_ok=True)
        (root / "a.py").write_text("def f(x):\n    return x\n", encoding="utf-8")

        def git(*args):
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)

        git("-c", "init.defaultBranch=main", "init", "-q")
        git("config", "user.email", "b@example.com")
        git("config", "user.name", "Bench")
        git("add", "-A")
        git("commit", "-q", "-m", "corpus")
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
        ).stdout.strip()

    def test_a_local_path_is_used_where_it_is(self, tmp_path):
        """A benchmark must never fetch into the tree someone is sitting in."""
        from code_atlas.bench import resolve_corpus

        sha = self._source_repo(tmp_path / "local")
        corpus = resolve_corpus(str(tmp_path / "local"))

        assert corpus.source == "local"
        assert corpus.commit == sha
        assert corpus.path == (tmp_path / "local").resolve()

    def test_a_url_is_cloned_and_pinned(self, tmp_path):
        from code_atlas.bench import resolve_corpus

        sha = self._source_repo(tmp_path / "src")
        corpus = resolve_corpus((tmp_path / "src").as_uri(), cache=tmp_path / "cache")

        assert corpus.commit == sha, "the clone was not pinned to the source commit"
        assert corpus.reused_cache is False
        assert (corpus.path / "a.py").exists()

    def test_a_second_resolve_reuses_the_clone(self, tmp_path):
        """Reuse is asserted on the flag, not on timing.

        Timing would make this flaky on a slow disk, and the property that matters is
        that no fetch happened — re-fetching would silently change the corpus under a
        comparison that assumes it is fixed.
        """
        from code_atlas.bench import resolve_corpus

        self._source_repo(tmp_path / "src")
        url = (tmp_path / "src").as_uri()
        first = resolve_corpus(url, cache=tmp_path / "cache")
        second = resolve_corpus(url, cache=tmp_path / "cache")

        assert first.reused_cache is False
        assert second.reused_cache is True
        assert first.commit == second.commit

    def test_two_urls_sharing_a_name_do_not_collide(self, tmp_path):
        """Two forks called `backend` must not share one cache directory."""
        from code_atlas.bench import resolve_corpus

        self._source_repo(tmp_path / "one" / "repo")
        self._source_repo(tmp_path / "two" / "repo")
        a = resolve_corpus((tmp_path / "one" / "repo").as_uri(), cache=tmp_path / "cache")
        b = resolve_corpus((tmp_path / "two" / "repo").as_uri(), cache=tmp_path / "cache")

        assert a.path != b.path, "two distinct URLs shared one cache directory"
        # Deliberately NOT asserting the commits differ. Two repositories with
        # identical content, author and timestamp produce the identical sha, which
        # made this fail under -n auto while passing serially. That would be a fact
        # about git, not about the cache key under test here.

    def test_comparability_is_gated_on_the_commit(self, tmp_path):
        from code_atlas.bench import resolve_corpus

        sha = self._source_repo(tmp_path / "local")
        corpus = resolve_corpus(str(tmp_path / "local"))

        assert corpus.comparable_with(sha)
        assert not corpus.comparable_with("0" * 40)

    def test_an_uncommitted_repo_is_not_comparable_with_anything(self, tmp_path):
        """An unpinnable corpus must refuse comparison rather than compare loosely.

        `git init` with no commit leaves HEAD unborn, so there is no sha. Reporting that
        as comparable-to-empty-string would make every such run compare equal.
        """
        import subprocess

        from code_atlas.bench import resolve_corpus

        root = tmp_path / "unborn"
        root.mkdir()
        subprocess.run(
            ["git", "-c", "init.defaultBranch=main", "init", "-q"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        corpus = resolve_corpus(str(root))

        assert corpus.commit == ""
        assert not corpus.comparable_with("")
        assert "no-commit" in corpus.label()


class TestCorpusProfile:
    def test_a_missing_required_shape_is_named(self):
        from code_atlas.bench import CorpusProfile

        code_only = CorpusProfile(labels={"Callable": 4, "Module": 4}, files=4, entities=8)
        assert code_only.missing() == ("DocSection",)
        assert code_only.doc_sections == 0

    def test_a_corpus_with_the_shape_reports_nothing_missing(self):
        from code_atlas.bench import CorpusProfile

        mixed = CorpusProfile(labels={"Callable": 4, "DocSection": 2}, files=5, entities=6)
        assert mixed.missing() == ()
        assert mixed.doc_sections == 2

    def test_the_summary_drops_empty_labels(self):
        from code_atlas.bench import CorpusProfile

        profile = CorpusProfile(labels={"Callable": 3, "Note": 0}, files=1, entities=3)
        summary = profile.summary()
        assert summary["Callable"] == 3
        assert "Note" not in summary, "a zero count reads as a shape the corpus has"


class TestOfflineBehaviour:
    """No network must degrade honestly: a cached corpus works, an uncached one says so."""

    def test_a_cached_corpus_resolves_after_the_source_is_gone(self, tmp_path):
        """The offline path, proved by deleting the remote rather than by mocking it.

        This is also what makes the cache trustworthy: resolving a cached corpus performs
        no fetch at all, so it cannot pick up a moved ref behind a comparison's back.
        """
        from code_atlas.bench import resolve_corpus

        src = tmp_path / "src"
        TestCorpusResolution._source_repo(src)
        url = src.as_uri()
        first = resolve_corpus(url, cache=tmp_path / "cache")

        # Renamed rather than deleted: git marks its object files read-only, so rmtree
        # raises PermissionError on Windows. Moving it makes the URL just as dead.
        src.rename(tmp_path / "src-moved")
        second = resolve_corpus(url, cache=tmp_path / "cache")

        assert second.reused_cache is True
        assert second.commit == first.commit
        assert (second.path / "a.py").exists()

    def test_an_uncached_unreachable_corpus_fails_with_a_named_reason(self, tmp_path):
        """Never a silent substitution — a benchmark that quietly measures nothing is worse
        than one that stops."""
        from code_atlas.bench import resolve_corpus

        missing = (tmp_path / "does-not-exist").as_uri()
        with pytest.raises(RuntimeError) as exc:
            resolve_corpus(missing, cache=tmp_path / "cache")

        assert "fetch" in str(exc.value).lower()
        assert missing in str(exc.value), "the error must name the corpus it could not get"


class TestPerLanguageCounts:
    def test_languages_are_counted_by_the_lookup_indexing_uses(self):
        from code_atlas.bench import CorpusProfile

        profile = CorpusProfile(
            labels={"Callable": 4},
            files=10,
            entities=4,
            files_by_language={"python": 9, "markdown": 1, "(none)": 0},
        )
        summary = profile.summary()
        assert summary["lang:python"] == 9
        assert summary["lang:markdown"] == 1
        assert "lang:(none)" not in summary, "a zero count reads as a language the corpus has"

    def test_embed_chunks_are_reported_but_not_required_by_default(self):
        from code_atlas.bench import CorpusProfile

        small = CorpusProfile(labels={"DocSection": 1}, files=2, entities=2)
        assert small.embed_chunks == 0
        assert small.missing() == (), "a small corpus must not fail for lacking oversized entities"
        assert small.missing(require=("EmbedChunk",)) == ("EmbedChunk",)
