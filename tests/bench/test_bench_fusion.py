"""The query path's pure-CPU stages, instruction-counted.

`hybrid_search` is mostly waiting on a database, and the existing query benchmarks
measure that: whole-call latency against wide p95 budgets, sized for co-tenancy rather
than for signal. Those budgets are honest about what they can detect, which is almost
nothing.

Three stages need no database at all — RRF fusion (`engine.py:437`), post-fusion
filtering (`:629`) and importance boosting (`:862`) — plus query routing (`:471`), which
decides the channel weights before any arm runs. They are the only part of the query path
that is comparable across machines, so they are the only part worth an instruction count.

Deliberately NOT applied to the arms themselves. A `graph_search` is dominated by a
round-trip; counting instructions in the Python process would measure the driver rather
than the query, which is the same mistake `test_vector_search_latency` documents avoiding.
"""

from __future__ import annotations

import random

import pytest

from code_atlas.search.engine import (
    SearchResult,
    _apply_filters,
    _boost_results,
    analyze_query,
    rrf_fuse,
)

pytestmark = [pytest.mark.bench, pytest.mark.slow]

# Sized to the shape a real fusion sees: three channels, overlapping but not identical,
# with the overlap that makes RRF do work rather than concatenate.
_CHANNELS = ("graph", "bm25", "vector")
_PER_CHANNEL = 200


def _ranked_lists(seed: int = 7) -> dict[str, list[str]]:
    """Three ranked lists with realistic partial overlap.

    Disjoint lists would make fusion a concatenation and the benchmark would measure the
    wrong thing — the cost is in the repeated dictionary hits for uids seen in more than
    one channel.
    """
    rng = random.Random(seed)
    pool = [f"proj:callable:mod{i // 10}.fn{i}" for i in range(_PER_CHANNEL * 2)]
    lists: dict[str, list[str]] = {}
    for channel in _CHANNELS:
        picked = rng.sample(pool, _PER_CHANNEL)
        lists[channel] = picked
    return lists


def _results(count: int = 400, seed: int = 11) -> list[SearchResult]:
    rng = random.Random(seed)
    kinds = ("function", "method", "class", "module")
    labels = (["Callable"], ["TypeDef"], ["Module"], ["DocSection"])
    out: list[SearchResult] = []
    for i in range(count):
        # A realistic mix of test, stub and generated paths, so the filters actually
        # reject some of the input instead of passing everything through.
        folder = ("src", "tests", "generated")[i % 3]
        out.append(
            SearchResult(
                uid=f"proj:callable:{folder}.mod{i // 10}.fn{i}",
                name=f"fn{i}",
                qualified_name=f"{folder}.mod{i // 10}.fn{i}",
                kind=kinds[i % len(kinds)],
                file_path=f"{folder}/mod{i // 10}.py",
                line_start=i,
                line_end=i + 8,
                signature=f"def fn{i}(x: int) -> int",
                docstring="Does a thing." if i % 4 else "",
                labels=list(labels[i % len(labels)]),
                rrf_score=rng.random(),
                sources={"graph": i % 20},
                visibility="public" if i % 5 else "private",
            )
        )
    return out


class TestFusionCost:
    def test_rrf_fuse(self, benchmark):
        """Pure dictionary arithmetic over three ranked lists. No I/O at all."""
        lists = _ranked_lists()
        weights = {"graph": 1.0, "bm25": 0.8, "vector": 1.2}

        fused = benchmark(lambda: rrf_fuse(lists, weights=weights))
        assert fused, "fusion produced nothing — the benchmark measured an empty input"

    def test_fusion_is_deterministic(self):
        """Same lists, same scores, same order.

        Ranking that shifts between identical runs would make every downstream comparison
        noise, and the dict ordering RRF relies on is a language guarantee worth pinning.
        """
        lists = _ranked_lists()
        first = rrf_fuse(lists, weights={"graph": 1.0})
        second = rrf_fuse(lists, weights={"graph": 1.0})
        assert list(first.items()) == list(second.items())

    def test_overlap_is_what_makes_fusion_work(self):
        """Non-vacuity for the corpus above.

        If the three channels were disjoint, fusion would be a concatenation and the
        instruction count would describe a case that never happens.
        """
        lists = _ranked_lists()
        seen = [set(uids) for uids in lists.values()]
        overlap = set.intersection(*seen)
        assert overlap, "the generated channels do not overlap, so fusion has nothing to combine"


class TestFilterCost:
    def test_apply_filters(self, benchmark):
        from code_atlas.settings import SearchSettings

        results = _results()
        settings = SearchSettings()

        kept = benchmark(lambda: _apply_filters(results, settings, exclude_tests=True, exclude_generated=True))
        assert kept, "every result was filtered out"

    def test_the_filters_actually_reject_some_input(self):
        """A filter that keeps everything costs nothing and proves nothing."""
        from code_atlas.settings import SearchSettings

        results = _results()
        kept = _apply_filters(results, SearchSettings(), exclude_tests=True, exclude_generated=True)
        assert len(kept) < len(results), (
            f"filters kept all {len(results)} results — the corpus has no test or generated paths, "
            "so the measured cost is the trivial path"
        )


class TestBoostCost:
    def test_boost_results(self, benchmark):
        results = _results()
        ranked = benchmark(lambda: _boost_results(results, secondary_projects=frozenset({"other"})))
        assert ranked

    def test_boosting_reorders(self):
        """Otherwise this is measuring a sort of an already-sorted list."""
        results = _results()
        before = [r.uid for r in results]
        after = [r.uid for r in _boost_results(results)]
        assert after != before, "boosting changed no order, so the visibility and label multipliers did nothing"


class TestQueryRoutingCost:
    def test_analyze_query(self, benchmark):
        """Runs before any arm, on every search, and decides the channel weights."""
        queries = [
            "how does the embed consumer dedup vectors",
            "SqliteGraphClient",
            "def _write_embedding_row",
            "why is the file hash gate slow",
            "atlas.toml",
        ]
        weights = benchmark(lambda: [analyze_query(q) for q in queries])
        assert all(w for w in weights)

    def test_routing_discriminates_between_query_shapes(self):
        """A router returning identical weights for a symbol and a question is not routing.

        Without this the instruction count above would still be stable and still be
        measuring a constant function.
        """
        symbol = analyze_query("SqliteGraphClient")
        question = analyze_query("why is the file hash gate slow")
        assert symbol != question, f"routing gave identical weights to both shapes: {symbol}"
