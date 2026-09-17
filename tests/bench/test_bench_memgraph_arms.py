"""Per-arm query cost on Memgraph, with results asserted rather than latency.

The SQLite arm of this story measured the embedded backend; this is the same measurement
against the network one, and the numbers are deliberately *not* compared between them.
Different engines, different neighbours, different tokenisers — only each against its own
history.

What is worth measuring here and not on SQLite is the **fan-out**. Both `text_search` and
`vector_search` query every index of their kind unless given a label, so one logical
search is many round-trips, and the count is set by how many labels carry that index
rather than by the query. A label added to the schema changes the cost of every search.

Results are asserted, never latency. On Memgraph the BM25 path has a live engine bug —
`text_search.search()` fails on Tantivy and only `text_search.search_all(idx, query,
{limit: N})` works — and a broken text index returns nothing quickly, which on a latency
chart is an improvement.

Needs the disposable stack, never the production graph on 7687:

    ATLAS_TEST_MEMGRAPH_PORT=7688 ATLAS_TEST_VALKEY_PORT=6380 uv run pytest tests/bench -m bench

Never under `-n`: `graph_client` wipes the database before each test.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from code_atlas.bench import capture, stub_provider
from code_atlas.search.ratelimit import unpaced

if TYPE_CHECKING:
    from pathlib import Path

    from code_atlas.events import EventBus
    from code_atlas.graph.client import GraphClient

pytestmark = [pytest.mark.bench, pytest.mark.integration]

_PROJECT = "bench-memgraph-arms"
_QUERY = "helper0"


def _ops_by_name(cap) -> dict[str, int]:
    totals: dict[str, int] = {}
    for (op, _kind), count in cap.round_trips().items():
        totals[op] = totals.get(op, 0) + count
    return totals


def _write_corpus(root: Path, count: int = 12) -> None:
    """Names carry a distinct token per module.

    `helper` would match nothing: FTS and Tantivy both index tokens, and a probe query
    that is a prefix of every indexed term matches none of them. That cost a debugging
    round on the SQLite arm and is not repeated here.
    """
    pkg = root / "src"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    for i in range(count):
        (pkg / f"m{i}.py").write_text(
            f'"""Module {i} with a helper."""\n\n\n'
            f'class Holder{i}:\n    """Holds things."""\n\n'
            f"    def run(self, x: int) -> int:\n        return helper{i}(x)\n\n\n"
            f'def helper{i}(x: int) -> int:\n    """A helper that helps."""\n    return x + {i}\n',
            encoding="utf-8",
        )
    (root / "README.md").write_text("# Corpus\n\nA helper is documented here.\n", encoding="utf-8")


@pytest.fixture
async def indexed(graph_client: GraphClient, event_bus: EventBus, tmp_path: Path):
    from code_atlas.indexing.orchestrator import index_project
    from code_atlas.settings import AtlasSettings

    root = tmp_path / "corpus"
    root.mkdir()
    _write_corpus(root)

    settings = AtlasSettings(project_root=root, embeddings={"dimension": graph_client.dimension})
    # Installed before the index: `capture()` is what enables telemetry, and calling it
    # afterwards finds an empty reader that reads as "this run issued no queries".
    # `index_project` does not create the schema; the CLI calls `ensure_schema`
    # separately. Without it Memgraph has no text or vector indices at all, every
    # `text_search.search_all` and `vector_search.search` fails with "index doesn't
    # exist", and the client catches and warns — so both arms return nothing and the
    # run still looks clean. Identical to the SQLite arm's failure, in a different
    # engine's error message.
    await graph_client.ensure_schema()

    cap = capture()
    cap.clear()
    with stub_provider(graph_client.dimension):
        await index_project(
            settings,
            graph_client,
            event_bus,
            full_reindex=True,
            project_name=_PROJECT,
            drain_timeout_s=180.0,
            limiter=unpaced(),
        )
    return graph_client


class TestEachArmOnMemgraph:
    async def test_arms_return_results_and_report_their_fan_out(self, indexed):
        """Each arm alone, because inside `hybrid_search` they run concurrently.

        The number reported is round-trips, which on this backend is the fan-out: one
        statement per index of that kind.
        """
        cap = capture()
        rows = []
        dimension = indexed.dimension

        for arm, call in (
            ("graph", lambda: indexed.graph_search(_QUERY, limit=10)),
            ("bm25", lambda: indexed.text_search(_QUERY, limit=10)),
            ("vector", lambda: indexed.vector_search([0.1] * dimension, limit=10)),
        ):
            cap.clear()
            before = sum(cap.round_trips().values())
            results = await call()
            after = sum(cap.round_trips().values())
            rows.append({"arm": arm, "results": len(results), "round_trips": after - before})

        print(f"\n{json.dumps({'memgraph_arms': rows}, indent=2)}")
        hits = {str(r["arm"]): int(r["results"]) for r in rows}

        assert hits["graph"] > 0, "the graph arm matched nothing — corpus or probe query is wrong"
        assert hits["bm25"] > 0, (
            "the BM25 arm matched nothing. On Memgraph `text_search.search()` fails outright "
            "(Tantivy) and only `search_all` works, so an empty result here may be the engine "
            "bug rather than the corpus."
        )
        assert hits["vector"] > 0, "the vector arm returned nothing at all"

    async def test_the_fan_out_is_more_than_one_statement(self, indexed):
        """The property that makes this arm worth measuring separately from SQLite.

        A search without a label queries every index of its kind, so its cost is set by
        the schema rather than the query. If this ever collapses to one round-trip, either
        a label filter was introduced or the fan-out was replaced — both change what every
        search costs.
        """
        cap = capture()
        cap.clear()
        before = sum(cap.round_trips().values())
        await indexed.text_search(_QUERY, limit=10)
        unlabelled = sum(cap.round_trips().values()) - before

        print(f"\nunlabelled BM25 search cost {unlabelled} round-trips")
        assert unlabelled > 1, (
            f"an unlabelled text search cost {unlabelled} round-trip(s) — the per-index fan-out "
            "is gone, which changes the cost of every search on this backend"
        )

    async def test_a_labelled_search_costs_less_than_an_unlabelled_one(self, indexed):
        """The fan-out proved by removing it, rather than asserted as a constant.

        Naming a label restricts the search to one index. If the labelled and unlabelled
        costs were equal, the label would not be narrowing anything and the number above
        would be describing something other than fan-out.
        """
        cap = capture()

        cap.clear()
        before = sum(cap.round_trips().values())
        await indexed.text_search(_QUERY, limit=10)
        unlabelled = sum(cap.round_trips().values()) - before

        cap.clear()
        before = sum(cap.round_trips().values())
        await indexed.text_search(_QUERY, label="Callable", limit=10)
        labelled = sum(cap.round_trips().values()) - before

        print(f"\nBM25 round-trips: unlabelled={unlabelled} labelled={labelled}")
        assert labelled < unlabelled, (
            f"naming a label did not reduce the search cost ({labelled} vs {unlabelled}), so the "
            "unlabelled number is not measuring a per-index fan-out"
        )


class TestBackendsAreNotCompared:
    def test_this_file_records_memgraph_only(self):
        """A guard on the reader, not on the code.

        Memgraph and SQLite numbers appear in sibling files and must never be diffed:
        Tantivy and FTS5 rank differently, and sqlite-vec is brute force while Memgraph's
        vector index is not. The project label on every stored baseline is what enforces
        this at comparison time; this states it where the numbers are produced.
        """
        assert _PROJECT.startswith("bench-memgraph"), "the project name must say which backend produced these"
