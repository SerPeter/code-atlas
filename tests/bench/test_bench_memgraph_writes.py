"""Memgraph's per-label round-trip multipliers, pinned to the label lists that cause them.

Memgraph has no label-free index, and `UNWIND ... MATCH (n {...}) WHERE n:Label` is
order-sensitive on this engine and **silently drops rows** — measured at 3 existing files
returning 0 when a non-matching path came first. So the label goes inline on the pattern,
and one logical read becomes one statement per label.

That is a deliberate design decision, not a defect, and it is the single largest
round-trip multiplier on the write path. It is therefore worth pinning to the constant
that produces it: a label added to one of these tuples changes the cost of every batch,
and the test says so by name rather than by a magic number drifting out of date.

The strong form of each assertion is the second one — monkeypatch the tuple and watch the
count follow. Asserting `== 10` alone would still pass if the loop were replaced by
something that happened to issue ten statements for an unrelated reason.

Needs a real Memgraph. Run against the disposable stack, never the production one:

    ATLAS_TEST_MEMGRAPH_PORT=7688 ATLAS_TEST_VALKEY_PORT=6380 uv run pytest tests/bench -m bench

Never with `-n`: `graph_client` wipes the database before every test, and under xdist the
workers would share one instance and erase each other mid-test.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

import code_atlas.graph.client as client_mod
from code_atlas.bench import capture, stub_provider
from code_atlas.graph.client import (
    _HASHED_ENTITY_LABELS,
    _INDEXED_FILE_LABELS,
)
from code_atlas.search.ratelimit import unpaced

if TYPE_CHECKING:
    from pathlib import Path

    from code_atlas.events import EventBus
    from code_atlas.graph.client import GraphClient

pytestmark = [pytest.mark.bench, pytest.mark.integration]

_PROJECT = "bench-memgraph-writes"


def _ops_by_name(cap) -> dict[str, int]:
    """Collapse the capture's (op, kind) keys to op totals."""
    totals: dict[str, int] = {}
    for (op, _kind), count in cap.round_trips().items():
        totals[op] = totals.get(op, 0) + count
    return totals


def _write_corpus(root: Path, count: int = 8) -> None:
    pkg = root / "src"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    for i in range(count):
        prev = f"from src.m{i - 1} import step{i - 1}\n" if i else ""
        body = f"    return step{i - 1}(x) + {i}\n" if i else f"    return x + {i}\n"
        (pkg / f"m{i}.py").write_text(
            f'{prev}\n\nclass Holder{i}:\n    """Holds."""\n\n'
            f"    def run(self, x: int) -> int:\n        return step{i}(x)\n\n\n"
            f'def step{i}(x: int) -> int:\n    """Step {i}."""\n{body}',
            encoding="utf-8",
        )
    (root / "README.md").write_text("# Corpus\n\nMarkdown so the doc path runs.\n", encoding="utf-8")


@pytest.fixture
async def indexed(graph_client: GraphClient, event_bus: EventBus, tmp_path: Path):
    """A small corpus indexed into the disposable Memgraph."""
    from code_atlas.indexing.orchestrator import index_project
    from code_atlas.settings import AtlasSettings

    root = tmp_path / "corpus"
    root.mkdir()
    _write_corpus(root)

    settings = AtlasSettings(project_root=root, embeddings={"dimension": graph_client.dimension})

    # Installed BEFORE the index, not after. `capture()` is what flips telemetry on; a
    # test that calls it afterwards finds an empty reader and reads that as "this run
    # issued no queries". Same ordering trap as the CLI, where a graph client built first
    # holds a lazy tracer resolved against a provider the harness never sees.
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


class TestTheTwoTransactionsAreSeparable:
    async def test_node_writes_and_relationship_writes_have_distinct_ops(self, indexed):
        """The split the span tree cannot make.

        `timed_phase("ast", "upsert")` brackets TX1 and TX2 together, so only the `op`
        label on `atlas_graph_query_seconds` — resolved by stack walk to the calling
        method — tells them apart. This is the assertion the SQLite arm could not make,
        because those are different method names on that backend.
        """
        # `TelemetryCapture.round_trips()` is keyed by (op, kind); `round_trips_by_op`
        # lives on `BenchReport`, not on the capture.
        ops = _ops_by_name(capture())
        assert ops, "no round-trips recorded at all"

        # Named exactly rather than matched on a substring. "entit" also catches
        # `count_entities` and `find_unembedded_entities`, which are reads — a filter that
        # loose would keep reporting the split as working even after TX1 stopped emitting.
        print(f"\nops seen: {sorted(ops)}")

        assert "_batch_create_entities" in ops, f"TX1 (node write) is not attributed: {sorted(ops)}"
        assert "_create_relationships" in ops, f"TX2 (relationship write) is not attributed: {sorted(ops)}"
        assert ops["_batch_create_entities"] > 0
        assert ops["_create_relationships"] > 0


class TestPerLabelMultipliers:
    """One statement per label, pinned to the tuple that causes it."""

    async def test_project_file_paths_costs_one_statement_per_indexed_label(self, indexed):
        cap = capture()
        cap.clear()
        before = sum(cap.round_trips().values())
        await indexed.get_project_file_paths(_PROJECT)
        issued = sum(cap.round_trips().values()) - before

        assert issued == len(_INDEXED_FILE_LABELS), (
            f"get_project_file_paths issued {issued} statements for "
            f"{len(_INDEXED_FILE_LABELS)} labels — the per-label loop has changed shape"
        )

    async def test_the_count_follows_the_label_tuple(self, indexed, monkeypatch):
        """The assertion that makes the one above mean something.

        `== 10` would still pass if the loop were replaced by something issuing ten
        statements for an unrelated reason. Shortening the tuple and watching the count
        follow is what actually pins the relationship.
        """
        cap = capture()
        cap.clear()
        before = sum(cap.round_trips().values())
        await indexed.get_project_file_paths(_PROJECT)
        full = sum(cap.round_trips().values()) - before

        monkeypatch.setattr(client_mod, "_INDEXED_FILE_LABELS", _INDEXED_FILE_LABELS[:3])
        cap.clear()
        before = sum(cap.round_trips().values())
        await indexed.get_project_file_paths(_PROJECT)
        trimmed = sum(cap.round_trips().values()) - before

        assert full == len(_INDEXED_FILE_LABELS)
        assert trimmed == 3, (
            f"trimming the label tuple to 3 left {trimmed} statements — the round-trip count "
            "is not driven by the tuple, so the multiplier is not what it appears to be"
        )

    async def test_content_hashes_cost_one_statement_per_hashed_label(self, indexed):
        cap = capture()
        cap.clear()
        before = sum(cap.round_trips().values())
        await indexed.get_file_content_hashes(_PROJECT, "src/m0.py")
        issued = sum(cap.round_trips().values()) - before

        assert issued == len(_HASHED_ENTITY_LABELS), (
            f"get_file_content_hashes issued {issued} statements for {len(_HASHED_ENTITY_LABELS)} labels"
        )


class TestTheMultipliersAreReported:
    async def test_report_the_cost_of_the_label_loops(self, indexed):
        """The table a person reads before deciding whether the design should change.

        Reported, never gated. Whether inline labels remain the right trade is a
        deliberate decision, and the epic's rule is that it be taken against a number.
        """
        rows = {
            "_HASHED_ENTITY_LABELS": len(_HASHED_ENTITY_LABELS),
            "_INDEXED_FILE_LABELS": len(_INDEXED_FILE_LABELS),
            "_IMPORT_TARGET_LABELS": len(client_mod._IMPORT_TARGET_LABELS),
        }
        ops = _ops_by_name(capture())
        top = dict(sorted(ops.items(), key=lambda kv: -kv[1])[:12])
        print(f"\n{json.dumps({'label_loop_widths': rows, 'round_trips_by_op': top}, indent=2)}")
        assert rows["_INDEXED_FILE_LABELS"] > 1, "the multiplier is 1, so there is nothing to report"
