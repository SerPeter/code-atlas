"""Write-path cost, with graph size swept independently of batch size.

The write path is where the time goes: a Feb-2026 profile put `upsert_file_entities` at
61% of total indexing time and `resolve_calls` at 20%.

**The design point that makes this story work.** Several stages cost in proportion to the
whole project rather than to the batch being written — `build_resolution_lookup`,
`build_anchor_lookup`, `build_citation_lookup`, `_create_file_ref_links` and
`get_project_file_paths` all pull the project on every flush. A benchmark that varies only
file count moves both variables together and can never separate "we wrote more" from "the
graph got bigger". So the corpus indexes a fixed N files into a graph already holding M
entities, and sweeps M.

That class is not hypothetical. It is the class of every write-path regression this
project has shipped: unlabelled uid scans at 78x and 58x, an `ENDS WITH` project scan, a
name-first REFERENCES resolution at 5.3s for one reference, `_create_doc_links` scanning
the whole graph per doc file, and a CALLS write whose cost grew with graph size and
reached a user's first real C++ project before anything noticed.

**Why the measured run is a delta, not a scoped `--full`.** A scoped full index computes
`stale = get_project_file_paths(project) - scanned` and deletes the difference
(`orchestrator.py:2089`), so scoping a full run to the measured files would delete the
entire ballast and measure a wipe. A delta run is both safe here and the shape a real
incremental index actually has.

Runs on the embedded backend, so no Docker and no network.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from code_atlas.bench import capture, stub_provider

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.bench, pytest.mark.slow]

# Ballast sizes to sweep. Kept small because each point costs a full index plus a delta,
# and the shape of the curve is what matters, not its resolution.
_BALLAST_SIZES = (8, 32)
_MEASURED_FILES = 4


def _module(index: int, prefix: str) -> str:
    """A module with a class, a method and a function, and a cross-file import.

    Call sites must resolve to project callables. A corpus whose bodies only call builtins
    has full node scale and no resolvable edges at all, which measures the wrong half of
    indexing — the reason the old `large` preset never earned its cost.
    """
    imp = f"from src.{prefix}.{prefix}_{index - 1} import K{prefix}_{index - 1}\n" if index else ""
    return (
        f'{imp}\n\nclass K{prefix}_{index}:\n    """Class {index}."""\n\n'
        f"    def run(self, x: int) -> int:\n        return helper_{prefix}_{index}(x)\n\n\n"
        f'def helper_{prefix}_{index}(x: int) -> int:\n    """Helper {index}."""\n    return x + {index}\n'
    )


def _git(root: Path, *args: str) -> None:
    """Run git in *root*. The corpus must be a real repository — see `_sweep_point`."""
    import subprocess

    subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True, check=False, timeout=60)


def _commit(root: Path, message: str) -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)


def _write_tree(root: Path, prefix: str, count: int) -> None:
    pkg = root / "src" / prefix
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    for i in range(count):
        (pkg / f"{prefix}_{i}.py").write_text(_module(i, prefix), encoding="utf-8")


async def _sweep_point(tmp_path: Path, ballast: int) -> dict[str, object]:
    """Load *ballast* files, then measure a fixed delta batch against that graph."""
    from code_atlas.backends import use_backends
    from code_atlas.indexing.orchestrator import index_project
    from code_atlas.settings import AtlasSettings

    root = tmp_path / f"corpus_{ballast}"
    root.mkdir(parents=True)
    (root / "README.md").write_text("# Corpus\n\nMarkdown so the doc path runs.\n", encoding="utf-8")
    _write_tree(root, "ballast", ballast)

    # A real repository, because delta mode is computed from `git diff` against the stored
    # `Project.git_hash`. Without one, `_decide_delta_mode` falls back to full and
    # re-publishes the entire corpus — which is what the first version of this test did:
    # the "measured" batch was 15 files at ballast=8 and 39 at ballast=32, so the thing
    # being held constant was not. The guard below is what caught it.
    _git(root, "-c", "init.defaultBranch=main", "init", "-q")
    _git(root, "config", "user.email", "bench@example.com")
    _git(root, "config", "user.name", "Bench")
    _commit(root, "ballast")

    settings = AtlasSettings(
        project_root=root,
        backend={"graph": "sqlite", "queue": "sqlite", "sqlite_data_dir": str(tmp_path / f"db_{ballast}")},
        embeddings={"dimension": 16},
        # Force the delta path at every sweep point. `_decide_delta_mode` falls back to a
        # full reindex when the changed ratio exceeds `delta_threshold`
        # (`orchestrator.py:1888`), and with a small ballast the fixed batch is a large
        # fraction of the corpus — so the low end of the sweep silently measured a full
        # index while the high end measured a delta. Two different operations compared as
        # though they were one, which is worse than no measurement.
        index={"delta_threshold": 1.0},
    )
    project = f"bench-sweep-{ballast}"

    cap = capture()
    with stub_provider(16):
        async with use_backends(settings, with_bus=True) as backends:
            # Load-bearing arm: build the graph to size M. Not measured.
            await index_project(
                settings,
                backends.graph,  # ty: ignore[invalid-argument-type]  # index_project accepts either backend
                backends.bus,  # ty: ignore[invalid-argument-type]  # index_project accepts either backend
                full_reindex=True,
                project_name=project,
            )

            # Now add the measured batch and re-index as a delta. Everything captured
            # from here belongs to N files written against a graph of size M.
            _write_tree(root, "measured", _MEASURED_FILES)
            _commit(root, "measured batch")
            cap.clear()
            result = await index_project(
                settings,
                backends.graph,  # ty: ignore[invalid-argument-type]  # index_project accepts either backend
                backends.bus,  # ty: ignore[invalid-argument-type]  # index_project accepts either backend
                project_name=project,
            )
            entities = await backends.graph.count_entities(project)

    report = cap.report(total_s=0.0, corpus=f"ballast={ballast}", backend="sqlite")
    by_op = report.round_trips_by_op()
    return {
        "ballast_files": ballast,
        "graph_entities": entities,
        "measured_files": result.files_published,
        "round_trips": report.total_round_trips,
        # Vectors written during the measured window. If this tracks graph size rather
        # than batch size, the run is finishing the ballast's embedding work and the
        # round-trip growth is leakage, not graph-size scaling. Surfaced rather than
        # assumed either way -- see TestEmbeddingLeakage.
        "embedding_rows": by_op.get("_write_embedding_row", 0),
        "by_op": by_op,
    }


@pytest.fixture(scope="module")
def sweep(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, object]]:
    """One point per ballast size. Module-scoped — each point is a full index."""
    import asyncio

    base = tmp_path_factory.mktemp("writepath")

    async def _run() -> list[dict[str, object]]:
        return [await _sweep_point(base, size) for size in _BALLAST_SIZES]

    return asyncio.run(_run())


class TestTheSweepIsValid:
    def test_graph_size_actually_grew(self, sweep):
        """Non-vacuity. Every claim below compares points on a curve that must exist."""
        sizes = [int(p["graph_entities"]) for p in sweep]
        assert sizes == sorted(sizes), f"graph did not grow across the sweep: {sizes}"
        assert sizes[-1] > sizes[0] * 2, f"the sweep is too flat to show anything: {sizes}"

    def test_the_measured_batch_is_the_same_size_at_every_point(self, sweep):
        """The whole design. If the batch moved too, the comparison means nothing."""
        published = {int(p["measured_files"]) for p in sweep}
        assert len(published) == 1, f"the measured batch differed across points: {published}"
        assert published.pop() > 0, "no files were published in the measured run"


class TestWhatScalesWithGraphSize:
    def test_report_the_curve(self, sweep):
        """The table a person reads: round-trips for a fixed batch, against graph size.

        Reported, not budgeted. The number that matters is the *shape* — an operation
        whose count climbs with graph size while the batch is held constant is doing work
        proportional to something it should not be.
        """
        rows = [
            {
                "ballast_files": p["ballast_files"],
                "graph_entities": p["graph_entities"],
                "measured_files": p["measured_files"],
                "round_trips": p["round_trips"],
            }
            for p in sweep
        ]
        first, last = sweep[0], sweep[-1]
        growth = {
            op: (int(last["by_op"].get(op, 0)), int(first["by_op"].get(op, 0)))
            for op in set(last["by_op"]) | set(first["by_op"])
        }
        climbing = {op: pair for op, pair in sorted(growth.items()) if pair[0] > pair[1]}

        print(f"\n{json.dumps({'sweep': rows, 'round_trips_that_climb_with_graph_size': climbing}, indent=2)}")
        assert rows

    def test_round_trips_are_attributed_to_real_methods(self, sweep):
        ops = set(sweep[-1]["by_op"])
        assert ops, "no round-trips recorded for the measured batch"
        assert not (ops & {"execute", "_record", "_timed_sql", "unknown", "None"}), (
            f"a round-trip was attributed to plumbing rather than its caller: {sorted(ops)}"
        )

    def test_node_and_relationship_writes_are_separable(self, sweep):
        """TX1 and TX2 must be distinguishable, which is the split the span tree cannot make.

        `timed_phase("ast", "upsert")` brackets both transactions together, so only the
        per-`op` attribution separates them.
        """
        ops = set(sweep[-1]["by_op"])
        entity_ops = {op for op in ops if "entit" in op.lower()}
        rel_ops = {op for op in ops if "relationship" in op.lower() or "rel" in op.lower()}
        assert entity_ops, f"no entity-write op in {sorted(ops)}"
        assert rel_ops, f"no relationship-write op in {sorted(ops)}"
        assert entity_ops != rel_ops, "entity and relationship writes are not separable"


class TestEmbeddingLeakage:
    """Is the round-trip growth graph-size scaling, or ballast work finishing late?

    `_write_embedding_row` climbing with the ballast would mean the measured window is
    completing embedding the *previous* run left undone — `_reconcile_until_embedded`
    sweeps the whole project for entities without vectors, so an incomplete ballast run
    lands its remainder on the next run's bill.

    This does not assert which it is. It puts the number where a reader can see it,
    because a benchmark that silently attributes ballast work to the measured batch is
    worse than one that says it cannot tell.
    """

    def test_embedding_writes_are_reported_against_batch_size(self, sweep):
        rows = [(int(p["ballast_files"]), int(p["embedding_rows"]), int(p["measured_files"])) for p in sweep]
        print(f"\nembedding rows per (ballast, rows, measured_files): {rows}")

        first, last = sweep[0], sweep[-1]
        leaked = int(last["embedding_rows"]) > int(first["embedding_rows"]) * 1.5
        if leaked:
            print(
                "  NOTE: embedding writes scale with the ballast, not the batch. Part of the "
                "round-trip growth above is the previous run's embedding work finishing here, "
                "not graph-size scaling. Treat the curve as an upper bound until the embed "
                "reconcile is excluded from the measured window."
            )
        assert rows
