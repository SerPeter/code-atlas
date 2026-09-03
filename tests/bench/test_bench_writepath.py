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

from code_atlas.backends.instrumentation import analyse_scans, capture_statements
from code_atlas.backends.sqlite_graph import SqliteGraphClient
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
            # `index_project` does not create the schema -- the CLI calls `ensure_schema`
            # separately. Without it every FTS5 and vec0 write is swallowed by
            # `_safe_exec` as "no such table", so the measured system has no text or
            # vector index at all and its write cost is understated.
            await backends.graph.ensure_schema()
            # Load-bearing arm: build the graph to size M. Not measured.
            await index_project(
                settings,
                backends.graph,  # ty: ignore[invalid-argument-type]  # index_project accepts either backend
                backends.bus,  # ty: ignore[invalid-argument-type]  # index_project accepts either backend
                full_reindex=True,
                project_name=project,
            )

            entities_before = await backends.graph.count_entities(project)
            vectors_before = (await backends.graph.count_embeddings_by_project()).get(project, 0)

            # Now add the measured batch and re-index as a delta. Everything captured
            # from here belongs to N files written against a graph of size M.
            _write_tree(root, "measured", _MEASURED_FILES)
            _commit(root, "measured batch")
            cap.clear()
            # Statement capture is a diagnostic and distorts what it measures -- the trace
            # callback fires per statement on the worker thread. It is inside the measured
            # window anyway because the scan report has to describe the SAME run the
            # round-trip counts describe; a scan profile of a different run is a different
            # fact. Timings from this run are therefore not comparable to an untraced one,
            # which is why nothing here asserts on time.
            # Narrowed explicitly: the scan analysis is SQLite-only by nature, and the
            # backend union has no `_get_conn`. Asserting the type here beats a blanket
            # ignore, because if this file ever runs against Memgraph it should stop
            # rather than silently skip the scan report.
            assert isinstance(backends.graph, SqliteGraphClient), "the scan report is SQLite-only"
            conn = await backends.graph._get_conn()
            async with capture_statements(conn) as statements:
                result = await index_project(
                    settings,
                    backends.graph,  # ty: ignore[invalid-argument-type]  # index_project accepts either backend
                    backends.bus,  # ty: ignore[invalid-argument-type]  # index_project accepts either backend
                    project_name=project,
                )
            scans = await analyse_scans(conn, statements)
            # A control, captured separately so it stays out of the measured window.
            # Every product statement is now index-served, and "no scans found" is also
            # what a broken detector reports -- so one statement that CANNOT be
            # index-served has to come back flagged in the same run. A json_extract on a
            # key nobody indexes cannot be optimised away by a later schema change, which
            # is why the control is written here rather than borrowed from the product.
            async with capture_statements(conn) as control_log:
                cur = await conn.execute(
                    "SELECT uid FROM nodes WHERE json_extract(props_json, '$.unindexed_control') = ?", ("x",)
                )
                await cur.close()
            control = await analyse_scans(conn, control_log)
            entities = await backends.graph.count_entities(project)
            vectors_after = (await backends.graph.count_embeddings_by_project()).get(project, 0)

    report = cap.report(total_s=0.0, corpus=f"ballast={ballast}", backend="sqlite")
    by_op = report.round_trips_by_op()
    return {
        "ballast_files": ballast,
        "graph_entities": entities,
        "measured_files": result.files_published,
        "round_trips": report.total_round_trips,
        # Vectors that actually appeared, and the SQL it took to write them. These are
        # different numbers and conflating them is easy: `_write_embedding_row` issues
        # several statements per vector (a rowid lookup, the node update, the vec0 sync),
        # so its round-trip count is NOT a vector count. An earlier version of this file
        # labelled it as one and read a fivefold "leak" that was mostly statements per row.
        "new_entities": entities - entities_before,
        "new_vectors": vectors_after - vectors_before,
        "embedding_statements": by_op.get("_write_embedding_row", 0),
        # Amplification is row visits: scanning executions times table rows. It read 64
        # -> 184 across this sweep while every node index was partial
        # (`WHERE labels = '<Label>'`), because a predicate that does not name one label
        # as a literal uses none of them. Three unqualified indices took it to 0; the
        # number is kept because it is the thing that would move if they were dropped.
        "scan_amplification": scans.amplification,
        "scanning_statements": len(scans.scanning),
        "scanning_executions": scans.scanning_executions,
        "non_scanning_statements": scans.non_scanning,
        "scanning_ops": sorted({op for op, _sql, _n in scans.scanning}),
        "detector_live": len(control.scanning) > 0,
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


class TestEmbeddingWriteVolume:
    """How many vector writes a fixed batch costs, and how that moves with graph size.

    Reported, not diagnosed. This number has been misread twice while writing this file
    and the record is worth keeping, because both misreadings are easy:

    1. `_write_embedding_row` round-trips (52 -> 200 across the sweep) were first read as
       a vector count. They are a CALL count -- the helper's own statement is one
       `SELECT rowid`, while its `DELETE`/`INSERT` attribute to `_safe_exec`.
    2. The correction then used `new_vectors`, the change in nodes holding a vector, and
       concluded there was no rewriting. That is also wrong: a node that already has a
       vector and gets a new one does not change that count, so the metric is blind to
       exactly the thing it was introduced to detect.

    What is established: a fixed 5-file batch triggers 52 embedding-row writes into a
    50-entity graph and 200 into a 146-entity one, while only 17 entities are new at both
    sizes. What is NOT established is why -- rewrites of existing vectors, `EmbedChunk`
    rows, and the reconcile sweep are all live candidates and none has been ruled out.

    So this asserts the shape and prints the numbers. Naming a cause here without
    checking it is how the first two readings happened.
    """

    def test_the_write_volume_is_reported_against_the_batch(self, sweep):
        rows = [
            (
                int(p["ballast_files"]),
                int(p["graph_entities"]),
                int(p["new_entities"]),
                int(p["new_vectors"]),
                int(p["embedding_statements"]),
            )
            for p in sweep
        ]
        print(f"\n(ballast, graph_entities, new_entities, new_vectors, embedding_row_writes): {rows}")

        for ballast, _graph, new_entities, new_vectors, _writes in rows:
            assert new_vectors > 0, f"ballast={ballast}: no vectors appeared, so the row is vacuous"
            assert new_vectors <= new_entities, (
                f"ballast={ballast}: {new_vectors} new vectors for {new_entities} new entities"
            )

    def test_write_volume_exceeding_new_entities_is_visible(self, sweep):
        """The open question, kept in front of a reader rather than buried.

        If embedding-row writes greatly exceed the entities that gained a vector, the
        measured window is doing work for something other than its batch. That is either
        a real inefficiency or a benchmark artefact; this says which numbers to look at.
        """
        excess = [
            (int(p["ballast_files"]), int(p["embedding_statements"]), int(p["new_vectors"]))
            for p in sweep
            if int(p["embedding_statements"]) > int(p["new_vectors"]) * 2
        ]
        if excess:
            print(
                f"\nOPEN: embedding-row writes far exceed newly-vectored entities {excess}. "
                "Cause not established — candidates are rewrites of existing vectors, EmbedChunk "
                "rows, and the unembedded-entity reconcile. The write-path curve above is an "
                "upper bound until this is resolved."
            )
        assert sweep


class TestScanAmplification:
    """That the hot write path plans as index seeks, and keeps doing so.

    This class was written the other way round. Every node index came from schema.py's
    registries and was therefore partial -- `CREATE INDEX ... WHERE labels = '<Label>'` --
    and SQLite can only use one of those when the predicate names that same label as a
    **literal**. The three shapes that dominate the write path never do:

        labels = ?              a bound parameter, unknown when the plan is built
        labels IN (...)         does not imply any single-label predicate
        labels NOT IN (...)     implies the opposite of one

    So the file-hash gate, the per-file diff, the resolution lookups, the worktree sweep
    and the ADR-0036 dedup all fell to `SCAN nodes`, and this class reported that with
    "reported, never fixed here -- whether 124 partial indices are the right shape is a
    separate decision that should be taken against a number." The number was 64 row
    visits at 50 entities and 184 at 146, on a batch held constant.

    The decision was then taken. Three unqualified indices -- `(project_name, file_path)`,
    `(labels, project_name, name)`, and a partial expression index on `embed_hash` --
    took twelve scanning shapes to zero. Not more partial indices, and not a change in
    schema.py either: those registries also drive Memgraph, which has no label-free index
    and genuinely needs per-label ones.

    So the class inverts. It now fails if a scan comes back, and it carries its own
    control statement, because "no scans found" is also what a silent detector reports.
    """

    def test_the_detector_is_live(self, sweep):
        """Checked first, because every assertion below reads a zero as good news."""
        for point in sweep:
            assert point["detector_live"], (
                f"ballast={point['ballast_files']}: the control statement, which cannot be index-served, "
                "was not flagged — the scan report is silent and its zeros mean nothing"
            )

    def test_amplification_is_reported_per_corpus_size(self, sweep):
        rows = [
            (
                int(p["ballast_files"]),
                int(p["graph_entities"]),
                int(p["scanning_statements"]),
                int(p["scanning_executions"]),
                int(p["scan_amplification"]),
            )
            for p in sweep
        ]
        print(f"\n(ballast, entities, scanning_stmts, scanning_execs, row_visits): {rows}")
        assert rows

    def test_nothing_on_the_write_path_scans(self, sweep):
        """The regression this exists to catch.

        A scan reappearing is not a slow query in isolation -- it is a query whose cost
        is the size of the whole graph, on a path that runs per file. That is invisible
        on a test corpus and fatal on a real one, which is the only reason it survived
        this long.
        """
        for point in sweep:
            ops = sorted(point["scanning_ops"])
            assert not ops, (
                f"ballast={point['ballast_files']}: {ops} plan as a full scan of nodes "
                f"({point['scan_amplification']} row visits). An index was dropped, or a new predicate "
                "uses `labels IN`/`NOT IN`/`= ?` in a shape the three unqualified indices do not cover."
            )

    def test_the_file_hash_gate_seeks_its_index(self, sweep):
        """Named specifically because it is the sharpest instance.

        `_get_batch_file_prop` exists to make an unchanged file cheap. Scanning the node
        table to decide that a file has not changed inverts the gate's entire purpose,
        and does so more expensively the larger the graph gets.
        """
        for point in sweep:
            assert "_get_batch_file_prop" not in set(point["scanning_ops"]), (
                f"ballast={point['ballast_files']}: the file-hash gate is scanning again"
            )
