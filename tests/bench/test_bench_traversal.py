"""Graph-traversal cost against graph size.

`trace_path_between`, `get_callers`/`get_callees` and the dead-code candidate query all
walk edges, and their cost is a property of the graph rather than of the query. A tool
whose cost is superlinear in graph size is fine on this repository and unusable on the
one the user actually has — which is the whole reason this suite measures against a
sweep rather than a single corpus.

Round-trips are the number, not milliseconds: `_bfs_reachable` issues one statement per
depth level, so a traversal that suddenly costs more round-trips has changed shape, while
one that costs more milliseconds may only have met a busier machine.

Runs on the embedded backend: no Docker, no network.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from code_atlas.bench import capture, stub_provider

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.bench, pytest.mark.slow]

# Two sizes, because the claim is about shape. A third point costs another full index
# for a curve that two points already make legible.
_SIZES = (10, 40)


def _write_chain(root: Path, count: int) -> None:
    """A call chain, so traversal has depth to walk rather than a flat fan.

    Each module calls the previous one's helper, giving a CALLS path of length *count*.
    A corpus of isolated functions would make every traversal terminate at depth 1 and
    the measurement would describe nothing.
    """
    pkg = root / "src"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    for i in range(count):
        prev = f"from src.m{i - 1} import step{i - 1}\n" if i else ""
        body = f"    return step{i - 1}(x) + {i}\n" if i else f"    return x + {i}\n"
        (pkg / f"m{i}.py").write_text(
            f'{prev}\n\ndef step{i}(x: int) -> int:\n    """Step {i}."""\n{body}',
            encoding="utf-8",
        )
    (root / "README.md").write_text("# Chain\n\nA call chain.\n", encoding="utf-8")


async def _point(tmp_path: Path, size: int) -> dict[str, object]:
    from code_atlas.backends import use_backends
    from code_atlas.indexing.orchestrator import index_project
    from code_atlas.settings import AtlasSettings

    root = tmp_path / f"chain_{size}"
    root.mkdir(parents=True)
    _write_chain(root, size)

    settings = AtlasSettings(
        project_root=root,
        backend={"graph": "sqlite", "queue": "sqlite", "sqlite_data_dir": str(tmp_path / f"db_{size}")},
        embeddings={"dimension": 16},
    )
    project = f"bench-chain-{size}"
    cap = capture()

    with stub_provider(16):
        async with use_backends(settings, with_bus=True) as backends:
            # `index_project` does not create the schema — the CLI calls `ensure_schema`
            # separately, and without it every side-table write is silently skipped.
            await backends.graph.ensure_schema()
            await index_project(
                settings,
                backends.graph,  # ty: ignore[invalid-argument-type]  # index_project accepts either backend
                backends.bus,  # ty: ignore[invalid-argument-type]  # index_project accepts either backend
                full_reindex=True,
                project_name=project,
            )

            graph = backends.graph
            entities = await graph.count_entities(project)
            # `{project}:{module_qn}.{name}` — no label segment, and `src` is a source
            # root rather than a package so it does not appear. Guessed wrong the first
            # time (`{project}:callable:src.m0.step0`), which made every traversal find
            # no node and return an empty walk in one round-trip. That reads as a
            # reassuringly flat curve, which is why the connectivity guard below exists.
            head = f"{project}:m{size - 1}.step{size - 1}"
            tail = f"{project}:m0.step0"

            measured: dict[str, dict[str, int]] = {}
            for name, call in (
                ("trace_path", lambda: graph.trace_path_between(head, tail, 12, ("CALLS",))),
                ("callers", lambda: graph.get_callers(tail, "Callable", 6, 50)),
                ("callees", lambda: graph.get_callees(head, "Callable", 6, 50)),
                ("dead_code", lambda: graph.get_dead_code_candidates(project, "")),
            ):
                cap.clear()
                before = sum(cap.round_trips().values())
                result = await call()
                after = sum(cap.round_trips().values())
                measured[name] = {
                    "round_trips": after - before,
                    "results": len(result) if isinstance(result, list) else 1,
                }

    return {"files": size, "entities": entities, "tools": measured}


@pytest.fixture(scope="module")
def sweep(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, object]]:
    import asyncio

    base = tmp_path_factory.mktemp("traversal")

    async def _run():
        return [await _point(base, size) for size in _SIZES]

    return asyncio.run(_run())


class TestTheSweepIsValid:
    def test_the_graph_grew(self, sweep):
        sizes = [int(p["entities"]) for p in sweep]
        assert sizes[-1] > sizes[0] * 2, f"the sweep is too flat to show a shape: {sizes}"

    def test_the_chain_is_actually_connected(self, sweep):
        """Non-vacuity for every claim below.

        A traversal over a graph with no edges costs one round-trip and finds nothing,
        which would make the whole file look reassuringly flat while measuring an empty
        walk.
        """
        tools = sweep[-1]["tools"]
        assert int(tools["callers"]["results"]) > 0 or int(tools["callees"]["results"]) > 0, (
            f"no CALLS edges were traversed at either end of the chain: {tools}"
        )


class TestTraversalCostAgainstGraphSize:
    def test_report_the_curve(self, sweep):
        """Reported, not budgeted. The shape is the deliverable."""
        rows = [
            {
                "files": p["files"],
                "entities": p["entities"],
                **{f"{tool}_rt": int(stats["round_trips"]) for tool, stats in p["tools"].items()},
                **{f"{tool}_n": int(stats["results"]) for tool, stats in p["tools"].items()},
            }
            for p in sweep
        ]
        print(f"\n{json.dumps({'traversal': rows}, indent=2)}")
        assert rows

    def test_bfs_round_trips_track_depth_not_graph_size(self, sweep):
        """`_bfs_reachable` issues one statement per depth level, so a bounded-depth walk
        should cost a bounded number of round-trips however large the graph gets.

        This is the property worth pinning: if it ever stops holding, a traversal has
        started doing per-node work and the tools that use it will degrade on exactly the
        graphs where they matter most.
        """
        first, last = sweep[0]["tools"], sweep[-1]["tools"]
        for tool in ("callers", "callees"):
            small = int(first[tool]["round_trips"])
            large = int(last[tool]["round_trips"])
            assert large <= small * 3, (
                f"{tool} round-trips grew {small} -> {large} for a 4x larger graph at the same "
                "depth limit — the walk is doing work proportional to the graph, not the path"
            )
