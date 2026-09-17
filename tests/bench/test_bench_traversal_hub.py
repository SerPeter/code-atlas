"""Traversal latency on a hub-heavy call graph, Memgraph against SQLite.

**Why this file compares backends when its siblings refuse to.** Every other bench file
measures one engine against its own history, because two engines' numbers differ for
reasons nobody chose (tokenisers, index kinds, neighbours). This file answers a one-off
design question about *algorithm shape*, not engine speed:

- Memgraph's `trace_path_between` / `compute_blast_radius` ran
  `MATCH p=(a)-[:T*1..N]->(b)` and aggregated `min(length(p))` / `max(reduce(...))` over
  **every path**, so their work grew with the number of paths, F^depth. `get_callers` /
  `get_callees` had the same shape followed by `DISTINCT`. They now run native `*BFS` /
  `*WSHORTEST` expansions, which should grow with the edges touched, like SQLite.
- SQLite's `_bfs_reachable` / `_bfs_shortest_path` walk a frontier, one indexed
  `IN (frontier)` statement per hop, keeping the best score per node, so their work
  should grow with the edges touched.

To isolate path explosion from graph size the corpus is a layered call DAG whose node
count is fixed while the fan-out F varies: from the entry there are F^d paths of length
d, but never more than W nodes per layer.

**The agreement guard is what makes a timing mean anything.** A traversal that returns
the wrong set — or nothing — is the fastest traversal there is. Every measured case
asserts that both backends reached the same uids at the same `min_depth` with the same
`confidence_score` (for traces, the same `hop_count` and `path_weight`; for callers and
callees, the same uids in the same order after the LIMIT). A
disagreement means one side is wrong and its number is meaningless.

Latency is reported, never asserted. A Memgraph query that exceeds the client's query
timeout is recorded as a `"timeout"` data point and deeper depths for that operation are
skipped: past that point the answer to the design question is already known.

Needs Docker (testcontainers). Never under `-n`: `graph_client` wipes the database
before each test, which is also why every test here indexes its own fan-out once.
"""

from __future__ import annotations

import json
import random
import statistics
import time
from typing import TYPE_CHECKING, Any

import pytest

from code_atlas.backends.sqlite_graph import SqliteGraphClient
from code_atlas.bench import capture, stub_provider
from code_atlas.graph.client import QueryTimeoutError
from tests.bench.conftest import write_bench_result

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from code_atlas.events import EventBus
    from code_atlas.graph.client import GraphClient
    from code_atlas.settings import AtlasSettings

pytestmark = [pytest.mark.bench, pytest.mark.integration]

_L = 7
"""Hops from the entry to the hub: entry (layer 0), W-wide layers 1..L-1, hub (layer L)."""
_W = 20
_FANOUTS = (2, 4, 8)
_DEPTHS = (3, 4, 5, 6)
_RUNS = 3
_MG_QUERY_TIMEOUT_S = 30.0
"""Client-side read timeout for Memgraph while measuring (the default is 10s)."""
_STEP = 7
"""Callee stride. Coprime with W, so the F callees of one function are distinct for F <= W."""

_SQLITE_DIM = 16
_NEIGHBOUR_LIMIT = 50
"""Callers/callees LIMIT: below the deepest reachable set (W per layer), so the cut is exercised."""

_ROWS: list[dict[str, Any]] = []
_CORPORA: list[dict[str, Any]] = []


def _callees(i: int, fanout: int) -> list[int]:
    return [(i + j * _STEP) % _W for j in range(fanout)]


def _fn(layer: int, i: int) -> str:
    return f"f_{layer}_{i}"


def _write_corpus(root: Path, fanout: int) -> int:
    """Write the layered DAG and return the CALLS edge count it implies.

    One function per module so every call is a plain `from src... import name`, which the
    resolver handles without receiver typing. Module files are named apart from their
    functions (`mod_hub.py` defines `hub`) so a lookup by name finds exactly one node.
    """
    src = root / "src"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("", encoding="utf-8")

    def module(path: Path, name: str, targets: list[tuple[str, str]]) -> None:
        imports = "".join(f"from {mod} import {fn}\n" for mod, fn in targets)
        body = " + ".join(f"{fn}(x)" for _mod, fn in targets) if targets else "x"
        path.write_text(
            f'{imports}\n\ndef {name}(x: int) -> int:\n    """{name}."""\n    return {body}\n',
            encoding="utf-8",
        )

    def target(layer: int, i: int) -> tuple[str, str]:
        return f"src.l{layer}.mod_{layer}_{i}", _fn(layer, i)

    module(src / "mod_entry.py", "entry", [target(1, c) for c in _callees(0, fanout)])
    for layer in range(1, _L):
        pkg = src / f"l{layer}"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        for i in range(_W):
            targets = (
                [target(layer + 1, c) for c in _callees(i, fanout)] if layer < _L - 1 else [("src.mod_hub", "hub")]
            )
            module(pkg / f"mod_{layer}_{i}.py", _fn(layer, i), targets)
    module(src / "mod_hub.py", "hub", [])
    (root / "README.md").write_text("# Hub corpus\n\nA layered call DAG.\n", encoding="utf-8")
    return fanout + (_L - 2) * _W * fanout + _W


def _round_trips() -> int:
    return sum(capture().round_trips().values())


async def _terminate_stragglers(client: GraphClient) -> int:
    """A client-side timeout cancels the await, not necessarily the server's work.

    Left running, an abandoned path enumeration would inflate every Memgraph number that
    follows it, so kill whatever else is still executing.
    """
    rows = await client.execute("SHOW TRANSACTIONS")
    killed = 0
    for row in rows:
        query = " ".join(str(q) for q in (row.get("query") or []))
        if "SHOW TRANSACTIONS" in query or "TERMINATE" in query:
            continue
        await client.execute(f'TERMINATE TRANSACTIONS "{row["transaction_id"]}"')
        killed += 1
    return killed


async def _measure(call: Callable[[], Awaitable[Any]]) -> dict[str, Any]:
    times: list[float] = []
    round_trips = 0
    result: Any = None
    for _ in range(_RUNS):
        before = _round_trips()
        start = time.perf_counter()
        result = await call()
        times.append((time.perf_counter() - start) * 1000.0)
        round_trips = _round_trips() - before
    return {"ms": round(statistics.median(times), 2), "round_trips": round_trips, "result": result}


def _blast_key(rows: list[dict[str, Any]]) -> dict[str, tuple[int, float]]:
    return {r["uid"]: (int(r["min_depth"]), float(r["confidence_score"])) for r in rows}


def _nodes_disagreement(sq: list[Any], mg: list[Any]) -> str | None:
    sq_uids, mg_uids = [n["uid"] for n in sq], [n["uid"] for n in mg]
    return None if sq_uids == mg_uids else f"ordered uids differ: sqlite={sq_uids[:5]} memgraph={mg_uids[:5]}"


def _disagreement(kind: str, sq: Any, mg: Any) -> str | None:
    if kind == "trace":
        for key in ("found", "hop_count"):
            if sq[key] != mg[key]:
                return f"{key}: sqlite={sq[key]!r} memgraph={mg[key]!r}"
        if sq["found"] and abs(float(sq["path_weight"]) - float(mg["path_weight"])) > 1e-9:
            return f"path_weight: sqlite={sq['path_weight']!r} memgraph={mg['path_weight']!r}"
        return None
    a, b = _blast_key(sq), _blast_key(mg)
    if set(a) != set(b):
        only_sq, only_mg = sorted(set(a) - set(b))[:5], sorted(set(b) - set(a))[:5]
        return f"reached sets differ ({len(a)} vs {len(b)}): only sqlite={only_sq} only memgraph={only_mg}"
    for uid, (depth, score) in a.items():
        mg_depth, mg_score = b[uid]
        if depth != mg_depth or abs(score - mg_score) > 1e-9:
            return f"{uid}: sqlite=(min_depth={depth}, score={score}) memgraph=(min_depth={mg_depth}, score={mg_score})"
    return None


def _count(kind: str, result: Any) -> int:
    """Reached entities for a blast radius or neighbour list; hop count of the found path (0 = none) for a trace."""
    if kind == "trace":
        return int(result["hop_count"] or 0) if result["found"] else 0
    return len(result)


async def _index_memgraph(root: Path, project: str, graph_client: GraphClient, event_bus: EventBus) -> float:
    from code_atlas.indexing.orchestrator import index_project
    from code_atlas.settings import AtlasSettings

    # `index_project` does not create the schema; without it every search index write is
    # skipped and the run still looks clean.
    await graph_client.ensure_schema()
    settings = AtlasSettings(project_root=root, embeddings={"dimension": graph_client.dimension})
    start = time.perf_counter()
    with stub_provider(graph_client.dimension):
        await index_project(
            settings, graph_client, event_bus, full_reindex=True, project_name=project, drain_timeout_s=600.0
        )
    return time.perf_counter() - start


async def _index_sqlite(root: Path, project: str, data_dir: Path) -> tuple[AtlasSettings, float]:
    from code_atlas.backends import use_backends
    from code_atlas.indexing.orchestrator import index_project
    from code_atlas.settings import AtlasSettings

    # `false` un-declares the network backends the Memgraph fixtures export through
    # ATLAS_* env vars; without it this would be a two-backend config error.
    settings = AtlasSettings(
        project_root=root,
        backend={
            "graph": {"sqlite": {}, "memgraph": False},
            "queue": {"sqlite": {}, "valkey": False},
            "sqlite_data_dir": str(data_dir),
        },
        embeddings={"dimension": _SQLITE_DIM},
    )
    start = time.perf_counter()
    with stub_provider(_SQLITE_DIM):
        async with use_backends(settings, with_bus=True) as backends:
            await backends.graph.ensure_schema()
            await index_project(
                settings,
                backends.graph,  # ty: ignore[invalid-argument-type]  # index_project accepts either backend
                backends.bus,  # ty: ignore[invalid-argument-type]  # index_project accepts either backend
                full_reindex=True,
                project_name=project,
            )
    return settings, time.perf_counter() - start


async def _assert_non_vacuous(sq: SqliteGraphClient, mg: GraphClient, expected_calls: int) -> dict[str, str]:
    """Pin the corpus on both backends before anything is timed; return the probe uids.

    A corpus whose calls did not resolve would time an empty walk, and an empty walk is
    the fastest walk there is.
    """
    conn = await sq._get_conn()
    cur = await conn.execute("SELECT count(*) FROM edges WHERE rel_type = 'CALLS'")
    row = await cur.fetchone()
    await cur.close()
    sq_calls = int(row[0]) if row else 0
    mg_calls = int((await mg.execute("MATCH ()-[r:CALLS]->() RETURN count(r) AS n"))[0]["n"])
    assert sq_calls == expected_calls, f"SQLite resolved {sq_calls} CALLS edges; the corpus implies {expected_calls}"
    assert mg_calls == expected_calls, f"Memgraph resolved {mg_calls} CALLS edges; the corpus implies {expected_calls}"

    uids: dict[str, str] = {}
    for name in ("entry", "hub", *(_fn(d, 0) for d in _DEPTHS)):
        cur = await conn.execute("SELECT uid FROM nodes WHERE name = ? AND labels = 'Callable'", (name,))
        sq_uids = [r[0] for r in await cur.fetchall()]
        await cur.close()
        mg_uids = [
            r["uid"] for r in await mg.execute("MATCH (n:Callable {name: $name}) RETURN n.uid AS uid", {"name": name})
        ]
        assert len(sq_uids) == 1, f"SQLite has {len(sq_uids)} callables named {name}"
        assert sq_uids == mg_uids, f"the backends disagree on the uid of {name}: {sq_uids} vs {mg_uids}"
        uids[name] = sq_uids[0]

    for label, client in (("sqlite", sq), ("memgraph", mg)):
        reach = await client.trace_path_between(uids["entry"], uids["hub"], _L, ("CALLS",))
        assert reach["found"], f"{label}: the entry does not reach the hub, so every walk below is vacuous"
        assert reach["hop_count"] == _L, f"{label}: the entry reaches the hub in {reach['hop_count']} hops, not {_L}"
    return uids


def _ops(client: Any, uids: dict[str, str], depth: int) -> dict[str, Callable[[], Awaitable[Any]]]:
    entry, hub, layer_d = uids["entry"], uids["hub"], uids[_fn(depth, 0)]
    return {
        "blast_out_entry": lambda: client.compute_blast_radius(entry, "out", ("CALLS",), depth),
        "blast_in_hub": lambda: client.compute_blast_radius(hub, "in", ("CALLS",), depth),
        # The hub sits L hops down, so for every depth measured here this is a miss —
        # which Memgraph still pays for by enumerating every path up to `depth`.
        "trace_entry_hub": lambda: client.trace_path_between(entry, hub, depth, ("CALLS",)),
        "trace_entry_layer_d": lambda: client.trace_path_between(entry, layer_d, depth, ("CALLS",)),
        "callers_hub": lambda: client.get_callers(hub, "Callable", depth, _NEIGHBOUR_LIMIT),
        "callees_entry": lambda: client.get_callees(entry, "Callable", depth, _NEIGHBOUR_LIMIT),
        # The other two directions are empty (nothing calls the entry, the hub calls nothing),
        # measured so a regression that makes an empty answer expensive would show.
        "callers_entry": lambda: client.get_callers(entry, "Callable", depth, _NEIGHBOUR_LIMIT),
        "callees_hub": lambda: client.get_callees(hub, "Callable", depth, _NEIGHBOUR_LIMIT),
    }


async def _measure_all(sq: SqliteGraphClient, mg: GraphClient, uids: dict[str, str], fanout: int) -> list[str]:
    disagreements: list[str] = []
    timed_out: set[str] = set()
    for depth in _DEPTHS:
        sq_ops, mg_ops = _ops(sq, uids, depth), _ops(mg, uids, depth)
        for op, sq_call in sq_ops.items():
            kind = "trace" if op.startswith("trace") else "nodes" if op.startswith("call") else "blast"
            base = {"fanout": fanout, "depth": depth, "op": op}
            sq_m = await _measure(sq_call)
            _ROWS.append(
                {
                    "backend": "sqlite",
                    **base,
                    "ms": sq_m["ms"],
                    "round_trips": sq_m["round_trips"],
                    "results": _count(kind, sq_m["result"]),
                }
            )
            row: dict[str, Any] = {"backend": "memgraph", **base}
            _ROWS.append(row)
            if op in timed_out:
                row["status"] = "skipped_after_timeout"
                continue
            try:
                mg_m = await _measure(mg_ops[op])
            except QueryTimeoutError:
                timed_out.add(op)
                terminated = await _terminate_stragglers(mg)
                row.update(status="timeout", limit_s=_MG_QUERY_TIMEOUT_S, terminated=terminated)
                continue
            row.update(ms=mg_m["ms"], round_trips=mg_m["round_trips"], results=_count(kind, mg_m["result"]))
            problem = (
                _nodes_disagreement(sq_m["result"], mg_m["result"])
                if kind == "nodes"
                else _disagreement(kind, sq_m["result"], mg_m["result"])
            )
            if problem:
                disagreements.append(f"F={fanout} depth={depth} {op}: {problem}")
    return disagreements


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("fanout", _FANOUTS)
async def test_traversal_latency_by_fanout(
    fanout: int, graph_client: GraphClient, event_bus: EventBus, tmp_path: Path
) -> None:
    from code_atlas.backends import use_backends

    root = tmp_path / "corpus"
    root.mkdir()
    expected_calls = _write_corpus(root, fanout)
    project = f"bench-hub-f{fanout}"

    # Installed before indexing: `capture()` is what enables telemetry, and installing it
    # afterwards finds an empty reader.
    capture()

    mg_index_s = await _index_memgraph(root, project, graph_client, event_bus)
    sq_settings, sq_index_s = await _index_sqlite(root, project, tmp_path / "sqlite")

    async with use_backends(sq_settings, with_bus=False) as backends:
        sq = backends.graph
        assert isinstance(sq, SqliteGraphClient), "the comparison side must be the embedded backend"
        uids = await _assert_non_vacuous(sq, graph_client, expected_calls)
        graph_client._query_timeout_s = _MG_QUERY_TIMEOUT_S  # the measured limit, not the setup one
        _CORPORA.append(
            {
                "fanout": fanout,
                "layers": _L,
                "width": _W,
                "callables": 2 + (_L - 1) * _W,
                "calls_edges": expected_calls,
                "index_s": {"memgraph": round(mg_index_s, 1), "sqlite": round(sq_index_s, 1)},
            }
        )
        disagreements = await _measure_all(sq, graph_client, uids, fanout)

    # Written after every fan-out, so a later one that dies still leaves the earlier data.
    payload = {"corpora": _CORPORA, "rows": _ROWS}
    write_bench_result("traversal_hub", payload)
    print(f"\n{json.dumps(payload, indent=2)}")

    assert not disagreements, (
        "BACKENDS DISAGREE — one side returns a wrong answer, so its timing is meaningless:\n  "
        + "\n  ".join(disagreements)
    )


# ---------------------------------------------------------------------------
# A high-degree target
# ---------------------------------------------------------------------------

_HD_PROJECT = "bench-high-degree"
_HD_DIRECT = 2000
"""Direct dependents of the target, landing on it over three edge types."""
_HD_UPSTREAM = 100
_HD_UPSTREAM_FANOUT = 500
"""Each upstream function calls this many direct dependents, so every direct dependent has
~25 callers and the region behind any one of them overlaps all the others."""
_HD_TOP = 20
_HD_DEPTHS = (3, 4, 5, 6)
_HD_TYPES = ("CALLS", "USES_TYPE", "IMPORTS")

_Edge = tuple[str, str, str, dict[str, Any]]


def _hd_graph() -> tuple[list[str], list[_Edge]]:
    """Target <- 2,000 direct dependents <- 100 upstream <- 20 top <- 1 root.

    Every edge is `(dependent)-[:T]->(dependency)`, so the measured traversal is
    `compute_blast_radius(target, "in", ...)`. A third of the direct dependents land over
    two edge types, so `via` has real sets to collect.
    """
    rng = random.Random(20260917)
    target = f"{_HD_PROJECT}:target"
    direct = [f"{_HD_PROJECT}:d{i:04d}" for i in range(_HD_DIRECT)]
    upstream = [f"{_HD_PROJECT}:u{i:03d}" for i in range(_HD_UPSTREAM)]
    top = [f"{_HD_PROJECT}:t{i:02d}" for i in range(_HD_TOP)]
    root = f"{_HD_PROJECT}:root"
    edges: list[_Edge] = []
    for i, d in enumerate(direct):
        first = _HD_TYPES[i % 3]
        edges.append((d, target, first, {"line": 10 + i % 7, "weight": 0.5 if first == "USES_TYPE" else 1.0}))
        if i % 3 == 0:
            edges.append((d, target, "USES_TYPE", {"line": 90, "weight": 0.5}))
    for u in upstream:
        edges.extend(
            (u, d, "CALLS", {"line": 3, "weight": rng.choice((1.0, 0.5, 0.25))})
            for d in rng.sample(direct, _HD_UPSTREAM_FANOUT)
        )
    edges.extend((t, u, "CALLS", {"weight": 1.0}) for t in top for u in upstream)
    edges.extend((root, t, "IMPORTS", {}) for t in top)
    return [target, *direct, *upstream, *top, root], edges


def _hd_oracle(nodes: list[str], edges: list[_Edge], start: str, depth: int) -> dict[str, tuple[Any, ...]]:
    """uid -> (min_depth, best weight, via, at_lines), by a level walk in plain Python.

    Walks rather than trails: every weight is in (0, 1] and the corpus has no cycles, so
    the two agree.
    """
    inbound: dict[str, list[tuple[str, str, dict[str, Any]]]] = {n: [] for n in nodes}
    for src, dst, rel, props in edges:
        inbound[dst].append((src, rel, props))
    min_depth: dict[str, int] = {}
    best: dict[str, float] = {}
    via: dict[str, set[str]] = {}
    lines: dict[str, set[int]] = {}
    frontier: dict[str, tuple[float, frozenset[str]]] = {}
    for src, rel, props in inbound[start]:
        min_depth.setdefault(src, 1)
        lines.setdefault(src, set()).add(props["line"])
        w, types = frontier.get(src, (0.0, frozenset()))
        frontier[src] = (max(w, props.get("weight", 1.0)), types | {rel})
    for node, (w, types) in frontier.items():
        best[node], via[node] = w, set(types)
    for level in range(2, depth + 1):
        nxt: dict[str, tuple[float, frozenset[str]]] = {}
        for node, (w, types) in frontier.items():
            for src, _rel, props in inbound[node]:
                score = w * props.get("weight", 1.0)
                min_depth.setdefault(src, level)
                grew = score > best.get(src, 0.0) or not types <= via.get(src, set())
                best[src] = max(best.get(src, 0.0), score)
                via.setdefault(src, set()).update(types)
                if grew:
                    pw, pt = nxt.get(src, (0.0, frozenset()))
                    nxt[src] = (max(pw, score), pt | types)
        frontier = nxt
    return {n: (d, best[n], sorted(via[n]), sorted(lines.get(n, ())) if d == 1 else []) for n, d in min_depth.items()}


def _hd_disagreement(expected: dict[str, tuple[Any, ...]], rows: list[dict[str, Any]]) -> str | None:
    got = {r["uid"]: (r["min_depth"], r["confidence_score"], r["via"], r.get("at_lines", [])) for r in rows}
    if set(got) != set(expected):
        return f"reached sets differ: {len(got)} vs expected {len(expected)}"
    for uid, (d, w, via, at) in expected.items():
        gd, gw, gvia, gat = got[uid]
        if gd != d or abs(gw - w) > 1e-9 or gvia != via or gat != at:
            return f"{uid}: got {got[uid]!r}, expected {(d, w, via, at)!r}"
    return None


async def _write_hd_graph(client: GraphClient, nodes: list[str], edges: list[_Edge]) -> None:
    await client.execute_write(
        "UNWIND $uids AS uid CREATE (:Callable:Entity {uid: uid, name: uid, project_name: $p})",
        {"uids": nodes, "p": _HD_PROJECT},
    )
    await client.execute_write("CREATE INDEX ON :Entity(uid)")
    for rel in sorted({e[2] for e in edges}):
        batch = [{"f": f, "t": t, "props": props} for f, t, r, props in edges if r == rel]
        for i in range(0, len(batch), 5000):
            await client.execute_write(
                f"UNWIND $rows AS row MATCH (a:Entity {{uid: row.f}}), (b:Entity {{uid: row.t}}) "
                f"CREATE (a)-[e:{rel}]->(b) SET e = row.props",
                {"rows": batch[i : i + 5000]},
            )
    stored = (await client.execute("MATCH ()-[r]->() RETURN count(r) AS n"))[0]["n"]
    assert stored == len(edges), f"wrote {stored} of {len(edges)} edges"


@pytest.mark.timeout(1800)
async def test_blast_radius_on_a_high_degree_target(graph_client: GraphClient) -> None:
    """Memgraph only, written straight to the graph: the question is how `via` scales with
    a target's in-degree, and the Python oracle above is the agreement guard."""
    nodes, edges = _hd_graph()
    await _write_hd_graph(graph_client, nodes, edges)
    capture()
    graph_client._query_timeout_s = _MG_QUERY_TIMEOUT_S
    target = nodes[0]
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for depth in _HD_DEPTHS:
        row: dict[str, Any] = {"backend": "memgraph", "depth": depth, "op": "blast_in_target"}
        rows.append(row)
        try:
            m = await _measure(lambda d=depth: graph_client.compute_blast_radius(target, "in", _HD_TYPES, d))
        except QueryTimeoutError:
            terminated = await _terminate_stragglers(graph_client)
            row.update(status="timeout", limit_s=_MG_QUERY_TIMEOUT_S, terminated=terminated)
            break
        row.update(ms=m["ms"], round_trips=m["round_trips"], results=len(m["result"]))
        problem = _hd_disagreement(_hd_oracle(nodes, edges, target, depth), m["result"])
        if problem:
            problems.append(f"depth={depth}: {problem}")

    payload = {"corpus": {"nodes": len(nodes), "edges": len(edges), "direct": _HD_DIRECT}, "rows": rows}
    write_bench_result("traversal_high_degree", payload)
    print(f"\n{json.dumps(payload, indent=2)}")
    assert not problems, "WRONG ANSWER, so the timing is meaningless:\n  " + "\n  ".join(problems)
