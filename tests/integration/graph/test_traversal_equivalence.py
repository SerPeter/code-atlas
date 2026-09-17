"""Pin Memgraph's traversal results against the all-paths reference.

`trace_path_between`, `compute_blast_radius`, `get_callers` and `get_callees` used to run
`MATCH p=(a)-[:T*1..N]->(b)` and aggregate over **every** path. That shape is exact by
construction and exponential in fan-out. This file keeps a verbatim copy of those queries
as the reference, and checks that whatever the client runs now returns the same answer,
field for field, on random small multigraphs and on hand-built graphs that aim at the
places a cheaper traversal gets wrong:

- a best-weight path LONGER than the shortest path;
- a best-weight path that only fits a larger hop budget;
- a node reachable only through ambiguous or only through test edges;
- equal-length ties with different weights (trace);
- `via` collected from several first-hop types.

Graphs stay at or under 10 nodes and 22 edges, so the reference's path enumeration is
cheap. Every example writes its graph under a `test`-prefixed project (the wipe guard)
and wipes the database first; `graph_client` is function-scoped and shared by all the
examples of one test.

Weights mirror what the writers produce: `_call_edge_weight` gives `1/candidate_count`,
damped by 0.5 (unverified receiver) and 0.25 (`from_test`), floored at 1e-6; USES_TYPE
and inferred IMPLEMENTS carry 1.0 or 0.5; IMPORTS carries none. So every stored weight is
in (0, 1], and an absent one counts as 1.0.

The reference is the old query with one deliberate divergence: `at_lines` for a direct
dependent holds only the lines of the edges joining it to the queried entity, where the
old query also took first-hop lines from longer routes (see `ref_compute_blast_radius`).

Never under `-n`: see `graph_client`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from hypothesis import HealthCheck, currently_in_test_context, event, given, note
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st
from neo4j.exceptions import ClientError

from code_atlas.graph.client import _DEFAULT_EDGE_WEIGHT, _LIMITED_QUERY_ORDER, _direct_call_lines, _format_path_hops
from code_atlas.schema import NodeLabel, RelType, primary_label_expr

if TYPE_CHECKING:
    from code_atlas.graph.client import GraphClient

pytestmark = pytest.mark.integration

_PROJECT = "test_traversal_eq"
_TOL = 1e-9
_TYPES = (RelType.CALLS.value, RelType.IMPORTS.value, RelType.USES_TYPE.value)

# Every weight a writer can produce, and "no property" (None).
_WEIGHTS = (None, 1.0, 0.5, 1 / 3, 0.25, 1 / 6, 0.125, 1e-6)


# ---------------------------------------------------------------------------
# Graph model
# ---------------------------------------------------------------------------


@dataclass
class Node:
    label: str
    marker: bool
    name: str
    qualified_name: str | None
    file_path: str
    kind: str


@dataclass
class Edge:
    src: int
    dst: int
    rel_type: str
    props: dict[str, Any] = field(default_factory=dict)


def _uid(i: int) -> str:
    return f"{_PROJECT}:n{i}"


async def _write_graph(client: GraphClient, nodes: list[Node], edges: list[Edge]) -> None:
    await client.execute_write("MATCH (n) DETACH DELETE n")
    if not nodes:
        return
    params: dict[str, Any] = {}
    parts: list[str] = []
    for i, nd in enumerate(nodes):
        labels = f":{nd.label}" + (f":{NodeLabel.ENTITY}" if nd.marker else "")
        props = {
            "uid": _uid(i),
            "project_name": _PROJECT,
            "name": nd.name,
            "file_path": nd.file_path,
            "kind": nd.kind,
        }
        if nd.qualified_name is not None:
            props["qualified_name"] = nd.qualified_name
        if i % 2:
            # A vector on some nodes: callers/callees must not ship it, and must keep the rest.
            props["embedding"] = [0.25, 0.5, 0.75]
            props["docstring"] = f"doc {i}"
        params[f"p{i}"] = props
        parts.append(f"(n{i}{labels} $p{i})")
    query = "CREATE " + ", ".join(parts)
    for j, e in enumerate(edges):
        params[f"e{j}"] = e.props
        query += f" CREATE (n{e.src})-[:{e.rel_type} $e{j}]->(n{e.dst})"
    await client.execute_write(query, params)


def _hop_view(nodes: list[Node], e: Edge) -> dict[str, Any]:
    """What `_format_path_hops` renders for one model edge."""

    class _Rel(dict):
        type = e.rel_type

    return _format_path_hops(
        [
            {"uid": _uid(e.src), "name": nodes[e.src].name},
            {"uid": _uid(e.dst), "name": nodes[e.dst].name},
        ],
        [_Rel(e.props)],
    )[0]


# ---------------------------------------------------------------------------
# Reference: the all-paths queries, verbatim from client.py at d68ba19, except where
# marked DELIBERATE DIVERGENCE (a semantic fixed on purpose, not an optimisation)
# ---------------------------------------------------------------------------


async def ref_trace_path_between(
    client: GraphClient, from_uid: str, to_uid: str, max_depth: int, edge_types: tuple[str, ...]
) -> dict[str, Any]:
    params: dict[str, Any] = {"from_uid": from_uid, "to_uid": to_uid}
    exist_raw = await client.execute(
        "OPTIONAL MATCH (a {uid: $from_uid}) OPTIONAL MATCH (b {uid: $to_uid}) "
        "RETURN a IS NOT NULL AS from_exists, b IS NOT NULL AS to_exists",
        params,
    )
    exists = exist_raw[0] if exist_raw else {"from_exists": False, "to_exists": False}
    if not exists["from_exists"] or not exists["to_exists"]:
        return {
            "from_exists": exists["from_exists"],
            "to_exists": exists["to_exists"],
            "found": False,
            "hop_count": None,
            "hops": [],
            "path_weight": None,
        }

    rel_pattern = "|".join(edge_types)
    records = await client.execute(
        f"MATCH p=(a {{uid: $from_uid}})-[:{rel_pattern}*1..{max_depth}]->(b {{uid: $to_uid}}) "
        "RETURN nodes(p) AS path_nodes, relationships(p) AS path_rels, length(p) AS hops, "
        f"reduce(w = 1.0, r IN relationships(p) | w * coalesce(r.weight, {_DEFAULT_EDGE_WEIGHT})) AS path_weight "
        "ORDER BY hops, path_weight DESC LIMIT 1",
        params,
    )
    if not records:
        return {
            "from_exists": True,
            "to_exists": True,
            "found": False,
            "hop_count": None,
            "hops": [],
            "path_weight": None,
        }

    record = records[0]
    return {
        "from_exists": True,
        "to_exists": True,
        "found": True,
        "hop_count": record["hops"],
        "hops": _format_path_hops(record["path_nodes"], record["path_rels"]),
        "path_weight": record["path_weight"],
    }


async def ref_compute_blast_radius(
    client: GraphClient, uid: str, direction_kind: str, edge_types: tuple[str, ...], max_depth: int
) -> list[dict[str, Any]]:
    rel_pattern = "|".join(edge_types)
    pattern = (
        f"-[:{rel_pattern}*1..{max_depth}]->" if direction_kind == "out" else f"<-[:{rel_pattern}*1..{max_depth}]-"
    )
    all_raw = await client.execute(
        f"MATCH p=(start {{uid: $uid}}){pattern}(affected) "
        "WHERE affected.uid <> $uid "
        "RETURN affected.uid AS uid, affected.name AS name, affected.qualified_name AS qn, "
        f"{primary_label_expr('affected')} AS label, affected.file_path AS file_path, "
        "affected.kind AS kind, "
        "min(length(p)) AS min_depth, "
        f"max(reduce(w = 1.0, r IN relationships(p) | w * coalesce(r.weight, {_DEFAULT_EDGE_WEIGHT}))) "
        "AS confidence_score, "
        "collect(DISTINCT type(relationships(p)[0])) AS via, "
        # DELIBERATE DIVERGENCE from d68ba19, which collected `relationships(p)[0].line` over
        # every path: a direct dependent also reachable by a longer route then reported the
        # long route's first-hop line, which sits in another file. Only length-1 paths count.
        "collect(DISTINCT CASE WHEN length(p) = 1 THEN relationships(p)[0].line END) AS via_lines",
        {"uid": uid},
    )
    resolved_raw = await client.execute(
        f"MATCH p=(start {{uid: $uid}}){pattern}(affected) "
        "WHERE affected.uid <> $uid "
        "AND all(r IN relationships(p) WHERE coalesce(r.confidence, 'resolved') = 'resolved') "
        "RETURN DISTINCT affected.uid AS uid",
        {"uid": uid},
    )
    production_raw = await client.execute(
        f"MATCH p=(start {{uid: $uid}}){pattern}(affected) "
        "WHERE affected.uid <> $uid AND all(r IN relationships(p) WHERE NOT coalesce(r.from_test, false)) "
        "RETURN DISTINCT affected.uid AS uid",
        {"uid": uid},
    )
    resolved_uids = {r["uid"] for r in resolved_raw}
    production_uids = {r["uid"] for r in production_raw}
    return [
        {
            "uid": r["uid"],
            "name": r["name"],
            "qualified_name": r["qn"],
            "label": r["label"],
            "file_path": r["file_path"],
            "kind": r.get("kind"),
            "min_depth": r["min_depth"],
            "direction": direction_kind,
            "via": sorted(r.get("via") or []),
            "ambiguous_only": r["uid"] not in resolved_uids,
            "confidence_score": r["confidence_score"],
            "test_only": r["uid"] not in production_uids,
            **_direct_call_lines(r),
        }
        for r in all_raw
    ]


async def ref_get_callers(client: GraphClient, uid: str, label: str, call_depth: int, limit: int) -> list[Any]:
    label_clause = f":{label}" if label else ""
    records = await client.execute(
        f"MATCH (caller:Callable)-[:{RelType.CALLS}*1..{call_depth}]->"
        f"(n{label_clause} {{uid: $uid}}) "
        f"WITH DISTINCT caller AS n "
        f"RETURN n {_LIMITED_QUERY_ORDER} LIMIT {limit}",
        {"uid": uid},
    )
    return [r["n"] for r in records]


async def ref_get_callees(client: GraphClient, uid: str, label: str, call_depth: int, limit: int) -> list[Any]:
    label_clause = f":{label}" if label else ""
    records = await client.execute(
        f"MATCH (n{label_clause} {{uid: $uid}})-[:{RelType.CALLS}*1..{call_depth}]->"
        f"(callee:Callable) WITH DISTINCT callee AS n "
        f"RETURN n {_LIMITED_QUERY_ORDER} LIMIT {limit}",
        {"uid": uid},
    )
    return [r["n"] for r in records]


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------


def _assert_blast_equal(ref: list[dict[str, Any]], got: list[dict[str, Any]], ctx: str) -> None:
    ref_by = {r["uid"]: r for r in ref}
    got_by = {r["uid"]: r for r in got}
    assert len(ref_by) == len(ref), f"{ctx}: reference returned duplicate uids"
    assert len(got_by) == len(got), f"{ctx}: implementation returned duplicate uids: {[r['uid'] for r in got]}"
    assert set(got_by) == set(ref_by), (
        f"{ctx}: reached sets differ: only reference={sorted(set(ref_by) - set(got_by))} "
        f"only implementation={sorted(set(got_by) - set(ref_by))}"
    )
    for uid, r in ref_by.items():
        g = got_by[uid]
        assert set(g) == set(r), f"{ctx} {uid}: keys differ: reference={sorted(r)} implementation={sorted(g)}"
        for key, expected in r.items():
            actual = g[key]
            if key == "confidence_score":
                assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=_TOL), (
                    f"{ctx} {uid}: confidence_score reference={expected!r} implementation={actual!r}"
                )
            else:
                assert actual == expected, f"{ctx} {uid}: {key} reference={expected!r} implementation={actual!r}"


def _assert_trace_equal(
    ref: dict[str, Any],
    got: dict[str, Any],
    nodes: list[Node],
    edges: list[Edge],
    from_uid: str,
    to_uid: str,
    edge_types: tuple[str, ...],
    ctx: str,
) -> None:
    assert set(got) == set(ref), f"{ctx}: keys differ: reference={sorted(ref)} implementation={sorted(got)}"
    for key in ("from_exists", "to_exists", "found", "hop_count"):
        assert got[key] == ref[key], f"{ctx}: {key} reference={ref[key]!r} implementation={got[key]!r}"
    if not ref["found"]:
        assert got["path_weight"] is None, f"{ctx}: path_weight should be None, got {got['path_weight']!r}"
        assert got["hops"] == [], f"{ctx}: hops should be empty, got {got['hops']!r}"
        return
    assert math.isclose(got["path_weight"], ref["path_weight"], rel_tol=0.0, abs_tol=_TOL), (
        f"{ctx}: path_weight reference={ref['path_weight']!r} implementation={got['path_weight']!r}"
    )
    # Several paths may tie on (hops, weight) and the reference picks any of them, so the
    # implementation's path is checked for being a real one with that length and weight.
    hops = got["hops"]
    assert len(hops) == got["hop_count"], f"{ctx}: {len(hops)} hops rendered for hop_count {got['hop_count']}"
    assert hops[0]["from"]["uid"] == from_uid, f"{ctx}: path starts at {hops[0]['from']['uid']}"
    assert hops[-1]["to"]["uid"] == to_uid, f"{ctx}: path ends at {hops[-1]['to']['uid']}"
    uid_to_idx = {_uid(i): i for i in range(len(nodes))}
    product = 1.0
    for k, hop in enumerate(hops):
        if k:
            assert hop["from"]["uid"] == hops[k - 1]["to"]["uid"], f"{ctx}: hop {k} is not contiguous"
        candidates = [
            e
            for e in edges
            if e.rel_type in edge_types
            and uid_to_idx.get(hop["from"]["uid"]) == e.src
            and uid_to_idx.get(hop["to"]["uid"]) == e.dst
            and _hop_view(nodes, e) == hop
        ]
        assert candidates, f"{ctx}: hop {k} {hop!r} matches no edge in the graph"
        product *= candidates[0].props.get("weight", _DEFAULT_EDGE_WEIGHT)
    assert math.isclose(product, got["path_weight"], rel_tol=0.0, abs_tol=_TOL), (
        f"{ctx}: the returned hops multiply to {product!r}, not path_weight {got['path_weight']!r}"
    )


def _node_view(n: Any) -> tuple[str, frozenset[str], dict[str, Any]]:
    """Reference rows are Nodes; the implementation returns property maps with `_labels`
    and without `embedding` (never read by a consumer). Both reduce to the same view."""
    if hasattr(n, "labels"):
        labels, props = frozenset(n.labels), dict(n.items())
    else:
        props = dict(n)
        labels = frozenset(props.pop("_labels"))
    props.pop("embedding", None)
    return (props["uid"], labels, props)


def _assert_nodes_equal(ref: list[Any], got: list[Any], ctx: str) -> None:
    shipped = [n["uid"] for n in got if "embedding" in n]
    assert not shipped, f"{ctx}: the implementation shipped embeddings for {shipped}"
    assert [_node_view(n) for n in got] == [_node_view(n) for n in ref], (
        f"{ctx}: reference={[n['uid'] for n in ref]} implementation={[n['uid'] for n in got]}"
    )


def _coverage_events(rows: list[dict[str, Any]]) -> None:
    """Record which interesting outcomes the random graphs actually produced (--hypothesis-show-statistics)."""
    if not rows or not currently_in_test_context():
        return
    event("blast reached something")
    for key in ("ambiguous_only", "test_only"):
        if any(r[key] for r in rows) and not all(r[key] for r in rows):
            event(f"blast {key} mixed")
    if any(r["confidence_score"] < 1.0 for r in rows):
        event("blast confidence_score < 1")
    if any(r["min_depth"] >= 2 for r in rows):
        event("blast min_depth >= 2")
    if any(len(r["via"]) > 1 for r in rows):
        event("blast via from several types")
    if any(len(r.get("at_lines", [])) > 1 for r in rows):
        event("blast several at_lines")


async def _compare_all(
    client: GraphClient,
    nodes: list[Node],
    edges: list[Edge],
    *,
    start: int,
    targets: list[int],
    depth: int,
    edge_types: tuple[str, ...],
    label: str,
    limit: int,
) -> None:
    s = _uid(start)
    for direction in ("out", "in"):
        ctx = f"blast {direction} from {s} depth={depth} types={edge_types}"
        ref = await ref_compute_blast_radius(client, s, direction, edge_types, depth)
        got = await client.compute_blast_radius(s, direction, edge_types, depth)
        _assert_blast_equal(ref, got, ctx)
        _coverage_events(ref)

    for target in targets:
        t = _uid(target)
        ctx = f"trace {s} -> {t} depth={depth} types={edge_types}"
        ref_t = await ref_trace_path_between(client, s, t, depth, edge_types)
        got_t = await client.trace_path_between(s, t, depth, edge_types)
        _assert_trace_equal(ref_t, got_t, nodes, edges, s, t, edge_types, ctx)
        if ref_t["found"] and currently_in_test_context():
            event(f"trace found, {min(ref_t['hop_count'], 3)}{'+' if ref_t['hop_count'] >= 3 else ''} hops")

    ctx = f"callers of {s} label={label!r} depth={depth} limit={limit}"
    ref_callers = await ref_get_callers(client, s, label, depth, limit)
    _assert_nodes_equal(ref_callers, await client.get_callers(s, label, depth, limit), ctx)
    if len(ref_callers) == limit and currently_in_test_context():
        event("callers cut by LIMIT")
    ctx = f"callees of {s} label={label!r} depth={depth} limit={limit}"
    _assert_nodes_equal(
        await ref_get_callees(client, s, label, depth, limit), await client.get_callees(s, label, depth, limit), ctx
    )


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


@st.composite
def _graphs(draw: st.DrawFn) -> tuple[list[Node], list[Edge]]:
    n = draw(st.integers(min_value=2, max_value=10))
    nodes = [
        Node(
            label=draw(st.sampled_from((NodeLabel.CALLABLE.value, NodeLabel.TYPE_DEF.value))),
            marker=draw(st.booleans()),
            name=f"n{i}",
            # Few distinct values and sometimes none, so the ORDER BY tie-break on uid matters.
            qualified_name=draw(st.one_of(st.none(), st.sampled_from(("m.a", "m.b", "m.c")))),
            file_path=draw(st.sampled_from(("src/a.py", "src/b.py", "tests/test_a.py"))),
            kind=draw(st.sampled_from(("function", "method", "class"))),
        )
        for i in range(n)
    ]
    edge_count = draw(st.integers(min_value=0, max_value=22))
    edges: list[Edge] = []
    for _ in range(edge_count):
        props: dict[str, Any] = {}
        weight = draw(st.sampled_from(_WEIGHTS))
        if weight is not None:
            props["weight"] = weight
        confidence = draw(st.sampled_from((None, "resolved", "ambiguous")))
        if confidence is not None:
            props["confidence"] = confidence
        from_test = draw(st.sampled_from((None, True, False)))
        if from_test is not None:
            props["from_test"] = from_test
        line = draw(st.one_of(st.none(), st.integers(min_value=1, max_value=6)))
        if line is not None:
            props["line"] = line
        if draw(st.booleans()):
            props["strategy"] = draw(st.sampled_from(("import", "same_file", "project_unique")))
        edges.append(
            Edge(
                src=draw(st.integers(min_value=0, max_value=n - 1)),
                dst=draw(st.integers(min_value=0, max_value=n - 1)),
                rel_type=draw(st.sampled_from(_TYPES)),
                props=props,
            )
        )
    return nodes, edges


@pytest.mark.timeout(900)
@hyp_settings(
    max_examples=400,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(graph=_graphs(), data=st.data())
async def test_traversals_match_the_all_paths_reference(
    graph_client: GraphClient, graph: tuple[list[Node], list[Edge]], data: st.DataObject
) -> None:
    nodes, edges = graph
    n = len(nodes)
    start = data.draw(st.integers(min_value=0, max_value=n - 1), label="start")
    depth = data.draw(st.integers(min_value=1, max_value=5), label="depth")
    edge_types = tuple(
        data.draw(st.lists(st.sampled_from(_TYPES), min_size=1, max_size=3, unique=True), label="edge_types")
    )
    label = data.draw(st.sampled_from(("", NodeLabel.CALLABLE.value, NodeLabel.TYPE_DEF.value)), label="label")
    limit = data.draw(st.integers(min_value=1, max_value=5), label="limit")
    note(f"nodes={nodes}")
    note(f"edges={edges}")
    await _write_graph(graph_client, nodes, edges)
    await _compare_all(
        graph_client,
        nodes,
        edges,
        start=start,
        # Every node, not one drawn target: one random target is mostly unreachable.
        targets=list(range(n)),
        depth=depth,
        edge_types=edge_types,
        label=label,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# Hand-written cases
# ---------------------------------------------------------------------------


def _fn(i: int, qn: str | None = None) -> Node:
    return Node(NodeLabel.CALLABLE.value, True, f"n{i}", qn or f"m.n{i}", "src/a.py", "function")


def _calls(src: int, dst: int, weight: float | None = None, **props: Any) -> Edge:
    p = dict(props)
    if weight is not None:
        p["weight"] = weight
    return Edge(src, dst, RelType.CALLS.value, p)


async def _blast_both(client: GraphClient, uid: str, direction: str, types: tuple[str, ...], depth: int) -> dict:
    ref = await ref_compute_blast_radius(client, uid, direction, types, depth)
    got = await client.compute_blast_radius(uid, direction, types, depth)
    _assert_blast_equal(ref, got, f"blast {direction} {uid} depth={depth}")
    return {r["uid"]: r for r in got}


async def test_best_weight_path_longer_than_the_shortest(graph_client: GraphClient) -> None:
    # 0 -> 1 directly at 0.25; 0 -> 2 -> 1 at 1.0 * 1.0.
    nodes = [_fn(i) for i in range(3)]
    edges = [_calls(0, 1, 0.25), _calls(0, 2, 1.0), _calls(2, 1, 1.0)]
    await _write_graph(graph_client, nodes, edges)
    out = await _blast_both(graph_client, _uid(0), "out", (RelType.CALLS.value,), 3)
    assert out[_uid(1)]["min_depth"] == 1
    assert out[_uid(1)]["confidence_score"] == 1.0
    inbound = await _blast_both(graph_client, _uid(1), "in", (RelType.CALLS.value,), 3)
    assert inbound[_uid(0)]["min_depth"] == 1
    assert inbound[_uid(0)]["confidence_score"] == 1.0
    await _compare_all(
        graph_client, nodes, edges, start=0, targets=[1], depth=3, edge_types=(RelType.CALLS.value,), label="", limit=5
    )


async def test_best_weight_path_beyond_the_hop_limit(graph_client: GraphClient) -> None:
    # 0 -> 1 at 0.25 fits depth 2; 0 -> 2 -> 3 -> 1 at 1.0 needs depth 3.
    nodes = [_fn(i) for i in range(4)]
    edges = [_calls(0, 1, 0.25), _calls(0, 2, 1.0), _calls(2, 3, 1.0), _calls(3, 1, 1.0)]
    await _write_graph(graph_client, nodes, edges)
    shallow = await _blast_both(graph_client, _uid(0), "out", (RelType.CALLS.value,), 2)
    assert shallow[_uid(1)]["confidence_score"] == 0.25
    deep = await _blast_both(graph_client, _uid(0), "out", (RelType.CALLS.value,), 3)
    assert deep[_uid(1)]["confidence_score"] == 1.0
    # Same shape seen from the far end, where the budget cuts at the other side.
    assert (await _blast_both(graph_client, _uid(1), "in", (RelType.CALLS.value,), 2))[_uid(0)][
        "confidence_score"
    ] == 0.25
    for depth in (1, 2, 3, 4):
        await _compare_all(
            graph_client,
            nodes,
            edges,
            start=0,
            targets=[1],
            depth=depth,
            edge_types=(RelType.CALLS.value,),
            label="",
            limit=5,
        )


async def test_reachable_only_through_ambiguous_or_test_edges(graph_client: GraphClient) -> None:
    # 0 -amb-> 1 -res-> 2          : 1 and 2 are ambiguous_only
    # 0 -test-> 3 -> 4             : 3 and 4 are test_only
    # 0 -amb-> 5 ; 0 -> 6 -> 7 -> 5 (all clean): 5 is ambiguous_only at depth 2, not at depth 3
    nodes = [_fn(i) for i in range(8)]
    edges = [
        _calls(0, 1, 0.5, confidence="ambiguous"),
        _calls(1, 2, 1.0, confidence="resolved"),
        _calls(0, 3, 0.25, confidence="resolved", from_test=True),
        _calls(3, 4, 1.0),
        _calls(0, 5, 0.5, confidence="ambiguous"),
        _calls(0, 6, 1.0, confidence="resolved", from_test=False),
        _calls(6, 7, 1.0),
        _calls(7, 5, 1.0, confidence="resolved"),
    ]
    await _write_graph(graph_client, nodes, edges)
    d2 = await _blast_both(graph_client, _uid(0), "out", (RelType.CALLS.value,), 2)
    assert d2[_uid(1)]["ambiguous_only"]
    assert d2[_uid(2)]["ambiguous_only"]
    assert not d2[_uid(1)]["test_only"]
    assert d2[_uid(3)]["test_only"]
    assert d2[_uid(4)]["test_only"]
    assert not d2[_uid(3)]["ambiguous_only"]
    assert d2[_uid(5)]["ambiguous_only"]
    d3 = await _blast_both(graph_client, _uid(0), "out", (RelType.CALLS.value,), 3)
    assert not d3[_uid(5)]["ambiguous_only"]
    assert d3[_uid(5)]["confidence_score"] == 1.0
    await _blast_both(graph_client, _uid(5), "in", (RelType.CALLS.value,), 3)
    await _blast_both(graph_client, _uid(4), "in", (RelType.CALLS.value,), 3)


async def test_trace_breaks_equal_length_ties_by_weight(graph_client: GraphClient) -> None:
    # Three 2-hop routes 0 -> {1,2,3} -> 4 weighing 0.5, 0.25, 1/3, and a 3-hop route at 1.0.
    nodes = [_fn(i) for i in range(7)]
    edges = [
        _calls(0, 1, 0.5),
        _calls(1, 4, 1.0),
        _calls(0, 2, 1.0),
        _calls(2, 4, 0.25),
        _calls(0, 3, 1 / 3),
        _calls(3, 4),
        _calls(0, 5, 1.0),
        _calls(5, 6, 1.0),
        _calls(6, 4, 1.0),
    ]
    await _write_graph(graph_client, nodes, edges)
    types = (RelType.CALLS.value,)
    for depth in (1, 2, 3, 4):
        ref = await ref_trace_path_between(graph_client, _uid(0), _uid(4), depth, types)
        got = await graph_client.trace_path_between(_uid(0), _uid(4), depth, types)
        _assert_trace_equal(ref, got, nodes, edges, _uid(0), _uid(4), types, f"trace depth={depth}")
        if depth == 1:
            assert got["found"] is False
        else:
            assert got["hop_count"] == 2
            assert got["path_weight"] == 0.5
            assert [h["to"]["uid"] for h in got["hops"]] == [_uid(1), _uid(4)]


async def test_via_collects_every_first_hop_type(graph_client: GraphClient) -> None:
    # 0 -CALLS(line 3)-> 1 -CALLS-> 4 ; 0 -IMPORTS(line 7)-> 4 ; 0 -USES_TYPE(line 9)-> 2 -IMPORTS-> 4
    # and a self-loop plus a cycle back to the start, which must never be reported.
    nodes = [_fn(i) for i in range(5)]
    edges = [
        _calls(0, 1, 1.0, line=3),
        _calls(1, 4, 0.5, line=11),
        Edge(0, 4, RelType.IMPORTS.value, {"line": 7}),
        Edge(0, 2, RelType.USES_TYPE.value, {"weight": 0.5, "line": 9}),
        Edge(2, 4, RelType.IMPORTS.value, {}),
        _calls(4, 4, 1.0, line=1),
        _calls(4, 0, 1.0, line=2),
    ]
    await _write_graph(graph_client, nodes, edges)
    types = _TYPES
    d2 = await _blast_both(graph_client, _uid(0), "out", types, 2)
    assert d2[_uid(4)]["via"] == sorted(_TYPES)
    assert d2[_uid(4)]["min_depth"] == 1
    assert d2[_uid(4)]["confidence_score"] == 1.0
    # The CALLS (line 3) and USES_TYPE (line 9) routes land on 4 two hops out.
    assert d2[_uid(4)]["at_lines"] == [7]
    assert _uid(0) not in d2
    d1 = await _blast_both(graph_client, _uid(0), "out", types, 1)
    assert d1[_uid(4)]["via"] == [RelType.IMPORTS.value]
    assert d1[_uid(4)]["at_lines"] == [7]
    await _blast_both(graph_client, _uid(4), "in", types, 3)
    for depth in (1, 2, 3):
        await _compare_all(
            graph_client, nodes, edges, start=0, targets=[4, 0, 2], depth=depth, edge_types=types, label="", limit=3
        )


async def test_via_counts_paths_that_pass_back_through_the_start(graph_client: GraphClient) -> None:
    # 0 -CALLS-> 1 -> 0 is a cycle; 0 -IMPORTS-> 2. The path 0 -CALLS-> 1 -> 0 -IMPORTS-> 2 is
    # three hops and starts with CALLS, so at depth 3 via(2) is [CALLS, IMPORTS]. A breadth-first
    # expansion from 0 never revisits 0, so an expansion per first-hop type alone misses it.
    # A self-loop 3 -USES_TYPE-> 3 does the same in one hop, seen from 3.
    nodes = [_fn(i) for i in range(5)]
    edges = [
        _calls(0, 1),
        _calls(1, 0),
        Edge(0, 2, RelType.IMPORTS.value, {}),
        Edge(3, 3, RelType.USES_TYPE.value, {}),
        Edge(3, 4, RelType.IMPORTS.value, {}),
    ]
    await _write_graph(graph_client, nodes, edges)
    d2 = await _blast_both(graph_client, _uid(0), "out", _TYPES, 2)
    assert d2[_uid(2)]["via"] == [RelType.IMPORTS.value]
    d3 = await _blast_both(graph_client, _uid(0), "out", _TYPES, 3)
    assert d3[_uid(2)]["via"] == sorted([RelType.CALLS.value, RelType.IMPORTS.value])
    assert d3[_uid(1)]["via"] == [RelType.CALLS.value]
    assert (await _blast_both(graph_client, _uid(3), "out", _TYPES, 1))[_uid(4)]["via"] == [RelType.IMPORTS.value]
    d2_loop = await _blast_both(graph_client, _uid(3), "out", _TYPES, 2)
    assert d2_loop[_uid(4)]["via"] == sorted([RelType.IMPORTS.value, RelType.USES_TYPE.value])
    for start in (0, 1, 2, 3, 4):
        for depth in (1, 2, 3, 4):
            for direction in ("out", "in"):
                await _blast_both(graph_client, _uid(start), direction, _TYPES, depth)


async def test_at_lines_come_only_from_the_direct_edge(graph_client: GraphClient) -> None:
    # 1 calls 0 directly at line 5, and also reaches 0 through 2 (1 -> 2 at line 30,
    # 2 -> 0 at line 40). Seen from 0, the long route's first hop is 2 -> 0, line 40, in
    # 2's file -- never one of 1's call sites. At d68ba19 1's at_lines was [5, 40].
    nodes = [_fn(i) for i in range(3)]
    edges = [_calls(1, 0, line=5), _calls(1, 2, line=30), _calls(2, 0, line=40)]
    await _write_graph(graph_client, nodes, edges)
    inbound = await _blast_both(graph_client, _uid(0), "in", (RelType.CALLS.value,), 2)
    assert inbound[_uid(1)]["min_depth"] == 1
    assert inbound[_uid(1)]["at_lines"] == [5]
    assert inbound[_uid(2)]["at_lines"] == [40]
    # And outbound: 1's callees 0 (line 5) and 2 (line 30); 0 is also reached via 2.
    outbound = await _blast_both(graph_client, _uid(1), "out", (RelType.CALLS.value,), 2)
    assert outbound[_uid(0)]["at_lines"] == [5]
    assert outbound[_uid(2)]["at_lines"] == [30]


async def test_callers_and_callees_cut_in_order(graph_client: GraphClient) -> None:
    # A hub reached by many callers at different depths, shared qualified names, a TypeDef in
    # the middle (traversed, never returned), and a cycle.
    nodes = [_fn(i, qn="m.same" if i % 2 else None) for i in range(9)]
    nodes[3] = Node(NodeLabel.TYPE_DEF.value, False, "n3", "m.a", "src/a.py", "class")
    edges = [
        _calls(1, 0),
        _calls(2, 0),
        _calls(3, 0),
        _calls(4, 3),
        _calls(5, 4),
        _calls(6, 5),
        _calls(0, 7),
        _calls(7, 8),
        _calls(8, 0),
        _calls(2, 1),
    ]
    await _write_graph(graph_client, nodes, edges)
    for depth in (1, 2, 3, 5):
        for limit in (1, 3, 8):
            for label in ("", NodeLabel.CALLABLE.value, NodeLabel.TYPE_DEF.value):
                for uid in (_uid(0), _uid(3)):
                    ctx = f"{uid} label={label!r} depth={depth} limit={limit}"
                    _assert_nodes_equal(
                        await ref_get_callers(graph_client, uid, label, depth, limit),
                        await graph_client.get_callers(uid, label, depth, limit),
                        f"callers {ctx}",
                    )
                    _assert_nodes_equal(
                        await ref_get_callees(graph_client, uid, label, depth, limit),
                        await graph_client.get_callees(uid, label, depth, limit),
                        f"callees {ctx}",
                    )
    callers = await graph_client.get_callers(_uid(0), "", 5, 8)
    assert _uid(3) not in [n["uid"] for n in callers]
    assert _uid(6) in [n["uid"] for n in callers]


@pytest.mark.parametrize("bad", [0.0, 1.5, -0.25])
async def test_a_stored_weight_outside_the_range_fails_the_query(graph_client: GraphClient, bad: float) -> None:
    """Writers refuse such a weight (`_checked_edge_weight`), but a graph written before that,
    or by hand, may hold one. A weight of 0 used to cost +inf and be kept silently; the
    traversals must refuse rather than mis-rank."""
    nodes = [_fn(i) for i in range(3)]
    edges = [_calls(0, 1, bad), _calls(1, 2, 1.0)]
    await _write_graph(graph_client, nodes, edges)
    types = (RelType.CALLS.value,)
    with pytest.raises(ClientError, match=r"edge weight outside \(0, 1\]"):
        await graph_client.compute_blast_radius(_uid(0), "out", types, 2)
    with pytest.raises(ClientError, match=r"edge weight outside \(0, 1\]"):
        await graph_client.trace_path_between(_uid(0), _uid(2), 3, types)


async def test_missing_endpoints(graph_client: GraphClient) -> None:
    nodes = [_fn(0), _fn(1)]
    edges = [_calls(0, 1)]
    await _write_graph(graph_client, nodes, edges)
    types = (RelType.CALLS.value,)
    for a, b in ((_uid(0), "nope"), ("nope", _uid(1)), (_uid(1), _uid(0)), (_uid(0), _uid(0))):
        ref = await ref_trace_path_between(graph_client, a, b, 3, types)
        got = await graph_client.trace_path_between(a, b, 3, types)
        assert got == ref, f"trace {a} -> {b}: reference={ref!r} implementation={got!r}"
    assert await graph_client.compute_blast_radius("nope", "out", types, 3) == []
    assert await graph_client.get_callers("nope", "", 3, 5) == []
    assert await graph_client.get_callees("nope", "", 3, 5) == []
