"""The SQLite mirror of warehouse-object resolution (ATL-169).

The shared conformance ledger classifies `resolve_warehouse_objects` as a resolution pass
and does not output-compare it, so without this file the SQLite implementation would be
asserted by nothing at all — it would typecheck, ship, and quietly produce no edges for
every embedded deployment.

Every fixture seeds through `resolve_imports`, never by writing an ExternalSymbol by hand.
That is deliberate: the first version of the Memgraph tests hand-built the stub with
`qualified_name = "warehouse.<obj>"`, passed, and matched nothing against a real index —
because `resolve_imports` mints external stubs under `ext/`. The fixtures and the code
agreed about a shape neither the parser nor the graph ever produces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from code_atlas.backends.sqlite_graph import SqliteGraphClient
from code_atlas.parsing.ast import ParsedEntity, ParsedRelationship
from code_atlas.schema import NodeLabel, RelType, Visibility

if TYPE_CHECKING:
    from pathlib import Path


def _entity(name: str, uid: str, kind: str, path: str) -> ParsedEntity:
    return ParsedEntity(
        name=name,
        qualified_name=uid,
        label=NodeLabel.TYPE_DEF,
        kind=kind,
        line_start=1,
        line_end=2,
        file_path=path,
        visibility=Visibility.PUBLIC,
        content_hash=f"h-{uid}",
    )


async def _seed_warehouse_import(client: SqliteGraphClient, project: str, table: str, obj: str) -> str:
    uid = f"{project}:pbi.model.M.table.{table}"
    await client.merge_project_node(project)
    await client.upsert_file_entities(
        project, f"tables/{table}.tmdl", [_entity(table, uid, "pbi_table", f"tables/{table}.tmdl")], []
    )
    await client.resolve_imports(
        project,
        [
            ParsedRelationship(
                from_qualified_name=uid,
                rel_type=RelType.IMPORTS,
                to_name=f"warehouse.{obj}",
                properties={"via": "partition"},
            )
        ],
    )
    return uid


async def _seed_dbt_model(client: SqliteGraphClient, project: str, name: str) -> str:
    uid = f"{project}:dbt.model.{name}"
    await client.merge_project_node(project)
    await client.upsert_file_entities(
        project, f"models/{name}.sql", [_entity(name, uid, "dbt_model", f"models/{name}.sql")], []
    )
    return uid


async def _rows(client: SqliteGraphClient, sql: str, args: tuple[Any, ...] = ()) -> list[Any]:
    conn = await client._get_conn()
    raw: Any = getattr(conn, "raw", conn)
    cur = await raw.execute(sql, args)
    out = list(await cur.fetchall())
    await cur.close()
    return out


async def _feeds(client: SqliteGraphClient) -> list[tuple[str, str]]:
    return [(a, b) for a, b in await _rows(client, "SELECT from_uid, to_uid FROM edges WHERE rel_type = 'FEEDS'")]


async def _stubs(client: SqliteGraphClient) -> list[str]:
    return [
        r[0]
        for r in await _rows(
            client,
            "SELECT qualified_name FROM nodes WHERE labels = 'ExternalSymbol' "
            "AND qualified_name LIKE 'ext/warehouse.%' ORDER BY qualified_name",
        )
    ]


@pytest.fixture
async def client(tmp_path: Path):
    async with SqliteGraphClient(tmp_path / "g.sqlite3", dimension=8) as c:
        await c.ensure_schema()
        yield c


async def test_the_seeded_stub_has_the_shape_the_resolver_looks_for(client):
    """A guard on the fixtures, not the resolver: everything below is worth its runtime
    only if the stub it seeds is the one a real index produces."""
    await _seed_warehouse_import(client, "bi", "Orders", "fct_orders")
    assert await _stubs(client) == ["ext/warehouse.fct_orders"]


async def test_a_dbt_model_feeds_the_bi_table_that_reads_it(client):
    table = await _seed_warehouse_import(client, "bi", "Orders", "fct_orders")
    model = await _seed_dbt_model(client, "transform", "fct_orders")

    assert await client.resolve_warehouse_objects(["bi", "transform"]) == 1
    assert await _feeds(client) == [(model, table)]
    assert await _stubs(client) == [], "the stub was rewired but not removed"


async def test_the_import_edge_is_replaced_not_left_beside_the_new_one(client):
    """A surviving IMPORTS edge to a deleted stub is a dangling reference, and one to a
    live stub makes the table look like it still reads an unresolved object."""
    table = await _seed_warehouse_import(client, "bi", "Orders", "fct_orders")
    await _seed_dbt_model(client, "transform", "fct_orders")
    await client.resolve_warehouse_objects(["bi", "transform"])

    remaining = await _rows(client, "SELECT to_uid FROM edges WHERE from_uid = ? AND rel_type = 'IMPORTS'", (table,))
    assert remaining == []


async def test_an_ambiguous_object_is_graded_never_guessed(client):
    table = await _seed_warehouse_import(client, "bi", "Orders", "orders")
    await _seed_dbt_model(client, "transform_a", "orders")
    await _seed_dbt_model(client, "transform_b", "orders")

    assert await client.resolve_warehouse_objects(["bi", "transform_a", "transform_b"]) == 0
    assert await _feeds(client) == []
    confidence = await _rows(
        client,
        "SELECT json_extract(props_json, '$.confidence') FROM edges WHERE from_uid = ? AND rel_type = 'IMPORTS'",
        (table,),
    )
    assert [c for (c,) in confidence] == ["ambiguous"]


async def test_resolve_imports_drops_whatever_the_parser_attached(client):
    """Pinning a surprising fact rather than a wish.

    `resolve_imports` builds its edge from from_uid/to_uid alone (plus `type_only`), on
    both backends — so any `properties` a parser puts on an IMPORTS relationship are gone
    by the time the edge exists. The warehouse parser used to attach a `via` marker that
    looked informative and was never stored; this is what stops it coming back.
    """
    table = await _seed_warehouse_import(client, "bi", "Orders", "fct_orders")
    props = await _rows(client, "SELECT props_json FROM edges WHERE from_uid = ? AND rel_type = 'IMPORTS'", (table,))
    assert props[0][0] in ("{}", None), f"IMPORTS unexpectedly carries parser properties: {props[0][0]}"


async def test_grading_writes_confidence_onto_the_surviving_edge(client):
    table = await _seed_warehouse_import(client, "bi", "Orders", "orders")
    await _seed_dbt_model(client, "transform_a", "orders")
    await _seed_dbt_model(client, "transform_b", "orders")
    await client.resolve_warehouse_objects(["bi", "transform_a", "transform_b"])

    props = await _rows(client, "SELECT props_json FROM edges WHERE from_uid = ? AND rel_type = 'IMPORTS'", (table,))
    assert '"confidence":"ambiguous"' in props[0][0].replace(" ", "")


async def test_an_unmatched_object_keeps_its_stub(client):
    await _seed_warehouse_import(client, "bi", "Manual", "hand_built_thing")
    assert await client.resolve_warehouse_objects(["bi"]) == 0
    assert await _stubs(client) == ["ext/warehouse.hand_built_thing"]


async def test_it_is_idempotent(client):
    table = await _seed_warehouse_import(client, "bi", "Orders", "fct_orders")
    model = await _seed_dbt_model(client, "transform", "fct_orders")

    assert await client.resolve_warehouse_objects(["bi", "transform"]) == 1
    for _ in range(2):
        await _seed_warehouse_import(client, "bi", "Orders", "fct_orders")
        await client.resolve_warehouse_objects(["bi", "transform"])
        assert await _feeds(client) == [(model, table)], "a re-run added a parallel FEEDS edge"


async def test_a_stub_two_tables_read_survives_a_partial_match(client):
    resolved = await _seed_warehouse_import(client, "bi", "Orders", "fct_orders")
    await _seed_warehouse_import(client, "bi", "Report", "no_such_model")
    model = await _seed_dbt_model(client, "transform", "fct_orders")

    await client.resolve_warehouse_objects(["bi", "transform"])

    assert await _feeds(client) == [(model, resolved)]
    assert await _stubs(client) == ["ext/warehouse.no_such_model"]


async def test_a_snapshot_produces_a_warehouse_object_too(client):
    """`dbt_snapshot` is the other kind that materialises a table. Excluding it would make
    every snapshot-backed BI table look hand-built."""
    table = await _seed_warehouse_import(client, "bi", "Hist", "snap_orders")
    uid = "transform:dbt.snapshot.snap_orders"
    await client.merge_project_node("transform")
    await client.upsert_file_entities(
        "transform",
        "snapshots/snap_orders.sql",
        [_entity("snap_orders", uid, "dbt_snapshot", "snapshots/snap_orders.sql")],
        [],
    )
    assert await client.resolve_warehouse_objects(["bi", "transform"]) == 1
    assert await _feeds(client) == [(uid, table)]
