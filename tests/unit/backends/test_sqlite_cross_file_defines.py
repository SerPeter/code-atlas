"""Re-parsing a file must not delete DEFINES edges another file's parse contributed.

``resolve_member_defines`` creates edges that run from a TypeDef to a member declared in a
**different** file: a C++ method in ``foo.cpp`` whose class is in ``foo.h``, a Go method
whose receiver struct is declared elsewhere in the package, a Rust ``impl`` split from its
``struct``, an SFDX ``objects/X/fields/Y.field-meta.xml`` under its object file. Four
language modules emit ``parent_type_name``, the property that routes them.

The edge is *sourced* at the owning file's node but *stated* by the member's file. Memgraph
carves that case out of its delete sweep and says why; SQLite had no counterpart, so
re-parsing the owner deleted the edge and only a re-parse of the member file could restore
it -- which the content-hash gate exists to prevent. It stayed gone, silently.

Nothing caught it because the backend conformance ledger classifies ``resolve_member_defines``
as a "resolution pass", which is not output-compared, and the Memgraph test for this
invariant takes the ``graph_client`` fixture and so never ran against SQLite.

**Note for anyone extending these tests:** ``upsert_file_entities`` early-returns when a file
classifies to no added/modified/deleted entities, so ``_recreate_file_relationships`` is never
reached. Re-upserting an identical entity passes while proving nothing -- the second upsert
must carry a genuinely different ``content_hash``, which is also the real case: the file
changed, which is why it is being re-parsed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from code_atlas.backends.sqlite_graph import SqliteGraphClient
from code_atlas.parsing.ast import ParsedEntity, ParsedRelationship
from code_atlas.schema import NodeLabel, RelType, Visibility

if TYPE_CHECKING:
    from pathlib import Path

HEADER = "src/shapes.h"
SOURCE = "src/shapes.cpp"


def _entity(
    name: str,
    qualified_name: str,
    *,
    path: str,
    label: NodeLabel,
    content_hash: str,
) -> ParsedEntity:
    return ParsedEntity(
        name=name,
        qualified_name=qualified_name,
        label=label,
        kind="class" if label is NodeLabel.TYPE_DEF else "method",
        line_start=1,
        line_end=2,
        file_path=path,
        visibility=Visibility.PUBLIC,
        content_hash=content_hash,
    )


def _klass(content_hash: str = "h1") -> ParsedEntity:
    return _entity("Circle", "p:shapes.Circle", path=HEADER, label=NodeLabel.TYPE_DEF, content_hash=content_hash)


def _method(content_hash: str = "m1") -> ParsedEntity:
    return _entity("area", "p:shapes.Circle.area", path=SOURCE, label=NodeLabel.CALLABLE, content_hash=content_hash)


def _member_rel() -> ParsedRelationship:
    """What the member file's parse contributes; ``resolve_member_defines`` routes it."""
    return ParsedRelationship(
        from_qualified_name="",
        to_name="p:shapes.Circle.area",
        rel_type=RelType.DEFINES,
        properties={"parent_type_name": "Circle"},
    )


@pytest.fixture
async def client(tmp_path: Path):
    async with SqliteGraphClient(tmp_path / "g.sqlite3", dimension=8) as graph:
        await graph.ensure_schema()
        yield graph


async def _defines(graph: SqliteGraphClient) -> set[tuple[str, str]]:
    conn = await graph._get_conn()
    cursor = await conn.execute("SELECT from_uid, to_uid FROM edges WHERE rel_type = 'DEFINES'")
    return {tuple(row) for row in await cursor.fetchall()}


async def _seed(graph: SqliteGraphClient) -> None:
    """Both files parsed, then the post-batch pass that mints the cross-file edge."""
    await graph.upsert_file_entities("p", HEADER, [_klass()], [])
    await graph.upsert_file_entities("p", SOURCE, [_method()], [])
    await graph.resolve_member_defines("p", [_member_rel()])


# uid is `qualified_name` verbatim; the `project:` prefix is a parser convention
# baked into the qualified name, so seeded entities must carry it too.
CROSS_FILE = ("p:shapes.Circle", "p:shapes.Circle.area")


async def test_the_cross_file_edge_survives_reparsing_the_owning_file(client: SqliteGraphClient):
    await _seed(client)
    assert CROSS_FILE in await _defines(client)

    # The header changed and is re-parsed alone; the .cpp is unchanged and the
    # content-hash gate skips it. This is the steady state, not an edge case.
    await client.upsert_file_entities("p", HEADER, [_klass("h2")], [])

    assert CROSS_FILE in await _defines(client)


async def test_the_cross_file_edge_survives_the_batch_sweep(client: SqliteGraphClient):
    """The batch path is the one a real index actually takes."""
    await _seed(client)

    await client.upsert_batch_entities("p", {HEADER: ([_klass("h2")], [])})

    assert CROSS_FILE in await _defines(client)


async def test_a_same_file_defines_is_still_deleted_and_recreated(client: SqliteGraphClient):
    """The carve-out must not be so broad that it strands stale intra-file edges.

    A DEFINES whose target lives in the re-parsed file is the parser's own output and
    is recreated from the new parse, so it must still be swept.
    """
    parent = _entity("Mod", "p:shapes.Mod", path=HEADER, label=NodeLabel.TYPE_DEF, content_hash="p1")
    child = _entity("Inner", "p:shapes.Mod.Inner", path=HEADER, label=NodeLabel.TYPE_DEF, content_hash="c1")
    rel = ParsedRelationship(from_qualified_name="p:shapes.Mod", rel_type=RelType.DEFINES, to_name="p:shapes.Mod.Inner")
    await client.upsert_file_entities("p", HEADER, [parent, child], [rel])
    assert ("p:shapes.Mod", "p:shapes.Mod.Inner") in await _defines(client)

    # Re-parsed with the child gone: the edge must go with it.
    await client.upsert_file_entities(
        "p", HEADER, [_entity("Mod", "p:shapes.Mod", path=HEADER, label=NodeLabel.TYPE_DEF, content_hash="p2")], []
    )

    assert ("p:shapes.Mod", "p:shapes.Mod.Inner") not in await _defines(client)


async def test_a_defines_to_a_node_that_does_not_exist_is_still_deleted(client: SqliteGraphClient):
    """Pins the dangling-target semantics deliberately.

    Memgraph cannot have a dangling edge at all -- its pattern simply does not match --
    so there is no behaviour to mirror. SQLite keeps deleting it, which is what it did
    before the carve-out existed: the carve-out preserves edges to a node in *another*
    file, not edges to no node at all.
    """
    orphan = ParsedRelationship(
        from_qualified_name="p:shapes.Circle", rel_type=RelType.DEFINES, to_name="p:shapes.Nowhere"
    )
    await client.upsert_file_entities("p", HEADER, [_klass()], [orphan])
    assert ("p:shapes.Circle", "p:shapes.Nowhere") in await _defines(client)

    await client.upsert_file_entities("p", HEADER, [_klass("h2")], [])

    assert ("p:shapes.Circle", "p:shapes.Nowhere") not in await _defines(client)
