"""Reclaiming a vector must take the vector and nothing else (ATL-166).

The embedding policy is applied at write time, which leaves every vector bought under the
old policy sitting in the graph. The reclaim sweep is what makes a policy change take
effect on the next ``atlas index`` instead of on a ``--reset-embeddings`` -- which would
re-bill every vector in the database for a dimension change nobody made.

Two things are easy to get wrong here and both are silent:

* **Taking the FTS row with it.** ``_cleanup_search_side_tables`` deletes the vec0 *and*
  the fts5 rows, because it exists for nodes that are going away. Reusing it here would
  make an excluded node unfindable by BM25 -- the exact opposite of the promise.
* **Leaving the vec0 row behind.** The project-scoped ``clear_embeddings`` can afford to,
  because what it clears is about to be re-embedded and the rows are re-keyed on the next
  write. These nodes never get a vector again, so a stale row would sit in the shortlist
  forever, spending ``k * _VEC_OVERSAMPLE`` slots the rescore then throws away.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from code_atlas.backends.sqlite_graph import SqliteGraphClient
from code_atlas.graph.client import EmbedChunkWrite
from code_atlas.parsing.ast import ParsedEntity
from code_atlas.schema import NodeLabel, Visibility

if TYPE_CHECKING:
    from pathlib import Path

DIM = 8


def _entity(name: str, *, kind: str, path: str, label: NodeLabel = NodeLabel.CALLABLE) -> ParsedEntity:
    return ParsedEntity(
        name=name,
        qualified_name=f"proj:{name}",
        label=label,
        kind=kind,
        line_start=1,
        line_end=2,
        file_path=path,
        visibility=Visibility.PUBLIC,
        content_hash=f"h-{name}",
        docstring=f"the {name} entity",
    )


async def _scalar(client: SqliteGraphClient, sql: str, args: tuple[Any, ...] = ()) -> Any:
    conn = await client._get_conn()
    raw: Any = getattr(conn, "raw", conn)
    cur = await raw.execute(sql, args)
    row = await cur.fetchone()
    await cur.close()
    return row[0] if row else None


@pytest.fixture
async def client(tmp_path: Path):
    async with SqliteGraphClient(tmp_path / "g.sqlite3", dimension=DIM) as c:
        await c.ensure_schema()
        await c.merge_project_node("proj")
        await c.upsert_file_entities(
            "proj",
            "conf/app.json",
            [_entity("blob", kind="config_setting", path="conf/app.json", label=NodeLabel.VALUE)],
            [],
        )
        await c.upsert_file_entities("proj", "src/mod.py", [_entity("fn", kind="function", path="src/mod.py")], [])
        await c.write_embeddings_and_hashes(
            [("proj:blob", [0.1] * DIM, "h1"), ("proj:fn", [0.2] * DIM, "h2")],
            model="m",
            labels=[NodeLabel.VALUE.value, NodeLabel.CALLABLE.value],
        )
        yield c


class TestFindEmbeddedEntities:
    async def test_it_reports_kind_and_path_for_every_vector(self, client):
        rows = await client.find_embedded_entities("proj")
        assert sorted(rows) == [
            ("proj:blob", "config_setting", "conf/app.json"),
            ("proj:fn", "function", "src/mod.py"),
        ]

    async def test_the_kind_filter_narrows_in_the_query(self, client):
        assert await client.find_embedded_entities("proj", kinds=["config_setting"]) == [
            ("proj:blob", "config_setting", "conf/app.json")
        ]

    async def test_an_embed_chunk_is_not_an_entity(self, client):
        """This table is unified where Memgraph's graph is not. There a chunk is skipped
        for free -- it carries no :Entity marker by design -- so without an explicit
        exclusion the two backends would sweep different sets, and a chunk returned here
        would be stripped of its vector and left behind as an unreachable row."""
        await client.write_embed_chunks(
            [
                EmbedChunkWrite(
                    uid="proj:fn#chunk1",
                    parent_uid="proj:fn",
                    project_name="proj",
                    chunk_index=1,
                    vector=[0.4] * DIM,
                    embed_hash="c1",
                )
            ],
            model="m",
        )
        uids = {uid for uid, _k, _p in await client.find_embedded_entities("proj")}
        assert "proj:fn#chunk1" not in uids
        assert "proj:fn" in uids, "precondition: the parent is still reported"

    async def test_a_node_with_no_vector_is_not_reported(self, client):
        await client.upsert_file_entities("proj", "b.py", [_entity("g", kind="function", path="b.py")], [])
        assert "proj:g" not in {uid for uid, _k, _p in await client.find_embedded_entities("proj")}


class TestClearEmbeddingsForUids:
    async def test_the_vector_and_its_hash_are_gone(self, client):
        assert await client.clear_embeddings_for_uids(["proj:blob"]) == 1
        assert await _scalar(client, "SELECT embedding FROM nodes WHERE uid = 'proj:blob'") is None
        assert (
            await _scalar(client, "SELECT json_extract(props_json, '$.embed_hash') FROM nodes WHERE uid = 'proj:blob'")
            is None
        )

    async def test_the_node_itself_survives(self, client):
        """Excluded from embeddings is not excluded from the index."""
        await client.clear_embeddings_for_uids(["proj:blob"])
        assert await _scalar(client, "SELECT name FROM nodes WHERE uid = 'proj:blob'") == "blob"

    async def test_the_fts_document_survives(self, client):
        """The whole promise: BM25 is the channel this node now lives in."""
        before = await _scalar(client, f"SELECT count(*) FROM text_{NodeLabel.VALUE.value.lower()}")
        await client.clear_embeddings_for_uids(["proj:blob"])
        assert await _scalar(client, f"SELECT count(*) FROM text_{NodeLabel.VALUE.value.lower()}") == before

    async def test_the_vec_row_goes_with_it(self, client):
        table = f"vec_{NodeLabel.VALUE.value.lower()}"
        assert await _scalar(client, f"SELECT count(*) FROM {table}") == 1
        await client.clear_embeddings_for_uids(["proj:blob"])
        assert await _scalar(client, f"SELECT count(*) FROM {table}") == 0

    async def test_it_touches_nothing_it_was_not_given(self, client):
        await client.clear_embeddings_for_uids(["proj:blob"])
        assert await _scalar(client, "SELECT embedding IS NOT NULL FROM nodes WHERE uid = 'proj:fn'") == 1

    async def test_overflow_chunks_are_deleted(self, client):
        """A chunk's entire content is its vector, so a stripped one is an unreachable row
        the next embed pass would have to recognise and reuse."""
        await client.write_embed_chunks(
            [
                EmbedChunkWrite(
                    uid="proj:blob#chunk1",
                    parent_uid="proj:blob",
                    project_name="proj",
                    chunk_index=1,
                    vector=[0.3] * DIM,
                    embed_hash="c1",
                )
            ],
            model="m",
        )
        assert await _scalar(client, "SELECT count(*) FROM nodes WHERE uid = 'proj:blob#chunk1'") == 1
        await client.clear_embeddings_for_uids(["proj:blob"])
        assert await _scalar(client, "SELECT count(*) FROM nodes WHERE uid = 'proj:blob#chunk1'") == 0

    async def test_an_empty_list_is_a_no_op(self, client):
        assert await client.clear_embeddings_for_uids([]) == 0

    async def test_clearing_twice_reports_the_second_pass_did_nothing(self, client):
        """Idempotent, and honest about it — the count is what the caller logs."""
        assert await client.clear_embeddings_for_uids(["proj:blob"]) == 1
        assert await client.clear_embeddings_for_uids(["proj:blob"]) == 0

    async def test_a_uid_that_does_not_exist_is_ignored(self, client):
        assert await client.clear_embeddings_for_uids(["proj:nope"]) == 0
