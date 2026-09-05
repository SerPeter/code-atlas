"""The embedding write path issues one statement per vector, not four.

It used to issue four: `UPDATE nodes`, then `SELECT rowid, labels` to find the row it had
just written, then a vec0 `DELETE` and `INSERT`. On a full index of this repo that was
10,948 vectors x 4 — the largest single block of SQL the backend produces, all of it
inside the write lock.

Measured per vector at 768 dimensions against a populated table: node UPDATE 0.279 ms,
the rowid lookup 0.478 ms, and the vec0 pair 4.977 ms. `RETURNING` removes the lookup
outright; sending the vec0 pair through `executemany` takes it to 1.632 ms, a 3.05x on
the same statements and the same rows, because what dominated was per-statement dispatch
rather than index work.

Counted rather than timed, per ADR-0043: the count reproduces exactly on any machine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from code_atlas.backends.instrumentation import capture_statements
from code_atlas.backends.sqlite_graph import SqliteGraphClient
from code_atlas.parsing.ast import ParsedEntity
from code_atlas.schema import NodeLabel, Visibility

if TYPE_CHECKING:
    from pathlib import Path

DIM = 8


def _raw(conn: object) -> Any:
    return getattr(conn, "raw", conn)


def _entity(i: int, label: NodeLabel = NodeLabel.CALLABLE) -> ParsedEntity:
    return ParsedEntity(
        name=f"fn_{i}",
        qualified_name=f"proj:mod.fn_{i}",
        label=label,
        kind="function" if label == NodeLabel.CALLABLE else "class",
        line_start=1,
        line_end=2,
        file_path="mod.py",
        visibility=Visibility.PUBLIC,
        content_hash=f"h{i}",
    )


def _vectors(n: int) -> list[tuple[str, list[float], str]]:
    return [(f"proj:mod.fn_{i}", [float(i) / 100] * DIM, f"hash{i}") for i in range(n)]


@pytest.fixture
async def client(tmp_path: Path):
    async with SqliteGraphClient(tmp_path / "g.sqlite3", dimension=DIM) as c:
        await c.ensure_schema()
        await c.merge_project_node("proj")
        await c.upsert_file_entities("proj", "mod.py", [_entity(i) for i in range(20)], [])
        yield c


class TestStatementShape:
    async def test_no_lookup_follows_the_update(self, client):
        """`RETURNING` or the lookup is back — there is no third option."""
        conn = await client._get_conn()
        async with capture_statements(conn) as log:
            await client.write_embeddings_and_hashes(_vectors(20), model="m")

        sql = [s for _op, s, _n in log.app]
        assert any("RETURNING" in s for s in sql), "the node update is no longer using RETURNING"
        assert not [s for s in sql if s.startswith("SELECT rowid, labels FROM nodes")], (
            "the write path is looking the row up again after updating it"
        )

    async def test_vector_statements_do_not_scale_with_batch_size(self, client):
        """Two vec0 statements per label per batch, whatever the batch holds."""
        conn = await client._get_conn()
        async with capture_statements(conn) as small:
            await client.write_embeddings_and_hashes(_vectors(2), model="m")
        async with capture_statements(conn) as large:
            await client.write_embeddings_and_hashes(_vectors(20), model="m")

        count = lambda log: len([s for _o, s, _n in log.app if "vec_callable" in s])  # noqa: E731
        assert count(small) > 0, "no vec0 statements recorded — nothing was measured"
        assert count(large) == count(small), (
            f"20 vectors cost {count(large)} vec0 statements against {count(small)} for 2 — "
            "the write path is back to one statement per vector"
        )


class TestItStillWrites:
    async def test_the_vector_and_the_hash_both_land(self, client):
        """Non-vacuity: a write path that issues no statements would pass everything above."""
        await client.write_embeddings_and_hashes(_vectors(3), model="m")
        conn = await client._get_conn()
        raw = _raw(conn)

        cur = await raw.execute(
            "SELECT count(*) FROM nodes WHERE embedding IS NOT NULL AND json_extract(props_json, '$.embed_model') = 'm'"
        )
        assert (await cur.fetchone())[0] == 3
        await cur.close()

        cur = await raw.execute("SELECT count(*) FROM vec_callable")
        assert (await cur.fetchone())[0] == 3, "the vec0 rows were not written"
        await cur.close()

    async def test_a_rewrite_replaces_rather_than_duplicating(self, client):
        """vec0 has no upsert, so the DELETE is load-bearing. Without it a re-embed
        leaves the old vector in the index answering searches beside the new one."""
        await client.write_embeddings_and_hashes(_vectors(3), model="m")
        await client.write_embeddings_and_hashes(_vectors(3), model="m")

        conn = await client._get_conn()
        cur = await _raw(conn).execute("SELECT count(*) FROM vec_callable")
        assert (await cur.fetchone())[0] == 3, "re-embedding duplicated the vec0 rows"
        await cur.close()

    async def test_an_unknown_uid_is_skipped_not_fatal(self, client):
        """A uid with no node yields no RETURNING row. Writing the rest of the batch
        matters more than the one that vanished — the alternative is losing every vector
        that travelled with it."""
        items = [*_vectors(2), ("proj:mod.gone", [0.5] * DIM, "hx")]
        await client.write_embeddings_and_hashes(items, model="m")

        conn = await client._get_conn()
        cur = await _raw(conn).execute("SELECT count(*) FROM vec_callable")
        assert (await cur.fetchone())[0] == 2
        await cur.close()

    async def test_labels_are_kept_apart(self, client):
        """Each label has its own vec table, and the batch is grouped by the label the
        UPDATE returned — not by anything the caller passed."""
        await client.upsert_file_entities("proj", "types.py", [_entity(100, NodeLabel.TYPE_DEF)], [])
        await client.write_embeddings_and_hashes([*_vectors(2), ("proj:mod.fn_100", [0.1] * DIM, "h100")], model="m")

        conn = await client._get_conn()
        raw = _raw(conn)
        cur = await raw.execute("SELECT count(*) FROM vec_callable")
        assert (await cur.fetchone())[0] == 2
        await cur.close()
        cur = await raw.execute("SELECT count(*) FROM vec_typedef")
        assert (await cur.fetchone())[0] == 1, "the TypeDef vector did not reach its own table"
        await cur.close()
