"""The vector side table is a bit index; the ranking it returns is still exact.

The float `vec0` table this replaces held a **second copy of every vector**, beside the
authoritative one in `nodes.embedding`, and used it only to answer KNN. sqlite-vec's KNN
is a brute-force scan either way, so the copy bought no algorithmic advantage — it only
made the scan read 32 bits per component instead of 1.

Measured on 9,118 real 1536-d embeddings from a local `.atlas` index:

    exact (float table)   24.23 ms/query   0.33 s to build
    bit + rescore(x8)      1.00 ms/query   0.09 s to build   recall@10 1.000

**What these tests do and do not prove.** They pin the *plumbing* — that the probe is
quantized the same way the stored vectors were, that the join reaches `nodes.embedding`,
that the ordering is by real cosine distance, and that the float fallback still works.
They deliberately size every fixture so the shortlist covers the whole table, which makes
recall 1.000 by construction.

They do **not** prove recall quality. That is a property of real embeddings and cannot be
measured on the toy vectors a unit test can build: random and clustered synthetic vectors
were measured at recall 0.20-0.32 where real ones scored 1.000, so a synthetic recall
assertion here would fail against a working index and pass against a broken one for
reasons unrelated to the code. Recall belongs in a measurement against real vectors.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pytest

from code_atlas.backends.sqlite_graph import SqliteGraphClient, _bit_quantizable, _vec_table_ddl
from code_atlas.parsing.ast import ParsedEntity
from code_atlas.schema import NodeLabel, Visibility

if TYPE_CHECKING:
    from pathlib import Path

DIM = 8


def _raw(conn: object) -> Any:
    return getattr(conn, "raw", conn)


def _entity(i: int) -> ParsedEntity:
    return ParsedEntity(
        name=f"fn_{i}",
        qualified_name=f"proj:mod.fn_{i}",
        label=NodeLabel.CALLABLE,
        kind="function",
        line_start=1,
        line_end=2,
        file_path="mod.py",
        visibility=Visibility.PUBLIC,
        content_hash=f"h{i}",
    )


def _unit(angle: float) -> list[float]:
    """A unit vector in the first two dimensions, so cosine order is angle order.

    Constructed rather than random: the point of these tests is that the ranking is
    exactly cosine, which needs a ground truth that can be written down.
    """
    return [math.cos(angle), math.sin(angle), *([0.0] * (DIM - 2))]


@pytest.fixture
async def client(tmp_path: Path):
    async with SqliteGraphClient(tmp_path / "g.sqlite3", dimension=DIM) as c:
        await c.ensure_schema()
        await c.merge_project_node("proj")
        yield c


class TestTheShape:
    def test_a_divisible_dimension_gets_a_bit_index(self):
        for dim in (8, 384, 768, 1024, 1536, 3072):
            assert _bit_quantizable(dim), f"dimension {dim} should quantize"
        assert "bit[768]" in _vec_table_ddl(768)[0]

    def test_an_indivisible_dimension_falls_back_to_float(self):
        """`vec_quantize_binary` requires a length divisible by 8. Every model in
        practical use is, but `dimension` is user configuration and a 100 must degrade
        rather than fail every write."""
        assert not _bit_quantizable(100)
        ddl = _vec_table_ddl(100)[0]
        assert "float[100]" in ddl
        assert "bit[" not in ddl

    async def test_the_table_really_is_a_bit_column(self, client):
        conn = await client._get_conn()
        cur = await _raw(conn).execute("SELECT sql FROM sqlite_master WHERE name = 'vec_callable'")
        ddl = (await cur.fetchone())[0]
        await cur.close()
        assert "bit[" in ddl, f"the vector table is not bit-quantized: {ddl}"


class TestRankingIsExact:
    async def test_results_come_back_in_true_cosine_order(self, client):
        """The rescore is the whole point: the shortlist may be approximate, the ranking
        must not be.

        Angles are spread so the correct order is unambiguous, and the shortlist covers
        the table — so a failure here is the rescore or the join, never the quantizer.
        """
        await client.upsert_file_entities("proj", "mod.py", [_entity(i) for i in range(6)], [])
        angles = [0.0, 0.3, 0.7, 1.2, 2.0, 3.0]
        await client.write_embeddings_and_hashes(
            [(f"proj:mod.fn_{i}", _unit(a), f"h{i}") for i, a in enumerate(angles)], model="m"
        )

        hits = await client.vector_search(_unit(0.0), label="Callable", limit=4, project="proj")
        got = [h["node"]["uid"] for h in hits]
        assert got == [f"proj:mod.fn_{i}" for i in range(4)], f"not in cosine order: {got}"

        sims = [h["similarity"] for h in hits]
        assert sims == sorted(sims, reverse=True), f"similarities not descending: {sims}"
        assert sims[0] == pytest.approx(1.0, abs=1e-5), (
            f"the identical vector did not score 1.0 ({sims[0]}) — the distance is not real cosine"
        )

    async def test_a_vector_with_no_node_cannot_be_returned(self, client):
        """The join is to `nodes`, so an orphaned index row is invisible rather than a
        crash or a half-populated result."""
        await client.upsert_file_entities("proj", "mod.py", [_entity(0), _entity(1)], [])
        await client.write_embeddings_and_hashes(
            [("proj:mod.fn_0", _unit(0.0), "h0"), ("proj:mod.fn_1", _unit(0.1), "h1")], model="m"
        )

        # Orphaned by deleting the node out from under a real index row, rather than by
        # hand-crafting one: a hand-written blob couples this test to vec0's on-disk
        # encoding, which is the one detail the rest of this file is careful never to
        # assume. Raw SQL, because the product delete path cleans the side tables — which
        # is the behaviour that makes this state rare, not impossible (a crash between the
        # two statements leaves exactly this).
        conn = await client._get_conn()
        await _raw(conn).execute("DELETE FROM nodes WHERE uid = 'proj:mod.fn_1'")
        await _raw(conn).commit()

        hits = await client.vector_search(_unit(0.0), label="Callable", limit=5, project="proj")
        assert [h["node"]["uid"] for h in hits] == ["proj:mod.fn_0"]


class TestTheProbeMatchesTheStore:
    async def test_an_exact_stored_vector_is_its_own_nearest_neighbour(self, client):
        """The sharpest check that both ends quantize identically.

        If the query probe were packed in a different bit order from the stored vectors,
        the shortlist would be effectively random — and with a small table the rescore
        would still return *something* in correct cosine order, so only asking for the
        vector itself catches it.
        """
        await client.upsert_file_entities("proj", "mod.py", [_entity(i) for i in range(5)], [])
        angles = [0.0, 1.0, 2.0, 3.0, 4.0]
        await client.write_embeddings_and_hashes(
            [(f"proj:mod.fn_{i}", _unit(a), f"h{i}") for i, a in enumerate(angles)], model="m"
        )

        for i, a in enumerate(angles):
            hits = await client.vector_search(_unit(a), label="Callable", limit=1, project="proj")
            got = [h["node"]["uid"] for h in hits]
            assert got == [f"proj:mod.fn_{i}"], (
                f"querying with the stored vector for fn_{i} returned {got} — the probe and the stored "
                "vectors are not quantized the same way"
            )


class TestTheRebuild:
    async def test_a_float_index_is_rebuilt_into_a_bit_index(self, tmp_path: Path):
        """An existing database has float vec tables. They cannot be altered in place, so
        the shape marker triggers a drop and a repopulate from `nodes.embedding` — which
        is why this costs time and never a provider bill.
        """
        db = tmp_path / "g.sqlite3"
        async with SqliteGraphClient(db, dimension=DIM) as c:
            await c.ensure_schema()
            await c.merge_project_node("proj")
            await c.upsert_file_entities("proj", "mod.py", [_entity(0)], [])
            await c.write_embeddings_and_hashes([("proj:mod.fn_0", _unit(0.0), "h0")], model="m")

            # Put the database back into the old shape.
            conn = await c._get_conn()
            raw = _raw(conn)
            await raw.execute("DROP TABLE vec_callable")
            await raw.execute(
                f"CREATE VIRTUAL TABLE vec_callable USING vec0(embedding float[{DIM}] distance_metric=cosine)"
            )
            await raw.execute("DELETE FROM meta WHERE key = 'vec_kind'")
            await raw.commit()

        async with SqliteGraphClient(db, dimension=DIM) as c:
            await c.ensure_schema()
            conn = await c._get_conn()
            cur = await _raw(conn).execute("SELECT sql FROM sqlite_master WHERE name = 'vec_callable'")
            assert "bit[" in (await cur.fetchone())[0], "the float table was not rebuilt"
            await cur.close()

            hits = await c.vector_search(_unit(0.0), label="Callable", limit=1, project="proj")
            assert [h["node"]["uid"] for h in hits] == ["proj:mod.fn_0"], (
                "the rebuild dropped the vector instead of re-quantizing it — note it was "
                "repopulated from nodes.embedding, so losing it here means losing it entirely"
            )

    async def test_the_rebuild_runs_once(self, tmp_path: Path):
        """`ensure_schema` runs on every startup; a rebuild that ran every time would
        drop and repopulate every vector on each daemon start."""
        db = tmp_path / "g.sqlite3"
        async with SqliteGraphClient(db, dimension=DIM) as c:
            await c.ensure_schema()
            await c.merge_project_node("proj")
            await c.upsert_file_entities("proj", "mod.py", [_entity(0)], [])
            await c.write_embeddings_and_hashes([("proj:mod.fn_0", _unit(0.0), "h0")], model="m")

            conn = await c._get_conn()
            from code_atlas.backends.instrumentation import capture_statements

            async with capture_statements(conn) as log:
                await c.ensure_schema()

            drops = [s for _o, s, _n in log.app if "DROP TABLE" in s and "vec_" in s]
            assert not drops, f"the vector rebuild ran again on an already-correct database: {drops}"
