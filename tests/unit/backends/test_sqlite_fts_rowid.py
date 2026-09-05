"""BM25 side-table writes must seek, not scan.

`fts5(uid UNINDEXED, text)` means `DELETE FROM text_<label> WHERE uid = ?` has no index
to use. The write path did exactly that, once per entity, while holding the write lock —
so every entity paid a full scan of the FTS table, and the per-row cost grew with the
table. Measured on this machine at 2k / 8k / 32k rows: 0.592 / 1.905 / 11.022 ms by uid
against 0.134 / 0.201 / 0.348 ms by rowid. Sixteen times the table, nineteen times the
per-row cost — quadratic in total.

The sharpest case is not a reindex. It is one file save, where every entity in the file
scans the whole table, so the incremental path degrades exactly as the graph grows.

These tests hold the plan and the round-trip shape rather than a duration, because a
timing assertion on a 2,000-row fixture would be noise on a fast machine and flaky on a
slow one. The plan is what actually changed.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from code_atlas.backends.instrumentation import capture_statements
from code_atlas.backends.sqlite_graph import SqliteGraphClient, _fts_document, _fts_document_from_entity
from code_atlas.parsing.ast import ParsedEntity
from code_atlas.schema import NodeLabel, Visibility

if TYPE_CHECKING:
    from pathlib import Path


def _raw(conn: object) -> Any:
    """The unwrapped aiosqlite connection behind the instrumentation proxy.

    EXPLAIN QUERY PLAN and the migration fixtures below are not product statements, and
    routing them through `TimedConnection` would put them in the round-trip counts the
    other tests assert on.
    """
    return getattr(conn, "raw", conn)


def _callable(i: int, *, doc: str = "", content_hash: str = "") -> ParsedEntity:
    return ParsedEntity(
        name=f"fn_{i}",
        qualified_name=f"proj:mod.fn_{i}",
        label=NodeLabel.CALLABLE,
        kind="function",
        line_start=1,
        line_end=2,
        file_path="mod.py",
        docstring=doc or f"does thing number {i}",
        visibility=Visibility.PUBLIC,
        content_hash=content_hash or f"h{i}",
    )


@pytest.fixture
async def client(tmp_path: Path):
    async with SqliteGraphClient(tmp_path / "g.sqlite3", dimension=8) as c:
        await c.ensure_schema()
        await c.merge_project_node("proj")
        yield c


class TestThePlan:
    async def test_the_delete_seeks_its_rowid(self, client):
        """The regression guard. `INDEX 0:` with no `=` is fts5 saying 'full scan'."""
        conn = await client._get_conn()
        raw = _raw(conn)

        cur = await raw.execute("EXPLAIN QUERY PLAN DELETE FROM text_callable WHERE rowid = ?", (1,))
        by_rowid = " ".join(str(r[-1]) for r in await cur.fetchall())
        await cur.close()
        cur = await raw.execute("EXPLAIN QUERY PLAN DELETE FROM text_callable WHERE uid = ?", ("x",))
        by_uid = " ".join(str(r[-1]) for r in await cur.fetchall())
        await cur.close()

        assert by_rowid.endswith("INDEX 0:="), f"the rowid delete is not an equality seek: {by_rowid}"
        assert not by_uid.endswith("="), (
            f"a uid delete now seeks too ({by_uid}) — this test can no longer tell the two apart, "
            "so it has stopped proving anything"
        )

    async def test_the_write_path_deletes_by_rowid(self, client):
        """What the plan test cannot see: which of the two the product actually issues."""
        conn = await client._get_conn()
        async with capture_statements(conn) as log:
            await client.upsert_file_entities("proj", "mod.py", [_callable(i) for i in range(5)], [])

        fts = [sql for _op, sql, _n in log.app if "text_callable" in sql]
        assert fts, "no FTS statement was issued at all — the capture missed the write path"
        assert not [s for s in fts if "WHERE uid" in s], f"the write path is deleting FTS rows by uid again: {fts}"


class TestRoundTrips:
    async def test_fts_statements_do_not_scale_with_entity_count(self, client):
        """Two statements per label per batch, not two per entity.

        Counted rather than timed: the count reproduces exactly on any machine, which is
        the whole basis of this project's benchmark comparisons (ADR-0043).
        """
        conn = await client._get_conn()

        async with capture_statements(conn) as small:
            await client.upsert_file_entities("proj", "a.py", [_callable(i) for i in range(2)], [])
        async with capture_statements(conn) as large:
            await client.upsert_file_entities("proj", "b.py", [_callable(i) for i in range(200, 260)], [])

        n_small = len([s for _o, s, _n in small.app if "text_callable" in s])
        n_large = len([s for _o, s, _n in large.app if "text_callable" in s])
        assert n_small > 0, "no FTS statements recorded — nothing was measured"
        assert n_large == n_small, (
            f"60 entities cost {n_large} FTS statements against {n_small} for 2 — the write path is "
            "back to one statement per entity"
        )


class TestSearchStillWorks:
    async def test_a_written_document_is_findable(self, client):
        """Non-vacuity for everything above: a rowid-keyed row that BM25 cannot find is
        a fast way of storing nothing."""
        await client.upsert_file_entities("proj", "mod.py", [_callable(1, doc="parses the frobnicator manifest")], [])
        hits = await client.text_search("frobnicator", limit=5, project="proj")
        assert any(h["node"]["uid"] == "proj:mod.fn_1" for h in hits), f"the document was not searchable: {hits}"

    async def test_a_rewrite_replaces_rather_than_duplicates(self, client):
        """The DELETE is the half that is easy to get wrong once it is keyed differently:
        a stale row left behind answers searches for text the entity no longer contains."""
        # Distinct content hashes, because the entity really did change: an unchanged
        # hash is the gate's cue to skip the write entirely, and reusing one would make
        # this test pass or fail for a reason that has nothing to do with FTS keys.
        await client.upsert_file_entities("proj", "mod.py", [_callable(1, doc="alpha marker", content_hash="h-a")], [])
        await client.upsert_file_entities("proj", "mod.py", [_callable(1, doc="beta marker", content_hash="h-b")], [])

        assert not await client.text_search("alpha", limit=5, project="proj"), (
            "the old FTS document survived the rewrite — searches still match deleted text"
        )
        assert await client.text_search("beta", limit=5, project="proj")


class TestTheDocumentIsBuiltOnce:
    def test_both_builders_agree(self):
        """The write path builds from a ParsedEntity, the migration from a stored row.

        If they diverge, a rebuilt index ranks differently from a freshly written one and
        nothing fails — search just quietly gets worse for older entities.
        """
        e = replace(_callable(7, doc="a docstring"), signature="def fn_7(x: int) -> str", tags=["a", "b"])
        from_entity = _fts_document_from_entity(e)
        from_row = _fts_document(
            e.name,
            "mod.fn_7",
            {"docstring": e.docstring, "signature": e.signature, "tags": e.tags},
        )
        assert from_entity == from_row, f"builders diverged:\n  entity={from_entity!r}\n  row   ={from_row!r}"


class TestTheMigration:
    async def test_an_existing_uid_keyed_database_is_rebuilt(self, tmp_path: Path):
        """A database written before this change has arbitrary FTS rowids.

        Simulated by re-keying the rows to something wrong and clearing the marker, which
        is the state an old database is in. Without the rebuild the delete would find
        nothing and stale documents would accumulate forever.
        """
        db = tmp_path / "g.sqlite3"
        async with SqliteGraphClient(db, dimension=8) as c:
            await c.ensure_schema()
            await c.merge_project_node("proj")
            await c.upsert_file_entities("proj", "mod.py", [_callable(1, doc="gamma marker")], [])
            conn = await c._get_conn()
            raw = _raw(conn)
            await raw.execute("DELETE FROM text_callable")
            await raw.execute(
                "INSERT INTO text_callable(rowid, uid, text) VALUES (999, 'proj:mod.fn_1', 'gamma marker')"
            )
            await raw.execute("DELETE FROM meta WHERE key = 'fts_key'")
            await raw.commit()

        async with SqliteGraphClient(db, dimension=8) as c:
            await c.ensure_schema()
            conn = await c._get_conn()
            raw = _raw(conn)
            cur = await raw.execute("SELECT rowid FROM text_callable")
            rowids = [r[0] for r in await cur.fetchall()]
            await cur.close()
            cur = await raw.execute("SELECT rowid FROM nodes WHERE uid = 'proj:mod.fn_1'")
            node_rowid = (await cur.fetchone())[0]
            await cur.close()

            assert rowids == [node_rowid], f"FTS rowid {rowids} does not match the node's {node_rowid}"
            assert await c.text_search("gamma", limit=5, project="proj"), (
                "the rebuild dropped the document instead of re-keying it"
            )

    async def test_the_rebuild_runs_once(self, tmp_path: Path):
        """`ensure_schema` runs on every startup; a rebuild that ran every time would
        make each start O(graph)."""
        db = tmp_path / "g.sqlite3"
        async with SqliteGraphClient(db, dimension=8) as c:
            await c.ensure_schema()
            await c.merge_project_node("proj")
            await c.upsert_file_entities("proj", "mod.py", [_callable(i) for i in range(3)], [])

            conn = await c._get_conn()
            async with capture_statements(conn) as log:
                await c.ensure_schema()

            rebuild_reads = [s for _o, s, _n in log.app if "FROM nodes WHERE labels = ?" in s]
            assert not rebuild_reads, f"the FTS rebuild ran again on an already-keyed database: {rebuild_reads}"
