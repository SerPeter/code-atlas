"""`resolve_doc_links` resolves a whole flush in two queries, with the same answers.

It used to run two queries **per relationship**, and both were unindexable by
construction: `labels NOT IN (...)` implies no single-label predicate so none of the
partial indices apply (ADR-0045), and `file_path LIKE '%suffix'` has a leading wildcard.
Measured on this repo it was a single call costing **124.08 seconds** — 72% of a full
index's wall clock, and the reason the teardown watchdog was cancelling the final
resolution flush four seconds before it would have finished.

What must not change is the never-multi-link discipline: an ambiguous match is left
unresolved rather than guessed at, and ambiguity is counted over *nodes*, not files. These
tests pin that, because it is the half a batching rewrite gets wrong.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from code_atlas.backends.instrumentation import capture_statements
from code_atlas.backends.sqlite_graph import SqliteGraphClient
from code_atlas.parsing.ast import ParsedEntity, ParsedRelationship
from code_atlas.schema import NodeLabel, RelType, Visibility

if TYPE_CHECKING:
    from pathlib import Path


def _entity(name: str, *, path: str, label: NodeLabel = NodeLabel.CALLABLE) -> ParsedEntity:
    return ParsedEntity(
        name=name,
        qualified_name=f"proj:{path.replace('/', '.')}.{name}",
        label=label,
        kind="function" if label == NodeLabel.CALLABLE else "class",
        line_start=1,
        line_end=2,
        file_path=path,
        visibility=Visibility.PUBLIC,
        content_hash=f"h-{path}-{name}",
    )


def _link(to_name: str, *, is_file_ref: bool = False, frm: str = "proj:note:doc") -> ParsedRelationship:
    return ParsedRelationship(
        from_qualified_name=frm,
        rel_type=RelType.DOCUMENTS,
        to_name=to_name,
        properties={"link_type": "wikilink", "confidence": 1.0, "is_file_ref": is_file_ref},
    )


async def _documents(client: SqliteGraphClient) -> list[tuple[str, str]]:
    conn = await client._get_conn()
    raw: Any = getattr(conn, "raw", conn)
    cur = await raw.execute("SELECT from_uid, to_uid FROM edges WHERE rel_type = 'DOCUMENTS' ORDER BY to_uid")
    rows = [(a, b) for a, b in await cur.fetchall()]
    await cur.close()
    return rows


@pytest.fixture
async def client(tmp_path: Path):
    async with SqliteGraphClient(tmp_path / "g.sqlite3", dimension=8) as c:
        await c.ensure_schema()
        await c.merge_project_node("proj")
        yield c


class TestResolution:
    async def test_a_unique_name_resolves(self, client):
        await client.upsert_file_entities("proj", "a.py", [_entity("only_one", path="a.py")], [])
        await client.resolve_doc_links("proj", [_link("only_one")])
        assert await _documents(client) == [("proj:note:doc", "proj:a.py.only_one")]

    async def test_an_ambiguous_name_resolves_to_nothing(self, client):
        """Two candidates means no edge. Guessing here would attach documentation to the
        wrong symbol silently, which is worse than leaving it unattached."""
        await client.upsert_file_entities("proj", "a.py", [_entity("shared", path="a.py")], [])
        await client.upsert_file_entities("proj", "b.py", [_entity("shared", path="b.py")], [])
        await client.resolve_doc_links("proj", [_link("shared")])
        assert await _documents(client) == []

    async def test_an_unknown_name_resolves_to_nothing(self, client):
        await client.upsert_file_entities("proj", "a.py", [_entity("present", path="a.py")], [])
        await client.resolve_doc_links("proj", [_link("absent")])
        assert await _documents(client) == []

    async def test_notes_and_doc_sections_are_never_candidates(self, client):
        """They are excluded so a note cannot resolve to another note. If the exclusion
        were dropped, this name would have two candidates and silently stop resolving —
        so the assertion is that it still resolves to the code entity."""
        await client.upsert_file_entities("proj", "a.py", [_entity("target", path="a.py")], [])
        await client.upsert_file_entities("proj", "n.md", [_entity("target", path="n.md", label=NodeLabel.NOTE)], [])
        await client.resolve_doc_links("proj", [_link("target")])
        assert await _documents(client) == [("proj:note:doc", "proj:a.py.target")]


class TestFileReferences:
    async def test_a_suffix_match_resolves_when_one_node_matches(self, client):
        await client.upsert_file_entities("proj", "src/pkg/mod.py", [_entity("fn", path="src/pkg/mod.py")], [])
        await client.resolve_doc_links("proj", [_link("pkg/mod.py", is_file_ref=True)])
        assert await _documents(client) == [("proj:note:doc", "proj:src.pkg.mod.py.fn")]

    async def test_a_file_with_two_entities_is_ambiguous(self, client):
        """Ambiguity is counted over nodes, not files — one matching path holding two
        entities is still two candidates. Counting files instead would resolve here, and
        that is the exact off-by-one a batching rewrite invites."""
        await client.upsert_file_entities(
            "proj", "src/mod.py", [_entity("one", path="src/mod.py"), _entity("two", path="src/mod.py")], []
        )
        await client.resolve_doc_links("proj", [_link("src/mod.py", is_file_ref=True)])
        assert await _documents(client) == []

    async def test_the_suffix_is_literal_not_a_wildcard(self, client):
        """`_` and `%` are LIKE wildcards, and the old query escaped them. Python's
        `endswith` is literal, so this must not match `mod-x.py`."""
        await client.upsert_file_entities("proj", "src/mod-x.py", [_entity("fn", path="src/mod-x.py")], [])
        await client.resolve_doc_links("proj", [_link("mod_x.py", is_file_ref=True)])
        assert await _documents(client) == []

    async def test_a_partial_segment_still_matches_as_a_suffix(self, client):
        """Deliberately pinning the *existing* semantics rather than improving them: the
        old predicate was a bare suffix with no path-separator anchoring, so "d.py"
        matches "src/mod.py". Changing that is a behaviour change and belongs in its own
        commit, not smuggled into a performance fix."""
        await client.upsert_file_entities("proj", "src/mod.py", [_entity("fn", path="src/mod.py")], [])
        await client.resolve_doc_links("proj", [_link("d.py", is_file_ref=True)])
        assert await _documents(client) == [("proj:note:doc", "proj:src.mod.py.fn")]


class TestBatching:
    async def test_queries_do_not_scale_with_relationship_count(self, client):
        """The whole point. Two lookups per flush, whatever the flush holds."""
        entities = [_entity(f"fn_{i}", path=f"m{i}.py") for i in range(40)]
        for e in entities:
            await client.upsert_file_entities("proj", e.file_path, [e], [])

        conn = await client._get_conn()
        async with capture_statements(conn) as few:
            await client.resolve_doc_links("proj", [_link("fn_0"), _link("fn_1")])
        async with capture_statements(conn) as many:
            await client.resolve_doc_links("proj", [_link(f"fn_{i}") for i in range(40)])

        n_few = len([s for _o, s, _n in few.app if "FROM nodes" in s])
        n_many = len([s for _o, s, _n in many.app if "FROM nodes" in s])
        assert n_few > 0, "no lookup was issued — nothing was measured"
        assert n_many == n_few, f"40 links cost {n_many} lookups against {n_few} for 2"

    async def test_forty_links_all_resolve(self, client):
        """Non-vacuity for the test above: a batched query that returned nothing would
        issue exactly as few statements."""
        entities = [_entity(f"fn_{i}", path=f"m{i}.py") for i in range(40)]
        for e in entities:
            await client.upsert_file_entities("proj", e.file_path, [e], [])
        await client.resolve_doc_links("proj", [_link(f"fn_{i}") for i in range(40)])
        assert len(await _documents(client)) == 40

    async def test_an_empty_flush_touches_nothing(self, client):
        await client.resolve_doc_links("proj", [])
        assert await _documents(client) == []
