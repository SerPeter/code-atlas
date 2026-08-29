"""Unit tests for the ADR/RFC citation resolver.

Two layers, both infrastructure-free:

* the pure matching logic in ``graph/client.py`` (canonical key derivation,
  document-node keying, candidate ranking) — the part *both* backends share
  verbatim;
* ``SqliteGraphClient.resolve_citations`` end-to-end, since the embedded
  backend runs in-process and needs no Docker. The Memgraph mirror of the same
  behaviours lives in ``tests/integration/graph/test_client.py``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from code_atlas.backends.sqlite_graph import SqliteGraphClient
from code_atlas.graph.client import (
    _citation_key,
    _CitationLookup,
    _directory_scheme,
    _document_citation_keys,
    _pick_citation_target,
    _render_citation_key,
)
from code_atlas.parsing.ast import ParsedEntity
from code_atlas.schema import NodeLabel

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path


# ---------------------------------------------------------------------------
# Canonical form
# ---------------------------------------------------------------------------


@pytest.fixture
async def client(tmp_path: Path) -> AsyncGenerator[SqliteGraphClient]:
    """An open client per test, closed even when the test fails.

    These tests built it identically and closed it on their last line, which meant
    any failing assertion skipped the close. Teardown does not skip.
    """
    async with SqliteGraphClient(tmp_path / "graph.sqlite3") as opened:
        yield opened


class TestCitationKey:
    def test_zero_padded_and_bare_adr_numbers_unify(self):
        """The whole point: ``see ADR 14`` and ``see ADR-0014`` are the same ADR."""
        assert _citation_key("ADR-14") == _citation_key("ADR-0014") == ("ADR", 14)

    def test_every_separator_form_the_extractor_can_emit_parses(self):
        for raw in ("ADR-0014", "ADR 0014", "ADR_0014", "ADR#0014", "ADR0014"):
            assert _citation_key(raw) == ("ADR", 14), raw

    def test_scheme_is_case_folded(self):
        assert _citation_key("adr-0014") == ("ADR", 14)
        assert _citation_key("Rfc 793") == ("RFC", 793)

    def test_rfc_numbers_are_never_zero_padded(self):
        """A blanket 4-digit pad would corrupt RFC 793 into RFC-0793."""
        assert _citation_key("RFC 793") == ("RFC", 793)
        assert _citation_key("RFC-793") != ("RFC", 793, 4)
        assert _render_citation_key(("RFC", 793)) == "RFC-793"

    def test_rendered_key_is_identical_for_both_adr_spellings(self):
        padded = _citation_key("ADR-0014")
        bare = _citation_key("ADR-14")
        assert padded is not None
        assert bare is not None
        assert _render_citation_key(padded) == _render_citation_key(bare) == "ADR-14"

    def test_unparseable_strings_yield_no_key(self):
        for raw in ("", "ADR", "0014", "ADR-", "ADR-12-34", "see ADR-0014", "ADR-1234567"):
            assert _citation_key(raw) is None, raw


class TestDirectoryScheme:
    def test_singular_and_plural_directory_names_agree(self):
        assert _directory_scheme("adr") == _directory_scheme("adrs") == _directory_scheme("ADRs") == "ADR"

    def test_rfc_directory(self):
        assert _directory_scheme("rfcs") == "RFC"

    def test_non_scheme_directories_yield_nothing(self):
        for name in ("", "2026-07", "a", "some_really_long_directory_name"):
            assert _directory_scheme(name) == "", name


# ---------------------------------------------------------------------------
# Document-node keying
# ---------------------------------------------------------------------------


class TestDocumentCitationKeys:
    def test_adr_docfile_is_keyed_from_filename_and_parent_dir(self):
        """``wiki/adr`` — not ``docs/adr`` — and nothing hardcodes either."""
        keys = _document_citation_keys(
            NodeLabel.DOC_FILE.value, "0014-calls-edge-confidence.md", "wiki/adr/0014-calls-edge-confidence.md"
        )
        assert keys == [(("ADR", 14), 0)]

    def test_the_same_file_under_docs_adr_keys_identically(self):
        keys = _document_citation_keys(NodeLabel.DOC_FILE.value, "0014-x.md", "docs/adr/0014-x.md")
        assert keys == [(("ADR", 14), 0)]

    def test_windows_separators_are_normalised(self):
        keys = _document_citation_keys(NodeLabel.DOC_FILE.value, "0014-x.md", r"wiki\adr\0014-x.md")
        assert keys == [(("ADR", 14), 0)]

    def test_numbered_file_outside_a_scheme_directory_gets_no_filename_key(self):
        assert _document_citation_keys(NodeLabel.DOC_FILE.value, "0014-x.md", "wiki/2026-07/0014-x.md") == []

    def test_top_level_heading_keys_as_the_documents_own_title(self):
        keys = _document_citation_keys(
            NodeLabel.DOC_SECTION.value, "ADR-0014: CALLS Edge Confidence", "wiki/adr/0014-x.md", 1
        )
        assert keys == [(("ADR", 14), 2)]

    def test_a_subsection_heading_is_a_mention_and_is_never_keyed(self):
        """The resolver's worst failure mode: ``## ADR-0014 rationale`` in some
        unrelated document was a perfectly good candidate at confidence 1.0
        whenever the real ADR lived outside a scheme-named directory."""
        for level in (2, 3, 6):
            keys = _document_citation_keys(
                NodeLabel.DOC_SECTION.value, "ADR-0014 rationale", "wiki/notes/design-log.md", level
            )
            assert keys == [], level

    def test_a_section_with_no_known_level_is_treated_as_a_mention(self):
        """Fail safe: a caller that forgets to pass the level loses a match
        rather than inventing a confident one."""
        assert _document_citation_keys(NodeLabel.DOC_SECTION.value, "ADR-0014: Title", "wiki/notes/log.md") == []

    def test_docsection_never_takes_a_filename_key(self):
        """Every section of a file shares its path; keying on it would make one
        document look like a dozen ambiguous candidates."""
        keys = _document_citation_keys(NodeLabel.DOC_SECTION.value, "Status", "wiki/adr/0014-x.md", 2)
        assert keys == []

    def test_note_title_keys_at_file_rank(self):
        keys = _document_citation_keys(NodeLabel.NOTE.value, "ADR 22 rollout notes", "wiki/notes/adr-22.md")
        assert keys == [(("ADR", 22), 1)]

    def test_heading_that_merely_mentions_a_scheme_is_not_keyed(self):
        keys = _document_citation_keys(NodeLabel.DOC_SECTION.value, "Accepted — amends ADR-0008", "wiki/adr/x.md", 1)
        assert keys == []

    def test_code_nodes_are_never_keyed(self):
        assert _document_citation_keys(NodeLabel.CALLABLE.value, "resolve_calls", "src/graph/client.py") == []


class TestPickCitationTarget:
    def test_the_docfile_wins_over_its_own_h1_section(self):
        """A DocFile and the heading inside it both answer to ADR-0014; that is
        one document described twice, not ambiguity."""
        lookup = _CitationLookup(by_key={("ADR", 14): [(2, "p:wiki/adr/0014-x.md#h1"), (0, "p:wiki/adr/0014-x.md")]})
        assert _pick_citation_target(("ADR", 14), lookup) == ("p:wiki/adr/0014-x.md", 1.0)

    def test_confidence_grades_down_with_the_strength_of_the_evidence(self):
        """A numbered file in an adr/ directory *is* the ADR; a title only
        suggests it. The edge says which of the two it got."""
        by_rank = {
            0: ("p:wiki/adr/0014-x.md", 1.0),
            1: ("p:note", 0.9),
            2: ("p:section", 0.8),
        }
        for rank, expected in by_rank.items():
            lookup = _CitationLookup(by_key={("ADR", 14): [(rank, expected[0])]})
            assert _pick_citation_target(("ADR", 14), lookup) == expected, rank

    def test_a_real_tie_at_the_best_rank_resolves_to_nothing(self):
        lookup = _CitationLookup(by_key={("ADR", 14): [(0, "p:wiki/adr/0014-a.md"), (0, "p:docs/adr/0014-b.md")]})
        assert _pick_citation_target(("ADR", 14), lookup) is None

    def test_duplicate_candidate_rows_for_one_uid_are_not_a_tie(self):
        lookup = _CitationLookup(by_key={("ADR", 14): [(0, "p:doc"), (0, "p:doc")]})
        assert _pick_citation_target(("ADR", 14), lookup) == ("p:doc", 1.0)

    def test_missing_key_resolves_to_nothing(self):
        assert _pick_citation_target(("RFC", 793), _CitationLookup(by_key={})) is None


# ---------------------------------------------------------------------------
# SqliteGraphClient.resolve_citations (backend mirror, no infrastructure)
# ---------------------------------------------------------------------------


def _adr_docfile(project: str, number: str, slug: str, *, directory: str = "wiki/adr") -> ParsedEntity:
    path = f"{directory}/{number}-{slug}.md"
    return ParsedEntity(
        name=f"{number}-{slug}.md",
        qualified_name=f"{project}:{path}",
        label=NodeLabel.DOC_FILE,
        kind="doc_file",
        line_start=1,
        line_end=10,
        file_path=path,
        content_hash=f"h-{number}",
    )


def _citing_callable(
    project: str, name: str, citations: list[str], *, content_hash: str = "", file_path: str = "src/mod.py"
) -> ParsedEntity:
    module = file_path.removesuffix(".py").replace("/", ".")
    return ParsedEntity(
        name=name,
        qualified_name=f"{project}:{module}.{name}",
        label=NodeLabel.CALLABLE,
        kind="function",
        line_start=1,
        line_end=5,
        file_path=file_path,
        citations=citations,
        content_hash=content_hash or f"h-{name}",
    )


async def _citation_edges(client: SqliteGraphClient) -> list[tuple[str, str, str]]:
    """``(document uid, citing uid, citation)`` for every citation-type DOCUMENTS edge.

    Direction is doc → code, like every other DOCUMENTS edge, so ``from_uid``
    is the cited document.
    """
    conn = await client._get_conn()
    cur = await conn.execute(
        "SELECT from_uid, to_uid, json_extract(props_json, '$.citation') FROM edges "
        "WHERE rel_type = 'DOCUMENTS' AND json_extract(props_json, '$.link_type') = 'citation' "
        "ORDER BY from_uid, to_uid"
    )
    rows = await cur.fetchall()
    await cur.close()
    return [(r[0], r[1], r[2]) for r in rows]


async def _unresolved(client: SqliteGraphClient, uid: str) -> list[str] | None:
    import json

    conn = await client._get_conn()
    cur = await conn.execute(
        "SELECT json_extract(props_json, '$.unresolved_citations') FROM nodes WHERE uid = ?", (uid,)
    )
    row = await cur.fetchone()
    await cur.close()
    if row is None or row[0] is None:
        return None
    return json.loads(row[0])


class TestSqliteResolveCitations:
    async def test_padded_and_bare_citations_both_reach_the_adr(self, client: SqliteGraphClient) -> None:
        await client.ensure_schema()
        project = "cit"
        adr = _adr_docfile(project, "0014", "calls-edge-confidence")
        await client.upsert_file_entities(project, adr.file_path, [adr], [])
        padded = _citing_callable(project, "padded", ["ADR-0014"])
        bare = _citing_callable(project, "bare", ["ADR-14"])
        await client.upsert_file_entities(project, "src/mod.py", [padded, bare], [])

        await client.resolve_citations(project, {padded.qualified_name: ["ADR-0014"], bare.qualified_name: ["ADR-14"]})

        assert await _citation_edges(client) == [
            (adr.qualified_name, bare.qualified_name, "ADR-14"),
            (adr.qualified_name, padded.qualified_name, "ADR-14"),
        ]

    async def test_rfc_stays_a_property_with_no_local_document(self, client: SqliteGraphClient) -> None:
        """RFCs are external — no node is invented for them, and the citation is
        recorded as unresolved rather than silently dropped."""
        await client.ensure_schema()
        project = "cit_rfc"
        entity = _citing_callable(project, "parse_headers", ["RFC-7231"])
        await client.upsert_file_entities(project, "src/mod.py", [entity], [])

        await client.resolve_citations(project, {entity.qualified_name: ["RFC-7231"]})

        assert await _citation_edges(client) == []
        assert await _unresolved(client, entity.qualified_name) == ["RFC-7231"]
        assert await client.count_entities(project) == 1  # no phantom RFC node

    async def test_a_vendored_rfc_document_resolves_with_no_special_casing(self, client: SqliteGraphClient) -> None:
        """The resolver is scheme-agnostic: if a repo does vendor the spec, the
        same code links it — which is why RFCs need no node of their own."""
        await client.ensure_schema()
        project = "cit_vendored"
        rfc = _adr_docfile(project, "793", "tcp", directory="wiki/rfc")
        await client.upsert_file_entities(project, rfc.file_path, [rfc], [])
        entity = _citing_callable(project, "handshake", ["RFC-793"])
        await client.upsert_file_entities(project, "src/mod.py", [entity], [])

        await client.resolve_citations(project, {entity.qualified_name: ["RFC-793"]})

        assert await _citation_edges(client) == [(rfc.qualified_name, entity.qualified_name, "RFC-793")]

    async def test_is_idempotent_across_replays(self, client: SqliteGraphClient) -> None:
        await client.ensure_schema()
        project = "cit_idem"
        adr = _adr_docfile(project, "0014", "x")
        await client.upsert_file_entities(project, adr.file_path, [adr], [])
        entity = _citing_callable(project, "f", ["ADR-0014"])
        await client.upsert_file_entities(project, "src/mod.py", [entity], [])

        payload = {entity.qualified_name: ["ADR-0014"]}
        await client.resolve_citations(project, payload)
        await client.resolve_citations(project, payload)

        assert len(await _citation_edges(client)) == 1

    async def test_a_citation_that_starts_resolving_clears_its_unresolved_entry(
        self, client: SqliteGraphClient
    ) -> None:
        await client.ensure_schema()
        project = "cit_late"
        entity = _citing_callable(project, "f", ["ADR-0014"])
        await client.upsert_file_entities(project, "src/mod.py", [entity], [])
        payload = {entity.qualified_name: ["ADR-0014"]}

        await client.resolve_citations(project, payload)
        assert await _unresolved(client, entity.qualified_name) == ["ADR-0014"]

        adr = _adr_docfile(project, "0014", "x")
        await client.upsert_file_entities(project, adr.file_path, [adr], [])
        await client.resolve_citations(project, payload)

        assert await _unresolved(client, entity.qualified_name) == []
        assert len(await _citation_edges(client)) == 1

    async def test_retry_sweep_links_documents_indexed_after_the_citing_file(self, client: SqliteGraphClient) -> None:
        """The ordering case that makes a cold full index work: src/ is published
        before wiki/, so the first pass cannot possibly find the ADR."""
        await client.ensure_schema()
        project = "cit_retry"
        entity = _citing_callable(project, "f", ["ADR-0014"])
        await client.upsert_file_entities(project, "src/mod.py", [entity], [])
        await client.resolve_citations(project, {entity.qualified_name: ["ADR-0014"]})
        assert await _citation_edges(client) == []

        adr = _adr_docfile(project, "0014", "x")
        await client.upsert_file_entities(project, adr.file_path, [adr], [])
        await client.resolve_citations(project, {}, retry_unresolved=True)

        assert await _citation_edges(client) == [(adr.qualified_name, entity.qualified_name, "ADR-14")]
        assert await _unresolved(client, entity.qualified_name) == []

    async def test_retry_sweep_clears_bookkeeping_for_a_deleted_citation(self, client: SqliteGraphClient) -> None:
        await client.ensure_schema()
        project = "cit_gone"
        entity = _citing_callable(project, "f", ["ADR-9999"])
        await client.upsert_file_entities(project, "src/mod.py", [entity], [])
        await client.resolve_citations(project, {entity.qualified_name: ["ADR-9999"]})
        assert await _unresolved(client, entity.qualified_name) == ["ADR-9999"]

        # The comment is edited away: the entity re-upserts with no citations
        # (and a new content_hash — citations feed the hash formula).
        stripped = _citing_callable(project, "f", [], content_hash="h-f-2")
        await client.upsert_file_entities(project, "src/mod.py", [stripped], [])
        await client.resolve_citations(project, {}, retry_unresolved=True)

        assert await _unresolved(client, entity.qualified_name) == []

    async def test_ambiguous_number_creates_no_edge(self, client: SqliteGraphClient) -> None:
        await client.ensure_schema()
        project = "cit_ambig"
        first = _adr_docfile(project, "0014", "a")
        second = _adr_docfile(project, "0014", "b", directory="docs/adr")
        await client.upsert_file_entities(project, first.file_path, [first], [])
        await client.upsert_file_entities(project, second.file_path, [second], [])
        entity = _citing_callable(project, "f", ["ADR-0014"])
        await client.upsert_file_entities(project, "src/mod.py", [entity], [])

        await client.resolve_citations(project, {entity.qualified_name: ["ADR-0014"]})

        assert await _citation_edges(client) == []
        assert await _unresolved(client, entity.qualified_name) == ["ADR-0014"]

    async def test_resolution_does_not_cross_project_boundaries(self, client: SqliteGraphClient) -> None:
        """Every repo has an ADR-0001; a cross-project lookup would collide."""
        await client.ensure_schema()
        other_adr = _adr_docfile("other", "0001", "x")
        await client.upsert_file_entities("other", other_adr.file_path, [other_adr], [])
        entity = _citing_callable("mine", "f", ["ADR-0001"])
        await client.upsert_file_entities("mine", "src/mod.py", [entity], [])

        await client.resolve_citations("mine", {entity.qualified_name: ["ADR-0001"]})

        assert await _citation_edges(client) == []

    async def test_empty_input_without_retry_is_a_no_op(self, client: SqliteGraphClient) -> None:
        await client.ensure_schema()

        await client.resolve_citations("cit_noop", {})

        assert await _citation_edges(client) == []

    async def test_the_edge_runs_document_to_code_like_every_other_documents_edge(
        self, client: SqliteGraphClient
    ) -> None:
        """Direction is the whole point of the read paths: get_linked_docs,
        get_module_summary and the module-summary renderer all treat the
        DOCUMENTS source as the documentation node."""
        await client.ensure_schema()
        project = "cit_dir"
        adr = _adr_docfile(project, "0014", "x")
        await client.upsert_file_entities(project, adr.file_path, [adr], [])
        entity = _citing_callable(project, "f", ["ADR-0014"])
        await client.upsert_file_entities(project, "src/mod.py", [entity], [])

        await client.resolve_citations(project, {entity.qualified_name: ["ADR-0014"]})

        assert await _citation_edges(client) == [(adr.qualified_name, entity.qualified_name, "ADR-14")]

    async def test_the_cited_document_surfaces_in_get_linked_docs(self, client: SqliteGraphClient) -> None:
        """The reader that expand_context/get_context uses. A DocFile is the
        usual citation target, so the doc-side label filter has to admit it."""
        await client.ensure_schema()
        project = "cit_ctx"
        adr = _adr_docfile(project, "0014", "x")
        await client.upsert_file_entities(project, adr.file_path, [adr], [])
        entity = _citing_callable(project, "f", ["ADR-0014"])
        await client.upsert_file_entities(project, "src/mod.py", [entity], [])
        await client.resolve_citations(project, {entity.qualified_name: ["ADR-0014"]})

        docs = await client.get_linked_docs(entity.qualified_name, "", 10)

        assert [(d["node"]["uid"], d["link_type"]) for d in docs] == [(adr.qualified_name, "citation")]

    async def test_a_title_only_match_does_not_claim_full_confidence(self, client: SqliteGraphClient) -> None:
        """The ADR lives outside a scheme-named directory, so only its H1
        identifies it — a real link, but an inferred one."""
        await client.ensure_schema()
        project = "cit_conf"
        section = ParsedEntity(
            name="ADR-0014: CALLS Edge Confidence",
            qualified_name=f"{project}:wiki/decisions/0014-calls.md > ADR-0014: CALLS Edge Confidence",
            label=NodeLabel.DOC_SECTION,
            kind="section",
            line_start=1,
            line_end=20,
            file_path="wiki/decisions/0014-calls.md",
            header_level=1,
            content_hash="h-sec",
        )
        await client.upsert_file_entities(project, section.file_path, [section], [])
        entity = _citing_callable(project, "f", ["ADR-14"])
        await client.upsert_file_entities(project, "src/mod.py", [entity], [])

        await client.resolve_citations(project, {entity.qualified_name: ["ADR-14"]})

        conn = await client._get_conn()
        cur = await conn.execute(
            "SELECT from_uid, json_extract(props_json, '$.confidence') FROM edges "
            "WHERE json_extract(props_json, '$.link_type') = 'citation'"
        )
        rows = await cur.fetchall()
        await cur.close()
        assert rows == [(section.qualified_name, 0.8)]

    async def test_a_subsection_mentioning_the_adr_is_not_linked(self, client: SqliteGraphClient) -> None:
        """A confidently wrong edge is worse than no edge: with no real ADR-0014
        document indexed, the passage discussing it must not stand in for one."""
        await client.ensure_schema()
        project = "cit_mention"
        mention = ParsedEntity(
            name="ADR-0014 rationale",
            qualified_name=f"{project}:wiki/notes/log.md > Design > ADR-0014 rationale",
            label=NodeLabel.DOC_SECTION,
            kind="section",
            line_start=10,
            line_end=20,
            file_path="wiki/notes/log.md",
            header_level=3,
            content_hash="h-mention",
        )
        await client.upsert_file_entities(project, mention.file_path, [mention], [])
        entity = _citing_callable(project, "f", ["ADR-0014"])
        await client.upsert_file_entities(project, "src/mod.py", [entity], [])

        await client.resolve_citations(project, {entity.qualified_name: ["ADR-0014"]})

        assert await _citation_edges(client) == []
        assert await _unresolved(client, entity.qualified_name) == ["ADR-0014"]

    async def test_deleting_the_citing_comment_revokes_the_edge(self, client: SqliteGraphClient) -> None:
        """The removal case. The edge points INTO the citing file's entity, so
        the file's own relationship-delete phase (outgoing edges only) can never
        reach it — only the file-scoped revoke pass can, and it has to fire on a
        parse that produced no citations at all, which is what removal looks
        like."""
        await client.ensure_schema()
        project = "cit_revoke"
        adr = _adr_docfile(project, "0014", "x")
        await client.upsert_file_entities(project, adr.file_path, [adr], [])
        entity = _citing_callable(project, "f", ["ADR-0014"])
        await client.upsert_file_entities(project, "src/mod.py", [entity], [])
        await client.resolve_citations(project, {entity.qualified_name: ["ADR-0014"]}, file_paths={"src/mod.py"})
        assert len(await _citation_edges(client)) == 1

        # The `see ADR-0014` comment is deleted: same entity, no citations.
        stripped = _citing_callable(project, "f", [], content_hash="h-f-2")
        await client.upsert_file_entities(project, "src/mod.py", [stripped], [])
        await client.resolve_citations(project, {}, file_paths={"src/mod.py"})

        assert await _citation_edges(client) == []

    async def test_reparsing_the_citing_file_keeps_a_citation_that_is_still_written(
        self, client: SqliteGraphClient
    ) -> None:
        """Delete-then-recreate, not delete: an untouched citation survives its
        own file being reparsed, exactly once."""
        await client.ensure_schema()
        project = "cit_keep"
        adr = _adr_docfile(project, "0014", "x")
        await client.upsert_file_entities(project, adr.file_path, [adr], [])
        entity = _citing_callable(project, "f", ["ADR-0014"])
        await client.upsert_file_entities(project, "src/mod.py", [entity], [])
        payload = {entity.qualified_name: ["ADR-0014"]}
        await client.resolve_citations(project, payload, file_paths={"src/mod.py"})
        await client.resolve_citations(project, payload, file_paths={"src/mod.py"})

        assert await _citation_edges(client) == [(adr.qualified_name, entity.qualified_name, "ADR-14")]

    async def test_only_the_dropped_citation_of_several_is_revoked(self, client: SqliteGraphClient) -> None:
        await client.ensure_schema()
        project = "cit_partial"
        first = _adr_docfile(project, "0014", "a")
        second = _adr_docfile(project, "0015", "b")
        await client.upsert_file_entities(project, first.file_path, [first], [])
        await client.upsert_file_entities(project, second.file_path, [second], [])
        entity = _citing_callable(project, "f", ["ADR-0014", "ADR-0015"])
        await client.upsert_file_entities(project, "src/mod.py", [entity], [])
        await client.resolve_citations(
            project, {entity.qualified_name: ["ADR-0014", "ADR-0015"]}, file_paths={"src/mod.py"}
        )
        assert len(await _citation_edges(client)) == 2

        await client.resolve_citations(project, {entity.qualified_name: ["ADR-0015"]}, file_paths={"src/mod.py"})

        assert await _citation_edges(client) == [(second.qualified_name, entity.qualified_name, "ADR-15")]

    async def test_the_revoke_pass_spares_files_outside_its_scope(self, client: SqliteGraphClient) -> None:
        """A batch reparsing one file must not touch another file's citations —
        the whole reason the scope is file paths and not the project."""
        await client.ensure_schema()
        project = "cit_scope"
        adr = _adr_docfile(project, "0014", "x")
        await client.upsert_file_entities(project, adr.file_path, [adr], [])
        mine = _citing_callable(project, "f", ["ADR-0014"])
        theirs = _citing_callable(project, "h", ["ADR-0014"], file_path="src/other.py")
        await client.upsert_file_entities(project, "src/mod.py", [mine], [])
        await client.upsert_file_entities(project, "src/other.py", [theirs], [])
        await client.resolve_citations(
            project,
            {mine.qualified_name: ["ADR-0014"], theirs.qualified_name: ["ADR-0014"]},
            file_paths={"src/mod.py", "src/other.py"},
        )
        assert len(await _citation_edges(client)) == 2

        # src/mod.py alone is reparsed, with its citation gone.
        await client.resolve_citations(project, {}, file_paths={"src/mod.py"})

        assert await _citation_edges(client) == [(adr.qualified_name, theirs.qualified_name, "ADR-14")]

    async def test_the_retry_sweep_revokes_nothing(self, client: SqliteGraphClient) -> None:
        """The sweep is project-wide and reparses nothing, so it gets no scope.
        Deleting there would wipe every citation in the project on the first
        newly-indexed ADR. The unresolved entity deliberately shares a file with
        a resolved one, so a scope wrongly derived from the sweep's own pending
        set would take the good edge with it."""
        await client.ensure_schema()
        project = "cit_sweep"
        adr = _adr_docfile(project, "0014", "x")
        await client.upsert_file_entities(project, adr.file_path, [adr], [])
        resolved = _citing_callable(project, "f", ["ADR-0014"])
        broken = _citing_callable(project, "g", ["ADR-9999"])
        elsewhere = _citing_callable(project, "h", ["ADR-0014"], file_path="src/other.py")
        await client.upsert_file_entities(project, "src/mod.py", [resolved, broken], [])
        await client.upsert_file_entities(project, "src/other.py", [elsewhere], [])
        await client.resolve_citations(
            project,
            {
                resolved.qualified_name: ["ADR-0014"],
                broken.qualified_name: ["ADR-9999"],
                elsewhere.qualified_name: ["ADR-0014"],
            },
            file_paths={"src/mod.py", "src/other.py"},
        )
        before = await _citation_edges(client)
        assert len(before) == 2

        await client.resolve_citations(project, {}, retry_unresolved=True)

        assert await _citation_edges(client) == before

    async def test_reparsing_the_cited_document_keeps_its_citation_edges(self, client: SqliteGraphClient) -> None:
        """The edge leaves the document's node but is owned by the citing file's
        parse — the document's own relationship-delete phase must skip it, or
        every ADR edit would silently drop every citation pointing at it."""
        await client.ensure_schema()
        project = "cit_reparse"
        adr = _adr_docfile(project, "0014", "x")
        await client.upsert_file_entities(project, adr.file_path, [adr], [])
        entity = _citing_callable(project, "f", ["ADR-0014"])
        await client.upsert_file_entities(project, "src/mod.py", [entity], [])
        await client.resolve_citations(project, {entity.qualified_name: ["ADR-0014"]})
        assert len(await _citation_edges(client)) == 1

        edited = replace(_adr_docfile(project, "0014", "x"), content_hash="h-0014-edited")
        await client.upsert_file_entities(project, edited.file_path, [edited], [])

        assert await _citation_edges(client) == [(adr.qualified_name, entity.qualified_name, "ADR-14")]

    async def test_the_revoke_pass_keeps_citations_when_the_cited_document_is_reparsed(
        self, client: SqliteGraphClient
    ) -> None:
        """Same carve-out, now against the revoke pass rather than the relationship
        delete phase: editing the ADR puts the ADR's OWN file in the scope, and
        the scope is read on the citing (target) side of the edge, so it must not
        match. Scoping it on the source side would drop every citation the ADR
        answers to on every edit — the failure the carve-out exists to prevent,
        reintroduced one layer down."""
        await client.ensure_schema()
        project = "cit_reparse_scoped"
        adr = _adr_docfile(project, "0014", "x")
        await client.upsert_file_entities(project, adr.file_path, [adr], [])
        entity = _citing_callable(project, "f", ["ADR-0014"])
        await client.upsert_file_entities(project, "src/mod.py", [entity], [])
        await client.resolve_citations(project, {entity.qualified_name: ["ADR-0014"]}, file_paths={"src/mod.py"})
        assert len(await _citation_edges(client)) == 1

        edited = replace(_adr_docfile(project, "0014", "x"), content_hash="h-0014-edited")
        await client.upsert_file_entities(project, edited.file_path, [edited], [])
        await client.resolve_citations(project, {}, file_paths={adr.file_path}, retry_unresolved=True)

        assert await _citation_edges(client) == [(adr.qualified_name, entity.qualified_name, "ADR-14")]
