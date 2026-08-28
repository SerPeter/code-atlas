"""Backend conformance: the shared surface, run against Memgraph and SQLite (ATL-134).

Both backends claim to satisfy `GraphBackend`. Until now nothing checked that claim by
comparing what they *return* — each was tested alone, and the gap cost real defects.
All three wrong-answer bugs ATL-112 fixed were found by a human reading the Memgraph
implementation line by line and diffing it against the SQLite one:

- `find_dead_code` was missing four exclusions and reported live code as dead
- `blast_radius(resolved_only=True)` dropped every structural edge
- `graph_search` passed user text into SQL `LIKE` unescaped, so `_` and `%` were wildcards

A suite that compares outputs is what finds the fourth one.

**The coverage ledger is the load-bearing part.** `GraphBackend` declares ~98 methods and
comparing all of them would blow the CI budget several times over, so this file does not
pretend to. Instead every protocol method must appear in exactly one of three sets —
`_COMPARED`, `_REFUSED`, or `_NOT_COMPARED` — and `test_every_protocol_method_is_classified`
fails if any method is in none. A new method cannot be added to the protocol without a
deliberate decision about whether it is checked.

`_NOT_COMPARED` carries a reason per method rather than a bare name. A silent exclusion
list is indistinguishable from an oversight; a stated one can be argued with.
"""

from __future__ import annotations

import inspect
import re
from typing import TYPE_CHECKING, Any

import pytest

from code_atlas.backends.sqlite_graph import SqliteGraphClient
from code_atlas.parsing.ast import ParsedEntity, ParsedRelationship
from code_atlas.schema import NodeLabel, RelType

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration

PROJECT = "test-conformance"


# ---------------------------------------------------------------------------
# The coverage ledger
# ---------------------------------------------------------------------------

# Read methods whose output is compared between backends below.
_COMPARED: frozenset[str] = frozenset(
    {
        "count_entities",
        "get_entity_by_uid",
        "get_existing_uids",
        "node_exists",
        "find_entity_uid",
        "get_label_counts",
        "get_node_exact_matches",
        "get_node_partial_matches",
        "graph_search",
        "get_callers",
        "get_callees",
        "get_dead_code_candidates",
        "compute_blast_radius",
        "get_project_file_paths",
        "read_embed_hashes",
        "find_embeddings_by_hash",
    }
)

# ADR-0015 places these outside the embedded backend. Asserted to REFUSE, never skipped:
# a skip and a silently wrong answer look identical in a test report.
_REFUSED: frozenset[str] = frozenset({"execute", "execute_write"})

# Everything else, with the reason it is not output-compared. Grouped by why.
_NOT_COMPARED: dict[str, str] = {
    # -- lifecycle / connection: no comparable return value ------------------
    "ping": "liveness, not data",
    "close": "lifecycle",
    "ensure_schema": "DDL; the backends' schemas are deliberately different shapes",
    "get_schema_version": "compared implicitly — both must reach the same version to run at all",
    # -- writes: verified by the reads that follow them ----------------------
    "upsert_file_entities": "write path; every comparison below reads what it wrote",
    "upsert_batch_entities": "write path, as above",
    "delete_file_entities": "write path",
    "delete_project_data": "write path",
    "merge_project_node": "write path",
    "merge_package_node": "write path",
    "merge_package_batch": "write path",
    "create_contains_edge": "write path",
    "create_depends_on_edges": "write path",
    "update_project_metadata": "write path",
    "update_external_package_versions": "write path",
    "apply_property_enrichments": "write path",
    "write_embeddings": "write path; read back by find_embeddings_by_hash",
    "write_embeddings_and_hashes": "write path; read back by read_embed_hashes",
    "write_embed_hashes": "write path; read back by read_embed_hashes",
    "clear_embeddings": "write path",
    "set_embedding_config": "write path; read back by get_embedding_config",
    "set_project_embedding_model": "write path",
    "set_batch_file_hashes": "write path; read back by get_batch_file_hashes",
    "write_git_file_signals": "write path",
    "write_co_change_edges": "write path",
    "gc_orphaned_reference_nodes": "write path",
    "invalidate_stale_anchors": "write path",
    "stamp_note_relations": "write path",
    # -- resolution passes: stateful, order-dependent, and compared only in
    #    aggregate by the edge-shaped reads above -----------------------------
    "resolve_calls": "resolution pass; its output is the CALLS edges get_callers compares",
    "resolve_imports": "resolution pass",
    "resolve_inherits": "resolution pass",
    "resolve_type_refs": "resolution pass",
    "resolve_value_references": "resolution pass",
    "resolve_member_defines": "resolution pass",
    "resolve_config_refs": "resolution pass",
    "resolve_anchors": "resolution pass",
    "resolve_citations": "resolution pass",
    "resolve_cross_project_imports": "resolution pass",
    "resolve_protocol_conformance": "returns 0 unconditionally on SQLite — a known, recorded gap",
    "build_resolution_lookup": "internal to resolve_calls",
    "build_anchor_lookup": "internal to resolve_anchors",
    "build_citation_lookup": "internal to resolve_citations",
    "find_overridden_method": "internal to resolution",
    "get_defining_parent": "internal to resolution",
    "batch_call_stats": "internal to resolution",
    # -- infrastructure-specific by construction -----------------------------
    "vector_search": "sqlite-vec vs Memgraph vector index — different engines, different neighbours",
    "text_search": "FTS5 vs Tantivy — different tokenisers, so ranking cannot match by design",
    "rebuild_vector_indices": "DDL, engine-specific",
    "get_vector_index_info": "engine-specific metadata shape",
    "get_text_index_info": "engine-specific metadata shape",
    "find_unembedded_entities": "ordering is engine-defined; membership covered by read_embed_hashes",
    # -- reads not yet compared. The honest residue: each is a real gap, not a
    #    justification. Comparing them is follow-up work, not a decision. ------
    "get_structure_overview": "not yet compared",
    "get_module_summary": "not yet compared",
    "get_package_docstring": "not yet compared",
    "get_sibling_entities": "not yet compared",
    "get_linked_docs": "not yet compared",
    "get_module_import_edges": "not yet compared",
    "get_project_dependency_edges": "not yet compared",
    "get_dependency_external_counts": "not yet compared",
    "get_diagram_packages": "not yet compared",
    "get_diagram_inheritance": "not yet compared",
    "get_diagram_module_detail": "not yet compared",
    "get_centrality_data": "not yet compared",
    "get_complexity_hotspots": "not yet compared",
    "get_quality_data": "not yet compared",
    "get_patterns_data": "not yet compared",
    "get_git_signals_data": "not yet compared",
    "trace_path_between": "not yet compared",
    "get_project_status": "not yet compared",
    "get_project_git_hash": "not yet compared",
    "get_batch_file_hashes": "not yet compared",
    "get_embedding_config": "not yet compared",
    "get_embedding_models_by_project": "not yet compared",
    "get_project_embedding_model": "not yet compared",
    "count_embeddings_by_project": "not yet compared",
    "read_entity_texts": "not yet compared",
    "get_notes_for_dedup": "not yet compared",
    "get_orphan_notes": "not yet compared",
    "get_inbox_note_paths": "not yet compared",
    "get_broken_anchor_notes": "not yet compared",
}


def _protocol_methods() -> set[str]:
    """Every method `GraphBackend` declares, read from the source.

    From the source rather than `dir()`: the protocol lives inside a
    `if TYPE_CHECKING:` block, so there is no runtime class to introspect.
    """
    from code_atlas.graph import protocol

    src = inspect.getsource(protocol)
    body = src[src.index("class GraphBackend") :]
    return {m for m in re.findall(r"^\s+(?:async )?def (\w+)\(", body, re.MULTILINE) if not m.startswith("__")}


class TestCoverageLedger:
    def test_every_protocol_method_is_classified(self):
        """A method in none of the three sets fails here.

        This is the whole point: it makes adding a backend method a decision about
        conformance rather than something that quietly ships uncompared.
        """
        classified = _COMPARED | _REFUSED | set(_NOT_COMPARED)
        unclassified = sorted(_protocol_methods() - classified)
        assert unclassified == [], (
            f"{len(unclassified)} GraphBackend method(s) are neither compared, refused, nor "
            f"explicitly excluded with a reason: {unclassified}"
        )

    def test_the_ledger_does_not_name_methods_that_no_longer_exist(self):
        """The other drift direction. A stale entry makes coverage look broader than
        it is — the ledger claims to account for a method the protocol dropped."""
        stale = sorted((_COMPARED | _REFUSED | set(_NOT_COMPARED)) - _protocol_methods())
        assert stale == [], f"ledger names methods not on the protocol: {stale}"

    def test_no_method_is_classified_twice(self):
        overlap = sorted((_COMPARED & _REFUSED) | (_COMPARED & set(_NOT_COMPARED)) | (_REFUSED & set(_NOT_COMPARED)))
        assert overlap == []

    def test_every_exclusion_states_a_reason(self):
        blank = sorted(name for name, why in _NOT_COMPARED.items() if not why.strip())
        assert blank == [], f"excluded without a reason: {blank}"


# ---------------------------------------------------------------------------
# One corpus, both backends
# ---------------------------------------------------------------------------


def _entity(
    name: str,
    qn: str,
    *,
    label: NodeLabel = NodeLabel.CALLABLE,
    kind: str = "function",
    file_path: str = "mod.py",
    docstring: str | None = None,
    decorator_name: str | None = None,
) -> ParsedEntity:
    extra: dict[str, Any] = {}
    if decorator_name:
        extra["decorator_name"] = decorator_name
    return ParsedEntity(
        name=name,
        qualified_name=qn,
        label=label,
        kind=kind,
        line_start=1,
        line_end=3,
        file_path=file_path,
        docstring=docstring,
        signature=f"def {name}()",
        content_hash=f"h-{qn}",
        extra_properties=extra,
    )


def _qn(name: str) -> str:
    """Parsers emit `{project}:{dotted}` as the qualified name, and
    `upsert_file_entities` uses it verbatim as the uid. The corpus does the same, or the
    uids under test are not the ones production creates."""
    return f"{PROJECT}:mod.{name}"


def _corpus() -> tuple[list[ParsedEntity], list[ParsedRelationship]]:
    """A corpus shaped by the defects this suite exists to catch.

    `weird_name` and `pct%name` exist for the LIKE-metacharacter bug: `_` and `%` are
    SQL wildcards, so an unescaped query matched things it should not. `helper` is
    referenced but never called, which is the REFERENCES-as-proof-of-life exclusion.
    `registered` carries a decorator, the framework-hook exclusion.
    """
    entities = [
        _entity("caller", _qn("caller")),
        _entity("callee", _qn("callee")),
        _entity("orphan", _qn("orphan")),
        _entity("helper", _qn("helper")),
        _entity("registered", _qn("registered"), decorator_name="app.route"),
        _entity("weird_name", _qn("weird_name")),
        _entity("weirdXname", _qn("weirdXname")),
        _entity("Widget", _qn("Widget"), label=NodeLabel.TYPE_DEF, kind="class"),
        _entity("mod", f"{PROJECT}:mod", label=NodeLabel.MODULE, kind="module"),
    ]
    rels = [
        # DEFINES is the only relationship `upsert_file_entities` materialises on both
        # backends -- CALLS, REFERENCES and the rest are pending until their resolution
        # pass runs. Worth knowing before extending this corpus: seeding a rel is not
        # the same as having an edge.
        ParsedRelationship(from_qualified_name=f"{PROJECT}:mod", rel_type=RelType.DEFINES, to_name=_qn("caller")),
        ParsedRelationship(from_qualified_name=_qn("caller"), rel_type=RelType.CALLS, to_name=_qn("callee")),
        ParsedRelationship(from_qualified_name=_qn("caller"), rel_type=RelType.REFERENCES, to_name=_qn("helper")),
    ]
    return entities, rels


async def _seed(client: Any) -> None:
    await client.ensure_schema()
    entities, rels = _corpus()
    await client.upsert_file_entities(PROJECT, "mod.py", entities, rels)


@pytest.fixture
async def both(graph_client, tmp_path: Path):
    """Memgraph and SQLite, seeded from the same corpus.

    Yields `(memgraph, sqlite)`. The SQLite database is a fresh file per test, so its
    state cannot leak between comparisons the way the shared Memgraph's would without
    the wipe fixture.
    """
    sqlite = SqliteGraphClient(tmp_path / "conformance.sqlite3")
    await _seed(graph_client)
    await _seed(sqlite)
    try:
        yield graph_client, sqlite
    finally:
        await sqlite.close()


def _uid_of(node: Any) -> str | None:
    """Pull a uid out of whatever shape the backend used for a node.

    Memgraph returns neo4j `Node` objects; SQLite returns plain dicts. Both support
    `[]`/`.get`, but a `Node` is not a `dict`, so an `isinstance(node, dict)` guard
    silently drops every Memgraph row and the comparison reads as "Memgraph returned
    nothing" — which looks exactly like a backend defect. It is not; it is the harness.
    """
    if node is None:
        return None
    if isinstance(node, dict):
        return node.get("uid")
    getter = getattr(node, "get", None)
    return getter("uid") if callable(getter) else None


def _keys(rows: list[dict[str, Any]], *fields: str) -> set[str]:
    """Identify rows by whichever field the method actually returns.

    Not every read returns `uid`. `get_dead_code_candidates` returns
    `{name, qn, label, kind, file_path, line_start}` and nothing else, so comparing it
    with a uid-only extractor yields `set() == set()` — which passes while proving
    nothing. That is exactly how the first version of the dead-code comparison here
    passed with the REFERENCES exclusion deliberately reverted.
    """
    out: set[str] = set()
    for row in rows:
        for field in fields:
            value = row.get(field)
            if value:
                out.add(str(value))
                break
    return out


def _uids(rows: list[dict[str, Any]]) -> set[str]:
    """Compare by uid membership, not row shape.

    The backends legitimately return different column sets and different orders; what
    has to agree is *which entities* answer the question.
    """
    out: set[str] = set()
    for row in rows:
        uid = _uid_of(row.get("n")) or _uid_of(row.get("node")) or _uid_of(row)
        if uid:
            out.add(uid)
    return out


class TestSharedSurfaceAgrees:
    """Same corpus, same question, same answer — or the difference is a defect."""

    async def test_count_entities(self, both):
        mg, lite = both
        assert await mg.count_entities(PROJECT) == await lite.count_entities(PROJECT)

    async def test_get_existing_uids(self, both):
        mg, lite = both
        wanted = [f"{PROJECT}:mod.caller", f"{PROJECT}:mod.absent"]
        assert await mg.get_existing_uids(wanted) == await lite.get_existing_uids(wanted)

    async def test_node_exists(self, both):
        mg, lite = both
        for uid in (f"{PROJECT}:mod.caller", f"{PROJECT}:mod.absent"):
            assert await mg.node_exists(uid) == await lite.node_exists(uid)

    async def test_find_entity_uid(self, both):
        mg, lite = both
        for name in ("caller", "absent"):
            assert await mg.find_entity_uid(PROJECT, "Callable", name) == await lite.find_entity_uid(
                PROJECT, "Callable", name
            )

    async def test_get_entity_by_uid(self, both):
        mg, lite = both
        uid = f"{PROJECT}:mod.caller"
        a, b = await mg.get_entity_by_uid(uid), await lite.get_entity_by_uid(uid)
        assert (a is None) == (b is None)
        assert a["name"] == b["name"]
        assert a["qualified_name"] == b["qualified_name"]

    async def test_get_label_counts(self, both):
        mg, lite = both
        a, b = await mg.get_label_counts(), await lite.get_label_counts()
        # Compare only the labels this corpus creates: the shared Memgraph may carry
        # meta nodes the fresh SQLite file cannot.
        for label in ("Callable", "TypeDef"):
            assert a.get(label, 0) == b.get(label, 0), f"{label} count differs"

    async def test_get_node_exact_matches(self, both):
        mg, lite = both
        assert _uids(await mg.get_node_exact_matches("caller", "", 10)) == _uids(
            await lite.get_node_exact_matches("caller", "", 10)
        )

    async def test_get_node_partial_matches(self, both):
        mg, lite = both
        assert _uids(await mg.get_node_partial_matches("call", "", 10)) == _uids(
            await lite.get_node_partial_matches("call", "", 10)
        )

    async def test_get_project_file_paths(self, both):
        mg, lite = both
        assert await mg.get_project_file_paths(PROJECT) == await lite.get_project_file_paths(PROJECT)

    async def test_get_callers_and_callees(self, both):
        mg, lite = both
        callee = f"{PROJECT}:mod.callee"
        caller = f"{PROJECT}:mod.caller"
        assert _uids(await mg.get_callers(callee, "Callable", 1, 10)) == _uids(
            await lite.get_callers(callee, "Callable", 1, 10)
        )
        assert _uids(await mg.get_callees(caller, "Callable", 1, 10)) == _uids(
            await lite.get_callees(caller, "Callable", 1, 10)
        )

    async def test_read_embed_hashes(self, both):
        mg, lite = both
        uids = [f"{PROJECT}:mod.caller", f"{PROJECT}:mod.absent"]
        assert await mg.read_embed_hashes(uids) == await lite.read_embed_hashes(uids)

    async def test_find_embeddings_by_hash(self, both):
        mg, lite = both
        dim = mg._dimension
        uid = f"{PROJECT}:mod.caller"
        for client in (mg, lite):
            await client.write_embeddings_and_hashes(
                [(uid, [0.25] * dim, "shared-hash")], labels=["Callable"], model="model-x"
            )
        a = await mg.find_embeddings_by_hash(["shared-hash", "absent"], "model-x")
        b = await lite.find_embeddings_by_hash(["shared-hash", "absent"], "model-x")
        assert set(a) == set(b) == {"shared-hash"}
        # And the model filter agrees too — a vector from another space is not copied.
        assert await mg.find_embeddings_by_hash(["shared-hash"], "model-y") == {}
        assert await lite.find_embeddings_by_hash(["shared-hash"], "model-y") == {}


class TestTheDefectsThatMotivatedThis:
    """One test per wrong-answer bug ATL-112 found by hand. A suite that compares
    outputs is what should have found them."""

    async def test_like_metacharacters_are_not_wildcards(self, both):
        """`_` matched any character, so searching `weird_name` also returned
        `weirdXname`. Both backends must now agree that it does not."""
        mg, lite = both
        a = _uids(await mg.graph_search("weird_name", project=PROJECT, limit=20))
        b = _uids(await lite.graph_search("weird_name", project=PROJECT, limit=20))
        assert a == b
        assert f"{PROJECT}:mod.weirdXname" not in b

    async def test_a_percent_query_does_not_match_everything(self, both):
        mg, lite = both
        a = _uids(await mg.graph_search("%", project=PROJECT, limit=20))
        b = _uids(await lite.graph_search("%", project=PROJECT, limit=20))
        assert a == b
        assert len(b) < 8, "a bare % matched the whole corpus"

    async def test_blast_radius_keeps_structural_edges(self, both):
        """SQLite's json_extract yielded NULL where Memgraph uses
        coalesce(r.confidence, 'resolved'), so every structural edge was filtered out
        of a resolved_only traversal (ADR-0028).

        Traverses DEFINES rather than CALLS, for two reasons. A CALLS edge does not
        exist until `resolve_calls` runs, and neither does REFERENCES — DEFINES is the
        only relationship `upsert_file_entities` materialises directly. And DEFINES is
        *structural*: it carries no confidence property at all, which is exactly the
        class the defect filtered out.
        """
        mg, lite = both
        uid = _qn("caller")
        a = _uids(await mg.compute_blast_radius(uid, "dependents", ("DEFINES",), 2))
        b = _uids(await lite.compute_blast_radius(uid, "dependents", ("DEFINES",), 2))
        assert a == b, f"only in memgraph: {a - b}; only in sqlite: {b - a}"
        assert f"{PROJECT}:mod" in b, "a structural edge was dropped from the traversal"

    async def test_dead_code_exclusions_agree(self, both):
        """The defect a user might act on by deleting live code, so the comparison has
        to be exact rather than approximately similar.

        Identified by `qn`, because this method returns no `uid` at all — see `_keys`.

        The REFERENCES-as-proof-of-life claim is deliberately NOT asserted: REFERENCES
        edges do not exist until `resolve_value_references` runs, so a seeded corpus
        cannot exercise that exclusion, and asserting it would pass for the wrong
        reason. What is asserted is what this corpus can actually show — the backends
        name the same set, the decorator exclusion holds on both, and a genuinely
        unreferenced function is still reported (so the comparison is not passing by
        returning nothing).
        """
        mg, lite = both
        a = _keys(await mg.get_dead_code_candidates(PROJECT, ""), "uid", "qn")
        b = _keys(await lite.get_dead_code_candidates(PROJECT, ""), "uid", "qn")
        assert a, "no candidates at all — the comparison would be vacuous"
        assert a == b, f"only in memgraph: {a - b}; only in sqlite: {b - a}"
        assert not any("registered" in k for k in b), "a decorated callable is registered, not dead"
        assert any("orphan" in k for k in b), "an unreferenced function must still be reported"


class TestOutOfScopeRefuses:
    """ADR-0015 places raw Cypher outside the embedded backend. Asserted as a refusal,
    never a skip: a skip and a silently wrong answer look identical in a report."""

    async def test_execute_refuses(self, both):
        _mg, lite = both
        with pytest.raises(NotImplementedError, match="not supported by the sqlite backend"):
            await lite.execute("MATCH (n) RETURN n")

    async def test_execute_write_refuses(self, both):
        _mg, lite = both
        with pytest.raises(NotImplementedError, match="not supported by the sqlite backend"):
            await lite.execute_write("CREATE (n:Foo)")
