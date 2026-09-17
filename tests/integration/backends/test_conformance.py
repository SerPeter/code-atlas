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
from code_atlas.graph.client import EmbedChunkWrite
from code_atlas.parsing.ast import ParsedEntity, ParsedRelationship
from code_atlas.schema import IMPORT_ATOMIC_NAME, NodeLabel, RelType

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
        "count_project_data",
        "get_entity_by_uid",
        "get_existing_uids",
        "node_exists",
        "find_entity_uids",
        "get_label_counts",
        "get_node_exact_matches",
        "get_node_partial_matches",
        "graph_search",
        "get_package_dependents",
        "classify_external_package_provenance",
        "upsert_external_stubs",
        "get_stubbed_package_versions",
        "resolve_cross_project_imports",
        "get_callers",
        "get_callees",
        "get_dead_code_candidates",
        "compute_blast_radius",
        "trace_path_between",
        "get_project_file_paths",
        "read_embed_hashes",
        "find_embeddings_by_hash",
        "find_embedded_entities",
        "clear_embeddings_for_uids",
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
    # Partly compared: its *scope* is, by TestTheDefectsThatMotivatedThis
    # .test_a_scoped_embedding_clear_reaches_the_same_projects. The counts it returns are
    # not, so it stays here rather than moving to _COMPARED.
    "clear_embeddings": "write path; scope compared by the scoped-clear defect test",
    "write_embed_chunks": "write path; read back by the vector search that resolves a chunk to its parent",
    "delete_embed_chunks": "write path",
    # Not in _COMPARED because that set is read as "output-compared for equal answers on
    # the seeded corpus"; this one seeds its own orphan first. The comparison is real —
    # see TestOutputEquivalence.test_gc_orphaned_embed_chunks.
    "gc_orphaned_embed_chunks": "write path; the count both return is compared in its own test",
    "set_embedding_config": "write path; read back by get_embedding_config",
    "set_project_embedding_model": "write path",
    "set_batch_file_hashes": "write path; read back by get_batch_file_hashes",
    "set_batch_rels_hashes": "write path; read back by get_batch_rels_hashes",
    "write_git_file_signals": "write path",
    "write_co_change_edges": "write path",
    "gc_orphaned_reference_nodes": "write path",
    "invalidate_stale_anchors": "write path",
    "stamp_note_relations": "write path",
    # -- resolution passes: stateful, order-dependent, and compared only in
    #    aggregate by the edge-shaped reads above -----------------------------
    #
    #    A bare "resolution pass" means NOT COMPARED. It does not mean "covered
    #    elsewhere" -- the entries that are say so and name what covers them. ATL-171
    #    reached main through this gap: SQLite deleted cross-file DEFINES edges that
    #    Memgraph carves out, and the only test for that invariant took the Memgraph
    #    fixture. If you add an entry here, prefer naming its cover.
    "resolve_calls": "resolution pass; its output is the CALLS edges get_callers compares",
    "resolve_imports": "resolution pass",
    "resolve_inherits": "resolution pass",
    "resolve_type_refs": "resolution pass",
    "resolve_value_references": "resolution pass",
    "resolve_member_defines": (
        "resolution pass; the cross-file DEFINES it creates are pinned on both backends by "
        "tests/unit/backends/test_sqlite_cross_file_defines.py and "
        "test_client.py::test_resolve_member_defines_cross_file"
    ),
    "resolve_config_refs": "resolution pass",
    "resolve_anchors": "resolution pass",
    "resolve_doc_links": "resolution pass",
    "resolve_citations": "resolution pass",
    "resolve_warehouse_objects": "resolution pass; its FEEDS output is compared by its own integration test",
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
    # Partly compared, so it stays here rather than moving to _COMPARED — that set means
    # "every field of the answer agrees on the seeded corpus", and only external_deps
    # does. See TestSharedSurfaceAgrees.test_external_dependency_versions: the v18 version
    # join is an OPTIONAL MATCH on one backend and a third LEFT JOIN on the other, which
    # is the part with no structural reason to stay in step.
    "get_structure_overview": "counts/packages/largest_modules not yet compared; external_deps is, in its own test",
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
    "get_project_status": "not yet compared",
    "get_project_git_hash": "not yet compared",
    "get_project_extraction_key": "not yet compared",
    "get_batch_file_hashes": "not yet compared",
    "get_batch_rels_hashes": "not yet compared",
    "get_embedding_config": "not yet compared",
    "get_embedding_models_by_project": "not yet compared",
    "get_project_embedding_model": "not yet compared",
    "count_embeddings_by_project": "compared by the scoped-clear defect test, which reads it on both backends",
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


LIMITED_PROJECT = "limitedproj"


def _limited_corpus() -> tuple[list[ParsedEntity], list[ParsedRelationship]]:
    """A corpus where every limited query's LIMIT actually binds.

    Three shapes, because the four methods cut on three different things:

    * **20 callers into one target, 20 callees out of it** -- `get_callers`/`get_callees`.
      Memgraph expands a variable-length CALLS pattern; SQLite runs a BFS and then selects
      the reached uids. Those produce genuinely different natural orders, which is what
      makes an unordered LIMIT diverge rather than merely be arbitrary.
    * **12 entities sharing the name `dupe`** -- `get_node_exact_matches`' second branch
      matches on `name`, so it needs duplicates to have anything to cut. One function name
      appearing in twelve modules is ordinary.
    * The caller/callee names all contain `edge`, so `get_node_partial_matches`' CONTAINS
      branch matches 40 and truncates.

    Lowercase throughout, for the reason `_order_corpus` documents: SQLite's LIKE is
    case-insensitive over ASCII and Cypher's CONTAINS is not, a pre-existing divergence
    that would make this red for an unrelated reason.
    """
    target_qn = f"{LIMITED_PROJECT}:mod.edge_target"
    entities = [
        _entity("mod", f"{LIMITED_PROJECT}:mod", label=NodeLabel.MODULE, kind="module"),
        _entity("edge_target", target_qn),
        *(_entity(f"edge_caller_{i:02d}", f"{LIMITED_PROJECT}:mod.edge_caller_{i:02d}") for i in range(20)),
        *(_entity(f"edge_callee_{i:02d}", f"{LIMITED_PROJECT}:mod.edge_callee_{i:02d}") for i in range(20)),
        *(_entity("dupe", f"{LIMITED_PROJECT}:pkg{i:02d}.dupe") for i in range(12)),
    ]
    rels = [
        *(
            ParsedRelationship(
                from_qualified_name=f"{LIMITED_PROJECT}:mod.edge_caller_{i:02d}",
                rel_type=RelType.CALLS,
                to_name="edge_target",
            )
            for i in range(20)
        ),
        *(
            ParsedRelationship(from_qualified_name=target_qn, rel_type=RelType.CALLS, to_name=f"edge_callee_{i:02d}")
            for i in range(20)
        ),
    ]
    return entities, rels


@pytest.fixture
async def limited_both(graph_client, tmp_path: Path):
    """Both backends holding `_limited_corpus`, seeded in OPPOSITE orders.

    The reversal is the whole point, and ATL-186 proved it: its first ordering test
    passed *without* the fix, because both backends were seeded identically and storage
    order coincided. Agreement is only evidence when the write paths disagree.

    `resolve_calls` has to run: `upsert_file_entities` materialises DEFINES and nothing
    else, so a seeded CALLS rel is not yet an edge and `get_callers` would compare two
    empty lists.
    """
    sqlite = SqliteGraphClient(tmp_path / "limited.sqlite3")
    entities, rels = _limited_corpus()
    await graph_client.ensure_schema()
    await graph_client.upsert_file_entities(LIMITED_PROJECT, "mod.py", entities, rels)
    await graph_client.resolve_calls(LIMITED_PROJECT, rels)
    await sqlite.ensure_schema()
    await sqlite.upsert_file_entities(LIMITED_PROJECT, "mod.py", list(reversed(entities)), rels)
    await sqlite.resolve_calls(LIMITED_PROJECT, list(reversed(rels)))
    try:
        yield graph_client, sqlite
    finally:
        await sqlite.close()


ORDER_PROJECT = "orderproj"


def _order_corpus() -> list[ParsedEntity]:
    """A corpus built so the per-stage LIMIT binds, and so the ordering key's tiers are
    all exercised.

    24 entities match "parse". At `limit=5` the cascade fetches 15 per stage, so stage 3
    truncates and *which* rows it keeps becomes observable -- the thing the 9-entity
    `_corpus` at limit=20 can never test.

    Everything is lowercase on purpose. SQLite's `LIKE` is case-insensitive over ASCII
    while Cypher's `CONTAINS` is not, a PRE-EXISTING divergence in the WHERE clauses that
    is not ATL-186's to fix; a mixed-case name here would make this test red for an
    unrelated reason and read as a false failure of the ordering.
    """
    named = ["parse", "parse_tree", "parser", "do_parse", "zparse", "aparse"]
    return [
        # tier 0: the name itself contains the query
        *(_entity(n, f"{ORDER_PROJECT}:mod.{n}") for n in named),
        # tier 1: only the module path contains it -- 18 of them, to overflow the window
        *(_entity(f"helper_{i:02d}", f"{ORDER_PROJECT}:parse.helper_{i:02d}") for i in range(18)),
    ]


@pytest.fixture
async def ordered_both(graph_client, tmp_path: Path):
    """Both backends, seeded from `_order_corpus` under its own project.

    A separate fixture rather than an extension of `_corpus`, because the shared corpus
    is deliberately tiny and shaped around other defects; growing it to 24 entities to
    make one LIMIT bind would change what every other comparison in this file is testing.
    """
    sqlite = SqliteGraphClient(tmp_path / "order.sqlite3")
    # Seeded in OPPOSITE orders on purpose. Insertion order is what storage order
    # follows, so seeding both the same way lets the backends agree by coincidence and
    # the comparison asserts nothing -- verified: with the ORDER BY removed, an
    # identically-seeded pair still matched. Reversing one makes agreement provable
    # evidence that the key decides the order rather than the write path.
    await graph_client.ensure_schema()
    await graph_client.upsert_file_entities(ORDER_PROJECT, "mod.py", _order_corpus(), [])
    await sqlite.ensure_schema()
    await sqlite.upsert_file_entities(ORDER_PROJECT, "mod.py", list(reversed(_order_corpus())), [])
    try:
        yield graph_client, sqlite
    finally:
        await sqlite.close()


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


def _ranked_uids(rows: list[dict[str, Any]]) -> list[str]:
    """Compare by uid *sequence*, for the methods that promise an order.

    Deliberately separate from `_uids` rather than making that ordered. Most compared
    methods legitimately return an unordered set, and several of them
    (`get_node_partial_matches`, `get_callers`, `get_callees`) still carry the very
    unordered-LIMIT defect ATL-186 fixed in `graph_search` -- making `_uids` ordered
    would assert a contract the product does not yet offer, and turn four unrelated
    methods red.
    """
    out: list[str] = []
    for row in rows:
        node = row.get("node") or row.get("n") or row
        value = node.get("uid") if hasattr(node, "get") else None
        if value:
            out.append(str(value))
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

    async def test_count_project_data(self, both):
        """The read a destructive run trusts to describe what it is about to remove.

        Compared field by field, not by membership: ADR-0042 makes a preflight that
        under-reports worse than no preflight at all, and the two implementations share
        nothing but this comparison — a Cypher UNWIND + WITH DISTINCT on one side, a
        UNION of two indexed joins on the other.

        Seeds its own corpus: the shared one has no sub-project, no vectors and no
        chunks, so three of the five columns would compare zero against zero and pass
        while proving nothing.
        """
        mg, lite = both
        dim = mg._dimension
        sub = f"{PROJECT}/sub"
        caller = f"{PROJECT}:mod.caller"
        for client in (mg, lite):
            await client.upsert_file_entities(
                sub, "sub.py", [_entity("subfn", f"{sub}:sub.subfn", file_path="sub.py")], []
            )
            await client.write_embeddings_and_hashes(
                [(caller, [0.25] * dim, "h-caller")], labels=["Callable"], model="model-x"
            )
            await client.write_embed_chunks(
                [EmbedChunkWrite(f"{caller}#chunk2", caller, PROJECT, 2, [0.5] * dim, "h-chunk")], model="model-x"
            )

        a = {r["name"]: r for r in await mg.count_project_data(PROJECT)}
        b = {r["name"]: r for r in await lite.count_project_data(PROJECT)}
        assert set(a) == set(b) == {PROJECT, sub}, "the prefix child is part of the blast radius"
        assert a == b, f"memgraph {a} != sqlite {b}"
        # Pinned as well as equal: two backends agreeing on zero would pass vacuously.
        assert a[PROJECT] == {
            "name": PROJECT,
            "nodes": len(_corpus()[0]) + 1,  # the corpus, plus the EmbedChunk
            "relationships": 1,  # DEFINES is the only edge upsert_file_entities materialises
            "embedded_nodes": 1,
            "embed_chunks": 1,
        }
        assert a[sub]["nodes"] == 1

    async def test_count_project_data_names_a_project_with_nothing_in_it(self, both):
        """An empty list would read as "the count failed", and ADR-0042 makes those two
        outcomes mean opposite things — abort versus proceed."""
        mg, lite = both
        zero = {"name": "absent", "nodes": 0, "relationships": 0, "embedded_nodes": 0, "embed_chunks": 0}
        assert await mg.count_project_data("absent") == await lite.count_project_data("absent") == [zero]

    async def test_get_existing_uids(self, both):
        mg, lite = both
        wanted = [f"{PROJECT}:mod.caller", f"{PROJECT}:mod.absent"]
        assert await mg.get_existing_uids(wanted) == await lite.get_existing_uids(wanted)

    async def test_node_exists(self, both):
        mg, lite = both
        for uid in (f"{PROJECT}:mod.caller", f"{PROJECT}:mod.absent"):
            assert await mg.node_exists(uid) == await lite.node_exists(uid)

    async def test_find_entity_uids(self, both):
        """Same answer, and a present name mixed with an absent one in one call.

        A batched lookup can agree on the easy shapes and still differ on the two that
        matter: a name with no match (must be absent from the mapping, not present with
        a null) and several labels in one request (each must be scoped to its own).
        """
        mg, lite = both
        wanted = [("Callable", "caller"), ("Callable", "absent"), ("TypeDef", "caller")]
        a, b = await mg.find_entity_uids(PROJECT, wanted), await lite.find_entity_uids(PROJECT, wanted)
        assert a == b, f"batched lookups diverged: memgraph={a} sqlite={b}"
        assert ("Callable", "caller") in a, "the present name was not found — the comparison is vacuous"
        assert ("Callable", "absent") not in a, "a missing name must be absent from the mapping, not null"
        assert ("TypeDef", "caller") not in a, "the Callable named 'caller' was returned under TypeDef"

    async def test_an_empty_batch_costs_nothing(self, both):
        mg, lite = both
        assert await mg.find_entity_uids(PROJECT, []) == {}
        assert await lite.find_entity_uids(PROJECT, []) == {}

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

    async def test_find_embedded_entities(self, both):
        """The embedding policy's reclaim sweep reads this and then strips what it names
        (ATL-166), so a backend that reported a different set would silently reclaim a
        different set of vectors."""
        mg, lite = both
        dim = mg._dimension
        uid = f"{PROJECT}:mod.caller"
        for client in (mg, lite):
            await client.write_embeddings_and_hashes(
                [(uid, [0.25] * dim, "h-caller")], labels=["Callable"], model="model-x"
            )
        assert sorted(await mg.find_embedded_entities(PROJECT)) == sorted(await lite.find_embedded_entities(PROJECT))
        assert sorted(await mg.find_embedded_entities(PROJECT, kinds=["function"])) == sorted(
            await lite.find_embedded_entities(PROJECT, kinds=["function"])
        )
        assert await mg.find_embedded_entities(PROJECT, kinds=["nothing_has_this_kind"]) == []
        assert await lite.find_embedded_entities(PROJECT, kinds=["nothing_has_this_kind"]) == []

    async def test_clear_embeddings_for_uids(self, both):
        """Both must strip the same vectors and leave the same node behind — the promise
        is that only the vector goes."""
        mg, lite = both
        dim = mg._dimension
        kept, doomed = f"{PROJECT}:mod.caller", f"{PROJECT}:mod.callee"
        for client in (mg, lite):
            await client.write_embeddings_and_hashes(
                [(kept, [0.25] * dim, "h-kept"), (doomed, [0.5] * dim, "h-doomed")],
                labels=["Callable", "Callable"],
                model="model-x",
            )
        assert await mg.clear_embeddings_for_uids([doomed]) == await lite.clear_embeddings_for_uids([doomed]) == 1
        assert await mg.read_embed_hashes([kept, doomed]) == await lite.read_embed_hashes([kept, doomed])
        # The node itself survives on both: excluded from embeddings is not unindexed.
        assert _uid_of(await mg.get_entity_by_uid(doomed)) == _uid_of(await lite.get_entity_by_uid(doomed)) == doomed
        # Idempotent on both, and honest about it — the count is what the sweep logs.
        assert await mg.clear_embeddings_for_uids([doomed]) == await lite.clear_embeddings_for_uids([doomed]) == 0

    async def test_gc_orphaned_embed_chunks(self, both):
        """A chunk has no edge to its parent, so nothing takes it along when the parent
        goes. Both backends must agree on which ones are then dead."""
        mg, lite = both
        dim = mg._dimension
        live = f"{PROJECT}:mod.caller"
        for client in (mg, lite):
            await client.write_embed_chunks(
                [
                    EmbedChunkWrite(f"{live}#chunk2", live, PROJECT, 2, [0.5] * dim, "h-live"),
                    EmbedChunkWrite(
                        f"{PROJECT}:mod.gone#chunk2", f"{PROJECT}:mod.gone", PROJECT, 2, [0.5] * dim, "h-dead"
                    ),
                ],
                model="model-x",
            )
        assert await mg.gc_orphaned_embed_chunks() == await lite.gc_orphaned_embed_chunks() == 1
        # And the live parent's chunk survives, in both.
        assert await mg.gc_orphaned_embed_chunks() == await lite.gc_orphaned_embed_chunks() == 0

    async def test_external_dependency_versions(self, both):
        """The manifest version comes off ``Project -[DEPENDS_ON]-> ExternalPackage`` (v18),
        and the two backends express that join in completely different shapes.

        Memgraph appends an ``OPTIONAL MATCH`` after the importer filter; SQLite adds a
        third ``LEFT JOIN`` to a query that is already grouping over two others. Nothing
        but this comparison holds them together, and the failure mode is quiet: a join
        that returns ``NULL`` for every version still returns every row, so it looks like
        a graph with no pinned dependencies rather than a broken read.

        Seeds its own corpus — the shared one has no Project node and no imports, so
        there is no dependency to report on.
        """
        mg, lite = both
        for client in (mg, lite):
            await client.merge_project_node(PROJECT)
            await client.resolve_imports(
                PROJECT,
                [
                    ParsedRelationship(from_qualified_name=f"{PROJECT}:mod", rel_type=RelType.IMPORTS, to_name=name)
                    for name in ("requests", "loguru")
                ],
            )
            # Only one of the two is declared: an imported-but-unpinned package must
            # still be reported, unversioned, by both.
            await client.update_external_package_versions(PROJECT, {"requests": "2.31.0"})

        a = {
            r["package"]: (r["version"], r["imported_by"])
            for r in (await mg.get_structure_overview(PROJECT, "", 20))["external_deps"]
        }
        b = {
            r["package"]: (r["version"], r["imported_by"])
            for r in (await lite.get_structure_overview(PROJECT, "", 20))["external_deps"]
        }
        assert a == {"requests": ("2.31.0", 1), "loguru": (None, 1)}
        assert a == b, f"only in memgraph: {a.items() - b.items()}; only in sqlite: {b.items() - a.items()}"


class TestTheDefectsThatMotivatedThis:
    """One test per wrong-answer bug ATL-112 found by hand. A suite that compares
    outputs is what should have found them."""

    async def test_a_scoped_embedding_clear_reaches_the_same_projects(self, both):
        """SQLite read ``project_name`` out of ``props_json``, where no writer puts it —
        it is a real column. So the scoped clear matched nothing at all while Memgraph's
        cleared the tree, and ``count_embeddings_by_project`` returned ``{}`` for every
        store. ``--reset-embeddings`` on that backend was a confirmed no-op that still
        recorded the new model, leaving old-space vectors under the new model's name.

        The prefix half is checked with an underscore in the project name on purpose: a
        raw ``LIKE 'code_atlas/%'`` reads ``_`` as a wildcard and reaches wider than
        Cypher's ``STARTS WITH``, which would make the destructive preflight understate.
        """
        family = {"code_atlas": 1, "code_atlas/sub": 1, "codeXatlas/sub": 1, "unrelated": 1}
        results = []
        for client in both:
            for name in family:
                uid = f"{name}:mod.embedded"
                await client.upsert_file_entities(name, "mod.py", [_entity("embedded", uid)], [])
                await client.write_embeddings([(uid, [0.05] * client._dimension)])
            before = {k: v for k, v in (await client.count_embeddings_by_project()).items() if k in family}
            cleared = await client.clear_embeddings("code_atlas")
            after = {k: v for k, v in (await client.count_embeddings_by_project()).items() if k in family}
            results.append((before, cleared, after))

        assert results[0] == results[1], "backends disagree about a destructive clear's scope"
        before, cleared, after = results[0]
        assert before == family, "count_embeddings_by_project did not see every seeded project"
        assert cleared == 2, "the clear must reach the project and its '{name}/' children, and stop there"
        assert set(after) == {"codeXatlas/sub", "unrelated"}

    @staticmethod
    async def _seed_dependency(client: Any, project: str, package: str, version: str | None) -> None:
        """A module in *project* importing *package*, plus the manifest pin if there is one.

        Both halves matter and they come from different writers:
        `resolve_imports` mints the ExternalPackage from the import, and
        `update_external_package_versions` MATCHes it to hang the version edge. Seeding
        only the first gives a null version; seeding only the second writes nothing at all.
        """
        entities = [_entity("mod", f"{project}:mod", label=NodeLabel.MODULE, kind="module")]
        await client.upsert_file_entities(project, "mod.py", entities, [])
        await client.merge_project_node(project)
        await client.resolve_imports(
            project,
            [ParsedRelationship(from_qualified_name=f"{project}:mod", rel_type=RelType.IMPORTS, to_name=package)],
        )
        if version is not None:
            await client.update_external_package_versions(project, {package: version})

    async def test_package_dependents_agree(self, both):
        """ATL-088 P1. The cross-repo dependency read must not be a Memgraph-only answer.

        Compared as an ordered structure rather than through `_uids`: the value here IS
        the grouping -- which projects, at which version, with how many import sites --
        and a set of uids would compare none of it.

        Two projects, one pinned and one not, because the null-version row is the half a
        plain (non-OPTIONAL) join would silently drop, and the two backends reach it by
        completely different routes -- a Cypher OPTIONAL MATCH versus a SQL correlated
        subquery.
        """
        mg, lite = both
        for client in (mg, lite):
            await self._seed_dependency(client, "dep_a", "sharedpkg", "1.2.3")
            await self._seed_dependency(client, "dep_b", "sharedpkg", None)

        a = await mg.get_package_dependents("sharedpkg")
        b = await lite.get_package_dependents("sharedpkg")

        assert a, "the fixture seeded nothing, so this comparison asserts nothing"
        assert a[0]["project_count"] == 2
        assert {p["version"] for p in a[0]["projects"]} == {"1.2.3", None}, (
            "the unpinned project was dropped, so the join is not OPTIONAL"
        )
        assert a == b, "backends disagree about who depends on a package"

    @staticmethod
    async def _seed_sibling_import(client: Any, *, importer: str, owner: str) -> None:
        """*owner* really defines `shared.thing`; *importer* imports it as an external stub.

        Both import shapes, because they take different arms of the resolver: the bare
        `import shared` goes to the ExternalPackage, `from shared import thing` to an
        ExternalSymbol the package CONTAINS.
        """
        await client.upsert_file_entities(
            owner,
            "shared.py",
            [_entity("thing", f"{owner}:shared.thing", label=NodeLabel.CALLABLE, kind="function")],
            [],
        )
        await client.merge_package_node(owner, "shared", "shared", "shared/__init__.py")
        await client.merge_project_node(owner)
        await client.upsert_file_entities(
            importer, "m.py", [_entity("m", f"{importer}:m", label=NodeLabel.MODULE, kind="module")], []
        )
        await client.merge_project_node(importer)
        await client.resolve_imports(
            importer,
            [
                ParsedRelationship(
                    from_qualified_name=f"{importer}:m", rel_type=RelType.IMPORTS, to_name="shared.thing"
                ),
                ParsedRelationship(from_qualified_name=f"{importer}:m", rel_type=RelType.IMPORTS, to_name="shared"),
            ],
        )

    async def test_a_sibling_symbol_import_is_rewired_on_both_backends(self, both):
        """`from shared import thing` must reach the real entity on either backend.

        SQLite derived the symbol's name by splitting its uid on "/", which yields the
        dotted `shared.thing`, while the writer stores `thing`. The lookup matched nothing,
        so the arm was unreachable for every input it can receive and a sibling symbol
        import was silently never rewired on the embedded backend.

        Asserted through `get_package_dependents` because it is the one observable both
        backends share: an un-rewired symbol stub keeps the package alive through its
        CONTAINS edge, so the package's delete stays guarded and the dependency is still
        reported. A clean rewire removes both.
        """
        mg, lite = both
        for client, label in ((mg, "memgraph"), (lite, "sqlite")):
            await self._seed_sibling_import(client, importer="imp_a", owner="own_b")
            assert await client.get_package_dependents("shared"), (
                f"{label}: the fixture minted no stub, so the transition below proves nothing"
            )
            await client.resolve_cross_project_imports(["imp_a", "own_b"])

        # The returned int is deliberately NOT compared: Memgraph counts resolved
        # candidates, SQLite counts importer edges actually rewritten, and they disagree
        # whenever a candidate has no importer. It feeds one logger.debug line and nothing
        # branches on it. What has to agree is the graph.
        assert await mg.get_package_dependents("shared") == await lite.get_package_dependents("shared")
        for client, label in ((mg, "memgraph"), (lite, "sqlite")):
            assert not await client.get_package_dependents("shared"), (
                f"{label}: the stub survived, so a sibling symbol import was left unrewired"
            )

    async def test_a_from_import_counts_as_an_importer(self, both):
        """`from pkg import thing` attaches IMPORTS to the ExternalSymbol, not the package.

        Counting only edges landing on the package reported **0 importers for pathlib in
        a project with 65** -- the node existed solely because something imported out of
        it. Both backends reach the symbol by different routes (a CONTAINS hop in Cypher,
        a self-join plus UNION in SQL), so the agreement is the assertion.
        """
        mg, lite = both
        for client in (mg, lite):
            entities = [_entity("mod", "sym_proj:mod", label=NodeLabel.MODULE, kind="module")]
            await client.upsert_file_entities("sym_proj", "mod.py", entities, [])
            await client.merge_project_node("sym_proj")
            await client.resolve_imports(
                "sym_proj",
                [
                    ParsedRelationship(
                        from_qualified_name="sym_proj:mod", rel_type=RelType.IMPORTS, to_name="frompkg.thing"
                    )
                ],
            )

        a = await mg.get_package_dependents("frompkg")
        b = await lite.get_package_dependents("frompkg")
        assert a, "the from-import never minted an ExternalPackage, so this proves nothing"
        assert a[0]["projects"][0]["import_sites"] == 1, (
            "a from-import was not counted; only bare `import pkg` edges are being seen"
        )
        assert a == b

    async def test_provenance_is_classified_identically(self, both):
        """ATL-191 P1. Declared beats stdlib beats undeclared, and both backends must agree.

        All three tiers come off data already in the graph: `declared` IS the DEPENDS_ON
        edge a manifest writes, `stdlib` is a name lookup, `undeclared` is neither. So the
        seeding below is the whole input -- one package with a manifest pin, one stdlib
        name, one imported and declared by nobody.
        """
        mg, lite = both
        for client in (mg, lite):
            await self._seed_dependency(client, "prov_a", "loguru", "0.7")
            await self._seed_dependency(client, "prov_a", "json", None)
            await self._seed_dependency(client, "prov_a", "somethingvendored", None)

        tallies = [await client.classify_external_package_provenance("prov_a") for client in (mg, lite)]
        assert tallies[0] == tallies[1], f"backends disagree on provenance: {tallies}"
        assert tallies[0] == {"declared": 1, "stdlib": 1, "undeclared": 1}, tallies[0]

    async def test_an_atomic_name_is_not_split_on_its_first_dot(self, both):
        """ATL-191 P2. A registry hostname is not a module path.

        `resolve_imports` derives an ExternalPackage by taking the part before the first
        dot, which is right for `os.path` and wrong for
        `ghcr.io/huggingface/text-embeddings-inference`: that indexed as a package called
        **`ghcr`** with the real name demoted to a sibling ExternalSymbol, so every image
        on a registry collected into one node and the manifest's tag joined against
        nothing.

        A parser that knows its name is opaque sets `IMPORT_ATOMIC_NAME`; nothing about
        the string could tell the resolver. Both halves are asserted, because the
        interesting failure is silent: the whole name appearing *as a symbol under a
        truncated package* still produces a node with the right text in it.
        """
        mg, lite = both
        image = "ghcr.io/huggingface/text-embeddings-inference"
        for client in (mg, lite):
            await client.upsert_file_entities(
                "atomic_proj",
                "Dockerfile",
                [_entity("stage", "atomic_proj:Dockerfile.builder", label=NodeLabel.TYPE_DEF, kind="docker_stage")],
                [],
            )
            await client.merge_project_node("atomic_proj")
            await client.resolve_imports(
                "atomic_proj",
                [
                    ParsedRelationship(
                        from_qualified_name="atomic_proj:Dockerfile.builder",
                        rel_type=RelType.IMPORTS,
                        to_name=image,
                        properties={IMPORT_ATOMIC_NAME: True},
                    ),
                    # The control: the same resolver, one rel apart, must still split a
                    # real dotted module path.
                    ParsedRelationship(
                        from_qualified_name="atomic_proj:Dockerfile.builder",
                        rel_type=RelType.IMPORTS,
                        to_name="os.path",
                    ),
                ],
            )
            await client.update_external_package_versions("atomic_proj", {image: "cpu-1.8"})

        for client, label in ((mg, "memgraph"), (lite, "sqlite")):
            packages = {r["package"] for r in await client.get_package_dependents()}
            assert image in packages, f"{label}: the image name was truncated at its first dot"
            assert "ghcr" not in packages, f"{label}: a registry hostname became a package"
            assert "os" in packages, f"{label}: the marker leaked onto an ordinary dotted import"

        a = await mg.get_package_dependents(image)
        b = await lite.get_package_dependents(image)
        assert a[0]["projects"][0]["version"] == "cpu-1.8", (
            "the manifest key no longer equals the minted node name, so the tag joined against nothing"
        )
        assert a == b, "backends disagree about an atomic external name"

    async def test_stub_symbols_are_written_identically(self, both):
        """ATL-191 P4. A stub is a signature on the ExternalSymbol the resolver already
        mints, not a new node type -- so the write has to survive meeting one that exists.

        Both halves are seeded: `pkgstub.known` is already in the graph because something
        imported it, and `pkgstub.unknown` is an entrypoint nobody has called yet. The
        first is the interesting one, because the two backends reach it by different
        routes -- a Cypher MERGE whose ON CREATE half must not fire again, and a SQLite
        upsert into a row that already exists.
        """
        mg, lite = both
        symbols = [
            {
                "uid": "stub_proj:ext/pkgstub.known",
                "qualified_name": "ext/pkgstub.known",
                "name": "known",
                "kind": "function",
                "signature": "def known(a: int) -> str",
                "docstring": "Already imported.",
            },
            {
                "uid": "stub_proj:ext/pkgstub.unknown",
                "qualified_name": "ext/pkgstub.unknown",
                "name": "unknown",
                "kind": "class",
                "signature": "",
                "docstring": "Never imported, but part of the public API.",
            },
        ]
        for client in (mg, lite):
            await self._seed_dependency(client, "stub_proj", "pkgstub", "1.0")
            await client.resolve_imports(
                "stub_proj",
                [
                    ParsedRelationship(
                        from_qualified_name="stub_proj:mod", rel_type=RelType.IMPORTS, to_name="pkgstub.known"
                    )
                ],
            )
            written = await client.upsert_external_stubs(
                "stub_proj", "pkgstub", symbols, version="1.0", source="bundled-pyi"
            )
            assert written == 2

        for client, label in ((mg, "memgraph"), (lite, "sqlite")):
            known = await client.get_entity_by_uid("stub_proj:ext/pkgstub.known")
            assert known is not None, f"{label}: the stub write lost a node the resolver had minted"
            assert known["signature"] == "def known(a: int) -> str", f"{label}: no signature"
            assert known["package"] == "pkgstub", f"{label}: the upsert lost the package link"
            versions = await client.get_stubbed_package_versions("stub_proj")
            assert versions == {"pkgstub": "1.0"}, f"{label}: {versions}"

        a = await mg.get_entity_by_uid("stub_proj:ext/pkgstub.unknown")
        b = await lite.get_entity_by_uid("stub_proj:ext/pkgstub.unknown")
        assert a is not None, "memgraph: an entrypoint nobody imported was not created"
        assert b is not None, "sqlite: an entrypoint nobody imported was not created"
        assert a["docstring"] == b["docstring"]
        assert a["kind"] == b["kind"] == "class"

    async def test_a_re_read_replaces_a_stale_signature(self, both):
        """The invalidation key is the package's `stub_version`, because the source is in
        site-packages and no `file_hash` gate covers it. A signature parsed from numpy 2.4
        must not survive an upgrade to 2.5 -- SET, never accumulate."""
        mg, lite = both
        old = [
            {
                "uid": "stub_v:ext/movingpkg.api",
                "qualified_name": "ext/movingpkg.api",
                "name": "api",
                "kind": "function",
                "signature": "def api() -> None",
                "docstring": "v1",
            }
        ]
        new = [{**old[0], "signature": "def api(added: bool) -> None", "docstring": "v2"}]
        for client in (mg, lite):
            await self._seed_dependency(client, "stub_v", "movingpkg", "2.4")
            await client.upsert_external_stubs("stub_v", "movingpkg", old, version="2.4", source="source")
            await client.upsert_external_stubs("stub_v", "movingpkg", new, version="2.5", source="source")

        for client, label in ((mg, "movingpkg/memgraph"), (lite, "movingpkg/sqlite")):
            node = await client.get_entity_by_uid("stub_v:ext/movingpkg.api")
            assert node is not None
            assert node["signature"] == "def api(added: bool) -> None", f"{label}: stale signature survived"
            assert await client.get_stubbed_package_versions("stub_v") == {"movingpkg": "2.5"}, label

    async def test_stubbing_a_package_with_no_symbols_writes_nothing(self, both):
        """`urllib`'s entrypoint is empty and several ecosystems' names resolve to nothing
        at all. An empty write must not stamp a `stub_version`, or the package would be
        recorded as read and never retried."""
        mg, lite = both
        for client in (mg, lite):
            await self._seed_dependency(client, "stub_empty", "emptypkg", None)
            assert await client.upsert_external_stubs("stub_empty", "emptypkg", [], version="1.0") == 0
            assert await client.get_stubbed_package_versions("stub_empty") == {}

    async def test_classifying_twice_changes_nothing(self, both):
        """It runs at the end of every index, so a second pass must not drift.

        SET rather than merge is what makes that true, and it is also what lets a package
        whose manifest entry disappeared stop reading as `declared` on the next index.
        That second half is not asserted here: removing a DEPENDS_ON edge has no portable
        API across the two backends, and a test that reached for Cypher would silently
        pass by doing nothing on SQLite.
        """
        mg, lite = both
        for client in (mg, lite):
            await self._seed_dependency(client, "prov_b", "loguru", "0.7")
            await self._seed_dependency(client, "prov_b", "json", None)
            first = await client.classify_external_package_provenance("prov_b")
            assert await client.classify_external_package_provenance("prov_b") == first
            assert first == {"declared": 1, "stdlib": 1}, first

    async def test_stdlib_is_flagged_and_filtered_identically(self, both):
        """`ext/json` and `ext/litellm` are the same kind of node, and on a real corpus the
        first kind dominates -- 59 of 90 shared names here.

        The filter lives in the backends rather than in the caller because `limit` has to
        count rows somebody asked for: filtering afterwards returned ONE package for
        `--no-stdlib --limit 6`, since six of the seven fetched were stdlib and the cut had
        already happened.
        """
        mg, lite = both
        for client in (mg, lite):
            await self._seed_dependency(client, "std_a", "json", None)
            await self._seed_dependency(client, "std_a", "loguru", "0.7")

        for client, label in ((mg, "memgraph"), (lite, "sqlite")):
            flagged = {r["package"]: r["stdlib"] for r in await client.get_package_dependents()}
            assert flagged.get("json") is True, f"{label}: a stdlib name was not flagged"
            assert flagged.get("loguru") is False, f"{label}: a real dependency was flagged as stdlib"

            kept = {r["package"] for r in await client.get_package_dependents(exclude_stdlib=True)}
            assert "loguru" in kept
            assert "json" not in kept, f"{label}: exclude_stdlib did not drop it"

        assert await mg.get_package_dependents(exclude_stdlib=True) == await lite.get_package_dependents(
            exclude_stdlib=True
        )

    async def test_the_limit_counts_rows_after_the_stdlib_filter(self, both):
        """Regression: the filter must be applied before the cut, not after it."""
        mg, lite = both
        for client in (mg, lite):
            for stdlib_name in ("json", "pathlib", "time", "collections"):
                await self._seed_dependency(client, "std_b", stdlib_name, None)
            await self._seed_dependency(client, "std_b", "loguru", None)
            await self._seed_dependency(client, "std_b", "typer", None)

            rows = await client.get_package_dependents(exclude_stdlib=True, limit=2)
            assert len(rows) == 2, "the limit was spent on rows the filter then removed"
            assert not any(r["stdlib"] for r in rows)

    async def test_package_dependents_min_projects_filters_identically(self, both):
        """The filter is what makes "shared across repos" answerable, so it is compared
        against a package that really is shared rather than against two empty lists."""
        mg, lite = both
        for client in (mg, lite):
            await self._seed_dependency(client, "dep_a", "sharedpkg", "1.2.3")
            await self._seed_dependency(client, "dep_b", "sharedpkg", None)
            await self._seed_dependency(client, "dep_a", "lonelypkg", None)

        for client in (mg, lite):
            shared = {r["package"] for r in await client.get_package_dependents(min_projects=2)}
            assert "sharedpkg" in shared
            assert "lonelypkg" not in shared, "a single-project package passed a min_projects=2 filter"

        assert await mg.get_package_dependents(min_projects=2) == await lite.get_package_dependents(min_projects=2)
        assert await mg.get_package_dependents(min_projects=99) == []
        assert await lite.get_package_dependents(min_projects=99) == []

    async def test_like_metacharacters_are_not_wildcards(self, both):
        """`_` matched any character, so searching `weird_name` also returned
        `weirdXname`. Both backends must now agree that it does not."""
        mg, lite = both
        a = _uids(await mg.graph_search("weird_name", project=PROJECT, limit=20))
        b = _uids(await lite.graph_search("weird_name", project=PROJECT, limit=20))
        assert a == b
        assert f"{PROJECT}:mod.weirdXname" not in b

    async def test_every_limited_query_cuts_the_same_rows(self, limited_both):
        """ATL-192. ATL-186 ordered `graph_search`; these four kept the same defect.

        An unordered LIMIT returns whichever rows the storage engine reaches first. With
        the two backends seeded in opposite orders that is a different set on each, so
        this comparison is what the defect looks like from outside -- and the assertion
        on length is what keeps it from passing vacuously if the corpus ever shrinks.
        """
        mg, lite = limited_both
        target = f"{LIMITED_PROJECT}:mod.edge_target"
        limit = 5

        cases = {
            "get_callers": (lambda c: c.get_callers(target, "", 1, limit), limit),
            "get_callees": (lambda c: c.get_callees(target, "", 1, limit), limit),
            "get_node_exact_matches": (lambda c: c.get_node_exact_matches("dupe", "", limit), limit),
            "get_node_partial_matches": (lambda c: c.get_node_partial_matches("edge", "", limit), limit),
        }
        for name, (call, expected) in cases.items():
            a = _ranked_uids(await call(mg))
            b = _ranked_uids(await call(lite))
            assert len(a) >= expected, f"{name}: the LIMIT never bound, so this compared nothing ({len(a)})"
            assert a == b, f"{name}: backends cut different rows\n  memgraph={a}\n  sqlite  ={b}"

    async def test_a_limited_query_is_stable_across_repeated_calls(self, limited_both):
        """Storage order is reproducible within one process, which is how this stayed
        invisible -- stability alone proves nothing. It is asserted anyway because a
        *non*-deterministic order would make the comparison above flaky rather than red,
        and those two failures want telling apart."""
        target = f"{LIMITED_PROJECT}:mod.edge_target"
        for client in self._both(limited_both):
            first = _ranked_uids(await client.get_callers(target, "", 1, 5))
            for _ in range(3):
                assert _ranked_uids(await client.get_callers(target, "", 1, 5)) == first

    async def test_graph_search_returns_the_same_order_from_both_backends(self, ordered_both):
        """ATL-186: every stage truncates at `limit * 3`, so *which* rows survive is part
        of the answer -- and the caller's stable sort carries that order into
        `rrf_fuse`, which reads list position as rank.

        `limit=5` against a corpus of 24 `parse`-matching entities means stage 3 fetches
        15 of them: the LIMIT binds, which is exactly what the pre-existing comparison
        could not do at limit=20 over a 9-entity corpus.
        """
        mg, lite = ordered_both
        for limit in (5, 8):
            a = _ranked_uids(await mg.graph_search("parse", project=ORDER_PROJECT, limit=limit))
            b = _ranked_uids(await lite.graph_search("parse", project=ORDER_PROJECT, limit=limit))
            assert a == b, f"backends disagree about graph_search order at limit={limit}"
            assert len(a) == limit, "the LIMIT must actually bind, or this compares nothing"

    async def test_a_name_match_outranks_a_path_only_match(self, ordered_both):
        """The ordering key's first tier, and the reason it is not new policy: the
        cascade already prices an exact name at 3.0 above a suffix at 2.0. Inside stage
        3's flat 1.0 bucket that rule simply did not apply, so a node matching only on
        its module path could outrank one matching on its own name."""
        for client in self._both(ordered_both):
            rows = await client.graph_search("parse", project=ORDER_PROJECT, limit=12)
            names = [(r["node"]["uid"].rsplit(".", 1)[-1]) for r in rows]
            name_hits = [i for i, n in enumerate(names) if "parse" in n]
            path_only = [i for i, n in enumerate(names) if "parse" not in n]
            assert name_hits, "no name-matching entity survived the fetch window"
            if path_only:
                assert max(name_hits) < min(path_only), f"a path-only match outranked a name match: {names}"

    async def test_the_order_is_stable_across_repeated_calls(self, ordered_both):
        """Storage order was reproducible run-to-run too, which is how this stayed
        invisible. The claim worth pinning is agreement *between* backends plus
        stability, not stability alone."""
        for client in self._both(ordered_both):
            first = _ranked_uids(await client.graph_search("parse", project=ORDER_PROJECT, limit=8))
            for _ in range(3):
                assert _ranked_uids(await client.graph_search("parse", project=ORDER_PROJECT, limit=8)) == first

    @staticmethod
    def _both(pair: Any) -> list[Any]:
        return [pair[0], pair[1]]

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


CYCLE_PROJECT = "cycleproj"


def _cycle_corpus() -> tuple[list[ParsedEntity], list[ParsedRelationship]]:
    """Recursion and mutual recursion: the shapes where "is X its own caller?" and "trace
    X to X" have an answer. `ping`/`pong` call each other, `a -> b -> c -> a` is a
    three-cycle with an exit to `d`, and `lone` has no edges at all. `rec` calls itself, but
    `resolve_calls` never resolves a call to its own caller, so no self-loop exists on
    either backend -- it is kept to pin that both agree it is not its own caller."""
    names = ("rec", "ping", "pong", "a", "b", "c", "d", "lone")
    entities = [
        _entity("mod", f"{CYCLE_PROJECT}:mod", label=NodeLabel.MODULE, kind="module"),
        *(_entity(n, f"{CYCLE_PROJECT}:mod.{n}") for n in names),
    ]
    calls = [("rec", "rec", 4), ("ping", "pong", 5), ("pong", "ping", 6), ("a", "b", 7), ("b", "c", 8)]
    calls += [("c", "a", 9), ("a", "d", 10)]
    rels = [
        ParsedRelationship(
            from_qualified_name=f"{CYCLE_PROJECT}:mod.{src}",
            rel_type=RelType.CALLS,
            to_name=dst,
            properties={"line": line},
        )
        for src, dst, line in calls
    ]
    return entities, rels


@pytest.fixture
async def cycle_both(graph_client, tmp_path: Path):
    """Both backends holding `_cycle_corpus`, CALLS resolved."""
    sqlite = SqliteGraphClient(tmp_path / "cycle.sqlite3")
    entities, rels = _cycle_corpus()
    for client in (graph_client, sqlite):
        await client.ensure_schema()
        await client.upsert_file_entities(CYCLE_PROJECT, "mod.py", entities, rels)
        await client.resolve_calls(CYCLE_PROJECT, rels)
    try:
        yield graph_client, sqlite
    finally:
        await sqlite.close()


class TestCyclesAgree:
    """A recursive function IS its own caller and callee when the cycle fits the depth, and
    a trace from a node to itself returns its shortest cycle. SQLite used to say neither."""

    @staticmethod
    def _uid(name: str) -> str:
        return f"{CYCLE_PROJECT}:mod.{name}"

    async def test_trace_path_between(self, cycle_both):
        mg, lite = cycle_both
        found = 0
        pairs = [("rec", "rec"), ("ping", "ping"), ("pong", "ping"), ("a", "a"), ("a", "c"), ("c", "d")]
        pairs += [("d", "a"), ("lone", "lone"), ("a", "missing")]
        for src, dst in pairs:
            for depth in (1, 2, 3, 4):
                a = await mg.trace_path_between(self._uid(src), self._uid(dst), depth, ("CALLS",))
                b = await lite.trace_path_between(self._uid(src), self._uid(dst), depth, ("CALLS",))
                assert a == b, f"trace {src} -> {dst} depth={depth}\n  memgraph={a}\n  sqlite  ={b}"
                found += bool(a["found"])
        assert found >= 12, f"only {found} traces found a path, so the comparison says little"
        ping = await lite.trace_path_between(self._uid("ping"), self._uid("ping"), 2, ("CALLS",))
        assert ping["found"], "mutual recursion is a two-hop cycle"
        assert [h["at_line"] for h in ping["hops"]] == [5, 6]
        three = await lite.trace_path_between(self._uid("a"), self._uid("a"), 3, ("CALLS",))
        assert three["hop_count"] == 3
        assert not (await lite.trace_path_between(self._uid("a"), self._uid("a"), 2, ("CALLS",)))["found"]

    async def test_callers_and_callees_include_the_recursive_function(self, cycle_both):
        mg, lite = cycle_both
        for name in ("rec", "ping", "a", "d", "lone"):
            for depth in (1, 2, 3):
                for method in ("get_callers", "get_callees"):
                    a = _ranked_uids(await getattr(mg, method)(self._uid(name), "", depth, 20))
                    b = _ranked_uids(await getattr(lite, method)(self._uid(name), "", depth, 20))
                    assert a == b, f"{method}({name}, depth={depth})\n  memgraph={a}\n  sqlite  ={b}"
        assert self._uid("a") in _ranked_uids(await lite.get_callers(self._uid("a"), "", 3, 20))
        assert self._uid("ping") not in _ranked_uids(await lite.get_callees(self._uid("ping"), "", 1, 20))
        assert self._uid("ping") in _ranked_uids(await lite.get_callees(self._uid("ping"), "", 2, 20))

    async def test_blast_radius_never_reports_the_entity_itself(self, cycle_both):
        mg, lite = cycle_both
        for name in ("rec", "ping", "a"):
            for direction in ("out", "in"):
                a = _uids(await mg.compute_blast_radius(self._uid(name), direction, ("CALLS",), 3))
                b = _uids(await lite.compute_blast_radius(self._uid(name), direction, ("CALLS",), 3))
                assert a == b, f"{name} {direction}: only in memgraph: {a - b}; only in sqlite: {b - a}"
                assert self._uid(name) not in b


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


# ---------------------------------------------------------------------------
# Observability parity (ATL-158)
# ---------------------------------------------------------------------------


def _graph_query_ops(reader: Any) -> set[tuple[str, str]]:
    """(op, kind) pairs seen on ``atlas_graph_query_seconds``."""
    seen: set[tuple[str, str]] = set()
    for rm in reader.get_metrics_data().resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name != "atlas_graph_query_seconds":
                    continue
                for point in metric.data.data_points:
                    seen.add((point.attributes.get("op"), point.attributes.get("kind")))
    return seen


class TestObservabilityParity:
    """Both backends must report on the same instrument, not just return the same rows.

    This is the check whose absence let the SQLite backend run for six weeks with no
    telemetry at all. The suite above compares what the two backends *return*; nothing
    compared what they *report*, so a backend emitting nothing was indistinguishable
    from one emitting everything. `atlas_graph_query_seconds` is the shared instrument,
    and its sample count per `op` is the round-trip count a benchmark reads.
    """

    async def test_both_backends_record_on_the_same_instrument(self, both, monkeypatch):
        pytest.importorskip("opentelemetry.sdk")
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader

        import code_atlas.telemetry as tel

        memgraph, sqlite = both
        results: dict[str, set[tuple[str, str]]] = {}

        for label, client in (("memgraph", memgraph), ("sqlite", sqlite)):
            reader = InMemoryMetricReader()
            # A local provider rather than the global one: `set_meter_provider` is
            # once-per-process and a second call is ignored, which would silently give
            # the second backend an empty reader and read as "sqlite emits nothing".
            meter = MeterProvider(metric_readers=[reader]).get_meter("code_atlas")
            with monkeypatch.context() as m:
                m.setattr(
                    tel,
                    "_metrics",
                    tel._Metrics(graph_query_seconds=meter.create_histogram("atlas_graph_query_seconds", unit="s")),
                )
                m.setattr(tel, "_enabled", True)
                m.setattr(tel, "_initialized", True)
                await client.count_entities(PROJECT)
            results[label] = _graph_query_ops(reader)

        for label, seen in results.items():
            assert seen, f"{label} recorded nothing on atlas_graph_query_seconds"
            assert any(op == "count_entities" for op, _ in seen), (
                f"{label} did not attribute the query to its calling method: {sorted(seen)}"
            )
            assert all(kind in {"read", "write", "read_tx", "write_tx"} for _, kind in seen), (
                f"{label} used a kind outside the shared vocabulary: {sorted(seen)}"
            )
