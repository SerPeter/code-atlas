"""SQLite-backed graph engine — in-process fallback for the Memgraph ``GraphClient``.

Implements the :class:`~code_atlas.graph.protocol.GraphBackend` structural
contract on top of a single ``aiosqlite`` connection (WAL mode), the
``sqlite-vec`` loadable extension for vector search, and SQLite's built-in
FTS5 extension for BM25 text search.

Unlike Memgraph's per-label node storage, all entities live in one unified
``nodes`` table (``labels`` column holds the single ``NodeLabel`` value) and
all relationships in one ``edges`` table — the small, fixed set of properties
every entity shares (``uid``, ``labels``, ``project_name``, ``qualified_name``,
``file_path``, ``name``, ``kind``, ``content_hash``) are real columns; every
other property (``line_start``, ``docstring``, ``signature``, the ``frontmatter``
map, etc.) lives in a ``props_json`` JSON1 column, merged via ``json_patch`` on
update to mirror Cypher's ``SET n += ...`` semantics -- except ``frontmatter``,
which is replaced rather than merged (see ``_batch_update_entities``).

Pure-Python matching logic with no Memgraph coupling
(``_resolve_one_call``, ``_resolve_one_path_anchor``, ``_classify_file``,
``GraphClient._classify_batch``) is reused directly from
:mod:`code_atlas.graph.client` rather than reimplemented — only the
*lookup-building* queries are ported from Cypher to SQL.

Known simplifications vs. ``GraphClient`` (spike scope, see ADR-0015 / the
embedded-backend plan's explicit "best-effort resolution" allowance):
  - ``resolve_imports`` skips the Python dotted-prefix fallback match.
  - Entity deletion is a straight delete — no zombie-node preservation for
    cross-file inbound edges (``GraphClient._batch_delete_entities``'s most
    intricate behavior).
  - Relationship recreation on file re-upsert doesn't special-case cross-file
    DEFINES edges (Go/C++ out-of-line members) — those get dropped until the
    *member's* own file is next reprocessed.
  - ``execute``/``execute_write`` (raw Cypher passthrough) always raise —
    there is no SQL translation layer. Every domain query is now behind a
    named ``GraphBackend`` method instead (see ``graph/protocol.py``); the
    sole remaining caller of raw ``execute()`` is ``cypher_query``/
    ``validate_cypher`` (arbitrary agent-authored Cypher has no meaningful SQL
    translation), and the MCP tool layer guards it explicitly — see
    ``server/mcp.py``'s ``SqliteGraphClient`` isinstance checks.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import struct
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Self

import aiosqlite
import sqlite_vec
from loguru import logger

from code_atlas.backends.instrumentation import TimedConnection
from code_atlas.backends.instrumentation import in_transaction as _txn

if TYPE_CHECKING:
    from code_atlas.backends.instrumentation import SqlConnection
from code_atlas.graph.client import (
    _CODE_ENTITY_KINDS,
    _DEFAULT_EDGE_WEIGHT,
    _DEFAULT_TEST_PATTERNS,
    _FILE_LOCAL_STRATEGIES,
    _POST_BATCH_REL_TYPES,
    _TYPE_REF_FACTS,
    _TYPE_REF_RANK,
    _UNVERIFIED_STRATEGIES,
    SCHEMA_VERSION,
    CallStats,
    EmbedChunkWrite,
    EntityHashData,
    ReplayableRels,
    UpsertResult,
    _AnchorLookup,
    _BatchClassification,
    _call_edge_weight,
    _CallEdgeFacts,
    _CallLookup,
    _checked_edge_weight,
    _citation_key,
    _CitationLookup,
    _combine_call_edge_facts,
    _document_citation_keys,
    _fuse_bm25_results,
    _pick_citation_target,
    _plan_config_refs,
    _project_data_rows,
    _rel_write_skips,
    _render_citation_key,
    _resolve_one_call,
    _resolve_one_path_anchor,
    _test_callable_uids,
    build_case_folded_map,
    resolve_component_alias,
)
from code_atlas.graph.client import (
    _classify_file as _classify_file_delta,
)
from code_atlas.schema import (
    _EMBEDDABLE_LABELS,
    _GRANT_ONLY_KINDS,
    _REFERENCE_COUNTED_LABELS,
    _TEXT_SEARCHABLE_LABELS,
    COMPOSITE_INDICES,
    FILE_HASH_LABELS,
    GLOBAL_PROJECT,
    IMPORT_ATOMIC_NAME,
    LABEL_PROPERTY_INDICES,
    PROVENANCE_DECLARED,
    PROVENANCE_STDLIB,
    PROVENANCE_UNDECLARED,
    STDLIB_MODULE_NAMES,
    TEXT_INDICES,
    NodeLabel,
    build_vector_index_specs,
)
from code_atlas.search.engine import matches_test_pattern

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Collection, Sequence
    from pathlib import Path

    from code_atlas.parsing.ast import ParsedEntity, ParsedRelationship
    from code_atlas.parsing.detectors import PropertyEnrichment

_VEC_LABEL_VALUES: frozenset[str] = frozenset(lbl.value for lbl in _EMBEDDABLE_LABELS)
_TEXT_LABEL_VALUES: frozenset[str] = frozenset(lbl.value for lbl in _TEXT_SEARCHABLE_LABELS)
_POST_BATCH_REL_VALUES: frozenset[str] = frozenset(r.value for r in _POST_BATCH_REL_TYPES)
# Rendered once, from the same tuple the Memgraph gate iterates, so the two backends
# cannot disagree about which nodes carry a file_hash.
_FILE_HASH_LABELS_SQL = ", ".join(f"'{lbl.value}'" for lbl in FILE_HASH_LABELS)

_NODE_COLUMNS = "uid, labels, project_name, qualified_name, file_path, name, kind, content_hash, props_json"

# A citation DOCUMENTS edge leaves the CITED document's node but is created by
# the citing file's parse (resolve_citations), so re-parsing the document must
# not delete it — nothing in that parse could put it back. Mirrors the same
# carve-out in ``GraphClient._recreate_{file,batch}_relationships``. The same
# predicate drives resolve_citations' own file-scoped revoke pass, which is
# where these edges DO get deleted (from the citing side, where they are
# inbound and the two _recreate_* sweeps below cannot see them).
_CITATION_EDGE_PREDICATE = "rel_type = 'DOCUMENTS' AND json_extract(props_json, '$.link_type') = 'citation'"

# The second carve-out, and the twin of Memgraph's
# `NOT (type(r) = 'DEFINES' AND coalesce(m.file_path, n.file_path) <> n.file_path)`.
#
# `resolve_member_defines` creates DEFINES edges that run from a TypeDef to a member
# declared in a DIFFERENT file -- a C++ method in `foo.cpp` whose class is in `foo.h`,
# a Go method whose receiver struct is declared elsewhere in the package, a Rust `impl`
# split from its `struct`, an SFDX `objects/X/fields/Y.field-meta.xml` under its
# `objects/X/X.object-meta.xml`. The edge is SOURCED at the owning file's node but
# STATED by the member's file, so re-parsing the owner alone would delete it and only a
# re-parse of the *member* file could restore it -- which the content-hash gate exists
# to prevent. It would stay gone.
#
# Both `nodes.uid` lookups are against the PRIMARY KEY, so this is two point lookups per
# candidate DEFINES edge rather than a scan (ADR-0045).
#
# Comparing against the SOURCE node's file_path rather than a bound parameter makes one
# predicate correct for both the single-file and the batch sweep: the single-file sweep
# already restricts `from_uid` to nodes in that file, so the two are equivalent there.
#
# A dangling `to_uid` (no node row yet) is NOT carved out, matching the behaviour before
# this predicate existed and matching Memgraph, where a pattern with no target simply
# does not match. A target whose `file_path` is NULL is likewise deleted, which is what
# Memgraph's `coalesce` collapses to.
_CROSS_FILE_DEFINES_PREDICATE = (
    "rel_type = 'DEFINES' AND EXISTS ("
    "SELECT 1 FROM nodes AS target, nodes AS source "
    "WHERE target.uid = edges.to_uid AND source.uid = edges.from_uid "
    "AND target.file_path IS NOT NULL AND target.file_path <> source.file_path)"
)


def _fts_document(name: str, qualified_name: str, props: dict[str, Any]) -> str:
    """The BM25 document for one node.

    Memgraph's text index is label-wide (``CREATE TEXT INDEX ... ON :Label``), so a new
    property becomes BM25-visible there for free; the SQLite FTS document is an explicit
    field list and has to name them.

    One function, two callers -- the write path (from a ParsedEntity) and the rowid
    migration (from a stored row). They must produce the same bytes, or a rebuilt index
    silently ranks differently from a freshly written one.
    """
    parts = [
        name,
        qualified_name,
        props.get("docstring"),
        props.get("signature"),
        props.get("source"),
        " ".join(props.get("tags") or []),
        props.get("rationale"),
        " ".join(props.get("citations") or []),
    ]
    return " ".join(p for p in parts if p)


def _fts_document_from_entity(e: ParsedEntity) -> str:
    uid = e.qualified_name
    qn = uid.split(":", 1)[1] if ":" in uid else uid
    return _fts_document(
        e.name,
        qn,
        {
            "docstring": e.docstring,
            "signature": e.signature,
            "source": e.source,
            "tags": e.tags,
            "rationale": e.rationale,
            "citations": e.citations,
        },
    )


def _node_columns(alias: str) -> str:
    """``_NODE_COLUMNS`` prefixed with a table alias, for JOIN queries — e.g. ``_node_columns("p")``."""
    return ", ".join(f"{alias}.{col.strip()}" for col in _NODE_COLUMNS.split(","))


_BASE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS nodes (
    uid TEXT PRIMARY KEY,
    labels TEXT NOT NULL,
    project_name TEXT NOT NULL,
    qualified_name TEXT,
    file_path TEXT,
    name TEXT,
    kind TEXT,
    content_hash TEXT,
    props_json TEXT NOT NULL DEFAULT '{}',
    embedding BLOB
);

CREATE TABLE IF NOT EXISTS edges (
    from_uid TEXT NOT NULL,
    to_uid TEXT NOT NULL,
    rel_type TEXT NOT NULL,
    props_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (from_uid, to_uid, rel_type)
);
CREATE INDEX IF NOT EXISTS ix_edges_to ON edges(to_uid, rel_type);
CREATE INDEX IF NOT EXISTS ix_edges_from_type ON edges(from_uid, rel_type);

-- The two indices every label shares. `_node_index_ddl` builds ~124 partial indices
-- from schema.py's registries, and each is `WHERE labels = '<Label>'` -- which SQLite
-- can only use when the predicate names that same label as a **literal**. Three of the
-- four hottest shapes never do:
--
--   labels = ?              a bound parameter; unknown when the plan is built
--   labels IN (...)         does not imply any single-label predicate
--   labels NOT IN (...)     implies the opposite of one
--
-- So the file-hash gate, the per-file diff, the resolution lookups and the worktree
-- sweep all fell to `SCAN nodes`, on a table that grows with the whole graph. Measured
-- with EXPLAIN QUERY PLAN, seven distinct hot statements scanned; with these two, none
-- do.
--
-- Not fixed by adding more partial indices, and not fixable in schema.py either: those
-- registries also drive Memgraph, which has no label-free index and genuinely needs
-- per-label ones. The unqualified pair is a SQLite-side answer to a SQLite-side planner
-- rule.
CREATE INDEX IF NOT EXISTS ix_nodes_project_file ON nodes(project_name, file_path);
CREATE INDEX IF NOT EXISTS ix_nodes_labels_project_name ON nodes(labels, project_name, name);

-- The dedup lookup (ADR-0036) asks whether ANY node -- any project, any label -- already
-- holds a vector for this text under this model, so no per-label index can serve it by
-- construction. It ran once per embed batch and walked the whole table each time; on the
-- graph sizes this backend is meant to reach, that is the expensive kind of once.
-- Partial on `embedding IS NOT NULL` because the query says exactly that as a literal,
-- which SQLite can prove -- and because it keeps the index to the rows that have vectors
-- rather than to every node in the graph.
CREATE INDEX IF NOT EXISTS ix_nodes_embed_hash ON nodes(
    json_extract(props_json, '$.embed_hash'),
    json_extract(props_json, '$.embed_model')
) WHERE embedding IS NOT NULL;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

_CHUNK_SIZE = 500

_FTS5_SAFE_CHARS = frozenset(" _.-")


def _chunks(items: list[Any], size: int = _CHUNK_SIZE) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)] or [[]]


def _sanitize_fts_query(query: str) -> str:
    """Quote every term so FTS5's query language (AND/OR/NOT, prefix `-`, column
    filters, `NEAR`) is never interpreted — mirrors ``_sanitize_bm25_query``'s
    "neutralize the query language, keep it a plain keyword search" intent.
    """
    cleaned = "".join(ch if ch.isalnum() or ch in _FTS5_SAFE_CHARS else " " for ch in query)
    terms = [t.replace('"', "") for t in cleaned.split() if t]
    if not terms:
        return '""'
    return " ".join(f'"{t}"' for t in terms)


def _props_weight(props: dict[str, Any]) -> float:
    """Numeric ``weight`` out of a decoded ``props_json`` dict.

    Mirrors how Memgraph/MAGE reads the property: a missing or non-numeric
    value is the neutral ``_DEFAULT_EDGE_WEIGHT``, never an error. ``bool`` is
    excluded explicitly since it is an ``int`` subclass in Python.
    """
    value = props.get("weight")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return _DEFAULT_EDGE_WEIGHT
    return _checked_edge_weight(float(value))


def _prefix_clause(column: str, path: str) -> tuple[str, list[Any]]:
    """SQL equivalent of Cypher's ``<column> STARTS WITH $path``.

    Compares a fixed-length ``substr`` instead of ``LIKE`` to avoid escaping
    ``%``/``_`` wildcard characters that can legitimately appear in file paths.
    Returns ``("", [])`` when *path* is empty (matches everything, same as
    the ``if path`` guard already used throughout analysis.py's Cypher).
    """
    if not path:
        return "", []
    return f" AND substr({column}, 1, ?) = ?", [len(path), path]


# LIKE's own wildcards. Backslash first, or escaping it again would double the ones the
# other two rules just inserted.
_LIKE_ESCAPES = (("\\", "\\\\"), ("%", "\\%"), ("_", "\\_"))


_ASCII_FOLD = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


def _ascii_fold(value: str) -> str:
    """Lower-case exactly the characters SQLite's ``LIKE`` folds, and no others.

    ``LIKE`` is case-insensitive by default, but only over ASCII -- ``'ÄÖ.md' LIKE
    '%äö.md'`` is 0 while ``'README.md' LIKE '%readme.md'`` is 1, and this file never sets
    ``PRAGMA case_sensitive_like``. So a Python suffix match has to fold the same subset:
    ``str.endswith`` alone is too strict and ``str.lower()`` is too permissive, and both
    are wrong in a way that shows up as edges rather than errors.

    Too strict is the dangerous direction. A doc reference to ``readme.md`` in a project
    holding both ``docs/readme.md`` and ``README.md`` has two candidates and is correctly
    refused; matching case-sensitively leaves one candidate and writes a confident
    DOCUMENTS edge to an arbitrary one of two equally plausible targets.
    """
    return value.translate(_ASCII_FOLD)


_WAREHOUSE_PREFIX = "ext/warehouse."
"""Mirror of ``GraphClient._WAREHOUSE_PREFIX``; see that constant for why it is ``ext/``."""

_WAREHOUSE_PRODUCER_KINDS: frozenset[str] = frozenset({"dbt_model", "dbt_snapshot"})
"""Mirror of ``GraphClient._WAREHOUSE_PRODUCER_KINDS``."""


def _like_literal(value: str) -> str:
    """Escape *value* for use inside a ``LIKE`` pattern.

    ``_`` and ``%`` are wildcards, and both are ordinary characters in the identifiers
    this backend searches: ``get_node`` would otherwise match ``getXnode``, and a query
    containing ``%`` would match everything. ``_prefix_clause`` already solved this for
    file paths by avoiding LIKE entirely; these call sites need real substring matching,
    so they escape instead and pair it with a ``LIKE ... ESCAPE`` clause.
    """
    for char, replacement in _LIKE_ESCAPES:
        value = value.replace(char, replacement)
    return value


def _prefix_clause_either(col_a: str, col_b: str, path: str) -> tuple[str, list[Any]]:
    """Like ``_prefix_clause``, but matching if *either* column has the prefix."""
    if not path:
        return "", []
    return (
        f" AND (substr({col_a}, 1, ?) = ? OR substr({col_b}, 1, ?) = ?)",
        [len(path), path, len(path), path],
    )


# Derived from _NODE_COLUMNS rather than re-listed: two hand-maintained lists of the
# same columns drift, and the failure mode is silent (an index on a column that is
# really a props_json key, or DDL that will not parse).
_REAL_NODE_COLUMNS: frozenset[str] = frozenset(c.strip() for c in _NODE_COLUMNS.split(","))


def _index_expr(prop: str) -> str:
    """Render *prop* as an indexable expression: a column, or a props_json extract.

    Everything schema.py names that is not a real column lives inside props_json, and
    indexing it needs an expression index. Rendered as a bare column reference it is
    not a slow index -- it is DDL that fails at ensure_schema, on all three branches,
    taking every already-initialised database down on every startup.
    """
    return prop if prop in _REAL_NODE_COLUMNS else f"json_extract(props_json, '$.{prop}')"


def _node_index_ddl() -> list[str]:
    """Per-(label, property) partial indices, sourced from schema.py's registries."""
    stmts = [
        f"CREATE INDEX IF NOT EXISTS idx_nodes_{spec.label.value.lower()}_{spec.property} "
        f"ON nodes({_index_expr(spec.property)}) WHERE labels = '{spec.label.value}';"
        for spec in LABEL_PROPERTY_INDICES
    ]
    stmts += [
        f"CREATE INDEX IF NOT EXISTS idx_nodes_{spec.label.value.lower()}_{'_'.join(spec.properties)} "
        f"ON nodes({', '.join(spec.properties)}) WHERE labels = '{spec.label.value}';"
        for spec in COMPOSITE_INDICES
    ]
    return stmts


# Candidates pulled from the bit index per result actually wanted, before exact
# rescoring. Measured on 9,118 real 1536-d embeddings, recall@10 against exact search is
# 1.000 at every factor from 4 to 64 -- so this is chosen for headroom at larger graphs
# rather than to buy back accuracy that was lost. 8 was 24x faster than exact search; 4
# was 33x, and the extra margin is cheap insurance against the recall question nobody has
# answered yet (see the module docstring on 1M).
_VEC_OVERSAMPLE = 8


def _bit_quantizable(dimension: int) -> bool:
    """Whether `vec_quantize_binary` accepts this dimension.

    It requires a length divisible by 8 -- one bit per component, packed into bytes.
    Every embedding model in practical use is a multiple of 8 (384, 768, 1024, 1536,
    3072), but `dimension` is user configuration, and a 100-dimension setting must
    degrade to the exact index rather than fail every write.
    """
    return dimension > 0 and dimension % 8 == 0


def _vec_table_ddl(dimension: int) -> list[str]:
    """Vector side tables: bit-quantized where possible, float otherwise.

    The float table this replaces was a **second copy of every vector**, stored beside
    the authoritative one in `nodes.embedding` and used only to answer KNN. sqlite-vec's
    KNN is a brute-force scan either way, so that copy bought no algorithmic advantage --
    it only made the scan read 32 bits per component instead of 1.

    A bit index plus an exact rescore of the candidates is strictly better on both axes,
    measured on 9,118 real 1536-d embeddings:

        exact (float table)   24.23 ms/query   0.33 s to build
        bit + rescore(x8)      1.00 ms/query   0.09 s to build   recall@10 1.000

    24x on retrieval, 3.7x on indexing, a thirty-second of the index storage, and the
    same answers. The exactness is not an approximation that happens to look good: the
    rescore step computes real cosine distance against `nodes.embedding`, so the ranking
    returned is exact for whatever the bit index shortlisted, and only the shortlist is
    approximate.

    Binary quantization keeps the sign of each component. Cosine distance is invariant to
    positive scaling, so the sign pattern is an equally good proxy whether or not the
    vectors are normalized -- which matters because this repo's index stores normalized
    vectors (mean L2 norm 1.000) and at least one production index does not (0.739).
    """
    if _bit_quantizable(dimension):
        return [
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {spec.name} USING vec0(embedding bit[{dimension}]);"
            for spec in build_vector_index_specs(dimension)
        ]
    return [
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {spec.name} "
        f"USING vec0(embedding float[{dimension}] distance_metric=cosine);"
        for spec in build_vector_index_specs(dimension)
    ]


def _fts_table_ddl() -> list[str]:
    return [f"CREATE VIRTUAL TABLE IF NOT EXISTS {spec.name} USING fts5(uid UNINDEXED, text);" for spec in TEXT_INDICES]


def _strip_uid(uid: str) -> str:
    """Strip project prefix from uid to get qualified_name (mirrors ``GraphClient._strip_uid``)."""
    return uid.split(":", 1)[1] if ":" in uid else uid


def _classify_batch(
    old_data: dict[str, dict[str, EntityHashData]],
    file_data: dict[str, tuple[list[ParsedEntity], list[ParsedRelationship]]],
) -> _BatchClassification:
    """Cross-file classification, mirroring ``GraphClient._classify_batch`` — kept as a
    free function here (rather than reaching into ``GraphClient``'s private static
    method) purely to avoid a cross-class private-member access.
    """
    all_added: list[ParsedEntity] = []
    all_modified: list[ParsedEntity] = []
    all_deleted_by_label: dict[str, list[str]] = defaultdict(list)
    all_shifted: list[ParsedEntity] = []
    per_file_results: dict[str, UpsertResult] = {}
    new_file_paths: set[str] = set()

    for file_path, (entities, _rels) in file_data.items():
        file_old = old_data.get(file_path, {})
        if not file_old:
            new_file_paths.add(file_path)

        fc = _classify_file_delta(file_old, entities, _strip_uid)

        all_added.extend(fc.added)
        all_modified.extend(fc.modified)
        for lbl, uids in fc.deleted_by_label.items():
            all_deleted_by_label[lbl].extend(uids)
        all_shifted.extend(fc.shifted)
        per_file_results[file_path] = fc.result

    return _BatchClassification(
        all_added=all_added,
        all_modified=all_modified,
        all_deleted_by_label=dict(all_deleted_by_label),
        all_shifted=all_shifted,
        per_file_results=per_file_results,
        new_file_paths=new_file_paths,
    )


def _json_default(value: Any) -> str:
    """Serialise the property types Bolt carries natively but JSON does not.

    `_coerce_property_value` in the markdown parser deliberately preserves temporals,
    because Bolt has a Date type and Memgraph stores an unquoted YAML `2026-08-30` as
    one. This backend has `json.dumps`, which does not, and the failure was not a lost
    property: `TypeError: Object of type date is not JSON serializable` propagates out of
    the whole **batch**, so every unrelated file travelling with the dated one was retried
    five times and then parked as poison. One dated ADR dropped a batch of this repo's own
    docs, silently, and only the benchmark's own output showed it.

    So the coercion belongs here rather than in the parser: the parser's value is correct
    for the backend it was written against, and narrowing it would cost Memgraph a real
    temporal to work around a JSON limitation. The two backends therefore return different
    types for a dated frontmatter key -- Date on Memgraph, ISO string here -- which is the
    honest consequence of one of them having temporals and the other not.
    """
    if isinstance(value, datetime.datetime | datetime.date | datetime.time):
        return value.isoformat()
    return str(value)


def _dumps(obj: Any) -> str:
    """`json.dumps` for anything stored in `props_json`.

    Every props_json write goes through this rather than `json.dumps` directly, because
    the crash above came from a value nobody thought of and the next one will too.
    """
    return json.dumps(obj, default=_json_default)


def _entity_props(e: ParsedEntity) -> dict[str, Any]:
    props: dict[str, Any] = {
        "line_start": e.line_start,
        "line_end": e.line_end,
        "visibility": e.visibility,
        "docstring": e.docstring,
        "signature": e.signature,
        "source": e.source,
        "tags": e.tags,
        "header_path": e.header_path,
        "header_level": e.header_level,
        # Always emitted, null when absent: the update path merges with
        # ``json_patch``, whose RFC 7396 semantics drop a key on null — that is
        # what makes a deleted ``# NOTE:`` clear the stored property.
        "rationale": e.rationale,
        "citations": e.citations or None,
    }
    props.update(e.extra_properties)
    return props


def _row_to_node(row: sqlite3.Row | tuple[Any, ...]) -> dict[str, Any]:
    """Reconstruct a Node-like dict (matches what ``search/engine.py``'s ``_extract_props``
    expects from a plain dict result) from a ``_NODE_COLUMNS``-ordered row.
    """
    uid, labels, project_name, qualified_name, file_path, name, kind, content_hash, props_json = row
    props = json.loads(props_json) if props_json else {}
    return {
        "uid": uid,
        "project_name": project_name,
        "qualified_name": qualified_name,
        "file_path": file_path,
        "name": name,
        "kind": kind,
        "content_hash": content_hash,
        **props,
        "_labels": [labels],
    }


def _is_chunk_record(record: dict[str, Any]) -> bool:
    """True when a search row is an overflow chunk rather than a node proper."""
    node = record.get("node")
    return isinstance(node, dict) and NodeLabel.EMBED_CHUNK.value in (node.get("_labels") or ())


def _best_similarity_per_uid(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse rows onto one per uid, keeping the highest similarity.

    Mirror of the same function in ``graph/client.py`` — see it for why max and not
    sum. Duplicated rather than shared because the two backends' "node" is a different
    shape and the uid read differs with it.
    """
    best: dict[str, dict[str, Any]] = {}
    for record in records:
        node = record.get("node") or {}
        uid = node.get("uid") if isinstance(node, dict) else None
        if uid is None:
            continue
        current = best.get(uid)
        if current is None or record.get("similarity", 0) > current.get("similarity", 0):
            best[uid] = record
    return list(best.values())


async def _rewire_stub_importers(conn: Any, *, stub_uid: str, real_uid: str, project: str) -> int:
    """Re-point *project*'s importers of *stub_uid* at *real_uid*. Returns how many moved.

    ``COALESCE(n.project_name, ?)``, not a plain join: scoping to the owning project is
    the ATL-088 P2 fix, but an inner join would also drop importers whose node row is
    missing, and those were previously rewired and cleaned up. An importer with no node
    belongs to no project, so it cannot be the sibling this predicate guards against --
    dropping it strands the stub and leaves an edge pointing at a deleted node.
    """
    cur = await conn.execute(
        "SELECT e.from_uid FROM edges e LEFT JOIN nodes n ON n.uid = e.from_uid "
        "WHERE e.to_uid = ? AND e.rel_type = 'IMPORTS' AND COALESCE(n.project_name, ?) = ?",
        (stub_uid, project, project),
    )
    importers = await cur.fetchall()
    await cur.close()
    for (src_uid,) in importers:
        await conn.execute(
            "INSERT OR IGNORE INTO edges(from_uid, to_uid, rel_type, props_json) VALUES (?, ?, 'IMPORTS', '{}')",
            (src_uid, real_uid),
        )
        await conn.execute(
            "DELETE FROM edges WHERE from_uid = ? AND to_uid = ? AND rel_type = 'IMPORTS'",
            (src_uid, stub_uid),
        )
    return len(importers)


async def _delete_stub_if_unimported(conn: Any, uid: str) -> None:
    """Drop an external stub only when nothing still imports it, taking its edges along.

    The mirror of Memgraph's ``WHERE NOT ()-[:IMPORTS]->(es) DETACH DELETE es``. An
    unguarded delete was safe only while every importer of the uid got rewired; once the
    rewrite is scoped to the stub's own project (ATL-088 P2), a deliberately-spared
    sibling importer is left pointing at a node that no longer exists -- and the edge row
    outlives it forever, because nothing sweeps for a dangling edge.
    """
    cur = await conn.execute("SELECT 1 FROM edges WHERE to_uid = ? AND rel_type = 'IMPORTS' LIMIT 1", (uid,))
    still_imported = await cur.fetchone()
    await cur.close()
    if still_imported is None:
        await conn.execute("DELETE FROM edges WHERE from_uid = ? OR to_uid = ?", (uid, uid))
        await conn.execute("DELETE FROM nodes WHERE uid = ?", (uid,))


class SqliteGraphClient:
    """Async SQLite-backed graph store — drop-in fallback for :class:`~code_atlas.graph.client.GraphClient`.

    Backed by a single lazily-opened ``aiosqlite`` connection (WAL mode) with
    the ``sqlite-vec`` extension loaded for vector search and SQLite's
    built-in FTS5 for BM25 text search.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        dimension: int = 768,
        embeddings_enabled: bool = True,
        conn: aiosqlite.Connection | None = None,
    ) -> None:
        self._db_path = db_path
        self._dimension = dimension
        # Decided once: the DDL, the write and the query all have to agree about which
        # index shape this database holds, and re-deriving it at three call sites is how
        # they come to disagree.
        self._bit_vectors = _bit_quantizable(dimension)
        self._embeddings_enabled = embeddings_enabled
        # Injected connections are used as-is (no PRAGMA/extension-load/schema
        # bootstrap) — the caller (real setup code or a test fake) owns that.
        self._conn: aiosqlite.Connection | None = conn
        # Instrumentation wraps only a connection this client opened. An injected one is
        # handed back untouched -- ADR-0038's "never what it was handed" applies to more
        # than closing: a caller that passes a connection gets that object back from
        # ``_get_conn``, not a decorated stand-in it never asked for.
        self._owns_conn = conn is None
        self._timed: TimedConnection | None = None
        self._connect_lock = asyncio.Lock()
        # Serializes the read-classify-write sequences in upsert_*: aiosqlite
        # funnels everything through one worker thread, but a multi-statement
        # logical operation still needs to run without another coroutine's
        # statements interleaving between its read and its write.
        self._write_lock = asyncio.Lock()

    @property
    def dimension(self) -> int:
        return self._dimension

    # -- Connection lifecycle / schema ---------------------------------------

    async def _get_conn(self) -> SqlConnection:
        if self._conn is None:
            async with self._connect_lock:
                if self._conn is None:
                    self._db_path.parent.mkdir(parents=True, exist_ok=True)
                    conn = await aiosqlite.connect(self._db_path)
                    await conn.enable_load_extension(True)
                    await conn.load_extension(sqlite_vec.loadable_path())
                    await conn.enable_load_extension(False)
                    await conn.execute("PRAGMA journal_mode=WAL")
                    await conn.execute("PRAGMA synchronous=NORMAL")
                    await conn.executescript(_BASE_SCHEMA_SQL)
                    await conn.commit()
                    self._conn = conn
        assert self._conn is not None
        if not self._owns_conn:
            return self._conn
        if self._timed is None:
            # Wrapped once, here, rather than at 213 call sites -- so a statement added
            # tomorrow is instrumented without anyone remembering to instrument it.
            self._timed = TimedConnection(self._conn)
        return self._timed

    async def ping(self) -> bool:
        """Health check — returns True if the local database is reachable."""
        conn = await self._get_conn()
        await conn.execute("SELECT 1")
        return True

    async def execute(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError(
            "Raw Cypher execution is not supported by the sqlite backend — arbitrary agent-authored "
            "Cypher (cypher_query/validate_cypher) has no SQL translation layer; this is a deliberate, "
            "permanent limitation, not a gap (see ADR-0015). The MCP tool layer guards these call sites "
            "explicitly rather than relying on this exception."
        )

    async def execute_write(self, query: str, params: dict[str, Any] | None = None) -> None:
        raise NotImplementedError("Raw Cypher execution is not supported by the sqlite backend.")

    async def get_schema_version(self) -> int | None:
        conn = await self._get_conn()
        cur = await conn.execute("SELECT value FROM meta WHERE key = 'schema_version'")
        row = await cur.fetchone()
        await cur.close()
        return int(row[0]) if row else None

    async def _upsert_meta(self, conn: SqlConnection, key: str, value: str) -> None:
        await conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    async def _set_schema_version(self, conn: SqlConnection, version: int) -> None:
        await self._upsert_meta(conn, "schema_version", str(version))

    async def _migrate_v6_clear_freshness_markers(self, conn: SqlConnection) -> None:
        """Mirror of ``GraphClient._migrate_v6_clear_freshness_markers``.

        CALLS edges gained numeric ``weight``/``candidate_count``/``from_test`` and entities
        gained ``rationale``/``citations``; both are produced at index time, and nothing would
        otherwise enumerate an unchanged file.

        Only ``git_hash`` is cleared here. The per-file ``file_hash`` clear these mirrors
        went with ATL-152: ``schema.EXTRACTION_EPOCH`` is folded into the gate's key, so a
        migration that needs a re-parse bumps the epoch and the stored hashes stop matching
        on their own. See ``GraphClient._run_data_migrations`` for why the git_hash clear is
        not subsumed by that and stays.
        """
        await conn.execute(
            "UPDATE nodes SET props_json = json_remove(props_json, '$.git_hash') "
            "WHERE json_extract(props_json, '$.git_hash') IS NOT NULL"
        )
        await conn.execute(
            "DELETE FROM edges WHERE rel_type = 'CALLS' AND json_extract(props_json, '$.weight') IS NULL"
        )
        await conn.commit()
        logger.info(
            "SQLite schema v6: cleared the stored git_hash and unweighted CALLS edges — "
            "run 'atlas index' to rebuild with edge weights and rationale"
        )
        await conn.commit()

    async def _migrate_v9_clear_for_abstract_bases(self, conn: SqlConnection) -> None:
        """Mirror of ``GraphClient._migrate_v9_clear_for_abstract_bases``."""
        await conn.execute(
            "DELETE FROM edges WHERE rel_type = 'CALLS' AND json_extract(props_json, '$.confidence') = 'ambiguous'"
        )
        await conn.execute(
            "UPDATE nodes SET props_json = json_remove(props_json, '$.git_hash') "
            "WHERE json_extract(props_json, '$.git_hash') IS NOT NULL"
        )
        await conn.commit()
        logger.info("SQLite graph schema v9: cleared ambiguous CALLS edges and the stored git_hash")

    async def _migrate_v8_drop_unverified_calls(self, conn: SqlConnection) -> None:
        """Mirror of ``GraphClient._migrate_v8_drop_unverified_calls``.

        project_unique edges were written confidence:"resolved" with full weight, so
        nothing downstream distrusts them; they must be removed rather than left to be
        overwritten by a re-parse.
        """
        await conn.execute(
            "DELETE FROM edges WHERE rel_type = 'CALLS' AND json_extract(props_json, '$.strategy') = 'project_unique'"
        )
        await conn.execute(
            "UPDATE nodes SET props_json = json_remove(props_json, '$.git_hash') "
            "WHERE json_extract(props_json, '$.git_hash') IS NOT NULL"
        )
        await conn.commit()
        logger.info("SQLite graph schema v8: dropped project_unique CALLS edges and cleared the git_hash")

    async def _migrate_v7_clear_freshness_markers(self, conn: SqlConnection) -> None:
        """Mirror of ``GraphClient._migrate_v7_clear_freshness_markers``.

        EnvVar/ResourceFile nodes exist only if a parser produced the reference,
        and nothing already in the graph can be used to derive them — so every
        file has to be re-parsed, and without this clear nothing would enumerate
        an unchanged one. See the v6 mirror for why only ``git_hash`` is cleared.
        """
        await conn.execute(
            "UPDATE nodes SET props_json = json_remove(props_json, '$.git_hash') "
            "WHERE json_extract(props_json, '$.git_hash') IS NOT NULL"
        )
        await conn.commit()
        logger.info(
            "SQLite schema v7: cleared the stored git_hash — run 'atlas index' to extract "
            "environment-variable and referenced-file nodes"
        )

    async def _migrate_v18_versions_moved_to_dependency_edge(self, conn: SqlConnection) -> None:
        """Mirror of ``GraphClient._migrate_v18_versions_moved_to_dependency_edge``.

        The first migration on this backend that moves data rather than clearing it —
        v6/v7/v8/v9 are all ``json_remove``/``DELETE``, because they hand the work back to
        the parser. That is not available here: a version comes from a manifest read only
        during ``atlas index``, so dropping the property and waiting would report every
        dependency as unversioned until someone happens to re-index.

        Both statements land in one commit and both are idempotent, because
        ``ensure_schema`` re-applies on every startup and a half-applied move is the one
        state with no reader — the INSERT alone leaves the stale property that
        get_structure_overview no longer reads, the UPDATE alone loses the version.
        """
        # Only for packages whose Project node still exists: the edge's from_uid is that
        # node's uid, and an orphan left by a deleted project would otherwise get a
        # dangling row. Those keep their node property rather than losing the value —
        # see the Cypher's docstring on why the REMOVE is gated instead of unconditional.
        await conn.execute(
            "INSERT INTO edges(from_uid, to_uid, rel_type, props_json) "
            "SELECT n.project_name, n.uid, 'DEPENDS_ON', "
            "json_object('version', json_extract(n.props_json, '$.version')) FROM nodes n "
            "WHERE n.labels = 'ExternalPackage' AND json_extract(n.props_json, '$.version') IS NOT NULL "
            "AND EXISTS (SELECT 1 FROM nodes p WHERE p.labels = 'Project' AND p.uid = n.project_name) "
            "ON CONFLICT(from_uid, to_uid, rel_type) DO UPDATE SET "
            "props_json = json_patch(edges.props_json, excluded.props_json)"
        )
        await conn.execute(
            "UPDATE nodes SET props_json = json_remove(props_json, '$.version') "
            "WHERE labels = 'ExternalPackage' AND json_extract(props_json, '$.version') IS NOT NULL "
            "AND EXISTS (SELECT 1 FROM edges d WHERE d.to_uid = nodes.uid AND d.rel_type = 'DEPENDS_ON' "
            "AND json_extract(d.props_json, '$.version') IS NOT NULL)"
        )
        await conn.commit()
        logger.info("SQLite graph schema v18: moved ExternalPackage versions onto Project -[DEPENDS_ON]-> edges")

    async def _apply_full_schema(self, conn: SqlConnection) -> None:
        for stmt in _node_index_ddl():
            await conn.execute(stmt)
        if self._embeddings_enabled:
            for stmt in _vec_table_ddl(self._dimension):
                await conn.execute(stmt)
        for stmt in _fts_table_ddl():
            await conn.execute(stmt)
        await conn.commit()
        await self._ensure_fts_keyed_by_rowid(conn)
        await self._ensure_vector_index_shape(conn)

    async def _ensure_vector_index_shape(self, conn: SqlConnection) -> None:
        """Rebuild the vector side tables when the index shape changes.

        A vec0 column type cannot be altered, so switching between the float and bit
        shapes means dropping and recreating. That costs nothing but time: the vectors
        themselves live in `nodes.embedding` and are never re-embedded, so a rebuild is a
        local re-index of data already on disk and never a provider bill.

        A backend-local `meta` marker rather than a `SCHEMA_VERSION` bump, for the reason
        `_ensure_fts_keyed_by_rowid` gives: that number is shared with Memgraph, where
        advancing it drops and recreates *its* vector indices.

        The marker records the shape, not the migration, so a database that predates it
        is rebuilt once and a future third shape is a different value rather than a
        version number nobody can interpret.
        """
        if not self._embeddings_enabled:
            return
        want = "bit" if self._bit_vectors else "float"
        cur = await conn.execute("SELECT value FROM meta WHERE key = 'vec_kind'")
        row = await cur.fetchone()
        await cur.close()
        if row is not None and row[0] == want:
            return

        rebuilt = 0
        for spec in build_vector_index_specs(self._dimension):
            await self._safe_exec(conn, f"DROP TABLE IF EXISTS {spec.name}")
        for stmt in _vec_table_ddl(self._dimension):
            await conn.execute(stmt)
        for spec in build_vector_index_specs(self._dimension):
            cur = await conn.execute(
                "SELECT rowid, embedding FROM nodes WHERE labels = ? AND embedding IS NOT NULL",
                (spec.label.value,),
            )
            payload = [(rid, blob) for rid, blob in await cur.fetchall()]
            await cur.close()
            if not payload:
                continue
            insert = (
                f"INSERT INTO {spec.name}(rowid, embedding) VALUES (?, vec_quantize_binary(?))"
                if self._bit_vectors
                else f"INSERT INTO {spec.name}(rowid, embedding) VALUES (?, ?)"
            )
            await self._safe_exec_many(conn, insert, payload)
            rebuilt += len(payload)

        await conn.execute(
            "INSERT INTO meta(key, value) VALUES ('vec_kind', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (want,),
        )
        await conn.commit()
        if rebuilt:
            logger.info("Rebuilt {} vector(s) into the {} index (no re-embedding)", rebuilt, want)

    async def _ensure_fts_keyed_by_rowid(self, conn: SqlConnection) -> None:
        """Rebuild the FTS tables once, so their rowid is the node's rowid.

        Deliberately **not** a `SCHEMA_VERSION` bump. That number is shared with Memgraph,
        where advancing it drops and recreates vector indices; taking semantic search down
        on the networked backend to re-key a table only the embedded one has would be a
        cost with nothing to buy. This is a storage detail of one backend, so it carries
        its own marker.

        Idempotent and self-describing: the marker records what the tables are keyed *by*,
        not which migration ran. A database with no FTS rows is marked without doing work,
        and a future re-keying gets a different value rather than a version number nobody
        can interpret later.
        """
        cur = await conn.execute("SELECT value FROM meta WHERE key = 'fts_key'")
        row = await cur.fetchone()
        await cur.close()
        if row is not None and row[0] == "rowid":
            return

        rebuilt = 0
        for spec in TEXT_INDICES:
            table = f"text_{spec.label.value.lower()}"
            try:
                await conn.execute(f"DELETE FROM {table}")
            except aiosqlite.OperationalError:
                continue  # no such table yet — nothing to re-key
            cur = await conn.execute(
                "SELECT rowid, uid, name, qualified_name, props_json FROM nodes WHERE labels = ?",
                (spec.label.value,),
            )
            rows = await cur.fetchall()
            await cur.close()
            payload = [
                (rowid, uid, _fts_document(name or "", qn or "", json.loads(props or "{}")))
                for rowid, uid, name, qn, props in rows
            ]
            if payload:
                await self._safe_exec_many(conn, f"INSERT INTO {table}(rowid, uid, text) VALUES (?, ?, ?)", payload)
                rebuilt += len(payload)

        await conn.execute(
            "INSERT INTO meta(key, value) VALUES ('fts_key', 'rowid') "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
        )
        await conn.commit()
        if rebuilt:
            logger.info("Re-keyed {} FTS document(s) by rowid (BM25 deletes no longer scan)", rebuilt)

    async def ensure_schema(self, *, force_drop_embeddings: bool = False) -> None:  # noqa: ARG002  # parity with GraphClient.ensure_schema; see docstring
        """Apply or migrate the graph schema.

        *force_drop_embeddings* is accepted for signature parity with
        ``GraphClient.ensure_schema`` and ignored: this backend's DDL is
        ``CREATE ... IF NOT EXISTS`` throughout and never drops a vec0 table, so
        running with embeddings disabled leaves an existing one intact rather than
        destroying it. There is nothing here to force past.

        Mirrors ``GraphClient.ensure_schema``'s fresh/current/older/newer
        branches conceptually, using a ``schema_version`` row in ``meta``
        instead of a ``SchemaVersion`` graph node. The embedded backend has
        no legacy data predating v5, so "older" just re-applies the
        (idempotent, ``CREATE ... IF NOT EXISTS``) DDL rather than running
        Memgraph's freshness-marker migrations.
        """
        conn = await self._get_conn()
        stored = await self.get_schema_version()
        if stored is None:
            logger.info("Fresh SQLite graph database — applying schema v{}", SCHEMA_VERSION)
            await self._apply_full_schema(conn)
            await self._set_schema_version(conn, SCHEMA_VERSION)
            logger.info("SQLite graph schema v{} applied", SCHEMA_VERSION)
        elif stored == SCHEMA_VERSION:
            logger.debug("SQLite graph schema v{} already current", SCHEMA_VERSION)
            # Re-apply rather than trust the version: a current version node does not
            # prove the vec0/FTS5 side tables still exist, and without them search
            # degrades silently. The DDL is CREATE ... IF NOT EXISTS throughout, so
            # this is a no-op when nothing is missing. Mirrors
            # GraphClient._reconcile_search_indices.
            await self._apply_full_schema(conn)
        elif stored < SCHEMA_VERSION:
            logger.info("SQLite graph schema v{} -> v{} (idempotent re-apply)", stored, SCHEMA_VERSION)
            await self._apply_full_schema(conn)
            if stored < 6:
                await self._migrate_v6_clear_freshness_markers(conn)
            if stored < 7:
                await self._migrate_v7_clear_freshness_markers(conn)
            if stored < 8:
                await self._migrate_v8_drop_unverified_calls(conn)
            if stored < 9:
                await self._migrate_v9_clear_for_abstract_bases(conn)
            # The gap from 10 to 17 is real, not an oversight: those are all Memgraph
            # freshness-marker and index-shape migrations with no embedded analogue. v18
            # moves data this backend actually stores, so it needs its own branch or a
            # SQLite graph keeps the version on the node while reading it off the edge.
            if stored < 18:
                await self._migrate_v18_versions_moved_to_dependency_edge(conn)
            await self._set_schema_version(conn, SCHEMA_VERSION)
        else:
            msg = (
                f"SQLite graph database schema v{stored} is newer than code v{SCHEMA_VERSION}. "
                f"Downgrade is not supported — update your Code Atlas installation."
            )
            raise RuntimeError(msg)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Close on the way out, including on an exception.

        The point is that closing stops being something each caller has to remember at
        every exit path. It was forgotten on four of them at once -- all four infra
        fixtures called `pytest.skip()` between constructing a client and closing it, so
        every skipped run abandoned a live connection and the resulting ResourceWarning
        was blamed on whichever unrelated test the GC happened to interrupt.
        """
        await self.close()

    # -- Internal helpers -----------------------------------------------------

    async def _safe_exec_many(self, conn: SqlConnection, stmt: str, rows: list[tuple[Any, ...]]) -> None:
        """`_safe_exec` for a batch -- one dispatch instead of one per row."""
        if not rows:
            return
        try:
            await conn.executemany(stmt, rows)
        except aiosqlite.OperationalError as exc:
            logger.debug("Side-table statement skipped ({}): {}", exc, stmt[:80])

    async def _safe_exec(self, conn: SqlConnection, stmt: str, params: Any = ()) -> None:
        """Execute a side-table (vec0/FTS5) statement, tolerating a missing virtual
        table (``ensure_schema`` not yet run) the way Memgraph tolerates writing
        node properties before its vector/text indices exist.
        """
        try:
            await conn.execute(stmt, params)
        except aiosqlite.OperationalError as exc:
            logger.debug("Side-table statement skipped ({}): {}", exc, stmt[:80])

    async def _nodes_by_uid(self, conn: SqlConnection, uids: list[str]) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        deduped = list(dict.fromkeys(uids))
        for chunk in _chunks(deduped):
            if not chunk:
                continue
            placeholders = ",".join("?" * len(chunk))
            cur = await conn.execute(f"SELECT {_NODE_COLUMNS} FROM nodes WHERE uid IN ({placeholders})", chunk)
            rows = await cur.fetchall()
            await cur.close()
            for row in rows:
                node = _row_to_node(row)
                result[node["uid"]] = node
        return result

    async def _cleanup_search_side_tables(self, conn: SqlConnection, uids: list[str]) -> None:
        for chunk in _chunks(uids):
            if not chunk:
                continue
            placeholders = ",".join("?" * len(chunk))
            cur = await conn.execute(f"SELECT uid, labels, rowid FROM nodes WHERE uid IN ({placeholders})", chunk)
            rows = await cur.fetchall()
            await cur.close()
            by_label: dict[str, list[tuple[str, int]]] = defaultdict(list)
            for uid, label, rowid in rows:
                by_label[label].append((uid, rowid))
            for label, items in by_label.items():
                if label in _VEC_LABEL_VALUES:
                    table = f"vec_{label.lower()}"
                    rowids = ",".join(str(rid) for _uid, rid in items)
                    await self._safe_exec(conn, f"DELETE FROM {table} WHERE rowid IN ({rowids})")
                if label in _TEXT_LABEL_VALUES:
                    # By rowid, like the vec table above it: `uid` is an UNINDEXED fts5
                    # column, so an `IN` over uids scans the whole FTS table.
                    table = f"text_{label.lower()}"
                    text_rowids = ",".join(str(rid) for _uid, rid in items)
                    await self._safe_exec(conn, f"DELETE FROM {table} WHERE rowid IN ({text_rowids})")

    async def _rowids_for(self, conn: SqlConnection, uids: list[str]) -> dict[str, int]:
        """``uid -> nodes.rowid``, via the uid primary key."""
        out: dict[str, int] = {}
        for chunk in _chunks(list(dict.fromkeys(uids))):
            if not chunk:
                continue
            placeholders = ",".join("?" * len(chunk))
            cur = await conn.execute(f"SELECT uid, rowid FROM nodes WHERE uid IN ({placeholders})", chunk)
            rows = await cur.fetchall()
            await cur.close()
            for uid, rowid in rows:
                out[uid] = rowid  # noqa: PERF403 — a comprehension here would rebuild the dict per chunk
        return out

    async def _sync_fts_rows(self, conn: SqlConnection, entities: list[ParsedEntity]) -> None:
        """Rewrite the FTS documents for *entities*, keyed by rowid, two statements per label.

        The version this replaced deleted by ``uid``, one entity at a time. ``uid`` is an
        ``UNINDEXED`` fts5 column, so that delete had no index to use and planned as
        ``SCAN text_<label> VIRTUAL TABLE INDEX 0:`` -- a full scan of the FTS table, per
        entity, while holding the write lock.

        The cost is not merely linear, it compounds: measured at 0.592 / 1.905 / 11.022 ms
        per row against tables of 2k / 8k / 32k rows, against 0.134 / 0.201 / 0.348 ms by
        rowid. Sixteen times the table gave nineteen times the per-row cost, so the total
        is quadratic in graph size -- and the sharpest case is not a reindex but a single
        file save, where every entity in the file pays a scan of the whole table.

        Keying on ``nodes.rowid`` puts no new constraint on the schema: the vec0 tables are
        already keyed that way. It does mean neither side table survives a ``VACUUM``,
        which renumbers rowids; nothing here runs one.
        """
        by_label: dict[str, list[ParsedEntity]] = defaultdict(list)
        for e in entities:
            if e.label.value in _TEXT_LABEL_VALUES:
                by_label[e.label.value].append(e)
        if not by_label:
            return

        rowids = await self._rowids_for(conn, [e.qualified_name for es in by_label.values() for e in es])
        for label, group in by_label.items():
            table = f"text_{label.lower()}"
            rows = [
                (rowids[e.qualified_name], e.qualified_name, _fts_document_from_entity(e))
                for e in group
                if e.qualified_name in rowids
            ]
            if not rows:
                continue
            await self._safe_exec_many(conn, f"DELETE FROM {table} WHERE rowid = ?", [(r[0],) for r in rows])
            await self._safe_exec_many(conn, f"INSERT INTO {table}(rowid, uid, text) VALUES (?, ?, ?)", rows)

    async def _get_file_content_hashes(
        self, conn: SqlConnection, project_name: str, file_path: str
    ) -> dict[str, EntityHashData]:
        cur = await conn.execute(
            "SELECT uid, labels, content_hash, props_json FROM nodes "
            "WHERE project_name = ? AND file_path = ? AND labels NOT IN ('Package', 'Project')",
            (project_name, file_path),
        )
        rows = await cur.fetchall()
        await cur.close()
        result: dict[str, EntityHashData] = {}
        for uid, label, content_hash, props_json in rows:
            props = json.loads(props_json) if props_json else {}
            result[uid] = EntityHashData(
                content_hash or "",
                props.get("line_start") or 0,
                props.get("line_end") or 0,
                props.get("signature"),
                props.get("docstring"),
                label or "",
            )
        return result

    async def _get_batch_file_content_hashes(
        self, conn: SqlConnection, project_name: str, file_paths: list[str]
    ) -> dict[str, dict[str, EntityHashData]]:
        if not file_paths:
            return {}
        result: dict[str, dict[str, EntityHashData]] = defaultdict(dict)
        for chunk in _chunks(file_paths):
            if not chunk:
                continue
            placeholders = ",".join("?" * len(chunk))
            cur = await conn.execute(
                f"SELECT file_path, uid, labels, content_hash, props_json FROM nodes "
                f"WHERE project_name = ? AND file_path IN ({placeholders}) AND labels NOT IN ('Package', 'Project')",
                (project_name, *chunk),
            )
            rows = await cur.fetchall()
            await cur.close()
            for fp, uid, label, content_hash, props_json in rows:
                props = json.loads(props_json) if props_json else {}
                result[fp][uid] = EntityHashData(
                    content_hash or "",
                    props.get("line_start") or 0,
                    props.get("line_end") or 0,
                    props.get("signature"),
                    props.get("docstring"),
                    label or "",
                )
        return dict(result)

    # -- Entity CRUD ------------------------------------------------------------

    async def _batch_create_entities(
        self, conn: SqlConnection, project_name: str, entities: list[ParsedEntity]
    ) -> None:
        if not entities:
            return
        rows = []
        for e in entities:
            uid = e.qualified_name
            qn = uid.split(":", 1)[1] if ":" in uid else uid
            rows.append(
                (
                    uid,
                    e.label.value,
                    project_name,
                    qn,
                    e.file_path,
                    e.name,
                    e.kind,
                    e.content_hash,
                    _dumps(_entity_props(e)),
                )
            )
        await conn.executemany(
            f"INSERT INTO nodes({_NODE_COLUMNS}) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(uid) DO UPDATE SET labels=excluded.labels, project_name=excluded.project_name, "
            "qualified_name=excluded.qualified_name, file_path=excluded.file_path, name=excluded.name, "
            "kind=excluded.kind, content_hash=excluded.content_hash, props_json=excluded.props_json",
            rows,
        )
        await self._sync_fts_rows(conn, entities)

    async def _batch_update_entities(self, conn: SqlConnection, entities: list[ParsedEntity]) -> None:
        if not entities:
            return
        for e in entities:
            await conn.execute(
                # ``frontmatter`` is dropped before the patch rather than merged into. RFC 7396
                # recurses into a nested object, so patching {"frontmatter": {"a": 1}} over a
                # stored {"a": 1, "b": 2} keeps ``b`` -- a key deleted from the file would survive
                # here while Cypher's ``SET n +=`` replaces the map wholesale. Removing it first
                # makes the patch authoritative and matches the other backend. ``json_remove`` on
                # an absent path is a no-op, and a null in the patch still clears the property.
                "UPDATE nodes SET name = ?, kind = ?, content_hash = ?, "
                "props_json = json_patch(json_remove(props_json, '$.frontmatter'), ?) "
                "WHERE uid = ?",
                (e.name, e.kind, e.content_hash, _dumps(_entity_props(e)), e.qualified_name),
            )
        await self._sync_fts_rows(conn, entities)

    async def _batch_update_positions(self, conn: SqlConnection, entities: list[ParsedEntity]) -> None:
        for e in entities:
            await conn.execute(
                "UPDATE nodes SET props_json = json_patch(props_json, ?) WHERE uid = ?",
                (_dumps({"line_start": e.line_start, "line_end": e.line_end}), e.qualified_name),
            )

    async def _batch_delete_entities(self, conn: SqlConnection, uids_by_label: dict[str, list[str]]) -> None:
        """Delete entity nodes by uid.

        Simplified vs. ``GraphClient._batch_delete_entities`` — always fully
        removes the node and its edges, with no zombie-node preservation for
        entities still referenced from a different file's untouched edges.
        """
        all_uids = [u for uids in uids_by_label.values() for u in uids]
        if not all_uids:
            return
        await self._cleanup_search_side_tables(conn, all_uids)
        for chunk in _chunks(all_uids):
            if not chunk:
                continue
            placeholders = ",".join("?" * len(chunk))
            await conn.execute(
                f"DELETE FROM edges WHERE from_uid IN ({placeholders}) OR to_uid IN ({placeholders})", (*chunk, *chunk)
            )
            await conn.execute(f"DELETE FROM nodes WHERE uid IN ({placeholders})", chunk)

    async def upsert_file_entities(
        self,
        project_name: str,
        file_path: str,
        entities: list[ParsedEntity],
        relationships: list[ParsedRelationship],
    ) -> UpsertResult:
        conn = await self._get_conn()
        async with self._write_lock, _txn():
            old_data = await self._get_file_content_hashes(conn, project_name, file_path)
            fc = _classify_file_delta(old_data, entities, _strip_uid)

            if not fc.added and not fc.modified and not fc.result.deleted:
                if fc.shifted:
                    await self._batch_update_positions(conn, fc.shifted)
                    await conn.commit()
                return fc.result

            if fc.deleted_by_label:
                await self._batch_delete_entities(conn, fc.deleted_by_label)
            if fc.added:
                await self._batch_create_entities(conn, project_name, fc.added)
            if fc.modified:
                await self._batch_update_entities(conn, fc.modified)
            if fc.shifted:
                await self._batch_update_positions(conn, fc.shifted)

            await self._recreate_file_relationships(
                conn, project_name, file_path, relationships, skip_delete=not old_data
            )
            await conn.commit()
            return fc.result

    async def upsert_batch_entities(
        self,
        project_name: str,
        file_data: dict[str, tuple[list[ParsedEntity], list[ParsedRelationship]]],
        *,
        rels_only: bool = False,
        rels_unchanged: Collection[str] = (),
    ) -> dict[str, UpsertResult]:
        if not file_data:
            return {}
        conn = await self._get_conn()
        async with self._write_lock, _txn():
            if rels_only:
                # See GraphClient.upsert_batch_entities for why this pass classifies to
                # nothing and why new_file_paths is empty.
                file_rels = {fp: rels for fp, (_entities, rels) in file_data.items()}
                await self._recreate_batch_relationships(conn, project_name, file_rels, set())
                await conn.commit()
                return {}
            file_paths = list(file_data)
            old_data = await self._get_batch_file_content_hashes(conn, project_name, file_paths)
            classification = _classify_batch(old_data, file_data)

            if classification.all_deleted_by_label:
                await self._batch_delete_entities(conn, classification.all_deleted_by_label)
            if classification.all_added:
                await self._batch_create_entities(conn, project_name, classification.all_added)
            if classification.all_modified:
                await self._batch_update_entities(conn, classification.all_modified)
            if classification.all_shifted:
                await self._batch_update_positions(conn, classification.all_shifted)

            file_rels = {fp: rels for fp, (_entities, rels) in file_data.items()}
            if rels_unchanged:
                # Same gate as GraphClient.upsert_batch_entities -- see _rel_write_skips.
                skips = _rel_write_skips(classification, file_rels, rels_unchanged)
                file_rels = {fp: rels for fp, rels in file_rels.items() if fp not in skips}
            if file_rels:
                await self._recreate_batch_relationships(conn, project_name, file_rels, classification.new_file_paths)
            await conn.commit()
            return classification.per_file_results

    async def delete_file_entities(self, project_name: str, file_path: str) -> list[str]:
        conn = await self._get_conn()
        old_data = await self._get_file_content_hashes(conn, project_name, file_path)
        if old_data:
            uids_by_label: dict[str, list[str]] = defaultdict(list)
            for uid, data in old_data.items():
                uids_by_label[data.label].append(uid)
            await self._batch_delete_entities(conn, dict(uids_by_label))
        await conn.execute(
            "UPDATE nodes SET props_json = json_remove(props_json, '$.file_hash', '$.rels_hash') "
            "WHERE labels = 'Package' AND project_name = ? AND file_path = ?",
            (project_name, file_path),
        )
        await conn.commit()
        return [_strip_uid(uid) for uid in old_data]

    # -- Relationships ------------------------------------------------------------

    async def _create_relationships(
        self, conn: SqlConnection, project_name: str, relationships: list[ParsedRelationship]
    ) -> None:
        if not relationships:
            return

        direct_rels: list[ParsedRelationship] = []
        implements_rels: list[ParsedRelationship] = []

        for r in relationships:
            # Post-batch types (CALLS/IMPORTS/USES_TYPE/READS_ENV/REFERENCES_FILE)
            # carry a bare target *name*, not a uid — the direct_rels path below
            # would happily insert a dangling edge to that name.
            if r.rel_type.value in _POST_BATCH_REL_VALUES:
                continue
            if r.rel_type.value == "DEFINES" and "parent_type_name" in r.properties:
                continue
            if r.rel_type.value == "IMPLEMENTS" and ":" not in r.to_name:
                implements_rels.append(r)
            else:
                direct_rels.append(r)

        if direct_rels:
            rows = [
                (r.from_qualified_name, r.to_name, r.rel_type.value, _dumps(r.properties or {})) for r in direct_rels
            ]
            await conn.executemany(
                "INSERT INTO edges(from_uid, to_uid, rel_type, props_json) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(from_uid, to_uid, rel_type) DO UPDATE SET "
                "props_json = json_patch(edges.props_json, excluded.props_json)",
                rows,
            )

        for name_rel_type, name_rels in (("IMPLEMENTS", implements_rels),):
            for r in name_rels:
                cur = await conn.execute(
                    "SELECT uid FROM nodes WHERE labels = 'TypeDef' AND project_name = ? AND name = ?",
                    (project_name, r.to_name),
                )
                target = await cur.fetchone()
                await cur.close()
                if target:
                    await conn.execute(
                        "INSERT OR IGNORE INTO edges(from_uid, to_uid, rel_type, props_json) VALUES (?, ?, ?, '{}')",
                        (r.from_qualified_name, target[0], name_rel_type),
                    )

    async def resolve_doc_links(self, project_name: str, doc_rels: list[ParsedRelationship]) -> None:
        """Simplified port of ``GraphClient.resolve_doc_links`` — per-relationship
        matching instead of batched queries; same never-multi-link discipline (an
        ambiguous match is left unresolved).

        Deferred to the resolution flush like its Memgraph counterpart: DOCUMENTS is
        in ``_POST_BATCH_REL_VALUES`` now, so the create phase above skips it. The
        ``file_path LIKE '%suffix'`` scan is unindexable here too, so the win is the
        same — it runs once per flush instead of once per doc file.
        """
        if not doc_rels:
            return
        conn = await self._get_conn()

        name_refs = [r for r in doc_rels if not r.properties.get("is_file_ref")]
        file_refs = [r for r in doc_rels if r.properties.get("is_file_ref")]

        # Two lookups for the whole flush, not two per relationship. Both of the old
        # per-rel queries were unindexable by construction: `labels NOT IN (...)` implies
        # no single-label predicate so none of the partial indices apply (ADR-0045), and
        # `file_path LIKE '%suffix'` has a leading wildcard, which the docstring above
        # already admitted. At ~40ms each that was the single most expensive call in the
        # entire index: measured at **124.08 seconds** for one call on this repo, 72% of
        # the run's wall clock, and it is the reason the final flush was being cancelled
        # by the teardown watchdog four seconds before it would have finished.
        by_name: dict[str, list[str]] = defaultdict(list)
        if name_refs:
            for chunk in _chunks(sorted({r.to_name for r in name_refs}), 900):
                if not chunk:
                    continue
                placeholders = ",".join("?" * len(chunk))
                cur = await conn.execute(
                    f"SELECT name, uid FROM nodes WHERE project_name = ? AND name IN ({placeholders}) "
                    "AND labels NOT IN ('Note', 'DocSection')",
                    (project_name, *chunk),
                )
                rows = await cur.fetchall()
                await cur.close()
                for name, uid in rows:
                    by_name[name].append(uid)

        # Suffix matching moves into Python, over distinct paths rather than nodes: a
        # file with 40 entities is one comparison here and was 40 rows in the old scan.
        # `_like_literal` escaped LIKE's `%`/`_`, so the old match was already literal in
        # that respect -- but it was NOT case-sensitive, and `str.endswith` is.
        uids_by_path: dict[str, list[str]] = defaultdict(list)
        if file_refs:
            cur = await conn.execute(
                "SELECT file_path, uid FROM nodes WHERE project_name = ? AND file_path IS NOT NULL "
                "AND labels NOT IN ('Note', 'DocSection')",
                (project_name,),
            )
            rows = await cur.fetchall()
            await cur.close()
            for file_path, uid in rows:
                uids_by_path[file_path].append(uid)

        def _file_candidates(suffix: str) -> list[str]:
            folded = _ascii_fold(suffix)
            return [uid for path, uids in uids_by_path.items() if _ascii_fold(path).endswith(folded) for uid in uids]

        writes: list[tuple[str, str, str]] = []
        for r in doc_rels:
            props = r.properties
            candidates = _file_candidates(r.to_name) if props.get("is_file_ref") else by_name.get(r.to_name, [])
            # The never-multi-link discipline, unchanged: an ambiguous match is left
            # unresolved rather than guessed at. Counted over *nodes*, as before — a file
            # reference matching one path with two entities in it is still ambiguous.
            if len(candidates) == 1:
                entry_props = _dumps(
                    {"link_type": props.get("link_type", ""), "confidence": props.get("confidence", 0.0)}
                )
                writes.append((r.from_qualified_name, candidates[0], entry_props))

        if writes:
            await conn.executemany(
                "INSERT OR IGNORE INTO edges(from_uid, to_uid, rel_type, props_json) VALUES (?, ?, 'DOCUMENTS', ?)",
                writes,
            )
        await conn.commit()

    async def _recreate_file_relationships(
        self,
        conn: SqlConnection,
        project_name: str,
        file_path: str,
        relationships: list[ParsedRelationship],
        *,
        skip_delete: bool = False,
    ) -> None:
        if not skip_delete:
            await conn.execute(
                "DELETE FROM edges WHERE from_uid IN "
                "(SELECT uid FROM nodes WHERE project_name = ? AND file_path = ? "
                "AND labels NOT IN ('Package', 'Project')) "
                f"AND NOT ({_CITATION_EDGE_PREDICATE}) "
                f"AND NOT ({_CROSS_FILE_DEFINES_PREDICATE})",
                (project_name, file_path),
            )
        await self._create_relationships(conn, project_name, relationships)

    async def _recreate_batch_relationships(
        self,
        conn: SqlConnection,
        project_name: str,
        file_rels: dict[str, list[ParsedRelationship]],
        new_file_paths: set[str],
    ) -> None:
        delete_fps = [fp for fp in file_rels if fp not in new_file_paths]
        for chunk in _chunks(delete_fps):
            if not chunk:
                continue
            placeholders = ",".join("?" * len(chunk))
            await conn.execute(
                f"DELETE FROM edges WHERE from_uid IN "
                f"(SELECT uid FROM nodes WHERE project_name = ? AND file_path IN ({placeholders}) "
                f"AND labels NOT IN ('Package', 'Project')) "
                f"AND NOT ({_CITATION_EDGE_PREDICATE}) "
                f"AND NOT ({_CROSS_FILE_DEFINES_PREDICATE})",
                (project_name, *chunk),
            )
        all_rels: list[ParsedRelationship] = [r for rels in file_rels.values() for r in rels]
        await self._create_relationships(conn, project_name, all_rels)

    # -- Project / package -----------------------------------------------------

    async def merge_project_node(self, project_name: str, **metadata: Any) -> None:
        conn = await self._get_conn()
        await conn.execute(
            f"INSERT INTO nodes({_NODE_COLUMNS}) "
            "VALUES (?, 'Project', ?, NULL, NULL, ?, NULL, NULL, ?) "
            "ON CONFLICT(uid) DO UPDATE SET project_name = excluded.project_name, name = excluded.name, "
            "props_json = json_patch(nodes.props_json, excluded.props_json)",
            (project_name, project_name, project_name, _dumps(metadata)),
        )
        await conn.commit()

    async def update_project_metadata(self, project_name: str, **metadata: Any) -> None:
        if not metadata:
            return
        conn = await self._get_conn()
        await conn.execute(
            "UPDATE nodes SET props_json = json_patch(props_json, ?) WHERE uid = ? AND labels = 'Project'",
            (_dumps(metadata), project_name),
        )
        await conn.commit()

    async def get_project_status(self, project_name: str | None = None) -> list[dict[str, Any]]:
        conn = await self._get_conn()
        if project_name:
            cur = await conn.execute(
                f"SELECT {_NODE_COLUMNS} FROM nodes WHERE labels = 'Project' AND uid = ?", (project_name,)
            )
        else:
            cur = await conn.execute(f"SELECT {_NODE_COLUMNS} FROM nodes WHERE labels = 'Project'")
        rows = await cur.fetchall()
        await cur.close()
        return [{"n": _row_to_node(row)} for row in rows]

    async def get_project_git_hash(self, project_name: str) -> str | None:
        conn = await self._get_conn()
        cur = await conn.execute("SELECT props_json FROM nodes WHERE labels = 'Project' AND uid = ?", (project_name,))
        row = await cur.fetchone()
        await cur.close()
        if not row:
            return None
        props = json.loads(row[0]) if row[0] else {}
        return props.get("git_hash")

    async def get_project_extraction_key(self, project_name: str) -> str | None:
        conn = await self._get_conn()
        cur = await conn.execute("SELECT props_json FROM nodes WHERE labels = 'Project' AND uid = ?", (project_name,))
        row = await cur.fetchone()
        await cur.close()
        if not row:
            return None
        props = json.loads(row[0]) if row[0] else {}
        return props.get("extraction_key")

    async def get_project_file_paths(self, project_name: str) -> set[str]:
        conn = await self._get_conn()
        cur = await conn.execute(
            # ResourceFile/EnvVar carry a file_path they only *reference*, never one this
            # project indexed. Counting them makes every referenced data file look like a
            # deleted source file on the next delta index. Mirrors GraphClient.
            "SELECT DISTINCT file_path FROM nodes WHERE project_name = ? "
            "AND labels NOT IN ('Project', 'SchemaVersion', 'ResourceFile', 'EnvVar') "
            "AND file_path IS NOT NULL",
            (project_name,),
        )
        rows = await cur.fetchall()
        await cur.close()
        return {r[0] for r in rows if r[0]}

    async def count_entities(self, project_name: str) -> int:
        conn = await self._get_conn()
        cur = await conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE project_name = ? "
            "AND labels IN ('Module', 'TypeDef', 'Callable', 'Value', 'Package')",
            (project_name,),
        )
        row = await cur.fetchone()
        await cur.close()
        return row[0] if row else 0

    async def count_project_data(self, project_name: str) -> list[dict[str, Any]]:
        """Mirror of ``GraphClient.count_project_data`` — see that docstring.

        Reads the ``project_name`` COLUMN, the way ``delete_project_data`` does — never
        ``json_extract(props_json, '$.project_name')``, a key no writer here ever sets.

        ``substr(...) = ?`` rather than ``LIKE ?`` for the prefix half: ``_`` and ``%``
        are LIKE wildcards and both are ordinary characters in a project name, so LIKE
        would claim a wider scope than Cypher's ``STARTS WITH`` reaches. Same reason
        ``_prefix_clause`` exists.
        """
        conn = await self._get_conn()
        scope = (project_name, len(project_name) + 1, f"{project_name}/")
        cur = await conn.execute(
            "SELECT project_name, COUNT(*), "
            f"SUM(CASE WHEN embedding IS NOT NULL AND labels <> '{NodeLabel.EMBED_CHUNK.value}' THEN 1 ELSE 0 END), "
            f"SUM(CASE WHEN labels = '{NodeLabel.EMBED_CHUNK.value}' THEN 1 ELSE 0 END) "
            "FROM nodes WHERE project_name = ? OR substr(project_name, 1, ?) = ? "
            "GROUP BY project_name",
            scope,
        )
        node_rows = await cur.fetchall()
        await cur.close()
        # Two indexed joins UNIONed rather than one `ON p.uid = e.from_uid OR p.uid =
        # e.to_uid`, which would use neither edge index. UNION (not UNION ALL) dedupes
        # the (project, edge) tuple, so an edge internal to one project is counted once
        # -- the SQL equivalent of the Cypher WITH DISTINCT.
        cur = await conn.execute(
            "SELECT name, COUNT(*) FROM ("
            "SELECT DISTINCT p.project_name AS name, e.from_uid AS f, e.to_uid AS t, e.rel_type AS rt "
            "FROM edges e JOIN nodes p ON p.uid = e.from_uid "
            "WHERE p.project_name = ? OR substr(p.project_name, 1, ?) = ? "
            "UNION "
            "SELECT DISTINCT p.project_name, e.from_uid, e.to_uid, e.rel_type "
            "FROM edges e JOIN nodes p ON p.uid = e.to_uid "
            "WHERE p.project_name = ? OR substr(p.project_name, 1, ?) = ?"
            ") GROUP BY name",
            (*scope, *scope),
        )
        edge_rows = await cur.fetchall()
        await cur.close()
        rels = {r[0]: r[1] for r in edge_rows}
        return _project_data_rows(
            [
                {"name": n, "nodes": c, "relationships": rels.get(n, 0), "embedded_nodes": emb, "embed_chunks": ch}
                for n, c, emb, ch in node_rows
            ],
            project_name,
        )

    async def delete_project_data(self, project_name: str) -> None:
        conn = await self._get_conn()
        cur = await conn.execute("SELECT uid FROM nodes WHERE project_name = ?", (project_name,))
        rows = await cur.fetchall()
        await cur.close()
        uids = [r[0] for r in rows]
        if uids:
            await self._cleanup_search_side_tables(conn, uids)
            for chunk in _chunks(uids):
                if not chunk:
                    continue
                placeholders = ",".join("?" * len(chunk))
                await conn.execute(
                    f"DELETE FROM edges WHERE from_uid IN ({placeholders}) OR to_uid IN ({placeholders})",
                    (*chunk, *chunk),
                )
        await conn.execute("DELETE FROM nodes WHERE project_name = ?", (project_name,))
        await conn.commit()

    async def _get_batch_file_prop(self, project_name: str, file_paths: list[str], prop: str) -> dict[str, str | None]:
        """Read one gate property (``file_hash`` / ``rels_hash``) off the file-level nodes."""
        conn = await self._get_conn()
        result: dict[str, str | None] = dict.fromkeys(file_paths)
        for chunk in _chunks(file_paths):
            if not chunk:
                continue
            placeholders = ",".join("?" * len(chunk))
            cur = await conn.execute(
                f"SELECT file_path, props_json FROM nodes WHERE project_name = ? AND file_path IN ({placeholders}) "
                f"AND labels IN ({_FILE_HASH_LABELS_SQL})",
                (project_name, *chunk),
            )
            rows = await cur.fetchall()
            await cur.close()
            for fp, props_json in rows:
                props = json.loads(props_json) if props_json else {}
                result[fp] = props.get(prop)
        return result

    async def _set_batch_file_prop(self, project_name: str, values: dict[str, str], prop: str) -> None:
        """Write one gate property onto the file-level nodes."""
        conn = await self._get_conn()
        for fp, v in values.items():
            await conn.execute(
                "UPDATE nodes SET props_json = json_patch(props_json, ?) "
                f"WHERE project_name = ? AND file_path = ? AND labels IN ({_FILE_HASH_LABELS_SQL})",
                (_dumps({prop: v}), project_name, fp),
            )
        await conn.commit()

    async def get_batch_file_hashes(self, project_name: str, file_paths: list[str]) -> dict[str, str | None]:
        if not file_paths:
            return {}
        return await self._get_batch_file_prop(project_name, file_paths, "file_hash")

    async def set_batch_file_hashes(self, project_name: str, file_hashes: dict[str, str]) -> None:
        if not file_hashes:
            return
        await self._set_batch_file_prop(project_name, file_hashes, "file_hash")

    async def get_batch_rels_hashes(self, project_name: str, file_paths: list[str]) -> dict[str, str | None]:
        if not file_paths:
            return {}
        return await self._get_batch_file_prop(project_name, file_paths, "rels_hash")

    async def set_batch_rels_hashes(self, project_name: str, rels_hashes: dict[str, str]) -> None:
        if not rels_hashes:
            return
        await self._set_batch_file_prop(project_name, rels_hashes, "rels_hash")

    async def merge_package_node(self, project_name: str, qualified_name: str, name: str, file_path: str) -> None:
        conn = await self._get_conn()
        uid = f"{project_name}:{qualified_name}"
        await conn.execute(
            f"INSERT INTO nodes({_NODE_COLUMNS}) "
            "VALUES (?, 'Package', ?, ?, ?, ?, NULL, NULL, '{}') "
            "ON CONFLICT(uid) DO UPDATE SET project_name = excluded.project_name, "
            "qualified_name = excluded.qualified_name, file_path = excluded.file_path, name = excluded.name",
            (uid, project_name, qualified_name, file_path, name),
        )
        await conn.commit()

    async def merge_package_batch(self, project_name: str, packages: list[tuple[str, str, str]]) -> None:
        if not packages:
            return
        conn = await self._get_conn()
        edges = []
        for qn, name, fp in packages:
            uid = f"{project_name}:{qn}"
            parent_qn = qn.rsplit(".", 1)[0] if "." in qn else None
            parent_uid = f"{project_name}:{parent_qn}" if parent_qn else project_name
            await conn.execute(
                f"INSERT INTO nodes({_NODE_COLUMNS}) "
                "VALUES (?, 'Package', ?, ?, ?, ?, NULL, NULL, '{}') "
                "ON CONFLICT(uid) DO UPDATE SET project_name = excluded.project_name, "
                "qualified_name = excluded.qualified_name, file_path = excluded.file_path, name = excluded.name",
                (uid, project_name, qn, fp, name),
            )
            edges.append((parent_uid, uid))
        await conn.executemany(
            "INSERT OR IGNORE INTO edges(from_uid, to_uid, rel_type, props_json) VALUES (?, ?, 'CONTAINS', '{}')", edges
        )
        await conn.commit()

    async def create_contains_edge(self, from_uid: str, to_uid: str) -> None:
        conn = await self._get_conn()
        await conn.execute(
            "INSERT OR IGNORE INTO edges(from_uid, to_uid, rel_type, props_json) VALUES (?, ?, 'CONTAINS', '{}')",
            (from_uid, to_uid),
        )
        await conn.commit()

    # -- Cross-file resolution ----------------------------------------------------

    async def _build_call_lookup(self, project_name: str) -> _CallLookup:
        conn = await self._get_conn()
        cur = await conn.execute(
            "SELECT uid, name, file_path, props_json FROM nodes WHERE labels = 'Callable' AND project_name = ?",
            (project_name,),
        )
        rows = await cur.fetchall()
        await cur.close()
        name_to_callables: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        uid_to_info: dict[str, tuple[str, str]] = {}
        for uid, name, raw_fp, props_json in rows:
            props = json.loads(props_json) if props_json else {}
            vis = props.get("visibility") or "public"
            fp = raw_fp or ""
            name_to_callables[name].append((uid, fp, vis))
            uid_to_info[uid] = (name, fp)

        import_map: dict[str, dict[str, str]] = defaultdict(dict)
        cur = await conn.execute(
            "SELECT e.from_uid, t.name, t.uid FROM edges e "
            "JOIN nodes m ON m.uid = e.from_uid AND m.project_name = ? AND m.labels IN ('Module', 'Package') "
            "JOIN nodes t ON t.uid = e.to_uid "
            "WHERE e.rel_type = 'IMPORTS'",
            (project_name,),
        )
        rows = await cur.fetchall()
        await cur.close()
        for from_uid, name, uid in rows:
            import_map[from_uid][name] = uid

        caller_to_parent: dict[str, str] = {}
        parent_children: dict[str, list[str]] = defaultdict(list)
        cur = await conn.execute(
            "SELECT e.from_uid, e.to_uid FROM edges e "
            "JOIN nodes a ON a.uid = e.from_uid AND a.labels = 'TypeDef' AND a.project_name = ? "
            "JOIN nodes b ON b.uid = e.to_uid AND b.labels = 'Callable' "
            "WHERE e.rel_type = 'DEFINES'",
            (project_name,),
        )
        rows = await cur.fetchall()
        await cur.close()
        for td_uid, c_uid in rows:
            caller_to_parent[c_uid] = td_uid
            parent_children[td_uid].append(c_uid)

        return _CallLookup(
            name_to_callables=dict(name_to_callables),
            import_map=dict(import_map),
            caller_to_parent=caller_to_parent,
            parent_children=dict(parent_children),
            uid_to_info=uid_to_info,
        )

    async def _name_to_typedefs(self, project_name: str) -> dict[str, list[tuple[str, str]]]:
        conn = await self._get_conn()
        cur = await conn.execute(
            "SELECT name, uid, file_path FROM nodes WHERE labels = 'TypeDef' AND project_name = ?", (project_name,)
        )
        rows = await cur.fetchall()
        await cur.close()
        result: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for name, uid, fp in rows:
            result[name].append((uid, fp or ""))
        return dict(result)

    async def build_resolution_lookup(self, project_name: str) -> tuple[_CallLookup, dict[str, list[tuple[str, str]]]]:
        lookup = await self._build_call_lookup(project_name)
        name_to_typedefs = await self._name_to_typedefs(project_name)
        return lookup, name_to_typedefs

    async def resolve_imports(self, project_name: str, import_rels: list[ParsedRelationship]) -> ReplayableRels:
        """Simplified vs. ``GraphClient.resolve_imports`` — no Python dotted-prefix
        fallback for re-exported names. The exact match, the ``cmp.`` alias widening and
        the Salesforce case fold all behave identically to the Memgraph side.

        Returns the rels with no exact in-project match, for the caller to retry
        once later batches have upserted more of the project.
        """
        if not import_rels:
            return ReplayableRels()
        conn = await self._get_conn()
        cur = await conn.execute(
            "SELECT qualified_name, uid FROM nodes WHERE project_name = ? "
            "AND labels NOT IN ('ExternalPackage', 'ExternalSymbol', 'EnvVar', 'ResourceFile', "
            "'SchemaVersion', 'Project')",
            (project_name,),
        )
        rows = await cur.fetchall()
        await cur.close()
        internal_map = {qn: uid for qn, uid in rows if qn}
        folded_map = build_case_folded_map((qn, uid) for qn, uid in rows)

        import_edges: list[tuple[str, str, bool]] = []
        ext_packages: dict[str, dict[str, str]] = {}
        ext_symbols: dict[str, dict[str, str]] = {}
        inexact: list[ParsedRelationship] = []

        for rel in import_rels:
            to_name = rel.to_name
            from_uid = rel.from_qualified_name
            is_type_only = bool(rel.properties.get("type_only", False))
            target_uid = internal_map.get(to_name) or resolve_component_alias(to_name, internal_map, folded_map)
            if target_uid is not None:
                import_edges.append((from_uid, target_uid, is_type_only))
                continue
            inexact.append(rel)
            # After `inexact.append`, matching the Memgraph side: a folded match is a
            # guess a later batch could improve on, so the rel stays replayable.
            folded_uid = folded_map.get(to_name.lower())
            if folded_uid is not None:
                import_edges.append((from_uid, folded_uid, is_type_only))
                continue

            # Mirror of GraphClient.resolve_imports: an atomic name is the package
            # name whole, rather than the part before the first dot.
            top_level = to_name if rel.properties.get(IMPORT_ATOMIC_NAME) else to_name.split(".")[0]
            if not top_level:
                continue
            pkg_uid = f"{project_name}:ext/{top_level}"
            ext_packages.setdefault(top_level, {"uid": pkg_uid, "name": top_level, "qn": f"ext/{top_level}"})

            if to_name == top_level:
                import_edges.append((from_uid, pkg_uid, is_type_only))
            else:
                sym_uid = f"{project_name}:ext/{to_name}"
                sym_name = to_name.rsplit(".", 1)[-1]
                ext_symbols.setdefault(
                    to_name, {"uid": sym_uid, "name": sym_name, "qn": f"ext/{to_name}", "package": top_level}
                )
                import_edges.append((from_uid, sym_uid, is_type_only))

        for pkg in ext_packages.values():
            await conn.execute(
                f"INSERT INTO nodes({_NODE_COLUMNS}) "
                "VALUES (?, 'ExternalPackage', ?, ?, NULL, ?, NULL, NULL, '{}') ON CONFLICT(uid) DO NOTHING",
                (pkg["uid"], project_name, pkg["qn"], pkg["name"]),
            )
        for sym in ext_symbols.values():
            await conn.execute(
                f"INSERT INTO nodes({_NODE_COLUMNS}) "
                "VALUES (?, 'ExternalSymbol', ?, ?, NULL, ?, NULL, NULL, ?) ON CONFLICT(uid) DO NOTHING",
                (sym["uid"], project_name, sym["qn"], sym["name"], _dumps({"package": sym["package"]})),
            )
            await conn.execute(
                "INSERT OR IGNORE INTO edges(from_uid, to_uid, rel_type, props_json) VALUES (?, ?, 'CONTAINS', '{}')",
                (f"{project_name}:ext/{sym['package']}", sym["uid"]),
            )
        for from_uid, to_uid, type_only in import_edges:
            props = _dumps({"type_only": True}) if type_only else "{}"
            await conn.execute(
                "INSERT INTO edges(from_uid, to_uid, rel_type, props_json) VALUES (?, ?, 'IMPORTS', ?) "
                "ON CONFLICT(from_uid, to_uid, rel_type) DO UPDATE SET props_json = excluded.props_json",
                (from_uid, to_uid, props),
            )
        await conn.commit()
        logger.debug(
            "Resolved {} imports ({} packages, {} symbols created, {} inexact)",
            len(import_rels),
            len(ext_packages),
            len(ext_symbols),
            len(inexact),
        )
        return ReplayableRels(stale_candidates=inexact)

    async def resolve_config_refs(self, project_name: str, ref_rels: list[ParsedRelationship]) -> None:
        """Full-parity port of ``GraphClient.resolve_config_refs``.

        Shares ``_plan_config_refs`` verbatim, so uid construction, path
        normalization and the names-only property allowlist cannot drift from
        the Memgraph path. ``DO NOTHING`` on conflict mirrors ``ON CREATE SET``.
        Also writes the FTS row that Memgraph's label-wide text index gives for
        free (``_sync_fts_row`` only runs for parser-produced entities, and
        these nodes are never parser-produced).
        """
        if not ref_rels:
            return
        conn = await self._get_conn()
        plan = _plan_config_refs(project_name, ref_rels)

        for label, nodes in (
            (NodeLabel.ENV_VAR.value, plan.env_nodes),
            (NodeLabel.RESOURCE_FILE.value, plan.file_nodes),
        ):
            for node in nodes.values():
                await conn.execute(
                    f"INSERT INTO nodes({_NODE_COLUMNS}) "
                    "VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, '{}') "
                    "ON CONFLICT(uid) DO UPDATE SET file_path = excluded.file_path",
                    (
                        node["uid"],
                        label,
                        node["project_name"],
                        node["qualified_name"],
                        node.get("file_path"),
                        node["name"],
                    ),
                )
                if label in _TEXT_LABEL_VALUES:
                    table = f"text_{label.lower()}"
                    text = f"{node['name']} {node['qualified_name']}"
                    await self._safe_exec(conn, f"DELETE FROM {table} WHERE uid = ?", (node["uid"],))
                    await self._safe_exec(conn, f"INSERT INTO {table}(uid, text) VALUES (?, ?)", (node["uid"], text))

        for from_uid, to_uid, rel_type in plan.edges:
            await conn.execute(
                "INSERT OR IGNORE INTO edges(from_uid, to_uid, rel_type, props_json) VALUES (?, ?, ?, '{}')",
                (from_uid, to_uid, rel_type),
            )
        await conn.commit()
        logger.debug(
            "Resolved {} config refs ({} env vars, {} resource files)",
            len(ref_rels),
            len(plan.env_nodes),
            len(plan.file_nodes),
        )

    async def gc_orphaned_reference_nodes(self) -> int:
        """Mirror of ``GraphClient.gc_orphaned_reference_nodes`` — see it for why
        incoming-edge count is a reference count and when this is safe to run.

        ``NOT EXISTS`` over ``ix_edges_to`` keeps the per-label pass an index
        probe rather than a table scan.
        """
        conn = await self._get_conn()
        total = 0
        for label in sorted(_REFERENCE_COUNTED_LABELS, key=lambda lbl: lbl.value):
            cur = await conn.execute(
                "SELECT uid FROM nodes n WHERE n.labels = ? "
                "AND NOT EXISTS (SELECT 1 FROM edges e WHERE e.to_uid = n.uid)",
                (label.value,),
            )
            rows = await cur.fetchall()
            await cur.close()
            uids = [r[0] for r in rows]
            if not uids:
                continue
            for chunk in _chunks(uids):
                if not chunk:
                    continue
                placeholders = ",".join("?" * len(chunk))
                await conn.execute(f"DELETE FROM nodes WHERE uid IN ({placeholders})", chunk)
                # Orphans have no incoming edges by definition, but they may
                # still own outgoing ones if a future schema gives them any —
                # mirror Cypher's DETACH rather than leaving dangling rows.
                await conn.execute(f"DELETE FROM edges WHERE from_uid IN ({placeholders})", chunk)
                if label.value in _TEXT_LABEL_VALUES:
                    await self._safe_exec(
                        conn, f"DELETE FROM text_{label.value.lower()} WHERE uid IN ({placeholders})", chunk
                    )
            total += len(uids)
        if total:
            await conn.commit()
            logger.debug("GC swept {} orphaned reference node(s)", total)
        return total

    async def resolve_calls(
        self,
        project_name: str,
        call_rels: list[ParsedRelationship],
        *,
        lookup: _CallLookup | None = None,
        name_to_typedefs: dict[str, list[tuple[str, str]]] | None = None,
        test_patterns: Sequence[str] | None = None,
    ) -> ReplayableRels:
        """Full-parity port — reuses ``_resolve_one_call`` (all 5 matching strategies),
        ``_combine_call_edge_facts`` and ``_call_edge_weight`` from ``graph.client``
        verbatim; only the lookup-building queries are SQL. Writes the same five
        edge properties as the Memgraph path (``confidence``, ``strategy``,
        ``candidate_count``, ``from_test``, ``weight``).

        Returns the unmatched rels, for the caller to retry on a later flush.
        """
        if not call_rels:
            return ReplayableRels()
        conn = await self._get_conn()
        if lookup is None:
            lookup = await self._build_call_lookup(project_name)
        if name_to_typedefs is None:
            name_to_typedefs = await self._name_to_typedefs(project_name)

        patterns = list(_DEFAULT_TEST_PATTERNS if test_patterns is None else test_patterns)
        test_callables = _test_callable_uids(lookup, patterns)
        caller_is_test: dict[str, bool] = {}
        edges: dict[tuple[str, str], _CallEdgeFacts] = {}
        resolved = ambiguous = 0
        replay = ReplayableRels()
        for rel in call_rels:
            result = _resolve_one_call(project_name, rel, lookup, name_to_typedefs, test_callables)
            if result is None:
                replay.unresolved.append(rel)
                continue
            candidate_uids, strategy = result
            if strategy not in _FILE_LOCAL_STRATEGIES:
                replay.stale_candidates.append(rel)
            # A lone candidate is not enough on its own: an unverified receiver yields
            # exactly one name match and still cannot be trusted, so candidate_count 1
            # with confidence "ambiguous" is a real and informative combination.
            confidence = (
                "resolved" if len(candidate_uids) == 1 and strategy not in _UNVERIFIED_STRATEGIES else "ambiguous"
            )
            resolved += confidence == "resolved"
            ambiguous += confidence == "ambiguous"
            caller_uid = rel.from_qualified_name
            from_test = caller_is_test.get(caller_uid)
            if from_test is None:
                caller_name, caller_fp = lookup.uid_to_info.get(caller_uid, ("", ""))
                from_test = matches_test_pattern(caller_fp, caller_name, patterns)
                caller_is_test[caller_uid] = from_test
            site_line = rel.properties.get("line")
            observed = _CallEdgeFacts(
                confidence,
                strategy,
                len(candidate_uids),
                from_test,
                site_line if isinstance(site_line, int) else None,
            )
            for target_uid in candidate_uids:
                key = (caller_uid, target_uid)
                prior = edges.get(key)
                edges[key] = observed if prior is None else _combine_call_edge_facts(prior, observed)

        if edges:
            rows = [
                (
                    f,
                    t,
                    "CALLS",
                    _dumps(
                        {
                            "confidence": facts.confidence,
                            "strategy": facts.strategy,
                            "candidate_count": facts.candidate_count,
                            "from_test": facts.from_test,
                            "weight": _call_edge_weight(facts.candidate_count, facts.from_test, facts.strategy),
                            "line": facts.line,
                            "site_count": facts.site_count,
                        }
                    ),
                )
                for (f, t), facts in edges.items()
            ]
            await conn.executemany(
                "INSERT INTO edges(from_uid, to_uid, rel_type, props_json) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(from_uid, to_uid, rel_type) DO UPDATE SET props_json = excluded.props_json",
                rows,
            )
            await conn.commit()
        logger.debug(
            "Resolved {} CALLS edges ({} ambiguous, {} unresolved)", resolved, ambiguous, len(replay.unresolved)
        )
        return replay

    async def build_anchor_lookup(self) -> _AnchorLookup:
        conn = await self._get_conn()
        cur = await conn.execute(
            "SELECT project_name, file_path, uid, content_hash FROM nodes WHERE labels IN ('Module', 'DocFile', 'Note')"
        )
        rows = await cur.fetchall()
        await cur.close()
        file_by_path: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
        for proj, fp, uid, chash in rows:
            file_by_path[proj][fp].append((uid, chash or ""))

        cur = await conn.execute(
            "SELECT project_name, file_path, name, uid, content_hash FROM nodes "
            "WHERE labels IN ('Callable', 'TypeDef', 'Value')"
        )
        rows = await cur.fetchall()
        await cur.close()
        symbols_by_path: dict[str, dict[str, dict[str, list[tuple[str, str]]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for proj, fp, name, uid, chash in rows:
            symbols_by_path[proj][fp or ""][name].append((uid, chash or ""))

        cur = await conn.execute("SELECT uid, props_json FROM nodes WHERE labels = 'Project'")
        rows = await cur.fetchall()
        await cur.close()
        project_roots: dict[str, str] = {}
        for uid, props_json in rows:
            props = json.loads(props_json) if props_json else {}
            root = props.get("root_path")
            if root:
                project_roots[uid] = root

        return _AnchorLookup(
            file_by_path={p: dict(v) for p, v in file_by_path.items()},
            symbols_by_path={p: {fp: dict(names) for fp, names in v.items()} for p, v in symbols_by_path.items()},
            project_roots=project_roots,
        )

    async def resolve_anchors(  # noqa: PLR0912
        self, anchor_rels: list[ParsedRelationship], *, lookup: _AnchorLookup | None = None
    ) -> None:
        """Reuses ``_resolve_one_path_anchor`` from ``graph.client`` verbatim."""
        if not anchor_rels:
            return
        conn = await self._get_conn()
        if lookup is None:
            lookup = await self.build_anchor_lookup()

        resolved: list[tuple[str, str, str]] = []
        unresolved_by_note: dict[str, list[str]] = defaultdict(list)
        uid_form: list[ParsedRelationship] = []

        for rel in anchor_rels:
            if rel.properties.get("anchor_form", "") == "uid":
                uid_form.append(rel)
                continue
            target = _resolve_one_path_anchor(rel, lookup)
            if target is None:
                raw = rel.properties.get("anchor_raw", rel.to_name)
                unresolved_by_note[rel.from_qualified_name].append(raw)
                continue
            file_uid, file_hash = target
            resolved.append((rel.from_qualified_name, file_uid, file_hash))

        if uid_form:
            target_uids = list({r.to_name for r in uid_form})
            hash_by_uid: dict[str, str] = {}
            for chunk in _chunks(target_uids):
                if not chunk:
                    continue
                placeholders = ",".join("?" * len(chunk))
                cur = await conn.execute(f"SELECT uid, content_hash FROM nodes WHERE uid IN ({placeholders})", chunk)
                rows = await cur.fetchall()
                await cur.close()
                hash_by_uid.update({u: (h or "") for u, h in rows})
            for rel in uid_form:
                found = hash_by_uid.get(rel.to_name)
                if found is None:
                    raw = rel.properties.get("anchor_raw", rel.to_name)
                    unresolved_by_note[rel.from_qualified_name].append(raw)
                else:
                    resolved.append((rel.from_qualified_name, rel.to_name, found))

        for from_uid, to_uid, to_hash in resolved:
            await conn.execute(
                "INSERT INTO edges(from_uid, to_uid, rel_type, props_json) VALUES (?, ?, 'DOCUMENTS', ?) "
                "ON CONFLICT(from_uid, to_uid, rel_type) DO UPDATE SET props_json = excluded.props_json",
                (
                    from_uid,
                    to_uid,
                    _dumps({"link_type": "anchor", "confidence": 1.0, "anchor_hash": to_hash, "stale": False}),
                ),
            )

        all_notes = {rel.from_qualified_name for rel in anchor_rels}
        for note_uid in all_notes:
            await conn.execute(
                "UPDATE nodes SET props_json = json_patch(props_json, ?) WHERE uid = ? AND labels = 'Note'",
                (_dumps({"unresolved_anchors": unresolved_by_note.get(note_uid, [])}), note_uid),
            )
        await conn.commit()
        total_unresolved = sum(len(v) for v in unresolved_by_note.values())
        logger.debug("Resolved {} anchor edges ({} unresolved)", len(resolved), total_unresolved)

    async def invalidate_stale_anchors(self, changed_uids: set[str]) -> int:
        if not changed_uids:
            return 0
        conn = await self._get_conn()
        count = 0
        for chunk in _chunks(list(changed_uids)):
            if not chunk:
                continue
            placeholders = ",".join("?" * len(chunk))
            cur = await conn.execute(
                f"SELECT e.from_uid, e.to_uid, e.props_json, n.content_hash FROM edges e "
                f"JOIN nodes n ON n.uid = e.to_uid "
                f"WHERE e.rel_type = 'DOCUMENTS' AND e.to_uid IN ({placeholders}) "
                f"AND json_extract(e.props_json, '$.link_type') = 'anchor'",
                chunk,
            )
            rows = await cur.fetchall()
            await cur.close()
            for from_uid, to_uid, props_json, content_hash in rows:
                props = json.loads(props_json) if props_json else {}
                if props.get("anchor_hash") != content_hash:
                    props["stale"] = True
                    await conn.execute(
                        "UPDATE edges SET props_json = ? WHERE from_uid = ? AND to_uid = ? AND rel_type = 'DOCUMENTS'",
                        (_dumps(props), from_uid, to_uid),
                    )
                    count += 1
        if count:
            await conn.commit()
            logger.debug("Marked {} anchor edge(s) stale", count)
        return count

    async def build_citation_lookup(self, project_name: str) -> _CitationLookup:
        conn = await self._get_conn()
        cur = await conn.execute(
            "SELECT labels, uid, name, file_path, json_extract(props_json, '$.header_level') FROM nodes "
            "WHERE project_name = ? AND labels IN ('DocFile', 'DocSection', 'Note')",
            (project_name,),
        )
        rows = await cur.fetchall()
        await cur.close()
        by_key: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
        for label, uid, name, file_path, header_level in rows:
            for key, rank in _document_citation_keys(label or "", name or "", file_path or "", header_level):
                by_key[key].append((rank, uid))
        return _CitationLookup(by_key=dict(by_key))

    async def resolve_citations(
        self,
        project_name: str,
        citations_by_uid: dict[str, list[str]],
        *,
        file_paths: Collection[str] | None = None,
        lookup: _CitationLookup | None = None,
        retry_unresolved: bool = False,
    ) -> None:
        """Reuses ``_citation_key``/``_pick_citation_target`` from ``graph.client`` verbatim.

        Edge direction matches Memgraph's — ``(document) -> (citing entity)``,
        see the DIRECTION note in ``graph.client``'s citation section. The one
        behavioural difference is structural: ``edges`` is unique on
        ``(from_uid, to_uid, rel_type)``, so a document that both mentions a
        symbol heuristically *and* is cited by it collapses to a single row,
        with the citation (explicit author intent) overwriting the heuristic
        guess. Memgraph keeps both as parallel edges.

        That collapse also colours the *file_paths* revoke pass (see the
        Memgraph docstring for what it is and why it exists): deleting the one
        collapsed row takes the heuristic link with it, where Memgraph would
        drop only the parallel citation edge. The heuristic link is owned by the
        document's parse and comes back the next time that document is indexed;
        the alternative — remembering a link this schema cannot represent — is
        the divergence getting worse, not better.
        """
        if not citations_by_uid and not retry_unresolved and not file_paths:
            return
        conn = await self._get_conn()

        if file_paths:
            # Non-empty by the guard above, so _chunks never yields its empty
            # placeholder chunk here.
            for chunk in _chunks(list(file_paths)):
                placeholders = ",".join("?" * len(chunk))
                await conn.execute(
                    f"DELETE FROM edges WHERE {_CITATION_EDGE_PREDICATE} AND to_uid IN "
                    f"(SELECT uid FROM nodes WHERE project_name = ? AND file_path IN ({placeholders}))",
                    (project_name, *chunk),
                )
            await conn.commit()

        pending: dict[str, list[str]] = {uid: list(raws) for uid, raws in citations_by_uid.items()}
        if retry_unresolved:
            cur = await conn.execute(
                "SELECT uid, json_extract(props_json, '$.citations') FROM nodes "
                "WHERE project_name = ? AND json_extract(props_json, '$.unresolved_citations') IS NOT NULL "
                "AND json_array_length(props_json, '$.unresolved_citations') > 0",
                (project_name,),
            )
            rows = await cur.fetchall()
            await cur.close()
            for uid, citations_json in rows:
                pending.setdefault(uid, json.loads(citations_json) if citations_json else [])

        if not pending:
            return
        if lookup is None:
            lookup = await self.build_citation_lookup(project_name)

        resolved: list[tuple[str, str, str, float]] = []
        unresolved_by_uid: dict[str, list[str]] = defaultdict(list)
        for entity_uid, raws in pending.items():
            for raw in raws:
                key = _citation_key(raw)
                target = _pick_citation_target(key, lookup) if key is not None else None
                if key is None or target is None or target[0] == entity_uid:
                    unresolved_by_uid[entity_uid].append(raw)
                    continue
                doc_uid, confidence = target
                resolved.append((doc_uid, entity_uid, _render_citation_key(key), confidence))

        for doc_uid, entity_uid, citation, confidence in resolved:
            await conn.execute(
                "INSERT INTO edges(from_uid, to_uid, rel_type, props_json) VALUES (?, ?, 'DOCUMENTS', ?) "
                "ON CONFLICT(from_uid, to_uid, rel_type) DO UPDATE SET props_json = excluded.props_json",
                (
                    doc_uid,
                    entity_uid,
                    _dumps({"link_type": "citation", "confidence": confidence, "citation": citation}),
                ),
            )

        for uid in pending:
            await conn.execute(
                "UPDATE nodes SET props_json = json_patch(props_json, ?) WHERE uid = ?",
                (_dumps({"unresolved_citations": unresolved_by_uid.get(uid, [])}), uid),
            )
        await conn.commit()

        total_unresolved = sum(len(v) for v in unresolved_by_uid.values())
        logger.debug(
            "Resolved {} citation edge(s) for project {} ({} unresolved)",
            len(resolved),
            project_name,
            total_unresolved,
        )

    async def resolve_value_references(self, project_name: str, ref_rels: list[ParsedRelationship]) -> None:
        """Same-file only. The import-scope pass is Memgraph-side: it walks the
        referring module's IMPORTS edges, which this schema has no cheap join for."""
        if not ref_rels:
            return
        conn = await self._get_conn()
        for r in ref_rels:
            # A table lookup or a constant read names a Value; everything else here
            # names a Callable — same split the Memgraph resolver makes.
            label = "Value" if r.properties.get("via") in {"table", "const"} else "Callable"
            cur = await conn.execute(
                "SELECT b.uid FROM nodes b JOIN nodes a ON a.uid = ? "
                "WHERE b.labels = ? AND b.project_name = ? AND b.name = ? "
                "AND b.file_path = a.file_path AND b.uid <> a.uid LIMIT 1",
                (r.from_qualified_name, label, project_name, r.to_name),
            )
            row = await cur.fetchone()
            await cur.close()
            if row:
                await conn.execute(
                    "INSERT OR IGNORE INTO edges(from_uid, to_uid, rel_type, props_json) "
                    "VALUES (?, ?, 'REFERENCES', '{}')",
                    (r.from_qualified_name, row[0]),
                )
        await conn.commit()

    async def resolve_protocol_conformance(self, project_name: str) -> int:
        """Not implemented on the embedded backend — the containment test needs a set
        comparison per (protocol, class) pair, which is a join this schema makes
        expensive. Returns 0 rather than silently claiming there are no protocols."""
        _ = project_name
        return 0

    async def resolve_inherits(self, project_name: str, inherit_rels: list[ParsedRelationship]) -> None:
        """In-project TypeDef wins; otherwise an imported ExternalSymbol of that name."""
        if not inherit_rels:
            return
        conn = await self._get_conn()
        for r in inherit_rels:
            target = None
            for label in ("TypeDef", "ExternalSymbol"):
                cur = await conn.execute(
                    "SELECT uid FROM nodes WHERE labels = ? AND project_name = ? AND name = ?",
                    (label, project_name, r.to_name),
                )
                row = await cur.fetchone()
                await cur.close()
                if row:
                    target = row[0]
                    break
            if target:
                await conn.execute(
                    "INSERT OR IGNORE INTO edges(from_uid, to_uid, rel_type, props_json) "
                    "VALUES (?, ?, 'INHERITS', '{}')",
                    (r.from_qualified_name, target),
                )
        await conn.commit()

    async def resolve_type_refs(  # noqa: PLR0912
        self,
        project_name: str,
        type_rels: list[ParsedRelationship],
        *,
        lookup: _CallLookup | None = None,
        name_to_typedefs: dict[str, list[tuple[str, str]]] | None = None,
    ) -> ReplayableRels:
        if not type_rels:
            return ReplayableRels()
        conn = await self._get_conn()
        if lookup is None:
            lookup = await self._build_call_lookup(project_name)
        if name_to_typedefs is None:
            name_to_typedefs = await self._name_to_typedefs(project_name)

        edges: dict[tuple[str, str], str] = {}
        replay = ReplayableRels()
        for rel in type_rels:
            from_uid = rel.from_qualified_name
            type_name = rel.to_name
            caller_info = lookup.uid_to_info.get(from_uid)
            caller_fp = caller_info[1] if caller_info else ""
            caller_qn = from_uid.split(":", 1)[1] if ":" in from_uid else from_uid
            parts = caller_qn.split(".")
            module_uid: str | None = None
            # Same rule as _resolve_one_call: a source that is not a Callable is the module
            # itself, so its own uid is the import scope rather than one segment up.
            for i in range(len(parts) if from_uid not in lookup.uid_to_info else len(parts) - 1, 0, -1):
                candidate = f"{project_name}:{'.'.join(parts[:i])}"
                if candidate in lookup.import_map:
                    module_uid = candidate
                    break

            target_uid: str | None = None
            strategy = ""
            if module_uid and type_name in lookup.import_map.get(module_uid, {}):
                target_uid = lookup.import_map[module_uid][type_name]
                strategy = "import"
            if target_uid is None and caller_fp:
                for uid, fp in name_to_typedefs.get(type_name, []):
                    if fp == caller_fp:
                        target_uid = uid
                        strategy = "same_file"
                        break
            if target_uid is None:
                candidates = name_to_typedefs.get(type_name, [])
                if len(candidates) == 1:
                    target_uid = candidates[0][0]
                    strategy = "project_unique"
            if target_uid is None:
                replay.unresolved.append(rel)
            else:
                key = (from_uid, target_uid)
                prior = edges.get(key)
                if prior is None or _TYPE_REF_RANK.index(strategy) < _TYPE_REF_RANK.index(prior):
                    edges[key] = strategy
                if strategy != "same_file":
                    replay.stale_candidates.append(rel)

        if edges:
            rows = [
                (
                    f,
                    t,
                    "USES_TYPE",
                    _dumps(
                        {
                            "strategy": st,
                            "confidence": _TYPE_REF_FACTS[st][0],
                            "weight": _TYPE_REF_FACTS[st][1],
                        }
                    ),
                )
                for (f, t), st in edges.items()
            ]
            await conn.executemany(
                "INSERT OR IGNORE INTO edges(from_uid, to_uid, rel_type, props_json) VALUES (?, ?, ?, ?)", rows
            )
            await conn.commit()
        logger.debug("Resolved {} USES_TYPE edges ({} unresolved)", len(edges), len(replay.unresolved))
        return replay

    async def resolve_member_defines(  # noqa: PLR0912
        self,
        project_name: str,
        member_rels: list[ParsedRelationship],
        *,
        lookup: _CallLookup | None = None,
        name_to_typedefs: dict[str, list[tuple[str, str]]] | None = None,
    ) -> None:
        if not member_rels:
            return
        conn = await self._get_conn()
        if lookup is None:
            lookup = await self._build_call_lookup(project_name)
        if name_to_typedefs is None:
            name_to_typedefs = await self._name_to_typedefs(project_name)

        type_edges: set[tuple[str, str]] = set()
        module_edges: set[tuple[str, str]] = set()
        for rel in member_rels:
            member_uid = rel.to_name
            type_name = rel.properties.get("parent_type_name", "")
            member_info = lookup.uid_to_info.get(member_uid)
            member_fp = member_info[1] if member_info else ""
            member_dir = member_fp.rsplit("/", 1)[0] if "/" in member_fp else ""

            candidates = name_to_typedefs.get(type_name, []) if type_name else []
            same_file = [uid for uid, fp in candidates if fp == member_fp]
            same_dir = [uid for uid, fp in candidates if (fp.rsplit("/", 1)[0] if "/" in fp else "") == member_dir]

            target_uid: str | None = None
            if len(same_file) == 1:
                target_uid = same_file[0]
            elif not same_file and len(same_dir) == 1:
                target_uid = same_dir[0]
            elif (
                not same_file
                and not same_dir
                and rel.properties.get("parent_scope") != "package"
                and len(candidates) == 1
            ):
                target_uid = candidates[0][0]

            if target_uid is not None:
                type_edges.add((target_uid, member_uid))
            else:
                module_edges.add((rel.from_qualified_name, member_uid))

        member_uids = sorted({rel.to_name for rel in member_rels})
        for chunk in _chunks(member_uids):
            if not chunk:
                continue
            placeholders = ",".join("?" * len(chunk))
            await conn.execute(
                f"DELETE FROM edges WHERE rel_type = 'DEFINES' AND to_uid IN ({placeholders}) "
                f"AND from_uid IN (SELECT uid FROM nodes WHERE labels IN ('TypeDef', 'Module'))",
                chunk,
            )

        if type_edges:
            await conn.executemany(
                "INSERT OR IGNORE INTO edges(from_uid, to_uid, rel_type, props_json) VALUES (?, ?, 'DEFINES', '{}')",
                list(type_edges),
            )
        if module_edges:
            await conn.executemany(
                "INSERT OR IGNORE INTO edges(from_uid, to_uid, rel_type, props_json) VALUES (?, ?, 'DEFINES', '{}')",
                list(module_edges),
            )
        await conn.commit()
        logger.debug("Resolved {} member DEFINES edges ({} fell back to module)", len(type_edges), len(module_edges))

    async def classify_external_package_provenance(self, project_name: str) -> dict[str, int]:
        """Mirror of ``GraphClient.classify_external_package_provenance`` -- see that docstring."""
        conn = await self._get_conn()
        placeholders = ",".join("?" * len(STDLIB_MODULE_NAMES))
        stdlib = sorted(STDLIB_MODULE_NAMES)
        await conn.execute(
            "UPDATE nodes SET props_json = json_set(COALESCE(props_json, '{}'), '$.provenance', "
            "  CASE "
            "    WHEN EXISTS (SELECT 1 FROM edges e WHERE e.to_uid = nodes.uid "
            "                 AND e.rel_type = 'DEPENDS_ON' AND e.from_uid = ?) THEN ? "
            f"    WHEN name IN ({placeholders}) THEN ? "
            "    ELSE ? END) "
            "WHERE labels = 'ExternalPackage' AND project_name = ?",
            [
                project_name,
                PROVENANCE_DECLARED,
                *stdlib,
                PROVENANCE_STDLIB,
                PROVENANCE_UNDECLARED,
                project_name,
            ],
        )
        await conn.commit()
        cur = await conn.execute(
            "SELECT json_extract(props_json, '$.provenance') AS provenance, count(*) AS n "
            "FROM nodes WHERE labels = 'ExternalPackage' AND project_name = ? GROUP BY provenance",
            (project_name,),
        )
        rows = await cur.fetchall()
        await cur.close()
        return {provenance: n for provenance, n in rows if provenance}

    async def get_package_dependents(
        self,
        name: str = "",
        *,
        min_projects: int = 1,
        exclude_stdlib: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Mirror of ``GraphClient.get_package_dependents`` -- see that docstring.

        One caveat is specific to this backend and belongs in front of the caller rather
        than here: the embedded graph is one ``.atlas/graph.sqlite3`` per checkout, so
        "across all my repos" reaches only the projects in *this* database. On a monorepo
        that is its sub-projects; on a single repo it is one row. The Memgraph backend is
        the one that spans checkouts.
        """
        conn = await self._get_conn()
        where = "WHERE n.labels = 'ExternalPackage'"
        params: list[Any] = []
        if name:
            where += " AND n.name = ?"
            params.append(name)
        cur = await conn.execute(
            "SELECT n.name AS package, n.project_name AS project, "
            # UNION, not two counts added: a module that does both `import pathlib` and
            # `from pathlib import Path` is one importer. The second arm is the one that
            # matters -- a from-import attaches to the ExternalSymbol, so counting only
            # direct edges reports 0 for a package with dozens of users.
            "  (SELECT count(*) FROM ("
            "     SELECT e.from_uid FROM edges e "
            "       WHERE e.to_uid = n.uid AND e.rel_type = 'IMPORTS' "
            "     UNION "
            "     SELECT i.from_uid FROM edges c JOIN edges i ON i.to_uid = c.to_uid "
            "       WHERE c.from_uid = n.uid AND c.rel_type = 'CONTAINS' AND i.rel_type = 'IMPORTS'"
            "  )) AS import_sites, "
            "  (SELECT json_extract(d.props_json, '$.version') FROM edges d "
            "     WHERE d.to_uid = n.uid AND d.rel_type = 'DEPENDS_ON' AND d.from_uid = n.project_name "
            "     LIMIT 1) AS version "
            f"FROM nodes n {where}",
            params,
        )
        rows = await cur.fetchall()
        await cur.close()

        grouped: dict[str, list[dict[str, Any]]] = {}
        for package, project, import_sites, version in rows:
            grouped.setdefault(package, []).append(
                {"project": project, "version": version, "import_sites": import_sites}
            )
        # Ordered on the typed mapping rather than on the result rows: those hold a
        # str/int/list union, so a key function over them cannot be checked.
        ranked = sorted(
            (
                (pkg, projects)
                for pkg, projects in grouped.items()
                if len(projects) >= min_projects and not (exclude_stdlib and pkg in STDLIB_MODULE_NAMES)
            ),
            key=lambda item: (-len(item[1]), item[0]),
        )
        return [
            {
                "package": pkg,
                "stdlib": pkg in STDLIB_MODULE_NAMES,
                "project_count": len(projects),
                "projects": sorted(projects, key=lambda p: (-(p["import_sites"] or 0), p["project"] or "")),
            }
            for pkg, projects in ranked[:limit]
        ]

    async def upsert_external_stubs(
        self,
        project_name: str,
        package: str,
        symbols: list[dict[str, Any]],
        *,
        version: str = "",
        source: str = "",
    ) -> int:
        """Mirror of ``GraphClient.upsert_external_stubs`` -- see that docstring.

        Two differences, both forced by the unified table. Signature, docstring and the
        `stub` marker live in ``props_json`` rather than as columns, so the upsert uses
        ``json_patch`` to merge them into whatever ``resolve_imports`` already wrote. That
        is defensive rather than load-bearing today -- ``package`` is currently the only
        property on the row and this write restates it -- but a replace would destroy any
        property a later pass adds, silently and only for the embedded backend.
        And the CONTAINS edge is a plain ``INSERT OR IGNORE`` selected out of ``nodes``,
        which reproduces the Cypher's ``MATCH`` on the package: nothing here has foreign
        keys, so an unconditional insert would hang symbols off a package uid that may
        not exist.
        """
        if not symbols:
            return 0
        conn = await self._get_conn()
        pkg_uid = f"{project_name}:ext/{package}"
        await conn.executemany(
            f"INSERT INTO nodes({_NODE_COLUMNS}) "
            "VALUES (?, 'ExternalSymbol', ?, ?, NULL, ?, ?, NULL, ?) "
            "ON CONFLICT(uid) DO UPDATE SET kind = excluded.kind, "
            "props_json = json_patch(nodes.props_json, excluded.props_json)",
            [
                (
                    sym["uid"],
                    project_name,
                    sym["qualified_name"],
                    sym["name"],
                    sym["kind"],
                    _dumps(
                        {
                            "package": package,
                            "signature": sym["signature"],
                            "docstring": sym["docstring"],
                            "stub": True,
                        }
                    ),
                )
                for sym in symbols
            ],
        )
        await conn.executemany(
            "INSERT OR IGNORE INTO edges(from_uid, to_uid, rel_type, props_json) "
            "SELECT ?, ?, 'CONTAINS', '{}' FROM nodes WHERE uid = ? AND labels = 'ExternalPackage'",
            [(pkg_uid, sym["uid"], pkg_uid) for sym in symbols],
        )
        await conn.execute(
            "UPDATE nodes SET props_json = json_patch(props_json, ?) WHERE uid = ? AND labels = 'ExternalPackage'",
            (_dumps({"stub_version": version, "stub_source": source, "stub_symbols": len(symbols)}), pkg_uid),
        )
        await conn.commit()
        return len(symbols)

    async def get_stubbed_package_versions(self, project_name: str) -> dict[str, str]:
        """Mirror of ``GraphClient.get_stubbed_package_versions`` -- see that docstring."""
        conn = await self._get_conn()
        cur = await conn.execute(
            "SELECT name, json_extract(props_json, '$.stub_version') AS version FROM nodes "
            "WHERE labels = 'ExternalPackage' AND project_name = ? "
            "AND json_extract(props_json, '$.stub_version') IS NOT NULL",
            (project_name,),
        )
        return {row[0]: row[1] for row in await cur.fetchall()}

    async def update_external_package_versions(self, project_name: str, versions: dict[str, str]) -> None:
        """Mirror of ``GraphClient.update_external_package_versions`` — the version lives
        on ``Project -[DEPENDS_ON]-> ExternalPackage`` (v18), not on the package node.

        ``INSERT ... SELECT`` rather than ``INSERT ... VALUES``, and that is the whole
        difference from the Cypher. Memgraph gets the "unmatched manifest key is a
        deliberate no-op" guarantee for free from ``MATCH``; here nothing enforces it —
        the edges table has no foreign keys and no ``PRAGMA foreign_keys`` — so a plain
        INSERT would write one Project -> nonexistent-uid row per declared coordinate in
        every Go/Java/PHP project, where the manifest names distributions that map to no
        import root (see indexing/orchestrator.py's manifest-parsing header). Selecting
        the target uid out of ``nodes`` reproduces the MATCH.

        SQLite requires the SELECT of an upsert to carry a WHERE clause to disambiguate
        the ``ON CONFLICT`` from the SELECT's own parse; ours does anyway.

        ``json_patch``, not ``= excluded.props_json``: the merge form (see
        _create_relationships) so a property added to this edge later is not silently
        destroyed by a version write.
        """
        if not versions:
            return
        conn = await self._get_conn()
        # Replace, not accumulate — see the Cypher's docstring. The subselect confines
        # the delete to ExternalPackage targets so it cannot eat create_depends_on_edges'
        # project-to-project rows, which share the rel_type and the from_uid.
        await conn.execute(
            "DELETE FROM edges WHERE rel_type = 'DEPENDS_ON' AND from_uid = ? "
            "AND to_uid IN (SELECT uid FROM nodes WHERE labels = 'ExternalPackage')",
            (project_name,),
        )
        await conn.executemany(
            "INSERT INTO edges(from_uid, to_uid, rel_type, props_json) "
            "SELECT ?, uid, 'DEPENDS_ON', ? FROM nodes WHERE uid = ? AND labels = 'ExternalPackage' "
            "ON CONFLICT(from_uid, to_uid, rel_type) DO UPDATE SET "
            "props_json = json_patch(edges.props_json, excluded.props_json)",
            [(project_name, _dumps({"version": ver}), f"{project_name}:ext/{pkg}") for pkg, ver in versions.items()],
        )
        await conn.commit()

    async def resolve_warehouse_objects(self, project_names: list[str]) -> int:
        """Mirror of ``GraphClient.resolve_warehouse_objects`` -- see that docstring.

        Same three outcomes and the same case-folded join; the difference is only that a
        unified table makes the reads plain SELECTs.
        """
        if not project_names:
            return 0
        conn = await self._get_conn()
        placeholders = ",".join("?" * len(project_names))

        cur = await conn.execute(
            f"SELECT e.from_uid, n.uid, n.qualified_name FROM edges e "
            f"JOIN nodes n ON n.uid = e.to_uid AND n.labels = 'ExternalSymbol' "
            f"JOIN nodes t ON t.uid = e.from_uid "
            f"WHERE e.rel_type = 'IMPORTS' AND t.project_name IN ({placeholders}) "
            f"AND n.qualified_name LIKE ? ESCAPE '\\'",
            (*project_names, f"{_like_literal(_WAREHOUSE_PREFIX)}%"),
        )
        stubs = list(await cur.fetchall())
        await cur.close()
        if not stubs:
            return 0

        kind_slots = ",".join("?" * len(_WAREHOUSE_PRODUCER_KINDS))
        cur = await conn.execute(
            f"SELECT name, uid FROM nodes WHERE labels = 'TypeDef' AND project_name IN ({placeholders}) "
            f"AND kind IN ({kind_slots})",
            (*project_names, *sorted(_WAREHOUSE_PRODUCER_KINDS)),
        )
        by_name: dict[str, set[str]] = defaultdict(set)
        for name, uid in await cur.fetchall():
            if name:
                by_name[name.strip().lower()].add(uid)
        await cur.close()

        written = 0
        async with self._write_lock, _txn():
            for table_uid, stub_uid, qn in stubs:
                obj = qn[len(_WAREHOUSE_PREFIX) :].strip().lower()
                candidates = by_name.get(obj, set())
                if len(candidates) != 1:
                    if candidates:
                        await conn.execute(
                            "UPDATE edges SET props_json = json_set(coalesce(props_json, '{}'), "
                            "'$.confidence', 'ambiguous') "
                            "WHERE from_uid = ? AND to_uid = ? AND rel_type = 'IMPORTS'",
                            (table_uid, stub_uid),
                        )
                    continue
                await conn.execute(
                    "INSERT OR IGNORE INTO edges(from_uid, to_uid, rel_type, props_json) VALUES (?, ?, 'FEEDS', '{}')",
                    (next(iter(candidates)), table_uid),
                )
                await conn.execute(
                    "DELETE FROM edges WHERE from_uid = ? AND to_uid = ? AND rel_type = 'IMPORTS'",
                    (table_uid, stub_uid),
                )
                written += 1

            # Only stubs nothing points at any more: an object read by two tables where
            # only one resolved must keep its stub for the other.
            await conn.execute(
                "DELETE FROM nodes WHERE labels = 'ExternalSymbol' "
                "AND qualified_name LIKE ? ESCAPE '\\' "
                "AND uid NOT IN (SELECT to_uid FROM edges WHERE rel_type = 'IMPORTS')",
                (f"{_like_literal(_WAREHOUSE_PREFIX)}%",),
            )
            await conn.commit()
        return written

    async def resolve_cross_project_imports(self, project_names: list[str]) -> int:
        # The int is a debug figure and deliberately not compared with Memgraph's --
        # this backend counts importer edges actually rewritten, Memgraph counts resolved
        # candidates. See GraphClient.resolve_cross_project_imports for why neither is
        # worth changing. The conformance suite compares the resulting graph instead.
        """Simplified port of ``GraphClient.resolve_cross_project_imports`` —
        per-stub rewiring rather than the bulk read-then-write-phase split.
        """
        if len(project_names) < 2:
            return 0
        conn = await self._get_conn()
        placeholders = ",".join("?" * len(project_names))
        cur = await conn.execute(
            f"SELECT name, project_name, qualified_name FROM nodes "
            f"WHERE labels = 'Package' AND project_name IN ({placeholders})",
            project_names,
        )
        rows = await cur.fetchall()
        await cur.close()
        pkg_to_project: dict[str, str] = {}
        for name, proj, qn in rows:
            top_name = qn.split(".")[0] if qn else name
            pkg_to_project.setdefault(top_name, proj)
        if not pkg_to_project:
            return 0

        cur = await conn.execute(
            f"SELECT name, uid, project_name FROM nodes "
            f"WHERE labels = 'ExternalPackage' AND project_name IN ({placeholders})",
            project_names,
        )
        ext_pkgs = await cur.fetchall()
        await cur.close()

        rewired = 0
        for name, ep_uid, proj in ext_pkgs:
            target_project = pkg_to_project.get(name)
            if target_project is None or target_project == proj:
                continue

            # `name` off the row, not derived from the uid. `es_uid.rsplit("/", 1)[-1]`
            # on `app:ext/shared.thing` yields `shared.thing`, while resolve_imports
            # stores `to_name.rsplit(".", 1)[-1]` -> `thing` -- so the lookup below
            # searched for a node nothing ever names and this arm never matched once.
            # An ExternalSymbol only exists when `to_name != top_level`, i.e. the name is
            # always dotted, so it was unreachable for every input it can receive: on this
            # backend `from sibling_pkg import thing` was never rewired to the real entity.
            # Memgraph carries `es.name` off the node and does not have the bug.
            cur = await conn.execute(
                "SELECT uid, name FROM nodes WHERE labels = 'ExternalSymbol' AND project_name = ? "
                "AND json_extract(props_json, '$.package') = ?",
                (proj, name),
            )
            ext_syms = await cur.fetchall()
            await cur.close()
            for es_uid, es_name in ext_syms:
                cur = await conn.execute(
                    "SELECT uid FROM nodes WHERE project_name = ? AND name = ? "
                    "AND labels NOT IN ('ExternalPackage', 'ExternalSymbol', 'EnvVar', 'ResourceFile', "
                    "'Project', 'SchemaVersion')",
                    (target_project, es_name),
                )
                real = await cur.fetchone()
                await cur.close()
                if real is None:
                    continue
                rewired += await _rewire_stub_importers(conn, stub_uid=es_uid, real_uid=real[0], project=proj)
                await _delete_stub_if_unimported(conn, es_uid)

            cur = await conn.execute(
                "SELECT uid FROM nodes WHERE labels = 'Package' AND project_name = ? AND name = ?",
                (target_project, name),
            )
            real_pkg = await cur.fetchone()
            await cur.close()
            if real_pkg is not None:
                rewired += await _rewire_stub_importers(conn, stub_uid=ep_uid, real_uid=real_pkg[0], project=proj)

            # DEPENDS_ON is excluded from the "is anything still pointing at this stub"
            # count: since v18 the stub's own project points at it with the manifest
            # version, and counting that would keep every rewired stub alive here while
            # Memgraph (which only asks about IMPORTS and CONTAINS) deletes it. The
            # version edge dies with the stub, exactly as the node property used to.
            cur = await conn.execute(
                "SELECT COUNT(*) FROM edges WHERE to_uid = ? AND rel_type <> 'DEPENDS_ON'", (ep_uid,)
            )
            count_row = await cur.fetchone()
            await cur.close()
            if (count_row[0] if count_row else 0) == 0:
                await conn.execute("DELETE FROM nodes WHERE uid = ?", (ep_uid,))
                # The DETACH half of Memgraph's DETACH DELETE. Until v18 nothing pointed
                # at a rewired stub, so dropping the node row alone left nothing behind;
                # the version edge does, and a dangling edge row outlives the node
                # forever because no sweep looks for one.
                await conn.execute("DELETE FROM edges WHERE from_uid = ? OR to_uid = ?", (ep_uid, ep_uid))

        await conn.commit()
        logger.debug(
            "Cross-project import resolution: {} imports rewired across {} projects", rewired, len(project_names)
        )
        return rewired

    async def create_depends_on_edges(self, project_names: list[str]) -> int:
        if len(project_names) < 2:
            return 0
        conn = await self._get_conn()
        placeholders = ",".join("?" * len(project_names))
        # Both endpoints are pinned to labels = 'Project' — that is what keeps this
        # delete disjoint from the Project -> ExternalPackage version edges sharing the
        # rel_type since v18, not an accident of which rows happen to exist.
        await conn.execute(
            f"DELETE FROM edges WHERE rel_type = 'DEPENDS_ON' AND "
            f"from_uid IN (SELECT uid FROM nodes WHERE labels = 'Project' AND project_name IN ({placeholders})) AND "
            f"to_uid IN (SELECT uid FROM nodes WHERE labels = 'Project' AND project_name IN ({placeholders}))",
            (*project_names, *project_names),
        )
        cur = await conn.execute(
            f"SELECT DISTINCT s.project_name, t.project_name FROM edges e "
            f"JOIN nodes s ON s.uid = e.from_uid JOIN nodes t ON t.uid = e.to_uid "
            f"WHERE e.rel_type = 'IMPORTS' AND s.project_name IN ({placeholders}) "
            f"AND t.project_name IN ({placeholders}) AND s.project_name <> t.project_name",
            (*project_names, *project_names),
        )
        rows = list(await cur.fetchall())
        await cur.close()
        if not rows:
            await conn.commit()
            return 0
        for from_proj, to_proj in rows:
            await conn.execute(
                "INSERT OR IGNORE INTO edges(from_uid, to_uid, rel_type, props_json) VALUES (?, ?, 'DEPENDS_ON', '{}')",
                (from_proj, to_proj),
            )
        await conn.commit()
        logger.debug("Created {} DEPENDS_ON edge(s) between projects", len(rows))
        return len(rows)

    async def apply_property_enrichments(self, enrichments: list[PropertyEnrichment]) -> None:
        items = [(e.qualified_name, _dumps(e.properties)) for e in enrichments if e.properties]
        if not items:
            return
        conn = await self._get_conn()
        await conn.executemany(
            "UPDATE nodes SET props_json = json_patch(props_json, ?) WHERE uid = ?",
            [(props, uid) for uid, props in items],
        )
        await conn.commit()

    # -- Embeddings ---------------------------------------------------------------

    async def get_embedding_config(self) -> tuple[str, int] | None:
        conn = await self._get_conn()
        cur = await conn.execute("SELECT key, value FROM meta WHERE key IN ('embedding_model', 'embedding_dimension')")
        rows = {k: v for k, v in await cur.fetchall()}  # noqa: C416 — dict() doesn't type-match Iterable[Row]
        await cur.close()
        model = rows.get("embedding_model")
        dim = rows.get("embedding_dimension")
        if model is None or dim is None:
            return None
        return (model, int(dim))

    async def get_project_embedding_model(self, project: str) -> str | None:
        """Embedding model *project* last indexed under, or ``None``.

        Stored in ``meta`` keyed by project rather than on a Project row: this
        backend is one database file per project root already, but a monorepo puts
        several sub-projects in one file, which is exactly the case a single global
        key gets wrong (ATL-135).
        """
        conn = await self._get_conn()
        cur = await conn.execute("SELECT value FROM meta WHERE key = ?", (f"embedding_model:{project}",))
        row = await cur.fetchone()
        await cur.close()
        return row[0] if row else None

    async def set_project_embedding_model(self, project: str, model: str) -> None:
        conn = await self._get_conn()
        async with self._write_lock, _txn():
            await conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (f"embedding_model:{project}", model),
            )
            await conn.commit()

    async def get_embedding_models_by_project(self) -> dict[str, str]:
        conn = await self._get_conn()
        cur = await conn.execute("SELECT key, value FROM meta WHERE key LIKE 'embedding_model:%'")
        rows = await cur.fetchall()
        await cur.close()
        return {k.split(":", 1)[1]: v for k, v in rows}

    async def set_embedding_config(self, model: str, dimension: int) -> None:
        conn = await self._get_conn()
        await self._upsert_meta(conn, "embedding_model", model)
        await self._upsert_meta(conn, "embedding_dimension", str(dimension))
        await conn.commit()

    async def read_entity_texts(
        self,
        uids: list[str],
        *,
        labels: list[str] | None = None,  # noqa: ARG002 — part of the GraphBackend contract, unused (unified table)
        chunk_size: int = 200,
    ) -> list[dict[str, Any]]:
        if not uids:
            return []
        conn = await self._get_conn()
        results: list[dict[str, Any]] = []
        for i in range(0, len(uids), chunk_size):
            chunk = uids[i : i + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            cur = await conn.execute(
                f"SELECT uid, qualified_name, name, kind, labels, file_path, props_json, embedding IS NOT NULL "
                f"FROM nodes WHERE uid IN ({placeholders})",
                chunk,
            )
            rows = await cur.fetchall()
            await cur.close()
            for uid, qn, name, kind, label, file_path, props_json, has_emb in rows:
                props = json.loads(props_json) if props_json else {}
                results.append(
                    {
                        "uid": uid,
                        "qualified_name": qn,
                        "name": name,
                        "signature": props.get("signature"),
                        "docstring": props.get("docstring"),
                        "source": props.get("source"),
                        "tags": props.get("tags"),
                        "kind": kind,
                        "_label": label,
                        "file_path": file_path,
                        "embed_hash": props.get("embed_hash"),
                        "has_embedding": bool(has_emb),
                    }
                )
        return results

    async def read_embed_hashes(
        self,
        uids: list[str],
        *,
        labels: list[str] | None = None,  # noqa: ARG002 — part of the GraphBackend contract, unused (unified table)
    ) -> dict[str, tuple[str | None, bool]]:
        if not uids:
            return {}
        conn = await self._get_conn()
        result: dict[str, tuple[str | None, bool]] = {}
        for chunk in _chunks(uids):
            if not chunk:
                continue
            placeholders = ",".join("?" * len(chunk))
            cur = await conn.execute(
                f"SELECT uid, props_json, embedding IS NOT NULL FROM nodes WHERE uid IN ({placeholders})", chunk
            )
            rows = await cur.fetchall()
            await cur.close()
            for uid, props_json, has_emb in rows:
                props = json.loads(props_json) if props_json else {}
                result[uid] = (props.get("embed_hash"), bool(has_emb))
        return result

    async def find_unembedded_entities(
        self, project_name: str, *, limit: int = 5000, exclude_kinds: Collection[str] = ()
    ) -> list[tuple[str, str, str, str]]:
        conn = await self._get_conn()
        placeholders = ",".join("?" * len(_VEC_LABEL_VALUES))
        skip = sorted(exclude_kinds)
        # `kind IS NULL OR` is not decoration: SQL's NOT IN yields NULL, not true, for a
        # NULL left operand, so without it every entity with no kind would be filtered out
        # by a policy that never named it.
        kind_clause = f"AND (kind IS NULL OR kind NOT IN ({','.join('?' * len(skip))})) " if skip else ""
        cur = await conn.execute(
            f"SELECT uid, labels, kind, file_path FROM nodes WHERE project_name = ? AND embedding IS NULL "
            f"AND labels IN ({placeholders}) {kind_clause}LIMIT ?",
            (project_name, *sorted(_VEC_LABEL_VALUES), *skip, limit),
        )
        rows = await cur.fetchall()
        await cur.close()
        return [(uid, label, kind or "", file_path or "") for uid, label, kind, file_path in rows]

    async def _sync_vector_rows(self, conn: SqlConnection, by_label: dict[str, list[tuple[int, bytes]]]) -> None:
        """Replace the vec0 rows for a whole batch, two statements per label.

        vec0 has no upsert, so a rewrite is delete-then-insert. Per row that was the
        dominant cost of the entire embed write path: measured at 768 dimensions against
        a populated table, 4.977 ms per vector, against 0.279 for the node UPDATE and
        0.478 for the rowid lookup that RETURNING has since removed. Sent through
        `executemany` instead it is 1.632 ms -- 3.05x, for the same statements and the
        same rows, because what dominated was per-statement dispatch rather than the
        index work.
        """
        if not self._embeddings_enabled:
            return
        for label, pairs in by_label.items():
            if label not in _VEC_LABEL_VALUES or not pairs:
                continue
            table = f"vec_{label.lower()}"
            await self._safe_exec_many(conn, f"DELETE FROM {table} WHERE rowid = ?", [(rid,) for rid, _b in pairs])
            insert = (
                f"INSERT INTO {table}(rowid, embedding) VALUES (?, vec_quantize_binary(?))"
                if self._bit_vectors
                else f"INSERT INTO {table}(rowid, embedding) VALUES (?, ?)"
            )
            await self._safe_exec_many(conn, insert, pairs)

    async def _write_embedding_row(self, conn: SqlConnection, uid: str, blob: bytes) -> None:
        """Single-row vec0 sync, for the paths that write one vector at a time."""
        cur = await conn.execute("SELECT rowid, labels FROM nodes WHERE uid = ?", (uid,))
        row = await cur.fetchone()
        await cur.close()
        if row is None:
            return
        rowid, label = row
        await self._sync_vector_rows(conn, {label: [(rowid, blob)]})

    async def write_embeddings(
        self,
        items: list[tuple[str, list[float]]],
        chunk_size: int = 50,  # noqa: ARG002 — part of the GraphBackend contract, unused (writes one row at a time)
        *,
        labels: list[str] | None = None,  # noqa: ARG002 — part of the GraphBackend contract, unused (unified table)
    ) -> None:
        if not items:
            return
        conn = await self._get_conn()
        for uid, vector in items:
            blob = sqlite_vec.serialize_float32(vector)
            await conn.execute("UPDATE nodes SET embedding = ? WHERE uid = ?", (blob, uid))
            await self._write_embedding_row(conn, uid, blob)
        await conn.commit()

    async def write_embed_hashes(
        self,
        items: list[tuple[str, str]],
        *,
        labels: list[str] | None = None,  # noqa: ARG002 — part of the GraphBackend contract, unused (unified table)
    ) -> None:
        if not items:
            return
        conn = await self._get_conn()
        for uid, h in items:
            await conn.execute(
                "UPDATE nodes SET props_json = json_patch(props_json, ?) WHERE uid = ?",
                (_dumps({"embed_hash": h}), uid),
            )
        await conn.commit()

    async def write_embeddings_and_hashes(
        self,
        items: list[tuple[str, list[float], str]],
        *,
        labels: list[str] | None = None,  # noqa: ARG002 — part of the GraphBackend contract, unused (unified table)
        model: str = "",
    ) -> None:
        if not items:
            return
        conn = await self._get_conn()
        # This backend has ONE connection, so a commit here would otherwise land in the
        # middle of another writer's transaction. It was safe only because the embed
        # consumer happened to serialise its callers; that is the consumer's concurrency
        # policy, not this backend's durability guarantee, and the two must not be the
        # same knob. Every other writer here already takes this lock.
        async with self._write_lock, _txn():
            props = {"embed_hash": "", "embed_model": model} if model else {"embed_hash": ""}
            # RETURNING, so the row is located once. The version this replaces followed
            # every UPDATE with `SELECT rowid, labels FROM nodes WHERE uid = ?` to find
            # the same row it had just written -- a second b-tree descent per vector, and
            # one of four statements per vector on the hottest write path in the backend.
            by_label: dict[str, list[tuple[int, bytes]]] = defaultdict(list)
            for uid, vector, h in items:
                blob = sqlite_vec.serialize_float32(vector)
                props["embed_hash"] = h
                cur = await conn.execute(
                    "UPDATE nodes SET embedding = ?, props_json = json_patch(props_json, ?) "
                    "WHERE uid = ? RETURNING rowid, labels",
                    (blob, _dumps(props), uid),
                )
                row = await cur.fetchone()
                await cur.close()
                if row is not None:
                    by_label[row[1]].append((row[0], blob))

            await self._sync_vector_rows(conn, by_label)
            await conn.commit()

    async def find_embeddings_by_hash(self, hashes: list[str], model: str) -> dict[str, list[float]]:
        """Return ``{embed_hash: vector}`` for texts already embedded under *model*.

        Same contract as the Memgraph implementation (ADR-0036), with one honest
        difference worth stating: this backend is one database file per project root,
        so its dedup reaches only within a root (its monorepo sub-projects included),
        never across separate repositories.
        """
        if not hashes:
            return {}
        conn = await self._get_conn()
        out: dict[str, list[float]] = {}
        unique = list(dict.fromkeys(hashes))
        # SQLITE_MAX_VARIABLE_NUMBER is 999 on older builds; +1 for the model.
        for i in range(0, len(unique), 900):
            chunk = unique[i : i + 900]
            placeholders = ",".join("?" * len(chunk))
            cur = await conn.execute(
                "SELECT json_extract(props_json, '$.embed_hash') AS h, embedding FROM nodes "
                f"WHERE json_extract(props_json, '$.embed_hash') IN ({placeholders}) "
                "AND embedding IS NOT NULL "
                "AND json_extract(props_json, '$.embed_model') = ?",
                (*chunk, model),
            )
            for h, blob in await cur.fetchall():
                if h in out or not blob:
                    continue
                out[h] = list(struct.unpack(f"<{len(blob) // 4}f", blob))
            await cur.close()
        return out

    async def stamp_note_relations(self, note_uids: list[str]) -> int:
        """Recompute ``superseded_by``/``contradicts_with`` for notes touched by a batch.

        Same contract as the Memgraph implementation, including the wider affected set:
        removing a ``supersedes:`` entry has to un-stamp a note the batch never
        mentions, so notes currently stamped *by* a batch note are recomputed too.
        """
        if not note_uids:
            return 0
        conn = await self._get_conn()
        placeholders = ",".join("?" * len(note_uids))

        cur = await conn.execute(
            f"SELECT to_uid FROM edges WHERE rel_type IN ('SUPERSEDES', 'CONTRADICTS') "
            f"AND from_uid IN ({placeholders})",
            tuple(note_uids),
        )
        affected = {r[0] for r in await cur.fetchall()}
        await cur.close()

        cur = await conn.execute(
            "SELECT uid FROM nodes WHERE labels = 'Note' AND ("
            f"json_extract(props_json, '$.superseded_by') IN ({placeholders}) "
            "OR EXISTS (SELECT 1 FROM json_each(json_extract(props_json, '$.contradicts_with')) "
            f"WHERE value IN ({placeholders})))",
            (*note_uids, *note_uids),
        )
        affected.update(r[0] for r in await cur.fetchall())
        await cur.close()
        affected.update(note_uids)

        async with self._write_lock, _txn():
            for uid in sorted(affected):
                cur = await conn.execute(
                    "SELECT from_uid FROM edges WHERE rel_type = 'SUPERSEDES' AND to_uid = ? LIMIT 1", (uid,)
                )
                row = await cur.fetchone()
                await cur.close()
                newer = row[0] if row else None

                cur = await conn.execute(
                    "SELECT to_uid FROM edges WHERE rel_type = 'CONTRADICTS' AND from_uid = ? "
                    "UNION SELECT from_uid FROM edges WHERE rel_type = 'CONTRADICTS' AND to_uid = ?",
                    (uid, uid),
                )
                others = sorted({r[0] for r in await cur.fetchall()})
                await cur.close()

                # Clear both first, so a note that lost its last incoming edge ends up
                # clean rather than keeping a stamp nothing points at any more.
                await conn.execute(
                    "UPDATE nodes SET props_json = json_remove(props_json, "
                    "'$.superseded_by', '$.contradicts_with') WHERE uid = ?",
                    (uid,),
                )
                patch: dict[str, object] = {}
                if newer:
                    patch["superseded_by"] = newer
                if others:
                    patch["contradicts_with"] = others
                if patch:
                    await conn.execute(
                        "UPDATE nodes SET props_json = json_patch(props_json, ?) WHERE uid = ?",
                        (_dumps(patch), uid),
                    )
            await conn.commit()
        return len(affected)

    async def clear_embeddings(self, project: str | None = None) -> int:
        """Strip vectors for one project, or the whole store when *project* is None.

        Database-wide is only correct for a dimension change; a model change belongs
        to one project, and clearing all of them for it destroyed other projects'
        vectors silently (ATL-135).
        """
        conn = await self._get_conn()
        # project_name is a COLUMN here, never a props_json key (_NODE_COLUMNS). Reading it
        # out of the blob matched nothing at all, so the scoped clear was a silent no-op
        # while _check_model_lock still recorded the new model -- old-space vectors under
        # the new model's name, and no later run able to notice. substr rather than LIKE:
        # LIKE reads `_` and `%` in a project name as wildcards and would reach wider than
        # Cypher's STARTS WITH, which is the radius count_project_data shows in the preflight.
        prefix = f"{project}/"
        where = (
            "(embedding IS NOT NULL OR json_extract(props_json, '$.embed_hash') IS NOT NULL)"
            if project is None
            # "{root}/{sub}" sub-projects belong to the same model as their root.
            else "(embedding IS NOT NULL OR json_extract(props_json, '$.embed_hash') IS NOT NULL) "
            "AND (project_name = ? OR substr(project_name, 1, ?) = ?)"
        )
        args: tuple[object, ...] = () if project is None else (project, len(prefix), prefix)
        cur = await conn.execute(f"SELECT count(*) FROM nodes WHERE {where}", args)
        row = await cur.fetchone()
        await cur.close()
        cleared = int(row[0]) if row else 0
        await conn.execute(
            "UPDATE nodes SET embedding = NULL, "
            "props_json = json_remove(props_json, '$.embed_hash', '$.embed_model') "
            f"WHERE {where}",
            args,
        )
        # Deleted, not stripped, mirroring GraphClient.clear_embeddings: a chunk's entire
        # content is its vector, so a stripped one is an unreachable row the next embed pass
        # would have to recognise and reuse.
        if project is None:
            # The vec0 shadow tables carry no project column, so only a database-wide clear
            # can prune them wholesale -- rows are re-keyed on the next write and a stale one
            # is unreachable once `nodes.embedding` is NULL.
            for spec in build_vector_index_specs(self._dimension):
                await self._safe_exec(conn, f"DELETE FROM {spec.name}")
            await conn.execute("DELETE FROM nodes WHERE labels = ?", (NodeLabel.EMBED_CHUNK.value,))
        else:
            cur = await conn.execute(
                "SELECT uid FROM nodes WHERE labels = ? AND (project_name = ? OR substr(project_name, 1, ?) = ?)",
                (NodeLabel.EMBED_CHUNK.value, project, len(prefix), prefix),
            )
            chunk_uids = [r[0] for r in await cur.fetchall()]
            await cur.close()
            await self._delete_chunk_uids(conn, chunk_uids)
        await conn.commit()
        return cleared

    async def find_embedded_entities(
        self, project_name: str, *, kinds: Collection[str] = ()
    ) -> list[tuple[str, str, str]]:
        """Mirror of ``GraphClient.find_embedded_entities`` — see that docstring."""
        conn = await self._get_conn()
        skip = sorted(kinds)
        kind_clause = f"AND kind IN ({','.join('?' * len(skip))}) " if skip else ""
        # EmbedChunk is excluded explicitly, because this table is unified where Memgraph's
        # graph is not: there, `MATCH (n:Entity)` skips chunks for free, since a chunk
        # deliberately carries no :Entity marker (schema.py). A chunk is not an entity the
        # policy has an opinion about — its vector belongs to its parent's lifecycle, and
        # `clear_embeddings_for_uids` removes it via `delete_embed_chunks`.
        cur = await conn.execute(
            f"SELECT uid, kind, file_path FROM nodes WHERE project_name = ? AND embedding IS NOT NULL "
            f"AND labels != ? {kind_clause}",
            (project_name, NodeLabel.EMBED_CHUNK.value, *skip),
        )
        rows = await cur.fetchall()
        await cur.close()
        return [(uid, kind or "", file_path or "") for uid, kind, file_path in rows]

    async def clear_embeddings_for_uids(self, uids: list[str]) -> int:
        """Mirror of ``GraphClient.clear_embeddings_for_uids`` — see that docstring.

        The vec0 rows go too, which the project-scoped ``clear_embeddings`` above leaves
        behind. It can afford to: what it clears is about to be re-embedded, so the rows
        are re-keyed on the next write. These nodes are not — the policy just said they
        never get a vector again — so a stale row would sit in the shortlist forever,
        spending `k * _VEC_OVERSAMPLE` slots that the rescore then discards.

        The FTS rows deliberately stay. Excluded from embeddings is not excluded from the
        index, and BM25 is the channel this node now lives in.
        """
        if not uids:
            return 0
        unique = list(dict.fromkeys(uids))
        conn = await self._get_conn()
        cleared = 0
        async with self._write_lock, _txn():
            for chunk in _chunks(unique):
                if not chunk:
                    continue
                placeholders = ",".join("?" * len(chunk))
                cur = await conn.execute(
                    f"SELECT uid, labels, rowid FROM nodes WHERE uid IN ({placeholders}) "
                    "AND (embedding IS NOT NULL OR json_extract(props_json, '$.embed_hash') IS NOT NULL)",
                    chunk,
                )
                rows = list(await cur.fetchall())
                await cur.close()
                if not rows:
                    continue
                cleared += len(rows)
                by_label: dict[str, list[int]] = defaultdict(list)
                for _uid, label, rowid in rows:
                    by_label[label].append(rowid)
                for label, rowids in by_label.items():
                    if label in _VEC_LABEL_VALUES:
                        ids = ",".join(str(r) for r in rowids)
                        await self._safe_exec(conn, f"DELETE FROM vec_{label.lower()} WHERE rowid IN ({ids})")
                await conn.execute(
                    "UPDATE nodes SET embedding = NULL, "
                    "props_json = json_remove(props_json, '$.embed_hash', '$.embed_model') "
                    f"WHERE uid IN ({placeholders})",
                    chunk,
                )
            await conn.commit()
        # Outside the lock: delete_embed_chunks takes it itself, and it is not reentrant.
        await self.delete_embed_chunks(unique)
        return cleared

    async def count_embeddings_by_project(self) -> dict[str, int]:
        conn = await self._get_conn()
        cur = await conn.execute(
            # The column, not props_json -- see clear_embeddings. Reading the blob returned
            # {} for every store, blanking the project list in the destructive preflight.
            "SELECT project_name AS p, count(*) FROM nodes WHERE embedding IS NOT NULL GROUP BY p"
        )
        rows = await cur.fetchall()
        await cur.close()
        return {p: c for p, c in rows if p}

    # -- Search -------------------------------------------------------------------

    async def graph_search(
        self, query: str, label: str = "", limit: int = 20, project: str = "", projects: list[str] | None = None
    ) -> list[dict[str, Any]]:
        conn = await self._get_conn()
        filter_projects = projects if projects is not None else ([project] if project else None)
        label_clause = "AND labels = ?" if label else ""
        label_params = [label] if label else []
        proj_sql = ""
        proj_params: list[str] = []
        if filter_projects:
            # GLOBAL_PROJECT rides along in the scope list — see text_search.
            proj_params = [*filter_projects, GLOBAL_PROJECT]
            placeholders = ",".join("?" * len(proj_params))
            proj_sql = f" AND project_name IN ({placeholders})"
        fetch_limit = limit * 3

        scored: dict[str, tuple[dict[str, Any], float]] = {}

        async def _run(sql: str, params: list[Any], score: float) -> None:
            cur = await conn.execute(sql, params)
            rows = await cur.fetchall()
            await cur.close()
            for row in rows:
                node = _row_to_node(row)
                uid = node["uid"]
                if uid not in scored or scored[uid][1] < score:
                    scored[uid] = (node, score)

        # Mirrors graph.client._GRAPH_SEARCH_ORDER exactly -- see the rationale there.
        # ``instr``, never ``LIKE``: LIKE is case-INSENSITIVE over ASCII in SQLite while
        # Cypher CONTAINS is case-sensitive, so the obvious spelling (reusing the LIKE the
        # WHERE clause already uses) silently reorders 165 of 400 real rows against
        # Memgraph, starting at rank 1, and no all-lowercase fixture can see it.
        # The bound parameter is the RAW query, matching CONTAINS $query -- not the
        # ``_like_literal`` form, whose escapes are a LIKE-pattern concern.
        order_sql = "ORDER BY CASE WHEN instr(coalesce(name,''), ?) > 0 THEN 0 ELSE 1 END, coalesce(name,''), uid"

        await _run(
            f"SELECT {_NODE_COLUMNS} FROM nodes WHERE name = ? {label_clause}{proj_sql} {order_sql} LIMIT ?",
            [query, *label_params, *proj_params, query, fetch_limit],
            3.0,
        )
        await _run(
            f"SELECT {_NODE_COLUMNS} FROM nodes WHERE qualified_name LIKE ? ESCAPE '\\' "
            f"{label_clause}{proj_sql} {order_sql} LIMIT ?",
            [f"%.{_like_literal(query)}", *label_params, *proj_params, query, fetch_limit],
            2.0,
        )
        await _run(
            f"SELECT {_NODE_COLUMNS} FROM nodes "
            f"WHERE (qualified_name LIKE ? ESCAPE '\\' OR name LIKE ? ESCAPE '\\') "
            f"{label_clause}{proj_sql} {order_sql} LIMIT ?",
            [
                f"%{_like_literal(query)}%",
                f"%{_like_literal(query)}%",
                *label_params,
                *proj_params,
                query,
                fetch_limit,
            ],
            1.0,
        )

        results = sorted(scored.values(), key=lambda item: item[1], reverse=True)
        return [{"node": node, "score": score} for node, score in results[:limit]]

    async def text_search(
        self, query: str, label: str = "", limit: int = 20, project: str = "", projects: list[str] | None = None
    ) -> list[dict[str, Any]]:
        conn = await self._get_conn()
        filter_projects = projects if projects is not None else ([project] if project else None)
        indices = [f"text_{label.lower()}"] if label else [spec.name for spec in TEXT_INDICES]
        fetch_limit = limit * 3 if filter_projects else limit
        safe_query = _sanitize_fts_query(query)

        async def _one(idx: str) -> list[dict[str, Any]]:
            try:
                cur = await conn.execute(
                    f"SELECT uid, bm25({idx}) AS rank FROM {idx} WHERE {idx} MATCH ? ORDER BY rank LIMIT ?",
                    (safe_query, fetch_limit),
                )
                rows = await cur.fetchall()
                await cur.close()
            except aiosqlite.OperationalError as exc:
                logger.warning("Text search on {} failed: {}", idx, exc)
                return []
            if not rows:
                return []
            node_by_uid = await self._nodes_by_uid(conn, [r[0] for r in rows])
            return [{"node": node_by_uid[uid], "score": -rank} for uid, rank in rows if uid in node_by_uid]

        results_per_index = list(await asyncio.gather(*(_one(idx) for idx in indices)))
        all_results = _fuse_bm25_results(results_per_index)

        if filter_projects:
            # GLOBAL_PROJECT always passes — see GraphClient.text_search.
            project_set = {*filter_projects, GLOBAL_PROJECT}
            all_results = [r for r in all_results if r["node"].get("project_name") in project_set]

        return all_results[:limit]

    async def vector_search(
        self,
        vector: list[float],
        label: str = "",
        limit: int = 20,
        project: str = "",
        threshold: float = 0.0,
        projects: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        conn = await self._get_conn()
        filter_projects = projects if projects is not None else ([project] if project else None)
        specs = [
            s for s in build_vector_index_specs(self._dimension) if not label or s.label.value.lower() == label.lower()
        ]
        # Unconditional over-fetch, matching the Memgraph backend: several rows can
        # collapse onto one node once EmbedChunk hits resolve to their parents.
        fetch_limit = limit * 3
        blob = sqlite_vec.serialize_float32(vector)

        # One statement, shortlist and rescore together. The shortlist comes from the
        # side table and the distance from `nodes.embedding`, so the number returned is a
        # real cosine distance in both index shapes -- with a bit index the approximation
        # is confined to *which* rows are considered, never to how they are then ranked.
        #
        # The float fallback runs the same SQL with `k = fetch_limit` and an unquantized
        # probe; its rescore is then a re-sort of rows already in cosine order, which
        # costs a few dozen distance computations and keeps this to one code path
        # instead of two.
        # The probe is quantized by the extension, not by us. A hand-written bit packer
        # that disagreed with `vec_quantize_binary` on bit order would return confidently
        # wrong neighbours with nothing failing, and the write side already uses the SQL
        # function -- so both ends are the same implementation by construction.
        match_expr = "vec_quantize_binary(?)" if self._bit_vectors else "?"
        sql = (
            f"WITH cand AS (SELECT rowid AS rid FROM {{table}} WHERE embedding MATCH {match_expr} AND k = ?) "
            f"SELECT n.rowid, vec_distance_cosine(n.embedding, ?) AS distance, {_node_columns('n')} "
            "FROM cand JOIN nodes n ON n.rowid = cand.rid "
            "WHERE n.embedding IS NOT NULL ORDER BY distance LIMIT ?"
        )
        candidates = fetch_limit * _VEC_OVERSAMPLE if self._bit_vectors else fetch_limit

        async def _one(spec: Any) -> list[dict[str, Any]]:
            try:
                cur = await conn.execute(sql.format(table=spec.name), (blob, candidates, blob, fetch_limit))
                rows = await cur.fetchall()
                await cur.close()
            except aiosqlite.OperationalError as exc:
                logger.warning("Vector search on {} failed: {}", spec.name, exc)
                return []
            return [{"node": _row_to_node(r[2:]), "similarity": 1.0 - r[1]} for r in rows]

        results_per_index = await asyncio.gather(*(_one(s) for s in specs))
        all_results: list[dict[str, Any]] = [r for batch in results_per_index for r in batch]

        if threshold > 0.0:
            all_results = [r for r in all_results if r.get("similarity", 0) >= threshold]
        # Before the project filter and the uid collapse, for the reason the Memgraph
        # backend gives: a chunk answers with its parent's identity, not its own.
        all_results = await self._resolve_embed_chunks(conn, all_results)
        all_results = _best_similarity_per_uid(all_results)
        if filter_projects:
            project_set = set(filter_projects)
            all_results = [r for r in all_results if r["node"].get("project_name") in project_set]

        all_results.sort(key=lambda r: r.get("similarity", 0), reverse=True)
        return all_results[:limit]

    async def write_embed_chunks(self, items: list[EmbedChunkWrite], *, model: str = "") -> None:
        """Mirror of ``GraphClient.write_embed_chunks`` — see that docstring.

        A chunk is an ordinary ``nodes`` row labelled EmbedChunk, so every side table
        this backend keeps (``vec_embedchunk`` via ``_write_embedding_row``, the delete
        sweep via ``_cleanup_search_side_tables``) reaches it without a special case:
        both are driven by ``schema.py``'s registries, which the label is now in.
        """
        if not items:
            return
        conn = await self._get_conn()
        async with self._write_lock, _txn():
            for item in items:
                props: dict[str, Any] = {
                    "parent_uid": item.parent_uid,
                    "chunk_index": item.chunk_index,
                    "embed_hash": item.embed_hash,
                    "snippet": item.snippet,
                    "line_start": item.line_start,
                    "line_end": item.line_end,
                }
                if model:
                    props["embed_model"] = model
                blob = sqlite_vec.serialize_float32(item.vector)
                # No name and no qualified_name on purpose: graph_search matches on both
                # with CONTAINS, and a chunk is not a thing anyone searches for by name.
                await conn.execute(
                    "INSERT INTO nodes (uid, labels, project_name, props_json, embedding) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(uid) DO UPDATE SET props_json = excluded.props_json, embedding = excluded.embedding",
                    (item.uid, NodeLabel.EMBED_CHUNK.value, item.project_name, _dumps(props), blob),
                )
                await self._write_embedding_row(conn, item.uid, blob)
            await conn.commit()

    async def delete_embed_chunks(self, parent_uids: list[str]) -> None:
        """Mirror of ``GraphClient.delete_embed_chunks`` — see that docstring."""
        if not parent_uids:
            return
        conn = await self._get_conn()
        async with self._write_lock, _txn():
            await self._delete_chunk_uids(conn, await self._chunk_uids_of(conn, parent_uids))
            await conn.commit()

    async def gc_orphaned_embed_chunks(self, project_name: str = "") -> int:
        """Mirror of ``GraphClient.gc_orphaned_embed_chunks`` — see that docstring."""
        conn = await self._get_conn()
        scope = " AND c.project_name = ?" if project_name else ""
        params: tuple[Any, ...] = (NodeLabel.EMBED_CHUNK.value, *((project_name,) if project_name else ()))
        cur = await conn.execute(
            "SELECT c.uid FROM nodes c "
            "LEFT JOIN nodes p ON p.uid = json_extract(c.props_json, '$.parent_uid') "
            f"WHERE c.labels = ?{scope} AND p.uid IS NULL",
            params,
        )
        dead = [r[0] for r in await cur.fetchall()]
        await cur.close()
        if not dead:
            return 0
        async with self._write_lock, _txn():
            await self._delete_chunk_uids(conn, dead)
            await conn.commit()
        return len(dead)

    async def _chunk_uids_of(self, conn: SqlConnection, parent_uids: list[str]) -> list[str]:
        out: list[str] = []
        for chunk in _chunks(list(dict.fromkeys(parent_uids))):
            if not chunk:
                continue
            placeholders = ",".join("?" * len(chunk))
            cur = await conn.execute(
                "SELECT uid FROM nodes WHERE labels = ? "
                f"AND json_extract(props_json, '$.parent_uid') IN ({placeholders})",
                (NodeLabel.EMBED_CHUNK.value, *chunk),
            )
            out.extend(r[0] for r in await cur.fetchall())
            await cur.close()
        return out

    async def _delete_chunk_uids(self, conn: SqlConnection, uids: list[str]) -> None:
        if not uids:
            return
        await self._cleanup_search_side_tables(conn, uids)
        for chunk in _chunks(uids):
            if not chunk:
                continue
            placeholders = ",".join("?" * len(chunk))
            await conn.execute(f"DELETE FROM nodes WHERE uid IN ({placeholders})", chunk)

    async def _resolve_embed_chunks(self, conn: SqlConnection, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Swap every EmbedChunk hit for the node it is a chunk of.

        Mirror of ``GraphClient._resolve_embed_chunks``; the difference is only that a
        node here is a dict carrying ``_labels`` rather than a driver Node object.
        """
        parent_uids = {p for r in records if _is_chunk_record(r) and (p := (r.get("node") or {}).get("parent_uid"))}
        if not parent_uids:
            return records
        placeholders = ",".join("?" * len(parent_uids))
        cur = await conn.execute(
            f"SELECT {_NODE_COLUMNS} FROM nodes WHERE uid IN ({placeholders})", sorted(parent_uids)
        )
        parents = {}
        for row in await cur.fetchall():
            node = _row_to_node(row)
            parents[node["uid"]] = node
        await cur.close()

        out: list[dict[str, Any]] = []
        for record in records:
            if not _is_chunk_record(record):
                out.append(record)
                continue
            parent = parents.get((record.get("node") or {}).get("parent_uid", ""))
            if parent is not None:
                chunk = record.get("node") or {}
                out.append(
                    {
                        **record,
                        "node": parent,
                        # Mirrors GraphClient._chunk_facts: the parent stands in for the
                        # chunk, but what the chunk knew travels with it.
                        "matched_chunk": {
                            "chunk_index": chunk.get("chunk_index"),
                            "snippet": chunk.get("snippet") or "",
                            "line_start": chunk.get("line_start"),
                            "line_end": chunk.get("line_end"),
                        },
                    }
                )
        return out

    async def get_vector_index_info(self) -> list[dict[str, Any]]:
        conn = await self._get_conn()
        result = []
        for spec in build_vector_index_specs(self._dimension):
            try:
                cur = await conn.execute(f"SELECT COUNT(*) FROM {spec.name}")
                row = await cur.fetchone()
                await cur.close()
                count = row[0] if row else 0
            except aiosqlite.OperationalError:
                count = 0
            result.append(
                {
                    "index_name": spec.name,
                    "label": spec.label.value,
                    "property": spec.property,
                    "dimension": spec.dimension,
                    "size": count,
                }
            )
        return result

    async def get_text_index_info(self) -> list[dict[str, Any]]:
        return [{"index_type": "fts5", "label": spec.label.value, "name": spec.name} for spec in TEXT_INDICES]

    async def rebuild_vector_indices(self, dimension: int) -> None:
        conn = await self._get_conn()
        for spec in build_vector_index_specs(self._dimension):
            await self._safe_exec(conn, f"DROP TABLE IF EXISTS {spec.name}")
        self._dimension = dimension
        if self._embeddings_enabled:
            for stmt in _vec_table_ddl(dimension):
                await conn.execute(stmt)
        await conn.commit()

    async def batch_call_stats(self, uids: list[str], *, top_n: int = 5) -> dict[str, CallStats]:
        if not uids:
            return {}
        conn = await self._get_conn()
        result: dict[str, CallStats] = {}
        for chunk in _chunks(uids):
            if not chunk:
                continue
            placeholders = ",".join("?" * len(chunk))
            cur = await conn.execute(
                f"SELECT e.to_uid, e.from_uid, c.name FROM edges e JOIN nodes c ON c.uid = e.from_uid "
                f"WHERE e.rel_type = 'CALLS' AND e.to_uid IN ({placeholders})",
                chunk,
            )
            caller_rows = await cur.fetchall()
            await cur.close()
            cur = await conn.execute(
                f"SELECT e.from_uid, e.to_uid, c.name FROM edges e JOIN nodes c ON c.uid = e.to_uid "
                f"WHERE e.rel_type = 'CALLS' AND e.from_uid IN ({placeholders})",
                chunk,
            )
            callee_rows = await cur.fetchall()
            await cur.close()

            callers_by_uid: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for to_uid, from_uid, name in caller_rows:
                callers_by_uid[to_uid].append((from_uid, name))
            callees_by_uid: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for from_uid, to_uid, name in callee_rows:
                callees_by_uid[from_uid].append((to_uid, name))

            for uid in chunk:
                callers = callers_by_uid.get(uid, [])
                callees = callees_by_uid.get(uid, [])
                result[uid] = CallStats(
                    caller_count=len({u for u, _n in callers}),
                    callee_count=len({u for u, _n in callees}),
                    caller_names=list(dict.fromkeys(n for _u, n in callers))[:top_n],
                    callee_names=list(dict.fromkeys(n for _u, n in callees))[:top_n],
                )
        return result

    # -- Analysis / diagram queries (server/analysis.py) ----------------------
    #
    # SQL ports of GraphClient's analysis/diagram query methods (see
    # graph/protocol.py's GraphBackend). Field names in each returned dict
    # match the Cypher versions exactly so server/analysis.py's downstream
    # Python shaping/aggregation logic is identical for both backends.

    async def node_exists(self, uid: str) -> bool:
        conn = await self._get_conn()
        cur = await conn.execute("SELECT 1 FROM nodes WHERE uid = ?", (uid,))
        row = await cur.fetchone()
        await cur.close()
        return row is not None

    async def _bfs_shortest_path(
        self, conn: SqlConnection, from_uid: str, to_uid: str, edge_types: tuple[str, ...], max_depth: int
    ) -> list[tuple[str, str, str, dict[str, Any]]] | None:
        """BFS shortest path over ``edges``, restricted to *edge_types*.

        Returns an ordered list of ``(from_uid, to_uid, rel_type, props)``
        hops, or ``None`` if no path exists within *max_depth*.

        Hop count still decides the winner (level-synchronous BFS, unchanged);
        among the equal-length paths a node can be reached by, the one with the
        highest running product of edge ``weight`` properties is kept. That is
        exact rather than greedy: every prefix of a shortest path is itself a
        shortest path, so keeping the best-scoring prefix per node at its
        min-depth level is a correct DP over shortest paths.
        """
        type_placeholders = ",".join("?" * len(edge_types))
        parent: dict[str, tuple[str, str, dict[str, Any]]] = {}
        best: dict[str, float] = {from_uid: 1.0}
        visited = {from_uid}
        frontier = [from_uid]
        for _ in range(max_depth):
            # `to_uid in parent`, not `in visited`: from == to is visited before any hop, and
            # a trace from a node to itself is its shortest cycle, as on Memgraph.
            if not frontier or to_uid in parent:
                break
            f_placeholders = ",".join("?" * len(frontier))
            cur = await conn.execute(
                f"SELECT from_uid, to_uid, rel_type, props_json FROM edges "
                f"WHERE rel_type IN ({type_placeholders}) AND from_uid IN ({f_placeholders})",
                (*edge_types, *frontier),
            )
            rows = await cur.fetchall()
            await cur.close()
            level_best: dict[str, float] = {}
            for f_uid, t_uid, rel_type, props_json in rows:
                if t_uid in visited and t_uid != to_uid:
                    continue
                props = json.loads(props_json) if props_json else {}
                score = best[f_uid] * _props_weight(props)
                if t_uid in level_best and score <= level_best[t_uid]:
                    continue
                level_best[t_uid] = score
                parent[t_uid] = (f_uid, rel_type, props)
            visited.update(level_best)
            best.update(level_best)
            frontier = list(level_best)

        if to_uid not in parent:
            return None
        path: list[tuple[str, str, str, dict[str, Any]]] = []
        cur_node = to_uid
        while not path or cur_node != from_uid:
            p_uid, rel_type, props = parent[cur_node]
            path.append((p_uid, cur_node, rel_type, props))
            cur_node = p_uid
        path.reverse()
        return path

    async def trace_path_between(
        self, from_uid: str, to_uid: str, max_depth: int, edge_types: tuple[str, ...]
    ) -> dict[str, Any]:
        conn = await self._get_conn()
        cur = await conn.execute("SELECT uid FROM nodes WHERE uid IN (?, ?)", (from_uid, to_uid))
        found_uids = {r[0] for r in await cur.fetchall()}
        await cur.close()
        from_exists = from_uid in found_uids
        to_exists = to_uid in found_uids
        if not from_exists or not to_exists:
            return {
                "from_exists": from_exists,
                "to_exists": to_exists,
                "found": False,
                "hop_count": None,
                "hops": [],
                "path_weight": None,
            }

        path = await self._bfs_shortest_path(conn, from_uid, to_uid, edge_types, max_depth)
        if path is None:
            return {
                "from_exists": True,
                "to_exists": True,
                "found": False,
                "hop_count": None,
                "hops": [],
                "path_weight": None,
            }

        path_uids = {u for hop in path for u in (hop[0], hop[1])}
        placeholders = ",".join("?" * len(path_uids))
        cur = await conn.execute(f"SELECT uid, name FROM nodes WHERE uid IN ({placeholders})", list(path_uids))
        names = {k: v for k, v in await cur.fetchall()}  # noqa: C416 — dict() doesn't type-match Iterable[Row]
        await cur.close()

        hops: list[dict[str, Any]] = []
        path_weight = 1.0
        for f_uid, t_uid, rel_type, props in path:
            hop: dict[str, Any] = {
                "from": {"uid": f_uid, "name": names.get(f_uid)},
                "to": {"uid": t_uid, "name": names.get(t_uid)},
                "edge_type": rel_type,
            }
            if "confidence" in props:
                hop["confidence"] = props["confidence"]
            if "strategy" in props:
                hop["strategy"] = props["strategy"]
            if "weight" in props:
                hop["weight"] = props["weight"]
            if "from_test" in props:
                hop["from_test"] = props["from_test"]
            if props.get("line") is not None:
                hop["at_line"] = props["line"]
                if props.get("site_count", 1) > 1:
                    hop["site_count"] = props["site_count"]
            path_weight *= _props_weight(props)
            hops.append(hop)
        return {
            "from_exists": True,
            "to_exists": True,
            "found": True,
            "hop_count": len(hops),
            "hops": hops,
            "path_weight": path_weight,
        }

    async def _bfs_reachable(
        self,
        conn: SqlConnection,
        uid: str,
        src_col: str,
        dst_col: str,
        edge_types: tuple[str, ...],
        max_depth: int,
        *,
        resolved_only: bool = False,
        production_only: bool = False,
        include_start: bool = False,
    ) -> dict[str, tuple[int, float]]:
        """BFS from *uid* following ``src_col -> dst_col`` edges.

        Returns ``{reached_uid: (min_depth, best_weight)}`` where ``best_weight``
        is the largest product of edge ``weight`` properties over any path of at
        most *max_depth* hops (missing weights count as ``_DEFAULT_EDGE_WEIGHT``,
        matching how MAGE reads an absent property). A node is re-expanded when
        a later round finds a better-scoring route to it — plain
        first-sighting BFS would lock in whichever path happened to be shortest
        and report its score, which is not the maximum.

        When *resolved_only* is set, only traverses edges whose ``confidence``
        property is ``"resolved"`` (ADR-0014); when *production_only* is set,
        only traverses edges not tagged ``from_test``. Those two filtered passes
        back ``compute_blast_radius``'s ``ambiguous_only``/``test_only`` flags
        and ignore the returned weights.

        *include_start* reports *uid* itself when a cycle through it fits *max_depth*:
        a recursive function is its own caller. Blast radius leaves it out.
        """
        type_placeholders = ",".join("?" * len(edge_types))
        filter_clause = ""
        if resolved_only:
            # coalesce, matching Memgraph's `coalesce(r.confidence, 'resolved')`. An absent
            # confidence means STRUCTURAL (ADR-0028) — DEFINES, IMPORTS, INHERITS are facts,
            # not guesses. Without it json_extract yields NULL for exactly those edges and
            # the comparison drops every one, so `resolved_only` hid the best evidence in
            # the graph while keeping the ambiguous CALLS that do carry the property.
            filter_clause += " AND coalesce(json_extract(props_json, '$.confidence'), 'resolved') = 'resolved'"
        if production_only:
            filter_clause += " AND coalesce(json_extract(props_json, '$.from_test'), 0) = 0"
        reached: dict[str, tuple[int, float]] = {}
        frontier: dict[str, float] = {uid: 1.0}
        depth = 0
        while frontier and depth < max_depth:
            depth += 1
            f_placeholders = ",".join("?" * len(frontier))
            cur = await conn.execute(
                f"SELECT {src_col}, {dst_col}, coalesce(json_extract(props_json, '$.weight'), "
                f"{_DEFAULT_EDGE_WEIGHT}) FROM edges "
                f"WHERE rel_type IN ({type_placeholders}) AND {src_col} IN ({f_placeholders}){filter_clause}",
                (*edge_types, *frontier),
            )
            rows = await cur.fetchall()
            await cur.close()
            next_frontier: dict[str, float] = {}
            for src, dst, edge_weight in rows:
                if dst == uid and not include_start:
                    continue
                score = frontier[src] * _checked_edge_weight(float(edge_weight))
                prior = reached.get(dst)
                if prior is None:
                    reached[dst] = (depth, score)
                elif score > prior[1]:
                    reached[dst] = (prior[0], score)
                else:
                    continue
                next_frontier[dst] = max(next_frontier.get(dst, 0.0), score)
            frontier = next_frontier
        return reached

    async def compute_blast_radius(
        self, uid: str, direction_kind: str, edge_types: tuple[str, ...], max_depth: int
    ) -> list[dict[str, Any]]:
        conn = await self._get_conn()
        src_col, dst_col = ("from_uid", "to_uid") if direction_kind == "out" else ("to_uid", "from_uid")

        reached = await self._bfs_reachable(conn, uid, src_col, dst_col, edge_types, max_depth)
        if not reached:
            return []
        resolved = await self._bfs_reachable(conn, uid, src_col, dst_col, edge_types, max_depth, resolved_only=True)
        production = await self._bfs_reachable(conn, uid, src_col, dst_col, edge_types, max_depth, production_only=True)

        affected_uids = list(reached)
        placeholders = ",".join("?" * len(affected_uids))
        cur = await conn.execute(
            f"SELECT uid, name, qualified_name, file_path, labels, kind FROM nodes WHERE uid IN ({placeholders})",
            affected_uids,
        )
        node_rows = await cur.fetchall()
        await cur.close()
        node_by_uid = {r[0]: r for r in node_rows}

        results: list[dict[str, Any]] = []
        for nuid, (depth, score) in reached.items():
            node = node_by_uid.get(nuid)
            results.append(
                {
                    "uid": nuid,
                    "name": node[1] if node else None,
                    "qualified_name": node[2] if node else None,
                    "label": node[4] if node else None,
                    "file_path": node[3] if node else None,
                    # Ranking needs it -- see `_GRANT_ONLY_KINDS`. `node` may be None:
                    # a dangling edge is tolerated here deliberately.
                    "kind": node[5] if node else None,
                    "min_depth": depth,
                    "direction": direction_kind,
                    "ambiguous_only": nuid not in resolved,
                    "confidence_score": score,
                    "test_only": nuid not in production,
                }
            )
        return results

    async def get_structure_overview(self, project: str, path: str, limit: int) -> dict[str, list[dict[str, Any]]]:
        conn = await self._get_conn()

        clause, extra = _prefix_clause("file_path", path)
        cur = await conn.execute(
            "SELECT labels AS label, kind, COUNT(*) AS cnt FROM nodes "
            f"WHERE project_name = ? AND labels NOT IN ('Project', 'SchemaVersion'){clause} "
            "GROUP BY labels, kind ORDER BY cnt DESC",
            [project, *extra],
        )
        counts_raw = [{"label": r[0], "kind": r[1], "cnt": r[2]} for r in await cur.fetchall()]
        await cur.close()

        clause, extra = _prefix_clause("m.file_path", path)
        cur = await conn.execute(
            "SELECT pkg.name, pkg.qualified_name, COUNT(*) AS modules FROM edges e "
            "JOIN nodes pkg ON pkg.uid = e.from_uid AND pkg.labels = 'Package' AND pkg.project_name = ? "
            "JOIN nodes m ON m.uid = e.to_uid AND m.labels = 'Module' "
            f"WHERE e.rel_type = 'CONTAINS'{clause} "
            "GROUP BY pkg.uid, pkg.name, pkg.qualified_name ORDER BY modules DESC LIMIT ?",
            [project, *extra, limit],
        )
        pkg_raw = [{"package": r[0], "qn": r[1], "modules": r[2]} for r in await cur.fetchall()]
        await cur.close()

        clause, extra = _prefix_clause("m.file_path", path)
        cur = await conn.execute(
            "SELECT m.name, m.qualified_name, m.file_path, COUNT(*) AS entities FROM edges e "
            "JOIN nodes m ON m.uid = e.from_uid AND m.labels = 'Module' AND m.project_name = ? "
            "JOIN nodes en ON en.uid = e.to_uid "
            f"WHERE e.rel_type = 'DEFINES'{clause} "
            "GROUP BY m.uid, m.name, m.qualified_name, m.file_path ORDER BY entities DESC LIMIT ?",
            [project, *extra, limit],
        )
        largest_raw = [{"module": r[0], "qn": r[1], "file_path": r[2], "entities": r[3]} for r in await cur.fetchall()]
        await cur.close()

        # Mirrors Cypher's "OPTIONAL MATCH (ep)<-[:IMPORTS]-(src) WHERE src IS NULL
        # OR src.file_path STARTS WITH $path": an ExternalPackage with zero
        # importers still appears (count 0); one whose importers all fall
        # outside *path* disappears entirely (not clamped to 0).
        if path:
            ext_where = "AND (src.uid IS NULL OR substr(src.file_path, 1, ?) = ?)"
            ext_extra: list[Any] = [len(path), path]
        else:
            ext_where = ""
            ext_extra = []
        # The version comes off the DEPENDS_ON edge since v18. LEFT, and pinned to this
        # project's from_uid: the join must stay 1:1 per ep or COUNT(src.uid) silently
        # multiplies. The edges primary key (from_uid, to_uid, rel_type) is what makes
        # that true, so dropping the from_uid predicate — the obvious "simplification"
        # once the node is globalized — would inflate every imported_by count.
        cur = await conn.execute(
            "SELECT ep.name, json_extract(dep.props_json, '$.version'), COUNT(src.uid) AS imported_by FROM nodes ep "
            "LEFT JOIN edges e ON e.to_uid = ep.uid AND e.rel_type = 'IMPORTS' "
            "LEFT JOIN nodes src ON src.uid = e.from_uid "
            "LEFT JOIN edges dep ON dep.to_uid = ep.uid AND dep.rel_type = 'DEPENDS_ON' AND dep.from_uid = ? "
            f"WHERE ep.labels = 'ExternalPackage' AND ep.project_name = ? {ext_where} "
            "GROUP BY ep.name ORDER BY imported_by DESC LIMIT ?",
            [project, project, *ext_extra, limit],
        )
        ext_raw = [{"package": r[0], "version": r[1], "imported_by": r[2]} for r in await cur.fetchall()]
        await cur.close()

        return {"counts": counts_raw, "packages": pkg_raw, "largest_modules": largest_raw, "external_deps": ext_raw}

    async def get_centrality_data(self, project: str, path: str, limit: int) -> dict[str, list[dict[str, Any]]]:
        conn = await self._get_conn()

        clause, extra = _prefix_clause("n.file_path", path)
        cur = await conn.execute(
            "SELECT n.name, n.qualified_name, n.labels, n.kind, n.file_path, COUNT(*) AS in_degree, "
            "SUM(CASE WHEN e.rel_type = 'IMPORTS' THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN e.rel_type = 'INHERITS' THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN e.rel_type = 'CALLS' THEN 1 ELSE 0 END) "
            "FROM edges e "
            "JOIN nodes n ON n.uid = e.to_uid AND n.project_name = ? "
            "WHERE e.rel_type IN ('IMPORTS', 'INHERITS', 'CALLS') "
            f"AND n.labels NOT IN ('ExternalPackage', 'ExternalSymbol'){clause} "
            "GROUP BY n.uid, n.name, n.qualified_name, n.labels, n.kind, n.file_path "
            "ORDER BY in_degree DESC LIMIT ?",
            [project, *extra, limit],
        )
        hubs_raw = [
            {
                "name": r[0],
                "qn": r[1],
                "label": r[2],
                "kind": r[3],
                "file_path": r[4],
                "in_degree": r[5],
                "imported_by": r[6],
                "inherited_by": r[7],
                "called_by": r[8],
            }
            for r in await cur.fetchall()
        ]
        await cur.close()

        clause, extra = _prefix_clause("m.file_path", path)
        cur = await conn.execute(
            "SELECT m.name, m.qualified_name, m.file_path, COUNT(*) AS imported_by FROM edges e "
            "JOIN nodes m ON m.uid = e.to_uid AND m.labels = 'Module' AND m.project_name = ? "
            f"WHERE e.rel_type = 'IMPORTS'{clause} "
            "GROUP BY m.uid, m.name, m.qualified_name, m.file_path ORDER BY imported_by DESC LIMIT ?",
            [project, *extra, limit],
        )
        hub_modules_raw = [
            {"name": r[0], "qn": r[1], "file_path": r[2], "imported_by": r[3]} for r in await cur.fetchall()
        ]
        await cur.close()

        clause, extra = _prefix_clause("n.file_path", path)
        cur = await conn.execute(
            "SELECT n.name, n.qualified_name, n.labels, n.kind, n.file_path FROM nodes n "
            # EnvVar/ResourceFile can never receive IMPORTS/INHERITS/CALLS, so a
            # label-only filter would report every one of them as a leaf.
            "WHERE n.project_name = ? AND n.labels NOT IN "
            "('Project', 'SchemaVersion', 'Package', 'ExternalPackage', 'ExternalSymbol', "
            "'EnvVar', 'ResourceFile')"
            f"{clause} "
            "AND NOT EXISTS (SELECT 1 FROM edges e WHERE e.to_uid = n.uid "
            "AND e.rel_type IN ('IMPORTS', 'INHERITS', 'CALLS')) "
            "LIMIT ?",
            [project, *extra, limit],
        )
        leaf_raw = [
            {"name": r[0], "qn": r[1], "label": r[2], "kind": r[3], "file_path": r[4]} for r in await cur.fetchall()
        ]
        await cur.close()

        return {"hubs": hubs_raw, "hub_modules": hub_modules_raw, "leaves": leaf_raw}

    async def _module_import_edges(
        self, conn: SqlConnection, project: str, clause: str, extra: list[Any]
    ) -> dict[str, list[dict[str, Any]]]:
        """Shared direct/indirect module-import query, parametrized by a
        pre-built path-scope clause — callers apply it to different columns
        (``m1`` only vs. ``m1`` or ``m2``).
        """
        cur = await conn.execute(
            "SELECT m1.qualified_name, m2.qualified_name, m1.file_path, m2.file_path FROM edges e "
            "JOIN nodes m1 ON m1.uid = e.from_uid AND m1.labels = 'Module' AND m1.project_name = ? "
            "JOIN nodes m2 ON m2.uid = e.to_uid AND m2.labels = 'Module' AND m2.project_name = ? "
            f"WHERE e.rel_type = 'IMPORTS' AND m1.uid <> m2.uid{clause}",
            [project, project, *extra],
        )
        direct_raw = [
            {"from_mod": r[0], "to_mod": r[1], "from_path": r[2], "to_path": r[3]} for r in await cur.fetchall()
        ]
        await cur.close()

        cur = await conn.execute(
            "SELECT m1.qualified_name, m2.qualified_name, m1.file_path, m2.file_path FROM edges imp "
            "JOIN nodes m1 ON m1.uid = imp.from_uid AND m1.labels = 'Module' AND m1.project_name = ? "
            "JOIN nodes ent ON ent.uid = imp.to_uid AND ent.labels <> 'Module' "
            "JOIN edges def ON def.to_uid = ent.uid AND def.rel_type = 'DEFINES' "
            "JOIN nodes m2 ON m2.uid = def.from_uid AND m2.labels = 'Module' AND m2.project_name = ? "
            f"WHERE imp.rel_type = 'IMPORTS' AND m1.uid <> m2.uid{clause}",
            [project, project, *extra],
        )
        indirect_raw = [
            {"from_mod": r[0], "to_mod": r[1], "from_path": r[2], "to_path": r[3]} for r in await cur.fetchall()
        ]
        await cur.close()
        return {"direct": direct_raw, "indirect": indirect_raw}

    async def get_module_import_edges(self, project: str, path: str) -> dict[str, list[dict[str, Any]]]:
        conn = await self._get_conn()
        clause, extra = _prefix_clause("m1.file_path", path)
        return await self._module_import_edges(conn, project, clause, extra)

    async def get_dependency_external_counts(self, project: str, path: str) -> dict[str, list[dict[str, Any]]]:
        conn = await self._get_conn()

        clause, extra = _prefix_clause("src.file_path", path)
        cur = await conn.execute(
            "SELECT ep.name, COUNT(*) AS cnt FROM edges e "
            "JOIN nodes src ON src.uid = e.from_uid AND src.project_name = ? "
            "JOIN nodes ep ON ep.uid = e.to_uid AND ep.labels = 'ExternalPackage' "
            f"WHERE e.rel_type = 'IMPORTS'{clause} "
            "GROUP BY ep.name",
            [project, *extra],
        )
        ext_pkg_raw = [{"package": r[0], "cnt": r[1]} for r in await cur.fetchall()]
        await cur.close()

        clause, extra = _prefix_clause("src.file_path", path)
        cur = await conn.execute(
            "SELECT json_extract(es.props_json, '$.package'), COUNT(*) AS cnt FROM edges e "
            "JOIN nodes src ON src.uid = e.from_uid AND src.project_name = ? "
            "JOIN nodes es ON es.uid = e.to_uid AND es.labels = 'ExternalSymbol' "
            f"WHERE e.rel_type = 'IMPORTS'{clause} "
            "GROUP BY 1",
            [project, *extra],
        )
        ext_sym_raw = [{"package": r[0], "cnt": r[1]} for r in await cur.fetchall()]
        await cur.close()

        return {"ext_packages": ext_pkg_raw, "ext_symbols": ext_sym_raw}

    async def get_quality_data(self, project: str, path: str) -> dict[str, list[dict[str, Any]]]:
        conn = await self._get_conn()

        clause, extra = _prefix_clause("m.file_path", path)
        cur = await conn.execute(
            "SELECT m.qualified_name, m.file_path, COUNT(*) AS entity_count FROM edges e "
            "JOIN nodes m ON m.uid = e.from_uid AND m.labels = 'Module' AND m.project_name = ? "
            "JOIN nodes en ON en.uid = e.to_uid AND en.labels <> 'Module' "
            f"WHERE e.rel_type = 'DEFINES'{clause} "
            "GROUP BY m.uid, m.qualified_name, m.file_path ORDER BY entity_count DESC",
            [project, *extra],
        )
        entity_raw = [{"module": r[0], "file_path": r[1], "entity_count": r[2]} for r in await cur.fetchall()]
        await cur.close()

        clause, extra = _prefix_clause_either("m1.file_path", "m2.file_path", path)
        edges = await self._module_import_edges(conn, project, clause, extra)

        return {"entities": entity_raw, "direct": edges["direct"], "indirect": edges["indirect"]}

    async def get_patterns_data(self, project: str, path: str, limit: int) -> dict[str, list[dict[str, Any]]]:
        conn = await self._get_conn()

        clause, extra = _prefix_clause("child.file_path", path)
        cur = await conn.execute(
            "SELECT child.name, child.qualified_name, parent.name, parent.qualified_name FROM edges e "
            "JOIN nodes child ON child.uid = e.from_uid AND child.labels = 'TypeDef' AND child.project_name = ? "
            "JOIN nodes parent ON parent.uid = e.to_uid "
            f"WHERE e.rel_type = 'INHERITS'{clause} LIMIT ?",
            [project, *extra, limit],
        )
        inherit_raw = [
            {"child": r[0], "child_qn": r[1], "parent": r[2], "parent_qn": r[3]} for r in await cur.fetchall()
        ]
        await cur.close()

        clause, extra = _prefix_clause("n.file_path", path)
        cur = await conn.execute(
            "SELECT n.name, n.qualified_name, n.file_path, COUNT(m.uid) AS members FROM nodes n "
            "LEFT JOIN edges e ON e.from_uid = n.uid AND e.rel_type = 'DEFINES' "
            "LEFT JOIN nodes m ON m.uid = e.to_uid AND m.labels = 'Value' "
            f"WHERE n.labels = 'TypeDef' AND n.kind = 'enum' AND n.project_name = ?{clause} "
            "GROUP BY n.uid, n.name, n.qualified_name, n.file_path ORDER BY n.name LIMIT ?",
            [project, *extra, limit],
        )
        enum_raw = [{"name": r[0], "qn": r[1], "file_path": r[2], "members": r[3]} for r in await cur.fetchall()]
        await cur.close()

        clause, extra = _prefix_clause("file_path", path)
        cur = await conn.execute(
            "SELECT json_extract(props_json, '$.visibility'), COUNT(*) AS cnt FROM nodes "
            f"WHERE project_name = ? AND json_extract(props_json, '$.visibility') IS NOT NULL{clause} "
            "GROUP BY 1 ORDER BY cnt DESC",
            [project, *extra],
        )
        vis_raw = [{"visibility": r[0], "cnt": r[1]} for r in await cur.fetchall()]
        await cur.close()

        clause, extra = _prefix_clause("file_path", path)
        cur = await conn.execute(
            "SELECT COUNT(*) AS total, COALESCE(SUM(CASE WHEN json_extract(props_json, '$.docstring') IS NOT NULL "
            "AND json_extract(props_json, '$.docstring') <> '' THEN 1 ELSE 0 END), 0) AS documented FROM nodes "
            f"WHERE project_name = ? AND labels IN ('Callable', 'TypeDef', 'Value'){clause}",
            [project, *extra],
        )
        doc_raw = [{"total": r[0], "documented": r[1]} for r in await cur.fetchall()]
        await cur.close()

        clause, extra = _prefix_clause("n.file_path", path)
        cur = await conn.execute(
            "SELECT e.rel_type, n.name, n.qualified_name, target.name FROM edges e "
            "JOIN nodes n ON n.uid = e.from_uid AND n.project_name = ? "
            "JOIN nodes target ON target.uid = e.to_uid "
            f"WHERE e.rel_type IN ('HANDLES_COMMAND', 'HANDLES_ROUTE', 'HANDLES_EVENT'){clause} "
            "ORDER BY e.rel_type, n.name LIMIT ?",
            [project, *extra, limit],
        )
        pattern_raw = [
            {"pattern_type": r[0], "name": r[1], "qn": r[2], "target_name": r[3]} for r in await cur.fetchall()
        ]
        await cur.close()

        return {
            "inheritance": inherit_raw,
            "enums": enum_raw,
            "visibility": vis_raw,
            "docstring": doc_raw,
            "detected_patterns": pattern_raw,
        }

    async def get_dead_code_candidates(self, project: str, path: str) -> list[dict[str, Any]]:
        conn = await self._get_conn()
        clause, extra = _prefix_clause("file_path", path)
        # Mirrors GraphClient.get_dead_code_candidates: the kind gate keeps
        # config/infra declarations (which can never be a CALLS target) out.
        code_kinds = sorted(_CODE_ENTITY_KINDS)
        kind_placeholders = ", ".join("?" * len(code_kinds))
        grant_kinds = sorted(_GRANT_ONLY_KINDS)
        grant_placeholders = ", ".join("?" * len(grant_kinds))
        cur = await conn.execute(
            "SELECT name, qualified_name, labels, kind, file_path, "
            "json_extract(props_json, '$.line_start') FROM nodes n "
            "WHERE project_name = ? AND labels IN ('Callable', 'TypeDef') "
            f"AND kind IN ({kind_placeholders}) AND substr(name, 1, 2) != '__'{clause} "
            # Parity with the Memgraph predicate, which this had drifted from: "unused"
            # cannot mean "no CALLS edge" (a class is used by being annotated, subclassed
            # or imported), and a function nested in another function is reached through
            # its enclosing scope rather than by name.
            # REFERENCES included, matching Memgraph: handed to a registry or a callback
            # slot counts as used, even though the call that eventually runs it belongs to
            # a framework rather than to this codebase.
            # Joined to the source node so grant-only kinds can be excluded: a permission
            # set gives every Apex class it grants an inbound IMPORTS, and counting that as
            # a reference returned zero dead Apex in any org where most classes are granted.
            "AND NOT EXISTS (SELECT 1 FROM edges e JOIN nodes s ON s.uid = e.from_uid "
            "WHERE e.to_uid = n.uid AND e.rel_type IN "
            "('CALLS', 'USES_TYPE', 'IMPORTS', 'INHERITS', 'IMPLEMENTS', 'OVERRIDES', 'REFERENCES') "
            f"AND COALESCE(s.kind, '') NOT IN ({grant_placeholders})) "
            "AND NOT EXISTS (SELECT 1 FROM edges d JOIN edges c ON c.to_uid = d.to_uid "
            "WHERE d.from_uid = n.uid AND d.rel_type = 'DEFINES' AND c.rel_type = 'CALLS') "
            "AND NOT EXISTS (SELECT 1 FROM edges d JOIN nodes p ON p.uid = d.from_uid "
            "WHERE d.to_uid = n.uid AND d.rel_type = 'DEFINES' AND p.labels = 'Callable') "
            # OVERRIDES added here: an override is reached through its base, so liveness is
            # an OUTBOUND test. Memgraph recorded this hook as 100% false-positive before it
            # was added there — the graph already held the disproof and the predicate
            # discarded it.
            "AND NOT EXISTS (SELECT 1 FROM edges i WHERE i.from_uid = n.uid "
            "AND i.rel_type IN ('IMPLEMENTS', 'REGISTERED_BY', 'OVERRIDES')) "
            # A decorator that is CALLED registers the function with whoever owns the
            # registry, which is the inbound edge no relationship records. Every Typer
            # command and every @mcp.tool() handler looked dead without this.
            "AND json_extract(props_json, '$.decorator_name') IS NULL "
            # A property is READ (`obj.ok`), and an attribute read is not a call — zero
            # inbound edges is the expected state for one, not evidence about it.
            "AND kind != 'property' "
            "ORDER BY file_path, json_extract(props_json, '$.line_start')",
            # Binding order follows the SQL: project, the kind gate, the path prefix, then
            # the grant kinds inside the NOT EXISTS.
            [project, *code_kinds, *extra, *grant_kinds],
        )
        rows = await cur.fetchall()
        await cur.close()
        return [
            {"name": r[0], "qn": r[1], "label": r[2], "kind": r[3], "file_path": r[4], "line_start": r[5]} for r in rows
        ]

    async def get_complexity_hotspots(self, project: str, path: str, limit: int) -> list[dict[str, Any]]:
        conn = await self._get_conn()
        clause, extra = _prefix_clause("file_path", path)
        cur = await conn.execute(
            "SELECT name, qualified_name, kind, file_path, "
            "json_extract(props_json, '$.line_start'), json_extract(props_json, '$.line_end'), "
            "(json_extract(props_json, '$.line_end') - json_extract(props_json, '$.line_start')) AS loc_span "
            "FROM nodes "
            f"WHERE labels = 'Callable' AND project_name = ? "
            "AND json_extract(props_json, '$.line_start') IS NOT NULL "
            f"AND json_extract(props_json, '$.line_end') IS NOT NULL{clause} "
            "ORDER BY loc_span DESC LIMIT ?",
            [project, *extra, limit],
        )
        rows = await cur.fetchall()
        await cur.close()
        return [
            {
                "name": r[0],
                "qn": r[1],
                "kind": r[2],
                "file_path": r[3],
                "line_start": r[4],
                "line_end": r[5],
                "loc_span": r[6],
            }
            for r in rows
        ]

    async def get_git_signals_data(
        self, project: str, path: str, limit: int, bus_factor_threshold: int
    ) -> dict[str, list[dict[str, Any]]]:
        conn = await self._get_conn()

        clause, extra = _prefix_clause("file_path", path)
        cur = await conn.execute(
            "SELECT name, qualified_name, file_path, json_extract(props_json, '$.git_commit_count'), "
            "json_extract(props_json, '$.git_author_count'), json_extract(props_json, '$.git_days_since_last_commit') "
            "FROM nodes "
            f"WHERE project_name = ? AND json_extract(props_json, '$.git_commit_count') IS NOT NULL{clause} "
            "ORDER BY 4 DESC LIMIT ?",
            [project, *extra, limit],
        )
        hotspots_raw = [
            {
                "name": r[0],
                "qn": r[1],
                "file_path": r[2],
                "commit_count": r[3],
                "author_count": r[4],
                "days_since_last_commit": r[5],
            }
            for r in await cur.fetchall()
        ]
        await cur.close()

        clause, extra = _prefix_clause("file_path", path)
        cur = await conn.execute(
            "SELECT name, qualified_name, file_path, json_extract(props_json, '$.git_commit_count'), "
            "json_extract(props_json, '$.git_author_count') FROM nodes "
            f"WHERE project_name = ? AND json_extract(props_json, '$.git_commit_count') IS NOT NULL "
            f"AND json_extract(props_json, '$.git_author_count') <= ?{clause} "
            "ORDER BY 4 DESC LIMIT ?",
            [project, bus_factor_threshold, *extra, limit],
        )
        bus_factor_raw = [
            {"name": r[0], "qn": r[1], "file_path": r[2], "commit_count": r[3], "author_count": r[4]}
            for r in await cur.fetchall()
        ]
        await cur.close()

        clause, extra = _prefix_clause_either("a.file_path", "b.file_path", path)
        cur = await conn.execute(
            "SELECT a.qualified_name, a.file_path, b.qualified_name, b.file_path, "
            "json_extract(e.props_json, '$.count') FROM edges e "
            "JOIN nodes a ON a.uid = e.from_uid AND a.project_name = ? "
            "JOIN nodes b ON b.uid = e.to_uid AND b.project_name = ? "
            f"WHERE e.rel_type = 'CO_CHANGES_WITH'{clause} "
            "ORDER BY 5 DESC LIMIT ?",
            [project, project, *extra, limit],
        )
        co_change_raw = [
            {"a_qn": r[0], "a_path": r[1], "b_qn": r[2], "b_path": r[3], "count": r[4]} for r in await cur.fetchall()
        ]
        await cur.close()

        return {"hotspots": hotspots_raw, "bus_factor": bus_factor_raw, "co_change": co_change_raw}

    async def get_diagram_packages(self, project: str, path: str, max_nodes: int) -> list[dict[str, Any]]:
        conn = await self._get_conn()
        clause, extra = _prefix_clause("child.file_path", path)
        cur = await conn.execute(
            "SELECT pkg.qualified_name, pkg.name, child.labels, child.qualified_name, child.name FROM edges e "
            "JOIN nodes pkg ON pkg.uid = e.from_uid AND pkg.labels = 'Package' AND pkg.project_name = ? "
            "JOIN nodes child ON child.uid = e.to_uid AND child.labels IN ('Package', 'Module') "
            f"WHERE e.rel_type = 'CONTAINS'{clause} "
            "ORDER BY pkg.qualified_name, child.qualified_name LIMIT ?",
            [project, *extra, max_nodes],
        )
        rows = await cur.fetchall()
        await cur.close()
        return [
            {"parent_qn": r[0], "parent_name": r[1], "child_label": r[2], "child_qn": r[3], "child_name": r[4]}
            for r in rows
        ]

    async def get_diagram_inheritance(self, project: str, path: str, max_nodes: int) -> list[dict[str, Any]]:
        conn = await self._get_conn()
        clause, extra = _prefix_clause("child.file_path", path)
        cur = await conn.execute(
            "SELECT child.name, child.qualified_name, child.kind, parent.name, parent.qualified_name FROM edges e "
            "JOIN nodes child ON child.uid = e.from_uid AND child.labels = 'TypeDef' AND child.project_name = ? "
            "JOIN nodes parent ON parent.uid = e.to_uid "
            f"WHERE e.rel_type = 'INHERITS'{clause} "
            "ORDER BY parent.qualified_name, child.qualified_name LIMIT ?",
            [project, *extra, max_nodes],
        )
        rows = await cur.fetchall()
        await cur.close()
        return [
            {"child_name": r[0], "child_qn": r[1], "child_kind": r[2], "parent_name": r[3], "parent_qn": r[4]}
            for r in rows
        ]

    async def get_diagram_module_detail(self, project: str, path: str, max_nodes: int) -> dict[str, Any] | None:
        conn = await self._get_conn()
        clause, extra = _prefix_clause("file_path", path)
        cur = await conn.execute(
            "SELECT name, qualified_name, uid FROM nodes "
            f"WHERE labels = 'Module' AND project_name = ?{clause} "
            "ORDER BY qualified_name LIMIT 1",
            [project, *extra],
        )
        row = await cur.fetchone()
        await cur.close()
        if row is None:
            return None
        mod = {"name": row[0], "qn": row[1], "uid": row[2]}

        cur = await conn.execute(
            "SELECT e.name, e.qualified_name, e.labels, e.kind, "
            "json_extract(e.props_json, '$.visibility'), json_extract(e.props_json, '$.signature') FROM edges rel "
            "JOIN nodes e ON e.uid = rel.to_uid "
            "WHERE rel.from_uid = ? AND rel.rel_type = 'DEFINES' "
            "ORDER BY json_extract(e.props_json, '$.line_start') LIMIT ?",
            (mod["uid"], max_nodes),
        )
        entities = [
            {"name": r[0], "qn": r[1], "label": r[2], "kind": r[3], "vis": r[4], "sig": r[5]}
            for r in await cur.fetchall()
        ]
        await cur.close()

        cur = await conn.execute(
            "SELECT td.qualified_name, td.name, method.name, "
            "json_extract(method.props_json, '$.visibility'), method.kind FROM edges rel1 "
            "JOIN nodes td ON td.uid = rel1.to_uid AND td.labels = 'TypeDef' "
            "JOIN edges rel2 ON rel2.from_uid = td.uid AND rel2.rel_type = 'DEFINES' "
            "JOIN nodes method ON method.uid = rel2.to_uid AND method.labels = 'Callable' "
            "WHERE rel1.from_uid = ? AND rel1.rel_type = 'DEFINES' "
            "ORDER BY td.name, json_extract(method.props_json, '$.line_start') LIMIT ?",
            (mod["uid"], max_nodes),
        )
        methods = [
            {"class_qn": r[0], "class_name": r[1], "name": r[2], "vis": r[3], "kind": r[4]}
            for r in await cur.fetchall()
        ]
        await cur.close()

        cur = await conn.execute(
            "SELECT td.qualified_name, td.name, parent.qualified_name, parent.name FROM edges rel1 "
            "JOIN nodes td ON td.uid = rel1.to_uid AND td.labels = 'TypeDef' "
            "JOIN edges rel2 ON rel2.from_uid = td.uid AND rel2.rel_type = 'INHERITS' "
            "JOIN nodes parent ON parent.uid = rel2.to_uid "
            "WHERE rel1.from_uid = ? AND rel1.rel_type = 'DEFINES' LIMIT ?",
            (mod["uid"], max_nodes),
        )
        inherits = [
            {"child_qn": r[0], "child_name": r[1], "parent_qn": r[2], "parent_name": r[3]} for r in await cur.fetchall()
        ]
        await cur.close()

        return {"module": mod, "entities": entities, "methods": methods, "inherits": inherits}

    async def get_module_summary(
        self, project: str, path: str, limit: int, edge_limit: int
    ) -> dict[str, list[dict[str, Any]]]:
        conn = await self._get_conn()

        clause, extra = _prefix_clause("file_path", path)
        cur = await conn.execute(
            "SELECT qualified_name, name, file_path, json_extract(props_json, '$.docstring') FROM nodes "
            f"WHERE project_name = ? AND labels = 'Module'{clause} ORDER BY file_path LIMIT ?",
            [project, *extra, limit],
        )
        modules = [{"qn": r[0], "name": r[1], "file_path": r[2], "docstring": r[3]} for r in await cur.fetchall()]
        await cur.close()

        clause, extra = _prefix_clause("e.file_path", path)
        cur = await conn.execute(
            "SELECT e.uid, e.name, e.qualified_name, e.labels, e.kind, "
            "json_extract(e.props_json, '$.visibility'), json_extract(e.props_json, '$.signature'), "
            "json_extract(e.props_json, '$.docstring'), json_extract(e.props_json, '$.line_start'), "
            "json_extract(e.props_json, '$.line_end'), e.file_path, p.qualified_name FROM nodes e "
            "LEFT JOIN edges rel ON rel.to_uid = e.uid AND rel.rel_type = 'DEFINES' "
            "LEFT JOIN nodes p ON p.uid = rel.from_uid "
            "WHERE e.project_name = ? AND e.labels IN ('TypeDef', 'Callable', 'Value')"
            f"{clause} ORDER BY e.file_path, json_extract(e.props_json, '$.line_start') LIMIT ?",
            [project, *extra, limit],
        )
        entities = [
            {
                "uid": r[0],
                "name": r[1],
                "qn": r[2],
                "label": r[3],
                "kind": r[4],
                "vis": r[5],
                "sig": r[6],
                "docstring": r[7],
                "line_start": r[8],
                "line_end": r[9],
                "file_path": r[10],
                "parent_qn": r[11],
            }
            for r in await cur.fetchall()
        ]
        await cur.close()

        structural = "('CALLS', 'INHERITS', 'IMPLEMENTS', 'USES_TYPE', 'OVERRIDES')"
        boundary = "('CALLS', 'INHERITS', 'IMPLEMENTS', 'USES_TYPE', 'OVERRIDES', 'IMPORTS')"
        clause_a, extra_a = _prefix_clause("a.file_path", path)
        clause_b, extra_b = _prefix_clause("b.file_path", path)
        # "not in scope" has no _prefix_clause equivalent. An empty *path* puts
        # everything in scope, which makes both boundary lists empty by definition.
        a_outside = " AND (a.file_path IS NULL OR substr(a.file_path, 1, ?) != ?)" if path else " AND 1 = 0"
        b_outside = " AND (b.file_path IS NULL OR substr(b.file_path, 1, ?) != ?)" if path else " AND 1 = 0"
        out_extra: list[Any] = [len(path), path] if path else []

        cur = await conn.execute(
            "SELECT a.qualified_name, b.qualified_name, r.rel_type, r.props_json FROM edges r "
            "JOIN nodes a ON a.uid = r.from_uid AND a.project_name = ? "
            "JOIN nodes b ON b.uid = r.to_uid AND b.project_name = ? "
            f"WHERE r.rel_type IN {structural} AND a.uid != b.uid{clause_a}{clause_b} "
            "ORDER BY 3, 1, 2 LIMIT ?",
            [project, project, *extra_a, *extra_b, edge_limit],
        )
        internal_edges = [
            {"from_qn": r[0], "to_qn": r[1], "rel_type": r[2], "props": json.loads(r[3]) if r[3] else {}}
            for r in await cur.fetchall()
        ]
        await cur.close()

        cur = await conn.execute(
            "SELECT a.qualified_name, a.name, a.file_path, a.labels, b.qualified_name, r.rel_type, r.props_json "
            "FROM edges r JOIN nodes a ON a.uid = r.from_uid AND a.project_name = ? "
            "JOIN nodes b ON b.uid = r.to_uid AND b.project_name = ? "
            f"WHERE r.rel_type IN {boundary}{clause_b}{a_outside} ORDER BY 6, 5, 1 LIMIT ?",
            [project, project, *extra_b, *out_extra, edge_limit],
        )
        fan_in = [
            {
                "from_qn": r[0],
                "from_name": r[1],
                "from_path": r[2],
                "from_label": r[3],
                "to_qn": r[4],
                "rel_type": r[5],
                "props": json.loads(r[6]) if r[6] else {},
            }
            for r in await cur.fetchall()
        ]
        await cur.close()

        cur = await conn.execute(
            "SELECT a.qualified_name, b.qualified_name, b.name, b.file_path, b.labels, r.rel_type, r.props_json "
            "FROM edges r JOIN nodes a ON a.uid = r.from_uid AND a.project_name = ? "
            "JOIN nodes b ON b.uid = r.to_uid AND b.project_name = ? "
            f"WHERE r.rel_type IN {boundary}{clause_a}{b_outside} ORDER BY 6, 1, 2 LIMIT ?",
            [project, project, *extra_a, *out_extra, edge_limit],
        )
        fan_out = [
            {
                "from_qn": r[0],
                "to_qn": r[1],
                "to_name": r[2],
                "to_path": r[3],
                "to_label": r[4],
                "rel_type": r[5],
                "props": json.loads(r[6]) if r[6] else {},
            }
            for r in await cur.fetchall()
        ]
        await cur.close()

        clause_e, extra_e = _prefix_clause("e.file_path", path)
        cur = await conn.execute(
            "SELECT d.qualified_name, d.name, d.labels, e.qualified_name, "
            "json_extract(r.props_json, '$.link_type') FROM edges r "
            "JOIN nodes d ON d.uid = r.from_uid "
            "JOIN nodes e ON e.uid = r.to_uid AND e.project_name = ? "
            f"WHERE r.rel_type = 'DOCUMENTS'{clause_e} ORDER BY 4, 1 LIMIT ?",
            [project, *extra_e, limit],
        )
        docs = [
            {"doc_qn": r[0], "doc_name": r[1], "doc_label": r[2], "to_qn": r[3], "link_type": r[4]}
            for r in await cur.fetchall()
        ]
        await cur.close()

        return {
            "modules": modules,
            "entities": entities,
            "internal_edges": internal_edges,
            "fan_in": fan_in,
            "fan_out": fan_out,
            "docs": docs,
        }

    # -- Context expansion / navigation (search/engine.py's expand_context) ---

    async def _label_matches(self, conn: SqlConnection, uid: str, label: str) -> bool:
        """Whether *uid* carries *label* (mirrors Cypher's inline ``:Label`` node-pattern filter).

        Always ``True`` when *label* is empty (no filter requested).
        """
        if not label:
            return True
        cur = await conn.execute("SELECT 1 FROM nodes WHERE uid = ? AND labels = ?", (uid, label))
        row = await cur.fetchone()
        await cur.close()
        return row is not None

    async def get_entity_by_uid(self, uid: str, label: str = "") -> dict[str, Any] | None:
        conn = await self._get_conn()
        label_clause = " AND labels = ?" if label else ""
        params: list[Any] = [uid, *([label] if label else [])]
        cur = await conn.execute(f"SELECT {_NODE_COLUMNS} FROM nodes WHERE uid = ?{label_clause}", params)
        row = await cur.fetchone()
        await cur.close()
        return _row_to_node(row) if row else None

    async def get_defining_parent(self, uid: str) -> dict[str, Any] | None:
        conn = await self._get_conn()
        cur = await conn.execute(
            f"SELECT {_node_columns('p')} FROM edges e JOIN nodes p ON p.uid = e.from_uid "
            "WHERE e.rel_type = 'DEFINES' AND e.to_uid = ? LIMIT 1",
            (uid,),
        )
        row = await cur.fetchone()
        await cur.close()
        return _row_to_node(row) if row else None

    async def get_sibling_entities(self, uid: str, limit: int) -> list[dict[str, Any]]:
        conn = await self._get_conn()
        cur = await conn.execute(
            f"SELECT {_node_columns('s')} FROM edges e1 "
            "JOIN edges e2 ON e2.from_uid = e1.from_uid AND e2.rel_type = 'DEFINES' "
            "JOIN nodes s ON s.uid = e2.to_uid "
            "WHERE e1.rel_type = 'DEFINES' AND e1.to_uid = ? AND s.uid <> ? LIMIT ?",
            (uid, uid, limit),
        )
        rows = await cur.fetchall()
        await cur.close()
        return [_row_to_node(r) for r in rows]

    async def get_package_docstring(self, uid: str) -> str | None:
        """Walks up the DEFINES chain up to 3 hops, mirroring Cypher's ``[:DEFINES*1..3]``
        variable-length pattern (no native equivalent over the flat ``edges`` table).
        """
        conn = await self._get_conn()
        current = uid
        for _ in range(3):
            cur = await conn.execute(
                "SELECT p.uid, p.labels, json_extract(p.props_json, '$.docstring') FROM edges e "
                "JOIN nodes p ON p.uid = e.from_uid WHERE e.rel_type = 'DEFINES' AND e.to_uid = ? LIMIT 1",
                (current,),
            )
            row = await cur.fetchone()
            await cur.close()
            if row is None:
                return None
            parent_uid, labels, docstring = row
            if labels == "Module":
                return docstring
            current = parent_uid
        return None

    async def get_callers(self, uid: str, label: str, call_depth: int, limit: int) -> list[dict[str, Any]]:
        """Mirror of ``GraphClient.get_callers``. The ORDER BY is the conformance
        requirement (ATL-192): a BFS returns its reachable set in traversal order, which
        is not the order Memgraph's expansion produces, so an unordered LIMIT makes the
        two backends disagree about *which* callers a truncated answer contains."""
        conn = await self._get_conn()
        if not await self._label_matches(conn, uid, label):
            return []
        reached = await self._bfs_reachable(conn, uid, "to_uid", "from_uid", ("CALLS",), call_depth, include_start=True)
        if not reached:
            return []
        uids = list(reached)
        placeholders = ",".join("?" * len(uids))
        cur = await conn.execute(
            f"SELECT {_NODE_COLUMNS} FROM nodes WHERE uid IN ({placeholders}) "
            f"AND labels = 'Callable' ORDER BY COALESCE(qualified_name, ''), uid LIMIT ?",
            [*uids, limit],
        )
        rows = await cur.fetchall()
        await cur.close()
        return [_row_to_node(r) for r in rows]

    async def get_callees(self, uid: str, label: str, call_depth: int, limit: int) -> list[dict[str, Any]]:
        """Mirror of ``GraphClient.get_callees`` — see ``get_callers`` for the ORDER BY."""
        conn = await self._get_conn()
        if not await self._label_matches(conn, uid, label):
            return []
        reached = await self._bfs_reachable(conn, uid, "from_uid", "to_uid", ("CALLS",), call_depth, include_start=True)
        if not reached:
            return []
        uids = list(reached)
        placeholders = ",".join("?" * len(uids))
        cur = await conn.execute(
            f"SELECT {_NODE_COLUMNS} FROM nodes WHERE uid IN ({placeholders}) "
            f"AND labels = 'Callable' ORDER BY COALESCE(qualified_name, ''), uid LIMIT ?",
            [*uids, limit],
        )
        rows = await cur.fetchall()
        await cur.close()
        return [_row_to_node(r) for r in rows]

    async def get_linked_docs(self, uid: str, label: str, limit: int) -> list[dict[str, Any]]:
        conn = await self._get_conn()
        if not await self._label_matches(conn, uid, label):
            return []
        cur = await conn.execute(
            f"SELECT {_node_columns('doc')}, json_extract(e.props_json, '$.link_type'), "
            "json_extract(e.props_json, '$.stale'), json_extract(e.props_json, '$.anchor_hash') "
            "FROM edges e JOIN nodes doc ON doc.uid = e.from_uid "
            "WHERE e.rel_type = 'DOCUMENTS' AND e.to_uid = ? "
            "AND doc.labels IN ('DocSection', 'Note', 'DocFile') LIMIT ?",
            (uid, limit),
        )
        rows = await cur.fetchall()
        await cur.close()
        return [
            {
                "node": _row_to_node(row[:9]),
                "link_type": row[9],
                "stale": bool(row[10]) if row[10] is not None else None,
                "anchor_hash": row[11],
            }
            for row in rows
        ]

    # -- get_node cascade / status queries (server/mcp.py, cli.py) ------------

    async def get_node_exact_matches(self, name: str, label: str, limit: int) -> list[dict[str, Any]]:
        """Mirror of ``GraphClient.get_node_exact_matches``.

        Only the second branch is ordered (ATL-192), matching the Cypher: ``uid`` is the
        primary key, so its LIMIT can never cut anything.
        """
        conn = await self._get_conn()
        label_clause = " AND labels = ?" if label else ""
        label_params = [label] if label else []
        results: list[dict[str, Any]] = []
        cur = await conn.execute(
            f"SELECT {_NODE_COLUMNS} FROM nodes WHERE uid = ?{label_clause} LIMIT ?", [name, *label_params, limit]
        )
        rows = await cur.fetchall()
        await cur.close()
        results.extend({"n": _row_to_node(r)} for r in rows)
        cur = await conn.execute(
            f"SELECT {_NODE_COLUMNS} FROM nodes WHERE name = ?{label_clause} "
            f"ORDER BY COALESCE(qualified_name, ''), uid LIMIT ?",
            [name, *label_params, limit],
        )
        rows = await cur.fetchall()
        await cur.close()
        results.extend({"n": _row_to_node(r)} for r in rows)
        return results

    async def get_node_partial_matches(self, name: str, label: str, limit: int) -> list[dict[str, Any]]:
        """Mirror of ``GraphClient.get_node_partial_matches``.

        ``score`` ranks the three branches against each other; the ORDER BY ranks within
        one, which is what each branch's LIMIT actually cuts on (ATL-192).
        """
        conn = await self._get_conn()
        label_clause = " AND labels = ?" if label else ""
        label_params = [label] if label else []
        results: list[dict[str, Any]] = []

        async def _branch(where: str, params: list[Any], score: int) -> None:
            cur = await conn.execute(
                f"SELECT {_NODE_COLUMNS} FROM nodes WHERE {where}{label_clause} "
                f"ORDER BY COALESCE(qualified_name, ''), uid LIMIT ?",
                [*params, *label_params, limit],
            )
            rows = await cur.fetchall()
            await cur.close()
            results.extend({"n": _row_to_node(r), "_match_score": score} for r in rows)

        escaped = _like_literal(name)
        await _branch("qualified_name LIKE ? ESCAPE '\\'", [f"%.{escaped}"], 3)
        await _branch("qualified_name LIKE ? ESCAPE '\\'", [f"{escaped}.%"], 2)
        await _branch(
            "(qualified_name LIKE ? ESCAPE '\\' OR name LIKE ? ESCAPE '\\')",
            [f"%{escaped}%", f"%{escaped}%"],
            1,
        )
        return results

    async def get_label_counts(self) -> dict[str, int]:
        conn = await self._get_conn()
        cur = await conn.execute("SELECT labels, COUNT(*) FROM nodes GROUP BY labels ORDER BY COUNT(*) DESC")
        rows = await cur.fetchall()
        await cur.close()
        return {r[0]: r[1] for r in rows}

    # -- Detector lookups (parsing/languages/*.py) -----------------------------

    async def find_entity_uids(self, project_name: str, wanted: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
        """One statement per distinct label; see the Memgraph docstring for why batched.

        ``(labels, project_name, name)`` is exactly `ix_nodes_labels_project_name`, so
        this seeks rather than scans even though `labels` is a bound parameter — which no
        partial index could have served (ADR-0045).
        """
        if not wanted:
            return {}
        by_label: dict[str, list[str]] = defaultdict(list)
        for label, name in wanted:
            by_label[label].append(name)

        conn = await self._get_conn()
        out: dict[tuple[str, str], str] = {}
        for label, names in by_label.items():
            for chunk in _chunks(sorted(set(names)), 900):
                if not chunk:
                    continue
                placeholders = ",".join("?" * len(chunk))
                cur = await conn.execute(
                    f"SELECT name, uid FROM nodes WHERE labels = ? AND project_name = ? AND name IN ({placeholders})",
                    (label, project_name, *chunk),
                )
                rows = await cur.fetchall()
                await cur.close()
                # First row wins, matching the singular version's LIMIT 1.
                for name, uid in rows:
                    out.setdefault((label, name), uid)
        return out

    async def find_overridden_method(
        self, project_name: str, bases: list[str], method_name: str
    ) -> tuple[str, list[str]] | None:
        if not bases:
            return None
        conn = await self._get_conn()
        placeholders = ",".join("?" * len(bases))
        cur = await conn.execute(
            "SELECT m.uid, m.props_json FROM edges e "
            "JOIN nodes base ON base.uid = e.from_uid AND base.labels = 'TypeDef' "
            "JOIN nodes m ON m.uid = e.to_uid AND m.labels = 'Callable' "
            f"WHERE e.rel_type = 'DEFINES' AND base.project_name = ? "
            f"AND base.name IN ({placeholders}) AND m.name = ? LIMIT 1",
            [project_name, *bases, method_name],
        )
        row = await cur.fetchone()
        await cur.close()
        if row is None:
            return None
        uid, props_json = row
        props = json.loads(props_json) if props_json else {}
        return uid, props.get("tags") or []

    async def get_project_dependency_edges(self) -> list[dict[str, Any]]:
        conn = await self._get_conn()
        cur = await conn.execute(
            "SELECT a.name, b.name FROM edges e "
            "JOIN nodes a ON a.uid = e.from_uid AND a.labels = 'Project' "
            "JOIN nodes b ON b.uid = e.to_uid AND b.labels = 'Project' "
            "WHERE e.rel_type = 'DEPENDS_ON'"
        )
        rows = await cur.fetchall()
        await cur.close()
        return [{"from_proj": r[0], "to_proj": r[1]} for r in rows]

    # -- Dream-mode lint queries (dream.py) ------------------------------------

    async def get_existing_uids(self, uids: list[str]) -> set[str]:
        if not uids:
            return set()
        conn = await self._get_conn()
        result: set[str] = set()
        for chunk in _chunks(uids):
            if not chunk:
                continue
            placeholders = ",".join("?" * len(chunk))
            cur = await conn.execute(f"SELECT uid FROM nodes WHERE uid IN ({placeholders})", chunk)
            rows = await cur.fetchall()
            await cur.close()
            result.update(r[0] for r in rows)
        return result

    async def get_orphan_notes(self) -> list[dict[str, Any]]:
        conn = await self._get_conn()
        cur = await conn.execute(
            "SELECT uid, name, project_name, file_path FROM nodes n "
            "WHERE labels = 'Note' AND NOT EXISTS ("
            "SELECT 1 FROM edges e WHERE e.rel_type = 'LINKS_TO' AND (e.from_uid = n.uid OR e.to_uid = n.uid))"
        )
        rows = await cur.fetchall()
        await cur.close()
        return [{"uid": r[0], "name": r[1], "project_name": r[2], "file_path": r[3]} for r in rows]

    async def get_broken_anchor_notes(self) -> list[dict[str, Any]]:
        """``has_broken_anchors`` is never set on this backend (no zombie-node preservation on
        delete — see the module docstring), so this only ever matches via ``unresolved_anchors``.
        """
        conn = await self._get_conn()
        cur = await conn.execute(
            "SELECT uid, name, project_name, file_path, json_extract(props_json, '$.unresolved_anchors') FROM nodes "
            "WHERE labels = 'Note' AND ("
            "json_extract(props_json, '$.has_broken_anchors') = 1 OR "
            "(json_extract(props_json, '$.unresolved_anchors') IS NOT NULL "
            "AND json_array_length(props_json, '$.unresolved_anchors') > 0))"
        )
        rows = await cur.fetchall()
        await cur.close()
        return [
            {
                "uid": r[0],
                "name": r[1],
                "project_name": r[2],
                "file_path": r[3],
                "unresolved_anchors": json.loads(r[4]) if r[4] else None,
            }
            for r in rows
        ]

    async def get_inbox_note_paths(self) -> list[str]:
        conn = await self._get_conn()
        cur = await conn.execute(
            "SELECT file_path FROM nodes WHERE labels = 'Note' "
            "AND (kind = 'draft' OR file_path LIKE '%/inbox/%') ORDER BY file_path"
        )
        rows = await cur.fetchall()
        await cur.close()
        return [r[0] for r in rows]

    async def get_notes_for_dedup(self) -> list[dict[str, Any]]:
        conn = await self._get_conn()
        cur = await conn.execute("SELECT uid, project_name, name, embedding FROM nodes WHERE labels = 'Note'")
        rows = await cur.fetchall()
        await cur.close()
        result: list[dict[str, Any]] = []
        for uid, project_name, name, blob in rows:
            vector = list(struct.unpack(f"<{len(blob) // 4}f", blob)) if blob else None
            result.append({"uid": uid, "project_name": project_name, "name": name, "embedding": vector})
        return result

    # -- Git signals write path (indexing/git_signals.py) -------------------------

    async def write_git_file_signals(self, project_name: str, label: str, items: list[dict[str, Any]]) -> int:
        if not items:
            return 0
        conn = await self._get_conn()
        matched = 0
        for item in items:
            cur = await conn.execute(
                "UPDATE nodes SET props_json = json_patch(props_json, ?) "
                "WHERE project_name = ? AND file_path = ? AND labels = ?",
                (
                    _dumps(
                        {
                            "git_commit_count": item["cc"],
                            "git_author_count": item["ac"],
                            "git_days_since_last_commit": item["days"],
                        }
                    ),
                    project_name,
                    item["fp"],
                    label,
                ),
            )
            matched += cur.rowcount
            await cur.close()
        await conn.commit()
        return matched

    async def write_co_change_edges(self, project_name: str, pairs: list[dict[str, Any]]) -> int:
        if not pairs:
            return 0
        conn = await self._get_conn()
        rows: list[tuple[str, str, str, str]] = []
        for pair in pairs:
            cur_a = await conn.execute(
                "SELECT uid FROM nodes WHERE labels = 'Module' AND project_name = ? AND file_path = ?",
                (project_name, pair["a"]),
            )
            a_row = await cur_a.fetchone()
            await cur_a.close()
            cur_b = await conn.execute(
                "SELECT uid FROM nodes WHERE labels = 'Module' AND project_name = ? AND file_path = ?",
                (project_name, pair["b"]),
            )
            b_row = await cur_b.fetchone()
            await cur_b.close()
            if a_row and b_row:
                rows.append((a_row[0], b_row[0], "CO_CHANGES_WITH", _dumps({"count": pair["cnt"]})))
        if rows:
            await conn.executemany(
                "INSERT INTO edges(from_uid, to_uid, rel_type, props_json) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(from_uid, to_uid, rel_type) DO UPDATE SET props_json = excluded.props_json",
                rows,
            )
            await conn.commit()
        return len(rows)
