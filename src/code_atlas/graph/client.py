"""Async Memgraph client for Code Atlas.

Handles connection lifecycle, schema application, and version management.
Uses the neo4j async driver (Bolt protocol) which is compatible with Memgraph.
"""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import contextvars
import re
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import groupby
from operator import attrgetter
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

from loguru import logger
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import TransientError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from code_atlas.schema import (
    _EMBEDDABLE_LABELS,
    _REFERENCE_COUNTED_LABELS,
    _TEXT_SEARCHABLE_LABELS,
    GLOBAL_PROJECT,
    RESOURCE_FILE_PREFIX,
    SCHEMA_VERSION,
    CallableKind,
    NodeLabel,
    NoteKind,
    RelType,
    TypeDefKind,
    env_var_uid,
    generate_composite_index_ddl,
    generate_drop_text_index_ddl,
    generate_drop_vector_index_ddl,
    generate_existence_constraint_ddl,
    generate_index_ddl,
    generate_text_index_ddl,
    generate_unique_constraint_ddl,
    generate_vector_index_ddl,
    resource_file_uid,
)
from code_atlas.search.engine import matches_test_pattern
from code_atlas.settings import SearchSettings
from code_atlas.telemetry import get_tracer

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Collection, Sequence

    from neo4j import AsyncDriver

    from code_atlas.parsing.ast import ParsedEntity, ParsedRelationship
    from code_atlas.parsing.detectors import PropertyEnrichment
    from code_atlas.settings import AtlasSettings

_tracer = get_tracer(__name__)

_VALID_LABELS: frozenset[str] = frozenset(lbl.value for lbl in NodeLabel)

# ---------------------------------------------------------------------------
# Relationship routing registry
#
# Every RelType must be creatable by *some* code path, or entities carrying
# it are silently dropped with no edges and no error (the exact failure
# class _validate_relationship_routing guards against — see schema.py's
# _validate_schema_completeness for the analogous node-label guarantee).
# ---------------------------------------------------------------------------

# Routed by _create_relationships via a direct uid-to-uid MATCH+MERGE.
_UID_ROUTED_REL_TYPES: frozenset[RelType] = frozenset(
    {
        RelType.DEFINES,
        RelType.CONTAINS,
        RelType.OVERRIDES,
        RelType.EXPORTS,
        RelType.TESTS,
        RelType.HANDLES_ROUTE,
        RelType.HANDLES_EVENT,
        RelType.HANDLES_COMMAND,
        RelType.INJECTED_INTO,
        RelType.LINKS_TO,
        RelType.DERIVED_FROM,
        RelType.SUPERSEDES,
    }
)

# Routed by _create_relationships via name/path matching (INHERITS/IMPLEMENTS
# by type name, DOCUMENTS by symbol or file-path suffix via _create_doc_links).
# IMPLEMENTS also has a uid-shaped path (detector-emitted target uids), but it
# always falls back to this name-matched route for parser-emitted bare names.
# DOCUMENTS has two further post-batch routes that carry no ParsedRelationship
# of their own: resolve_anchors (link_type='anchor') and resolve_citations
# (link_type='citation', driven by the entity's `citations` property).
_NAME_ROUTED_REL_TYPES: frozenset[RelType] = frozenset(
    {
        RelType.IMPLEMENTS,
        RelType.DOCUMENTS,
    }
)

# Resolved post-batch, after all files in a batch are upserted (see
# GraphClient.resolve_calls / resolve_imports / resolve_uses_type /
# resolve_config_refs) — not part of _create_relationships at all.
# READS_ENV/REFERENCES_FILE belong here because their *target* node does not
# exist until resolution MERGEs it, exactly like an ExternalPackage stub.
_POST_BATCH_REL_TYPES: frozenset[RelType] = frozenset(
    {
        RelType.CALLS,
        RelType.IMPORTS,
        RelType.USES_TYPE,
        RelType.READS_ENV,
        RelType.REFERENCES_FILE,
        RelType.REFERENCES,
        # Same reason as REFERENCES: the registrar is named, not uid'd, and must resolve
        # in the decorated entity's own scope rather than by a project-wide match.
        RelType.REGISTERED_BY,
        # INHERITS is here, not name-routed, because a base is very often external
        # (StrEnum, ABC, Protocol, BaseSettings) and the ExternalSymbol it must point at
        # is MERGEd by resolve_imports — which runs in the deferred flush, after
        # _create_relationships. Written at create time the MATCH simply found nothing,
        # so 43 of 45 classes with a base carried no inheritance edge at all.
        RelType.INHERITS,
    }
)

# Config references, split out of _POST_BATCH_REL_TYPES so callers that only
# care about the EnvVar/ResourceFile pair (the AST consumer's partitioning,
# the SQLite backend's create-phase skip list) do not have to re-enumerate it.
_CONFIG_REF_REL_TYPES: frozenset[RelType] = frozenset(
    {
        RelType.READS_ENV,
        RelType.REFERENCES_FILE,
    }
)

# Created by dedicated, out-of-band methods rather than per-file parsing:
# DEPENDS_ON is project-to-project (monorepo dependency graph); SIMILAR_TO
# is materialized by the (future) dream-mode consolidation pass, not parsing;
# CO_CHANGES_WITH is written by indexing/git_signals.py's write_git_signals,
# triggered by the one-shot `atlas mine-git-history` CLI command, not by the
# parsing/indexing pipeline.
_OUT_OF_BAND_REL_TYPES: frozenset[RelType] = frozenset(
    {
        RelType.DEPENDS_ON,
        RelType.SIMILAR_TO,
        RelType.CO_CHANGES_WITH,
    }
)


def _validate_relationship_routing() -> None:
    """Ensure every RelType is routed by some code path.

    Raises RuntimeError at import time if any RelType is missing — prevents
    a new relationship type from silently producing zero edges.
    """
    routed = _UID_ROUTED_REL_TYPES | _NAME_ROUTED_REL_TYPES | _POST_BATCH_REL_TYPES | _OUT_OF_BAND_REL_TYPES
    missing = set(RelType) - routed
    if missing:
        raise RuntimeError(f"RelTypes not routed anywhere in GraphClient: {missing}")


_validate_relationship_routing()


# ---------------------------------------------------------------------------
# Config references (EnvVar / ResourceFile)
#
# SECURITY INVARIANT — capture NAMES, never VALUES.
#
# ``os.getenv("API_KEY", "sk-live-abc123")`` puts a live secret in the default
# argument, and a referenced config file's *contents* are secrets far more
# often than its path is.  If a value or a default ever reached a node
# property, it would be persisted in the graph AND — for any label that is
# embeddable — shipped verbatim to a third-party embedding API.
#
# The invariant is enforced structurally rather than by review: _plan_config_refs
# builds each node from a fixed four-key allowlist derived only from the
# reference's *target name*, and never reads ``rel.properties`` at all.  Edges
# are written bare for the same reason — a parser that starts attaching a
# ``default=`` property cannot leak it through this path.  Neither label is in
# _EMBEDDABLE_LABELS, so nothing here can reach an embedding provider even if
# the allowlist were widened later.
# ---------------------------------------------------------------------------


class _ConfigRefPlan(NamedTuple):
    """Nodes to MERGE and edges to create for one batch of config references."""

    env_nodes: dict[str, dict[str, str]]  # uid -> allowlisted node properties
    file_nodes: dict[str, dict[str, str]]  # uid -> allowlisted node properties
    edges: list[tuple[str, str, str]]  # (from_uid, to_uid, rel_type)


def _normalize_resource_path(raw: str) -> str:
    """Canonicalize a referenced path so one file yields one node.

    Backslashes fold to forward slashes and leading ``./`` segments are
    stripped, so ``./data\\fixtures.json`` and ``data/fixtures.json`` converge.
    Nothing else is rewritten: the path is not resolved against the filesystem
    (the reference may well point at a file that does not exist) and ``..`` is
    left intact rather than collapsed, since collapsing it would silently merge
    references made from different directories.
    """
    path = raw.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def _plan_config_refs(project_name: str, ref_rels: list[ParsedRelationship]) -> _ConfigRefPlan:
    """Turn READS_ENV/REFERENCES_FILE references into nodes + edges to write.

    Pure and backend-agnostic — both graph backends share it so they cannot
    drift on uid construction, normalization, or the names-only allowlist.
    """
    env_nodes: dict[str, dict[str, str]] = {}
    file_nodes: dict[str, dict[str, str]] = {}
    edges: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for rel in ref_rels:
        if rel.rel_type == RelType.READS_ENV:
            name = rel.to_name.strip()
            if not name:
                continue
            uid = env_var_uid(name)
            # GLOBAL_PROJECT, not project_name: one node per variable name
            # across every indexed repo (see schema.env_var_uid).
            env_nodes.setdefault(
                uid,
                {"uid": uid, "project_name": GLOBAL_PROJECT, "name": name, "qualified_name": uid},
            )
        elif rel.rel_type == RelType.REFERENCES_FILE:
            path = _normalize_resource_path(rel.to_name)
            if not path:
                continue
            uid = resource_file_uid(project_name, path)
            file_nodes.setdefault(
                uid,
                {
                    "uid": uid,
                    "project_name": project_name,
                    "name": path.rsplit("/", 1)[-1],
                    "qualified_name": f"{RESOURCE_FILE_PREFIX}{path}",
                    # Without this the path survives only inside the uid, so nothing that
                    # filters or renders by file_path can see the node at all.
                    "file_path": path,
                },
            )
        else:
            continue

        key = (rel.from_qualified_name, uid, rel.rel_type.value)
        if key not in seen:
            seen.add(key)
            edges.append(key)

    return _ConfigRefPlan(env_nodes=env_nodes, file_nodes=file_nodes, edges=edges)


def _assert_valid_label(label: str) -> None:
    """Raise ValueError if *label* is not a known NodeLabel value.

    Prevents Cypher injection when labels are interpolated into queries.
    """
    if label not in _VALID_LABELS:
        msg = f"Invalid node label: {label!r}"
        raise ValueError(msg)


def _build_graph_search_query(
    label: str,
    project_clause: str,
    fetch_limit: int,
) -> str:
    """Build a UNION ALL Cypher query for the 3-stage graph search cascade.

    Collapses exact / suffix / contains matching into a single round-trip.
    """
    label_filter = f":{label}" if label else ""

    return (
        f"MATCH (n{label_filter}) WHERE n.name = $query{project_clause} "
        f"RETURN n AS node, 3.0 AS score LIMIT {fetch_limit} "
        f"UNION ALL "
        f"MATCH (n{label_filter}) WHERE n.qualified_name ENDS WITH $suffix{project_clause} "
        f"RETURN n AS node, 2.0 AS score LIMIT {fetch_limit} "
        f"UNION ALL "
        f"MATCH (n{label_filter}) WHERE (n.qualified_name CONTAINS $query OR n.name CONTAINS $query)"
        f"{project_clause} RETURN n AS node, 1.0 AS score LIMIT {fetch_limit}"
    )


def _format_path_hops(path_nodes: list[Any], path_rels: list[Any]) -> list[dict[str, Any]]:
    """Render a Cypher path's nodes/relationships into per-hop dicts for ``trace_path_between``.

    Includes the CALLS ``confidence``/``strategy`` edge properties (ADR-0014)
    and the ``weight``/``from_test`` properties that amend it, when present on
    the traversed edge — ``weight`` explains why this path won an equal-hop-count
    tie-break, ``from_test`` shows the hop runs through a test caller.
    """
    hops: list[dict[str, Any]] = []
    for i, rel in enumerate(path_rels):
        from_props = dict(path_nodes[i].items()) if hasattr(path_nodes[i], "items") else {}
        to_props = dict(path_nodes[i + 1].items()) if hasattr(path_nodes[i + 1], "items") else {}
        rel_props = dict(rel.items()) if hasattr(rel, "items") else {}
        hop: dict[str, Any] = {
            "from": {"uid": from_props.get("uid"), "name": from_props.get("name")},
            "to": {"uid": to_props.get("uid"), "name": to_props.get("name")},
            "edge_type": getattr(rel, "type", None),
        }
        if "confidence" in rel_props:
            hop["confidence"] = rel_props["confidence"]
        if "strategy" in rel_props:
            hop["strategy"] = rel_props["strategy"]
        if "weight" in rel_props:
            hop["weight"] = rel_props["weight"]
        if "from_test" in rel_props:
            hop["from_test"] = rel_props["from_test"]
        hops.append(hop)
    return hops


class QueryTimeoutError(Exception):
    """Raised when a read query exceeds the configured timeout."""

    def __init__(self, timeout_s: float, query_prefix: str = "") -> None:
        self.timeout_s = timeout_s
        self.query_prefix = query_prefix
        super().__init__(f"Query timed out after {timeout_s}s: {query_prefix}")


class EntityHashData(NamedTuple):
    """Stored entity data used for delta comparison during upsert."""

    content_hash: str
    line_start: int
    line_end: int
    signature: str | None
    docstring: str | None
    label: str


@dataclass(frozen=True)
class UpsertResult:
    """Result of a delta-aware upsert for a single file."""

    added: list[str] = field(default_factory=list)  # qualified_names of new entities
    modified: list[str] = field(default_factory=list)  # qualified_names with changed content_hash
    deleted: list[str] = field(default_factory=list)  # qualified_names removed from file
    unchanged: list[str] = field(default_factory=list)  # qualified_names with matching content_hash
    modified_significance: dict[str, str] = field(default_factory=dict)  # qualified_name → Significance value


@dataclass
class _FileClassification:
    """Delta classification for a single file's entities."""

    added: list[ParsedEntity]
    modified: list[ParsedEntity]
    deleted_by_label: dict[str, list[str]]  # label → [uids]
    shifted: list[ParsedEntity]
    result: UpsertResult


def _classify_file(
    old_data: dict[str, EntityHashData],
    entities: list[ParsedEntity],
    strip_uid: Callable[[str], str],
) -> _FileClassification:
    """Pure-Python classification of entities for a single file.

    Compares new entities against *old_data* and classifies each as
    added/modified/deleted/unchanged.  Shared by both single-file and
    batched upsert paths.
    """
    new_hashes = {e.qualified_name: e.content_hash for e in entities}
    new_entity_map = {e.qualified_name: e for e in entities}

    old_uids = set(old_data)
    new_uids = set(new_hashes)

    added_uids = new_uids - old_uids
    deleted_uids = old_uids - new_uids
    common_uids = old_uids & new_uids
    modified_uids = {uid for uid in common_uids if old_data[uid].content_hash != new_hashes[uid]}
    unchanged_uids = common_uids - modified_uids

    # Deleted entities grouped by label for index-backed deletion
    deleted_by_label: dict[str, list[str]] = defaultdict(list)
    for uid in deleted_uids:
        deleted_by_label[old_data[uid].label].append(uid)

    # Significance per modified entity
    mod_significance: dict[str, str] = {}
    for uid in modified_uids:
        old = old_data[uid]
        new_entity = new_entity_map[uid]
        qn = strip_uid(uid)
        if (new_entity.signature or "") != (old.signature or ""):
            mod_significance[qn] = "HIGH"
        elif (new_entity.docstring or "") != (old.docstring or ""):
            mod_significance[qn] = "MODERATE"
        else:
            mod_significance[qn] = "HIGH"

    # Position shifts for unchanged entities
    shifted: list[ParsedEntity] = []
    for uid in unchanged_uids:
        entity = new_entity_map[uid]
        if (entity.line_start, entity.line_end) != (old_data[uid].line_start, old_data[uid].line_end):
            shifted.append(entity)

    return _FileClassification(
        added=[new_entity_map[uid] for uid in added_uids],
        modified=[new_entity_map[uid] for uid in modified_uids],
        deleted_by_label=dict(deleted_by_label),
        shifted=shifted,
        result=UpsertResult(
            added=[strip_uid(uid) for uid in added_uids],
            modified=[strip_uid(uid) for uid in modified_uids],
            deleted=[strip_uid(uid) for uid in deleted_uids],
            unchanged=[strip_uid(uid) for uid in unchanged_uids],
            modified_significance=mod_significance,
        ),
    )


@dataclass
class _BatchClassification:
    """Cross-file classification of entities for batched upsert."""

    all_added: list[ParsedEntity]
    all_modified: list[ParsedEntity]
    all_deleted_by_label: dict[str, list[str]]  # label → [uids]
    all_shifted: list[ParsedEntity]
    per_file_results: dict[str, UpsertResult]  # file_path → UpsertResult
    new_file_paths: set[str]  # files with no prior data (skip rel delete)


def _node_project_name(record: dict[str, Any]) -> str:
    """Extract project_name from a record containing a neo4j Node."""
    node = record.get("node") or record.get("n")
    if node is None:
        return ""
    if hasattr(node, "get"):
        return node.get("project_name", "")
    return ""


def _record_uid(record: dict[str, Any]) -> str | None:
    """Extract uid from a record containing a neo4j Node."""
    node = record.get("node") or record.get("n")
    if node is None or not hasattr(node, "get"):
        return None
    return node.get("uid")


_BM25_RRF_K = 60
_BM25_UNSAFE_CHARS = frozenset('(){}[]^":')


def _sanitize_bm25_query(query: str) -> str:
    """Neutralize Tantivy query-syntax characters that make
    ``text_search.search_all`` raise on syntax-invalid input.

    Parens, brackets, braces, caret, colon, and quote are common in
    code-shaped queries (``embed_batch(texts)``, ``dict[str, Any]``,
    ``std::vector``) but are reserved Tantivy query syntax; an unbalanced
    or nested occurrence raises inside Memgraph. Replaced with spaces
    (not stripped) to keep word boundaries sane.
    """
    return "".join(" " if ch in _BM25_UNSAFE_CHARS else ch for ch in query)


def _fuse_bm25_results(
    results_per_index: list[list[dict[str, Any]]],
    k: int = _BM25_RRF_K,
) -> list[dict[str, Any]]:
    """Fuse per-index BM25 result lists via reciprocal rank fusion.

    Raw BM25 scores from ``text_search.search_all`` are not comparable
    across different label indices — each label has its own separate
    Tantivy index with its own corpus statistics (document frequency,
    average document length). Each index's own ranking (by its own score)
    is valid, so fusing by rank position avoids the cross-corpus bias of
    comparing raw scores directly. Returns records sorted by fused score
    descending, with ``score`` replaced by the fused value.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    rec_by_uid: dict[str, dict[str, Any]] = {}
    for batch in results_per_index:
        for rank, rec in enumerate(batch):
            uid = _record_uid(rec)
            if uid is None:
                continue
            rrf_scores[uid] += 1.0 / (k + rank + 1)
            rec_by_uid.setdefault(uid, rec)
    fused = [{**rec_by_uid[uid], "score": score} for uid, score in rrf_scores.items()]
    fused.sort(key=lambda r: r["score"], reverse=True)
    return fused


@dataclass(frozen=True)
class _AnchorLookup:
    """Pre-built, cross-project lookup tables for anchor (path-form) resolution.

    Anchors may target code in any project (uid/project-prefixed/absolute
    path forms), so — unlike CALLS/IMPORTS resolution — this lookup spans
    the whole graph rather than one project's own entities.
    """

    # project → file_path → [(uid, content_hash)]
    file_by_path: dict[str, dict[str, list[tuple[str, str]]]]
    # project → file_path → name → [(uid, content_hash)]
    symbols_by_path: dict[str, dict[str, dict[str, list[tuple[str, str]]]]]
    project_roots: dict[str, str]  # project → root_path (posix, resolved, no trailing slash)


def _resolve_anchor_path(project: str, path: str, lookup: _AnchorLookup) -> tuple[str, str] | None:
    """Resolve a path anchor to ``(file_uid, file_content_hash)`` within *project*.

    Exact ``file_path`` match first; falls back to a unique-suffix match
    (mirrors the suffix convention ``_create_doc_links`` uses for file
    refs). An ambiguous exact or suffix match resolves to nothing rather
    than guessing.
    """
    files = lookup.file_by_path.get(project)
    if not files:
        return None
    exact = files.get(path)
    if exact is not None:
        return exact[0] if len(exact) == 1 else None
    suffix = "/" + path
    candidates = [v for fp, vs in files.items() if fp.endswith(suffix) for v in vs]
    return candidates[0] if len(candidates) == 1 else None


def _resolve_anchor_symbol(project: str, file_path: str, symbol: str, lookup: _AnchorLookup) -> tuple[str, str] | None:
    """Resolve a ``#Symbol`` refinement to ``(uid, content_hash)``; ``None`` if missing/ambiguous."""
    candidates = lookup.symbols_by_path.get(project, {}).get(file_path, {}).get(symbol, [])
    return candidates[0] if len(candidates) == 1 else None


def _resolve_absolute_anchor(target: str, lookup: _AnchorLookup) -> tuple[str, str] | None:
    """Resolve an absolute-path anchor to ``(project, relative_path)`` via longest-prefix match."""
    best_project: str | None = None
    best_len = -1
    for project, root in lookup.project_roots.items():
        prefix = root.rstrip("/") + "/"
        if target.startswith(prefix) and len(prefix) > best_len:
            best_project, best_len = project, len(prefix)
    if best_project is None:
        return None
    return best_project, target[best_len:]


def _resolve_one_path_anchor(rel: ParsedRelationship, lookup: _AnchorLookup) -> tuple[str, str] | None:
    """Resolve a single path/project_path/absolute_path anchor to ``(uid, content_hash)``.

    Uid-form anchors are handled separately (a direct batched graph read) —
    this covers only the three path-shaped forms.
    """
    form = rel.properties.get("anchor_form", "")
    target: tuple[str, str] | None = None
    project_for_symbol = ""

    if form == "path":
        project_for_symbol = rel.from_qualified_name.split(":", 1)[0]
        target = _resolve_anchor_path(project_for_symbol, rel.to_name, lookup)
    elif form == "project_path":
        project_for_symbol = rel.properties.get("anchor_project", "")
        target = _resolve_anchor_path(project_for_symbol, rel.to_name, lookup)
    elif form == "absolute_path":
        resolved_loc = _resolve_absolute_anchor(rel.to_name, lookup)
        if resolved_loc is not None:
            project_for_symbol, rel_path = resolved_loc
            target = _resolve_anchor_path(project_for_symbol, rel_path, lookup)

    if target is None:
        return None

    symbol = rel.properties.get("anchor_symbol")
    if symbol:
        # Ambiguous/missing symbol falls back to the file-level anchor
        # rather than failing outright (decision Q9).
        sym_target = _resolve_anchor_symbol(project_for_symbol, rel.to_name, symbol, lookup)
        if sym_target is not None:
            return sym_target
    return target


# ---------------------------------------------------------------------------
# Citation resolution (ADR/RFC references captured by extract_rationale)
#
# CANONICAL FORM
# --------------
# ``extract_rationale`` stores what the author wrote, verbatim-ish, in the
# entity's ``citations`` list ("ADR-14", "ADR-0014", "RFC-793"). That string is
# the *evidence* and is never rewritten — normalising at capture time would
# throw away the only record of what the comment actually said, and any
# padding rule that unifies "ADR 14" with "ADR-0014" would also turn "RFC 793"
# into the nonexistent "RFC-0793".
#
# So the canonical form lives here, at resolution time, and is not a string at
# all: a citation's identity is the pair ``(scheme, number)`` — scheme
# case-folded, number compared as an *integer*. Leading zeros stop mattering
# without anything being padded or stripped destructively:
#
#     "ADR-14" "ADR 0014" "adr#014"  -> ("ADR", 14)
#     "RFC 793" "rfc-793"            -> ("RFC", 793)
#
# Document nodes are keyed into the same space (``_document_citation_keys``),
# so an ADR whose filename is ``0014-calls-edge-confidence.md`` answers to
# ``("ADR", 14)`` and both spellings above find it. Edges record the canonical
# pair rendered back as ``"<SCHEME>-<number>"`` (unpadded, e.g. ``"ADR-14"``)
# in their ``citation`` property, so the edge is self-describing and identical
# regardless of which spelling produced it.
#
# DIRECTION
# ---------
# The emitted edge runs ``(document) -[:DOCUMENTS {link_type:'citation'}]->
# (citing entity)`` — document to code, the same way every other DOCUMENTS
# edge runs. The evidence was found on the code side ("see ADR-14"), but the
# fact it establishes is the ordinary one: that ADR documents this function.
# Writing it the other way round would make DOCUMENTS the only relationship in
# the schema whose direction depends on which parser noticed it, and every
# reader (get_linked_docs, get_module_summary, the module-summary renderer,
# the zombie-preservation carve-out — in two backends) would have to special-
# case the exception forever. Provenance is not lost: ``link_type='citation'``
# says the edge came from a reference in the code, and ``citation`` records the
# identifier as canonicalised. See ``resolve_citations`` for the ownership
# consequence — the edge is owned by the CITING file's parse, not by the
# document's, so it is carved out of the document's relationship-delete phase.
# ---------------------------------------------------------------------------

# A whole captured citation string. The separator class matches the one
# ``parsing.ast._citation_pattern`` accepts, so any string that extractor can
# emit round-trips, plus the raw forms a human might hand-write. The scheme is
# letters-only so the separator can be absent ("ADR0014") without the scheme
# greedily swallowing the leading digits.
_CITATION_KEY_RE = re.compile(r"^(?P<scheme>[A-Za-z]{2,16})[ \t\-_#]?(?P<number>\d{1,6})$")

# Same shape but anchored only at the start, for document *titles*
# ("# ADR-0014: CALLS Edge Confidence").
_DOC_TITLE_RE = re.compile(r"^(?P<scheme>[A-Za-z]{2,16})[ \t\-_#]?(?P<number>\d{1,6})\b")

# The near-universal numbered-document filename convention: ``0014-slug.md``.
# Deliberately not tied to any parent directory — see ``_directory_scheme``.
_DOC_FILENAME_RE = re.compile(r"^(?P<number>\d{1,6})[-_]")

# Candidate strength, lowest wins. A whole numbered file beats the heading
# inside it, so "cite ADR-0014" links the document, not its H1 section.
_CITATION_RANK_FILENAME = 0
_CITATION_RANK_FILE_TITLE = 1
_CITATION_RANK_DOC_HEADING = 2

# Confidence stamped on the edge, per winning rank. Only the filename form is
# structural evidence — a file *named* ``0014-*.md`` inside an ``adr/``
# directory is that ADR, there is nothing to infer. The two title forms are
# inference from prose ("ADR 22 rollout notes" is a note *about* ADR-22 as
# plausibly as it is ADR-22), and the edge says so rather than claiming 1.0.
_CITATION_RANK_CONFIDENCE: dict[int, float] = {
    _CITATION_RANK_FILENAME: 1.0,
    _CITATION_RANK_FILE_TITLE: 0.9,
    _CITATION_RANK_DOC_HEADING: 0.8,
}

# Only a document's own top-level heading is treated as naming that document.
# A deeper heading that starts with an identifier ("## ADR-0014 rationale",
# "### ADR-0014 was rejected") is a passage *discussing* the record, and
# matching it produced the resolver's worst failure mode: when the real ADR
# lived outside a scheme-named directory, the only candidate left was some
# unrelated document that merely mentions the number, linked at confidence
# 1.0. A confidently wrong edge is worse than no edge, so those citations are
# now recorded as unresolved instead. The cost is the "one file, many records"
# layout (a changelog of ``## ADR-NNNN`` sections), which stops resolving —
# that layout is served by giving each record its own file, and an unresolved
# citation is still reported on the citing node.
_CITATION_DOC_HEADING_LEVEL = 1

_FILE_LEVEL_DOC_LABELS: frozenset[str] = frozenset({NodeLabel.DOC_FILE.value, NodeLabel.NOTE.value})


def _citation_key(citation: str) -> tuple[str, int] | None:
    """Canonical match key for one captured citation string, or ``None`` if unparseable.

    ``"ADR-0014"`` and ``"ADR 14"`` both yield ``("ADR", 14)``; ``"RFC 793"``
    yields ``("RFC", 793)`` and is never zero-padded. See the module-section
    comment above for why normalisation happens here rather than at capture.
    """
    match = _CITATION_KEY_RE.match(citation.strip())
    if match is None:
        return None
    return match.group("scheme").upper(), int(match.group("number"))


def _render_citation_key(key: tuple[str, int]) -> str:
    """Canonical key rendered back to a string for the edge's ``citation`` property."""
    return f"{key[0]}-{key[1]}"


def _directory_scheme(directory: str) -> str:
    """Scheme a document *directory* name implies — ``adr``/``ADRs`` → ``ADR``.

    This is how ``wiki/adr/0014-foo.md`` gets an ``ADR`` scheme without the
    ADR directory being hardcoded anywhere: whatever the containing folder is
    called *is* the scheme, singularised and upper-cased. A repo using
    ``docs/adr``, ``doc/adrs`` or ``notes/rfc`` works identically, and a
    directory that is not a plausible scheme token (``notes``, ``2026-07``)
    simply yields a key nothing cites.
    """
    token = directory.strip().upper()
    if token.endswith("S") and len(token) > 1:
        token = token[:-1]
    return token if token.isalpha() and 2 <= len(token) <= 16 else ""


def _document_citation_keys(
    label: str, name: str, file_path: str, header_level: int | None = None
) -> list[tuple[tuple[str, int], int]]:
    """Citation keys a document node answers to, each paired with a match rank.

    Three forms, none requiring a configured ADR path, in descending strength:

    * **filename shape** — a file-level node (DocFile/Note) named
      ``NNNN-slug.md`` inside a scheme-named directory (``adr/``, ``rfcs/``).
      Strongest: it identifies a whole document structurally.
    * **file title shape** — a file-level node whose own name *starts* with a
      scheme+number (``ADR-0014: CALLS Edge Confidence``, ``ADR-0014-foo.md``).
      Catches repos whose decision records live in a differently-named folder.
    * **document heading shape** — a DocSection that is its file's top-level
      heading, i.e. the document's title by another route.

    *header_level* is the DocSection's heading depth; a section only qualifies
    at depth ``_CITATION_DOC_HEADING_LEVEL``. Deeper headings, and sections
    passed with no level at all, are treated as mentions and produce no key —
    see ``_CITATION_DOC_HEADING_LEVEL`` for why that is worth the miss.
    """
    keys: list[tuple[tuple[str, int], int]] = []
    is_file_level = label in _FILE_LEVEL_DOC_LABELS

    if is_file_level and file_path:
        posix = PurePosixPath(file_path.replace("\\", "/"))
        filename_match = _DOC_FILENAME_RE.match(posix.stem)
        scheme = _directory_scheme(posix.parent.name)
        if filename_match is not None and scheme:
            keys.append(((scheme, int(filename_match.group("number"))), _CITATION_RANK_FILENAME))

    titles_document = is_file_level or header_level == _CITATION_DOC_HEADING_LEVEL
    title_match = _DOC_TITLE_RE.match(name.strip()) if titles_document else None
    if title_match is not None:
        rank = _CITATION_RANK_FILE_TITLE if is_file_level else _CITATION_RANK_DOC_HEADING
        keys.append(((title_match.group("scheme").upper(), int(title_match.group("number"))), rank))

    return keys


@dataclass(frozen=True)
class _CitationLookup:
    """Per-project index from canonical citation key to candidate document uids.

    ``by_key[("ADR", 14)]`` is a list of ``(rank, uid)`` candidates — see
    ``_document_citation_keys`` for what the ranks mean.
    """

    by_key: dict[tuple[str, int], list[tuple[int, str]]]


def _pick_citation_target(key: tuple[str, int], lookup: _CitationLookup) -> tuple[str, float] | None:
    """Best ``(document uid, edge confidence)`` for *key*, or ``None`` when missing or ambiguous.

    The strongest available rank wins outright — a DocFile and the H1
    DocSection inside it both answer to ``("ADR", 14)``, and that is not
    ambiguity, it is one document described twice. A genuine tie *within* the
    winning rank (two ``0014-*.md`` files in two ``adr/`` directories) resolves
    to nothing rather than guessing, matching the never-multi-link discipline
    ``resolve_anchors``/``_create_doc_links`` already use.

    The winning rank also fixes the confidence written onto the edge
    (``_CITATION_RANK_CONFIDENCE``): a weaker form of evidence still links,
    but it does not get to claim certainty.
    """
    candidates = lookup.by_key.get(key)
    if not candidates:
        return None
    best_rank = min(rank for rank, _ in candidates)
    winners = {uid for rank, uid in candidates if rank == best_rank}
    if len(winners) != 1:
        return None
    return next(iter(winners)), _CITATION_RANK_CONFIDENCE[best_rank]


@dataclass(frozen=True)
class _CallLookup:
    """Pre-built lookup tables for CALLS resolution."""

    name_to_callables: dict[str, list[tuple[str, str, str]]]  # name → [(uid, file_path, vis)]
    import_map: dict[str, dict[str, str]]  # module_uid → {imported_name: target_uid}
    caller_to_parent: dict[str, str]  # callable_uid → parent TypeDef uid
    parent_children: dict[str, list[str]]  # parent_uid → [child_uids]
    uid_to_info: dict[str, tuple[str, str]]  # uid → (name, file_path)
    # Callables whose body is literally `...` or which are @abstractmethod. Per-method,
    # not per-class: an ABC's concrete methods are real code and must stay resolvable.
    stub_callables: frozenset[str] = frozenset()
    # Every TypeDef name in the project, so a receiver whose declared type is NOT one of
    # them can be recognised as leaving the project entirely.
    typedef_names: frozenset[str] = frozenset()


def _typedef_init_uid(typedef_uid: str, lk: _CallLookup) -> str | None:
    """Return the uid of *typedef_uid*'s ``__init__`` Callable child, if any."""
    for child_uid in lk.parent_children.get(typedef_uid, []):
        child_info = lk.uid_to_info.get(child_uid)
        if child_info and child_info[0] == "__init__":
            return child_uid
    return None


# Receiver expressions that denote the enclosing instance, where an attribute call is
# still lexically grounded: `self.helper()` is exactly as resolvable as `helper()`.
# Anything else — `client.scan()`, `self._valkey.scan()` — names a member of a type the
# indexer may never have seen.
_SELF_RECEIVERS = frozenset({"self", "cls", "this"})

# Strategies whose match is a name coincidence rather than a lexical resolution. Kept as
# edges (ADR-0014 materializes rather than discards) but never marked resolved.
_UNVERIFIED_STRATEGIES = frozenset({"unverified_receiver", "unverified_wide"})


def _is_abstract_stub(uid: str, lk: _CallLookup) -> bool:
    """Whether *uid*'s body can never be what runs — a `...` body or @abstractmethod.

    Asked of the method, not its class. Asking the class conflated Protocol (all stubs)
    with ABC (one abstractmethod among many real ones) and deleted true callees.
    """
    return uid in lk.stub_callables


def _resolve_one_call(  # noqa: PLR0911, PLR0912
    project_name: str,
    rel: ParsedRelationship,
    lk: _CallLookup,
    name_to_typedefs: dict[str, list[tuple[str, str]]] | None = None,
) -> tuple[list[str], str] | None:
    """Resolve a single CALLS relationship to one or more candidate target uids.

    Returns ``(candidate_uids, strategy)``, or ``None`` if nothing matched at all.
    A single-element ``candidate_uids`` is an unambiguous match (the caller tags
    the edge ``confidence: "resolved"``); more than one element means the bare
    name matched multiple candidates that could not be disambiguated (the caller
    tags every resulting edge ``confidence: "ambiguous"`` instead of discarding
    them, per ADR-0014).
    """
    caller_uid = rel.from_qualified_name
    bare_name = rel.to_name

    # Before any strategy: a known receiver type that names no class in this project means
    # the callee is not in this project either — `seen.add(x)` on a `set` is not a call to
    # any project method named `add`. This has to precede the name-matching strategies,
    # because a same-file match on `add` is exactly the false positive. The collisions it
    # removes produced 78 edges each onto two unrelated project classes, and inflated one
    # blast_radius answer from 5 entities to 26.
    receiver_type = str(rel.properties.get("receiver_type") or "")
    if receiver_type and lk.typedef_names and receiver_type not in lk.typedef_names:
        return None

    # Derive caller's module uid — find the longest module prefix in import_map
    caller_qn = caller_uid.split(":", 1)[1] if ":" in caller_uid else caller_uid
    parts = caller_qn.split(".")
    module_uid: str | None = None
    for i in range(len(parts) - 1, 0, -1):
        candidate = f"{project_name}:{'.'.join(parts[:i])}"
        if candidate in lk.import_map:
            module_uid = candidate
            break

    # Strategy 1: Import match. import_map is shared with USES_TYPE resolution
    # (client.py:resolve_type_refs), so its targets aren't Callable-scoped — an
    # imported class resolves here to the TypeDef's own uid, which the CALLS
    # edge-creation Cypher (:Callable-scoped) would silently drop. Redirect a
    # non-Callable import target to its __init__ (constructor call) before
    # falling through to the remaining strategies.
    if module_uid and bare_name in lk.import_map.get(module_uid, {}):
        target_uid = lk.import_map[module_uid][bare_name]
        if target_uid in lk.uid_to_info:
            return ([target_uid], "import")
        init_uid = _typedef_init_uid(target_uid, lk)
        if init_uid is not None:
            return ([init_uid], "import")

    # Strategy 2: Same-class sibling
    if caller_uid in lk.caller_to_parent:
        parent_uid = lk.caller_to_parent[caller_uid]
        for sibling_uid in lk.parent_children.get(parent_uid, []):
            if sibling_uid == caller_uid:
                continue
            sib_info = lk.uid_to_info.get(sibling_uid)
            if sib_info and sib_info[0] == bare_name and not _is_abstract_stub(sibling_uid, lk):
                return ([sibling_uid], "sibling")

    # Strategy 3: Same-file match
    caller_info = lk.uid_to_info.get(caller_uid)
    caller_fp = caller_info[1] if caller_info else ""
    if caller_fp:
        for uid, fp, _vis in lk.name_to_callables.get(bare_name, []):
            if fp == caller_fp and uid != caller_uid and not _is_abstract_stub(uid, lk):
                return ([uid], "same_file")

    # Strategy 4: Project-wide match. Previously only fired when exactly 1
    # candidate existed (ambiguous names like run/close/get were left
    # unresolved to avoid false positives from external attribute calls such
    # as asyncio.run(), session.run()). Now every candidate is returned —
    # unambiguous (len==1) resolves normally; ambiguous (len>1) still
    # materializes an edge to each candidate, tagged confidence:"ambiguous"
    # by the caller instead of being discarded (ADR-0014).
    candidates = lk.name_to_callables.get(bare_name, [])
    non_self = [uid for uid, _fp, _vis in candidates if uid != caller_uid]

    # Strategy 3.5: the receiver's declared type. Most of what looks like polymorphism is
    # not — measured, 772 of 915 fanned-out sites call exactly ONE concrete class, and
    # only 24 the Protocol. They are monomorphic calls on concretely-typed receivers that
    # a name-only resolver spreads across every implementation. Knowing the type removes
    # the false edges rather than re-weighting them, which is why it leaves total graph
    # weight unchanged where a containment heuristic inflated it 16%.
    # Drop stub candidates FIRST, before any strategy consults the list. Doing this after
    # the receiver-type branch let 36 edges resolve straight onto a `...` Protocol body at
    # full confidence — the exact outcome the filter exists to prevent, and worse than the
    # ambiguity it replaced, because a resolved edge is trusted and it displaced the real
    # target. A filter that a strategy can return past is not a filter.
    real = [uid for uid in non_self if not _is_abstract_stub(uid, lk)]

    declared = str(rel.properties.get("receiver_type") or "")
    if declared and len(real) > 1:
        # The parent is a TypeDef, and uid_to_info holds Callables only — its class name
        # comes from the uid's last dotted segment, e.g. "proj:pkg.mod.Store" -> "Store".
        owned = [uid for uid in real if lk.caller_to_parent.get(uid, "").rsplit(".", 1)[-1] == declared]
        if len(owned) == 1:
            return (owned, "receiver_type")

    if real and len(real) < len(non_self):
        # Exactly one implementation behind a declaration is a resolution, not a guess,
        # and must NOT fall through to the single-candidate branch below: the receiver
        # test there would re-tag it unverified at half weight. A Protocol declaring this
        # very name IS the project-namespace evidence that branch looks for, so
        # re-damping it would be precisely backwards.
        return (real, "polymorphic_unique" if len(real) == 1 else "polymorphic")
    non_self = real or non_self

    if len(non_self) == 1:
        # Uniqueness within the project is evidence of identity only if the name was
        # looked up in the project's namespace. For `client.scan()` it was not: the
        # receiver's type may never have been indexed, so the single same-named entity
        # is a coincidence, not the callee. The edge is still worth recording — it is
        # the best guess available — but it must not claim to be resolved. Verified in
        # the field: EmbedCache.clear called Valkey's .scan() and this branch pointed it
        # at FileScope.scan with confidence "resolved" and full weight.
        receiver = str(rel.properties.get("receiver") or "")
        if receiver and receiver not in _SELF_RECEIVERS:
            return (non_self, "unverified_receiver")
        return (non_self, "project_unique")
    if len(non_self) > 1:
        # The receiver is just as unverifiable here as it is with one candidate, and
        # ATL-091 only damped the single-candidate case: 1498 of 1506 multi-candidate
        # sites in the reference index have a non-self receiver and none were damped.
        # Applying the same doubt to the same evidence is the point.
        receiver = str(rel.properties.get("receiver") or "")
        if receiver and receiver not in _SELF_RECEIVERS:
            return (non_self, "unverified_wide")
        return (non_self, "project_wide")

    # Strategy 5: Constructor call (bare_name is a class, not a function) — a
    # `ClassName(...)` call's bare name is the class name, but the constructor
    # itself is named `__init__`, so it never matches any strategy above. Runs
    # last so it never steals priority from a real function-name match. Same
    # "return every candidate" discipline as Strategy 4: multiple classes
    # sharing the bare name now all resolve (ambiguously) instead of being
    # dropped.
    if name_to_typedefs is not None:
        typedef_candidates = name_to_typedefs.get(bare_name, [])
        init_uids = [
            init_uid for td_uid, _fp in typedef_candidates if (init_uid := _typedef_init_uid(td_uid, lk)) is not None
        ]
        if init_uids:
            return (init_uids, "constructor")

    return None


# ---------------------------------------------------------------------------
# CALLS edge weighting (amends ADR-0014)
#
# ADR-0014 tags every CALLS edge with two *categorical* facts (`confidence`,
# `strategy`) and explicitly rejected a numeric score as premature precision.
# Three consumers since needed a magnitude rather than a category: MAGE's
# weighted Leiden (which silently runs unweighted unless a numeric property is
# persisted on the relationship), blast_radius impact ranking, and trace_path's
# equal-hop-count tie-break.
#
# ADR-0014's objection is honored by storing the *raw observed facts* on the
# edge — `candidate_count` (how many targets the winning strategy returned) and
# `from_test` (whether the caller lives in test code) — and deriving `weight`
# from them here, in one place. Consumers that want the underlying evidence
# read the facts; consumers that need a scalar read the derived weight.
#
# Caveat inherited from _resolve_one_call: strategies 1-3 early-return a
# single-element candidate list, so `candidate_count == 1` means "this strategy
# committed to one target", not "provably unique".

# Base weight of one fully-resolved, non-test call edge. Deliberately 1.0
# because MAGE reads a *missing* weight property as 1.0 — IMPORTS edges carry
# no weight, so this makes "a certain production call" worth exactly one
# import, which is the reference point _analyze_communities documents. Change
# this and that equivalence silently stops holding.
_CALL_WEIGHT_BASE = 1.0
# Test callers do exercise their callee, so their edges are damped rather than
# dropped: a test-only caller ranks below every production caller, above none.
_CALL_WEIGHT_TEST_DAMPING = 0.25

# An unverified-receiver edge names one project entity, but the real callee may be a
# method of a type that was never indexed. Halving says exactly that: at best an even
# split between the name match and something outside the graph. Without it the edge is
# marked ambiguous and still weighted like a certainty, because 1/candidate_count with
# candidate_count 1 is 1.0 — the flag alone never reaches Leiden or blast_radius ranking.
_CALL_WEIGHT_UNVERIFIED_DAMPING = 0.5
# Strictly-positive floor. MAGE's Leiden normalizes gamma by the sum of edge
# weights, so a zero total yields NaN and silently meaningless communities.
_MIN_CALL_WEIGHT = 1e-6
# What a missing/unweighted edge counts as when traversals multiply weights
# together. Matches MAGE's own default so weighted Leiden and the Python-side
# consumers agree about untagged edges (IMPORTS, USES_TYPE, and CALLS edges
# written before this change).
_DEFAULT_EDGE_WEIGHT = 1.0

# Default test-path patterns behind the `from_test` flag. Single-sourced from
# SearchSettings rather than re-listed here so the graph layer cannot drift
# from the search layer's rule; resolve_calls takes an override for projects
# that configure their own patterns.
_DEFAULT_TEST_PATTERNS: tuple[str, ...] = tuple(SearchSettings().test_patterns)

# Both weight numbers above are heuristics, not measurements — they are
# expected to be retuned once there is evidence about what ranks well.

# Callable/TypeDef kinds that denote *invocable code*, i.e. entities for which
# "nothing calls this" is evidence rather than a tautology.
#
# Derived from the schema enums rather than listed by hand, because the two
# families of parsers already differ exactly along that line:
#   - Code parsers (python, typescript, go, rust, jvm, cpp, ruby, php, shell)
#     assign every Callable/TypeDef a CallableKind/TypeDefKind member.
#   - Infra/config parsers (hcl, sql, config, containerfile) mint free-form
#     kind strings for things that are declarations, not code —
#     'terraform_resource', 'k8s_resource', 'sql_table', 'docker_stage',
#     'ci_job', 'ansible_task', 'xml_element', ...
# Those declarations can never be the target of a resolved CALLS edge, so a
# label-only dead-code filter reports every last one of them as dead.
#
# A kind deny-list or a file-extension/language allow-list would both have to
# grow with every language added — the same drift trap as `_DEFAULT_INCLUDE`
# in indexing/orchestrator.py. This set instead grows automatically: a new code
# parser reuses the enums and is included for free, while a new infra parser
# inventing its own kind vocabulary is excluded for free.
#
# Plain ``str`` values, not enum members: this set is handed to the Bolt driver
# and to sqlite3 as a query parameter.
_CODE_ENTITY_KINDS: frozenset[str] = frozenset(str(k) for k in (*CallableKind, *TypeDefKind))


def _call_edge_weight(candidate_count: int, from_test: bool, strategy: str = "") -> float:
    """Derive a CALLS edge's numeric weight from its raw observed facts.

    ``1 / candidate_count`` spreads one call's worth of evidence across the
    candidates the resolver could not disambiguate; ``from_test`` then damps
    the result. The return value is always strictly positive.

    An unverified receiver is damped separately because candidate_count cannot express
    it: the resolver found exactly one name match, so the count is 1 and the quotient is
    a full-confidence 1.0, even though the real callee may not be in the graph at all.
    """
    weight = _CALL_WEIGHT_BASE / max(candidate_count, 1)
    if strategy in _UNVERIFIED_STRATEGIES:
        weight *= _CALL_WEIGHT_UNVERIFIED_DAMPING
    if from_test:
        weight *= _CALL_WEIGHT_TEST_DAMPING
    return max(weight, _MIN_CALL_WEIGHT)


class _CallEdgeFacts(NamedTuple):
    """Observed facts for one ``(caller, callee)`` CALLS edge.

    N call sites from the same caller to the same callee collapse into a single
    edge, so these are the *combined* observations — see
    :func:`_combine_call_edge_facts`.
    """

    confidence: str
    strategy: str
    candidate_count: int
    from_test: bool


def _combine_call_edge_facts(existing: _CallEdgeFacts, observed: _CallEdgeFacts) -> _CallEdgeFacts:
    """Combine two observations of the same ``(caller, callee)`` pair.

    The prior rule was last-write-wins, which made a pair's stored confidence
    depend on parse order — a site resolved unambiguously by strategy 3 could
    be overwritten by a later ambiguous strategy-4 site, or the reverse. This
    keeps the *best-evidenced* observation instead (lowest ``candidate_count``;
    ties keep the one seen first, so the result is order-stable), and treats
    ``from_test`` as true only when *every* observed call site was in test code
    — one production caller is enough to make the edge production-relevant.
    """
    best = observed if observed.candidate_count < existing.candidate_count else existing
    return _CallEdgeFacts(
        best.confidence, best.strategy, best.candidate_count, existing.from_test and observed.from_test
    )


@dataclass(frozen=True)
class CallStats:
    """Caller/callee statistics for a single entity."""

    caller_count: int = 0
    callee_count: int = 0
    caller_names: list[str] = field(default_factory=list)
    callee_names: list[str] = field(default_factory=list)


# Per-task transaction context — prevents leaking between concurrent asyncio tasks.
_active_tx_var: contextvars.ContextVar[Any] = contextvars.ContextVar("_active_tx_var", default=None)

_T = TypeVar("_T")


class GraphClient:
    """Async Memgraph client wrapping the neo4j Bolt driver.

    Follows the same lifecycle pattern as EventBus: construct → ping → use → close.
    """

    def __init__(self, settings: AtlasSettings, *, driver: AsyncDriver | None = None) -> None:
        mg = settings.memgraph
        self._uri = f"bolt://{mg.host}:{mg.port}"
        auth = (mg.username, mg.password) if mg.username else None
        self._driver: AsyncDriver = driver if driver is not None else AsyncGraphDatabase.driver(self._uri, auth=auth)
        self._dimension = settings.embeddings.dimension or 768
        self._embeddings_enabled = settings.embeddings.enabled
        self._query_timeout_s = mg.query_timeout_s
        self._write_timeout_s = mg.write_timeout_s

    @property
    def dimension(self) -> int:
        """Current embedding vector dimension used for vector indices."""
        return self._dimension

    async def ping(self) -> bool:
        """Health check — returns True if Memgraph is reachable."""
        records = await self.execute("RETURN 1 AS n")
        return len(records) == 1 and records[0]["n"] == 1

    async def execute(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a read query and return results as a list of dicts."""
        active_tx = _active_tx_var.get()
        if active_tx is not None:
            result = await active_tx.run(query, params or {})
            return [dict(record) async for record in result]
        with _tracer.start_as_current_span("graph.execute", attributes={"db.statement": query[:200]}):
            try:
                return await asyncio.wait_for(self._execute_inner(query, params), timeout=self._query_timeout_s)
            except TimeoutError:
                raise QueryTimeoutError(self._query_timeout_s, query[:120]) from None

    async def _execute_inner(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Inner execute without timeout — used by ``execute()``."""
        async with self._driver.session() as session:
            result = await session.run(query, params or {})  # type: ignore[arg-type]  # dynamic Cypher
            return [dict(record) async for record in result]

    async def execute_write(self, query: str, params: dict[str, Any] | None = None) -> None:
        """Execute a write query with automatic retry on transient conflicts.

        Consumes the result to ensure server-side errors (e.g. constraint
        violations) are raised instead of being silently dropped.

        When called inside a managed transaction (``_active_tx_var`` is set),
        errors propagate directly — the managed transaction handles retries.
        """
        active_tx = _active_tx_var.get()
        if active_tx is not None:
            result = await active_tx.run(query, params or {})
            await result.consume()
            return
        await self._execute_write_with_retry(query, params)

    @retry(
        retry=retry_if_exception_type(TransientError),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=2),
        before_sleep=lambda rs: logger.warning(
            "Transient conflict, retrying {} in {:.1f}s (attempt {}): {}",
            rs.fn.__qualname__,
            rs.next_action.sleep,
            rs.attempt_number,
            rs.outcome.exception(),
        ),
        reraise=True,
    )
    async def _execute_write_with_retry(self, query: str, params: dict[str, Any] | None = None) -> None:
        """Standalone write with retry + timeout (not used inside managed transactions)."""
        with _tracer.start_as_current_span("graph.execute_write", attributes={"db.statement": query[:200]}):
            try:
                await asyncio.wait_for(self._execute_write_inner(query, params), timeout=self._write_timeout_s)
            except TimeoutError:
                raise QueryTimeoutError(self._write_timeout_s, query[:120]) from None

    async def _execute_write_inner(self, query: str, params: dict[str, Any] | None = None) -> None:
        """Inner execute_write without timeout."""
        async with self._driver.session() as session:
            result = await session.run(query, params or {})  # type: ignore[arg-type]  # dynamic Cypher
            await result.consume()

    async def get_schema_version(self) -> int | None:
        """Read the current schema version from the SchemaVersion node.

        Returns the MAX across nodes — defensive against duplicate
        SchemaVersion nodes left behind by the pre-fix migration MERGE.
        """
        records = await self.execute(f"MATCH (sv:{NodeLabel.SCHEMA_VERSION}) RETURN max(sv.version) AS version")
        if not records:
            return None
        return records[0]["version"]

    async def ensure_schema(self) -> None:
        """Apply or migrate the graph schema.

        - Fresh DB (no version): apply all DDL, create version node.
        - Same version: no-op.
        - Older version: drop & recreate vector/text indices, bump version.
        - Newer version: raise RuntimeError (downgrade not supported).
        """
        stored = await self.get_schema_version()

        if stored is None:
            logger.info("Fresh database — applying schema v{}", SCHEMA_VERSION)
            await self._apply_full_schema()
            await self._set_schema_version(SCHEMA_VERSION)
            logger.info("Schema v{} applied successfully", SCHEMA_VERSION)

        elif stored == SCHEMA_VERSION:
            logger.debug("Schema v{} already current — no migration needed", SCHEMA_VERSION)
            await self._reconcile_search_indices()

        elif stored < SCHEMA_VERSION:
            logger.info("Migrating schema v{} → v{}", stored, SCHEMA_VERSION)
            await self._migrate_indices()
            if stored < 3:  # v3 data migration threshold
                await self._migrate_v3_clear_freshness_markers()
            if stored < 4:  # v4 data migration threshold
                await self._migrate_v4_clear_freshness_markers()
            if stored < 5:  # v5 data migration threshold
                await self._migrate_v5_clear_freshness_markers()
            if stored < 6:  # v6 data migration threshold
                await self._migrate_v6_clear_freshness_markers()
            if stored < 7:  # v7 data migration threshold
                await self._migrate_v7_clear_freshness_markers()
            if stored < 8:  # v8 data migration threshold
                await self._migrate_v8_drop_unverified_calls()
            if stored < 9:  # v9 data migration threshold
                await self._migrate_v9_clear_for_abstract_bases()
            if stored < 10:  # v10 data migration threshold
                await self._migrate_v10_stub_flag_moved_to_methods()
            await self._set_schema_version(SCHEMA_VERSION)
            logger.info("Schema migrated to v{}", SCHEMA_VERSION)

        else:
            msg = (
                f"Database schema v{stored} is newer than code v{SCHEMA_VERSION}. "
                f"Downgrade is not supported — update your Code Atlas installation."
            )
            raise RuntimeError(msg)

    async def get_file_content_hashes(self, project_name: str, file_path: str) -> dict[str, EntityHashData]:
        """Return ``{uid: EntityHashData}`` for non-structural nodes."""
        records = await self.execute(
            f"MATCH (n {{project_name: $p, file_path: $f}}) "
            f"WHERE NOT n:{NodeLabel.PACKAGE} AND NOT n:{NodeLabel.PROJECT} "
            "RETURN n.uid AS uid, n.content_hash AS hash, n.line_start AS ls, n.line_end AS le, "
            "n.signature AS sig, n.docstring AS doc, labels(n)[0] AS lbl",
            {"p": project_name, "f": file_path},
        )
        return {
            r["uid"]: EntityHashData(r["hash"] or "", r["ls"] or 0, r["le"] or 0, r["sig"], r["doc"], r["lbl"] or "")
            for r in records
        }

    async def get_batch_file_content_hashes(
        self,
        project_name: str,
        file_paths: list[str],
    ) -> dict[str, dict[str, EntityHashData]]:
        """Return ``{file_path: {uid: EntityHashData}}`` for multiple files in one RTT."""
        if not file_paths:
            return {}
        records = await self.execute(
            f"UNWIND $fps AS fp "
            f"MATCH (n {{project_name: $p, file_path: fp}}) "
            f"WHERE NOT n:{NodeLabel.PACKAGE} AND NOT n:{NodeLabel.PROJECT} "
            "RETURN n.file_path AS fp, n.uid AS uid, n.content_hash AS hash, "
            "n.line_start AS ls, n.line_end AS le, "
            "n.signature AS sig, n.docstring AS doc, labels(n)[0] AS lbl",
            {"p": project_name, "fps": file_paths},
        )
        result: dict[str, dict[str, EntityHashData]] = defaultdict(dict)
        for r in records:
            result[r["fp"]][r["uid"]] = EntityHashData(
                r["hash"] or "", r["ls"] or 0, r["le"] or 0, r["sig"], r["doc"], r["lbl"] or ""
            )
        return dict(result)

    async def upsert_file_entities(
        self,
        project_name: str,
        file_path: str,
        entities: list[ParsedEntity],
        relationships: list[ParsedRelationship],
    ) -> UpsertResult:
        """Delta-aware upsert: only write changed entities to the graph.

        Compares ``content_hash`` of new entities against stored values to
        classify each as added/modified/deleted/unchanged.  Unchanged entities
        are skipped entirely — their embed data is never touched.

        All graph reads and writes run inside a single managed transaction
        for atomicity and reduced session overhead.  ``session.execute_write``
        auto-retries on ``TransientError`` (MVCC conflicts).

        Returns an ``UpsertResult`` describing what changed.
        """

        async def _tx_fn(tx: Any) -> UpsertResult:
            token = _active_tx_var.set(tx)
            try:
                return await self._upsert_file_entities_inner(project_name, file_path, entities, relationships)
            finally:
                _active_tx_var.reset(token)

        async with self._driver.session() as session:
            return await session.execute_write(_tx_fn)

    async def _upsert_file_entities_inner(
        self,
        project_name: str,
        file_path: str,
        entities: list[ParsedEntity],
        relationships: list[ParsedRelationship],
    ) -> UpsertResult:
        """Core upsert logic — runs inside a managed transaction via ``_active_tx_var``."""
        old_data = await self.get_file_content_hashes(project_name, file_path)
        fc = _classify_file(old_data, entities, self._strip_uid)

        # Fast path: no content changes — apply position-only drift and skip
        # relationship recreation (rels can't change if no entity content changed)
        if not fc.added and not fc.modified and not fc.result.deleted:
            if fc.shifted:
                await self._batch_update_positions(fc.shifted)
            logger.debug("Delta skip (no content changes) for {}", file_path)
            return fc.result

        # Apply delta (all writes route through _active_tx_var)
        if fc.deleted_by_label:
            await self._batch_delete_entities(fc.deleted_by_label)
        if fc.added:
            await self._batch_create_entities(project_name, fc.added)
        if fc.modified:
            await self._batch_update_entities(fc.modified)
        if fc.shifted:
            await self._batch_update_positions(fc.shifted)

        # Recreate ALL relationships for the file (delete old, create new).
        await self._recreate_file_relationships(project_name, file_path, relationships, skip_delete=not old_data)

        logger.debug(
            "Upserted {} (added={}, modified={}, deleted={}, unchanged={}) for {}",
            len(entities),
            len(fc.result.added),
            len(fc.result.modified),
            len(fc.result.deleted),
            len(fc.result.unchanged),
            file_path,
        )
        return fc.result

    @staticmethod
    def _strip_uid(uid: str) -> str:
        """Strip project prefix from uid to get qualified_name."""
        return uid.split(":", 1)[1] if ":" in uid else uid

    @staticmethod
    def _classify_batch(
        old_data: dict[str, dict[str, EntityHashData]],
        file_data: dict[str, tuple[list[ParsedEntity], list[ParsedRelationship]]],
    ) -> _BatchClassification:
        """Pure-Python classification of entities across multiple files.

        Compares new entities against *old_data* (from ``get_batch_file_content_hashes``)
        and produces cross-file lists for batched graph writes.
        """
        all_added: list[ParsedEntity] = []
        all_modified: list[ParsedEntity] = []
        all_deleted_by_label: dict[str, list[str]] = defaultdict(list)
        all_shifted: list[ParsedEntity] = []
        per_file_results: dict[str, UpsertResult] = {}
        new_file_paths: set[str] = set()

        strip = GraphClient._strip_uid

        for file_path, (entities, _rels) in file_data.items():
            file_old = old_data.get(file_path, {})
            if not file_old:
                new_file_paths.add(file_path)

            fc = _classify_file(file_old, entities, strip)

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

    async def upsert_batch_entities(
        self,
        project_name: str,
        file_data: dict[str, tuple[list[ParsedEntity], list[ParsedRelationship]]],
    ) -> dict[str, UpsertResult]:
        """Batched multi-file upsert using two sequential managed transactions.

        TX1 (Entity CRUD): batch hash read → classify → delete/create/update/positions.
        TX2 (Relationships): delete old rels → create new rels → INHERITS → DOCUMENTS.

        Returns ``{file_path: UpsertResult}`` for each file in *file_data*.
        """
        if not file_data:
            return {}

        file_paths = list(file_data)

        # --- TX1: Entity CRUD ---
        async def _entity_tx(tx: Any) -> _BatchClassification:
            token = _active_tx_var.set(tx)
            try:
                old_data = await self.get_batch_file_content_hashes(project_name, file_paths)
                classification = self._classify_batch(old_data, file_data)

                if classification.all_deleted_by_label:
                    await self._batch_delete_entities(classification.all_deleted_by_label)
                if classification.all_added:
                    await self._batch_create_entities(project_name, classification.all_added)
                if classification.all_modified:
                    await self._batch_update_entities(classification.all_modified)
                if classification.all_shifted:
                    await self._batch_update_positions(classification.all_shifted)

                return classification
            finally:
                _active_tx_var.reset(token)

        async with self._driver.session() as session:
            classification = await session.execute_write(_entity_tx)

        # --- TX2: Relationships ---
        file_rels = {fp: rels for fp, (_, rels) in file_data.items()}

        async def _rel_tx(tx: Any) -> None:
            token = _active_tx_var.set(tx)
            try:
                await self._recreate_batch_relationships(
                    project_name,
                    file_rels,
                    classification.new_file_paths,
                )
            finally:
                _active_tx_var.reset(token)

        async with self._driver.session() as session:
            await session.execute_write(_rel_tx)

        total_added = sum(len(r.added) for r in classification.per_file_results.values())
        total_modified = sum(len(r.modified) for r in classification.per_file_results.values())
        total_deleted = sum(len(r.deleted) for r in classification.per_file_results.values())
        logger.debug(
            "Batch upsert {} files (added={}, modified={}, deleted={})",
            len(file_data),
            total_added,
            total_modified,
            total_deleted,
        )

        return classification.per_file_results

    async def delete_file_entities(self, project_name: str, file_path: str) -> list[str]:
        """Delete all non-structural entity nodes for a file. Returns deleted uids."""
        old_data = await self.get_file_content_hashes(project_name, file_path)
        if old_data:
            uids_by_label: dict[str, list[str]] = defaultdict(list)
            for uid, data in old_data.items():
                uids_by_label[data.label].append(uid)
            await self._batch_delete_entities(dict(uids_by_label))
        # A Package node for this path (e.g. __init__.py) is intentionally not
        # deleted — the directory hierarchy may still need it — but its stored
        # file_hash must be cleared. Otherwise an identically re-created file
        # is silently skipped forever by the hash gate (get_batch_file_hashes).
        await self.execute_write(
            f"MATCH (n:{NodeLabel.PACKAGE} {{project_name: $p, file_path: $f}}) SET n.file_hash = NULL",
            {"p": project_name, "f": file_path},
        )
        return [self._strip_uid(uid) for uid in old_data]

    async def merge_project_node(self, project_name: str, **metadata: Any) -> None:
        """Create or update a Project node by uid."""
        uid = project_name
        props = {"uid": uid, "project_name": project_name, "name": project_name, **metadata}
        set_clause = ", ".join(f"n.{k} = ${k}" for k in props)
        await self.execute_write(
            f"MERGE (n:{NodeLabel.PROJECT} {{uid: $uid}}) SET {set_clause}",
            props,
        )

    async def get_batch_file_hashes(
        self,
        project_name: str,
        file_paths: list[str],
    ) -> dict[str, str | None]:
        """Return ``{file_path: file_hash}`` for Module/Package nodes in one RTT.

        Returns ``None`` for files that have no stored hash.
        """
        if not file_paths:
            return {}
        # The label filter MUST be inline on the node pattern, not a post-MATCH WHERE.
        # `UNWIND ... MATCH (n {...}) WHERE n:Label` is order-sensitive in Memgraph and
        # silently drops rows: measured, a batch of 3 existing files with one
        # non-matching path FIRST returned 0 rows. That made the hash gate read back
        # nothing and re-parse everything. One statement per label because the inline
        # form takes a single label.
        result: dict[str, str | None] = dict.fromkeys(file_paths)
        for label in (NodeLabel.MODULE, NodeLabel.PACKAGE):
            records = await self.execute(
                f"UNWIND $fps AS fp "
                f"MATCH (n:{label} {{project_name: $p, file_path: fp}}) "
                "RETURN n.file_path AS fp, n.file_hash AS fh",
                {"p": project_name, "fps": file_paths},
            )
            for r in records:
                result[r["fp"]] = r["fh"]
        return result

    async def set_batch_file_hashes(
        self,
        project_name: str,
        file_hashes: dict[str, str],
    ) -> None:
        """Write ``file_hash`` on Module/Package nodes for each file path."""
        if not file_hashes:
            return
        params = [{"fp": fp, "fh": fh} for fp, fh in file_hashes.items()]
        # Inline label, one statement per label — see get_batch_file_hashes. With the
        # post-MATCH `WHERE n:Module OR n:Package` form this wrote only the FIRST
        # file's hash per call (measured: 1 of 3, with every file's node present), so
        # the incremental hash gate recorded almost nothing and re-parsed the repo on
        # every run.
        for label in (NodeLabel.MODULE, NodeLabel.PACKAGE):
            await self.execute_write(
                f"UNWIND $items AS item "
                f"MATCH (n:{label} {{project_name: $p, file_path: item.fp}}) "
                "SET n.file_hash = item.fh",
                {"p": project_name, "items": params},
            )

    async def merge_package_node(self, project_name: str, qualified_name: str, name: str, file_path: str) -> None:
        """Create or update a Package node by uid."""
        uid = f"{project_name}:{qualified_name}"
        await self.execute_write(
            f"MERGE (n:{NodeLabel.PACKAGE} {{uid: $uid}}) "
            f"SET n.project_name = $project_name, n.name = $name, "
            f"n.qualified_name = $qualified_name, n.file_path = $file_path",
            {
                "uid": uid,
                "project_name": project_name,
                "name": name,
                "qualified_name": qualified_name,
                "file_path": file_path,
            },
        )

    async def create_contains_edge(self, from_uid: str, to_uid: str) -> None:
        """Create an idempotent CONTAINS relationship between two nodes."""
        await self.execute_write(
            f"MATCH (a {{uid: $from_uid}}), (b {{uid: $to_uid}}) MERGE (a)-[:{RelType.CONTAINS}]->(b)",
            {"from_uid": from_uid, "to_uid": to_uid},
        )

    async def merge_package_batch(
        self,
        project_name: str,
        packages: list[tuple[str, str, str]],
    ) -> None:
        """Create/update Package nodes and CONTAINS edges in two batched queries.

        Each tuple is ``(qualified_name, name, file_path)``.  Parent UIDs are
        derived from the dotted *qualified_name* prefix; top-level packages
        point to the Project node.
        """
        if not packages:
            return

        params = []
        for qn, name, fp in packages:
            uid = f"{project_name}:{qn}"
            parent_qn = qn.rsplit(".", 1)[0] if "." in qn else None
            parent_uid = f"{project_name}:{parent_qn}" if parent_qn else project_name
            params.append(
                {
                    "uid": uid,
                    "project_name": project_name,
                    "name": name,
                    "qualified_name": qn,
                    "file_path": fp,
                    "parent_uid": parent_uid,
                }
            )

        await self.execute_write(
            f"UNWIND $pkgs AS p "
            f"MERGE (n:{NodeLabel.PACKAGE} {{uid: p.uid}}) "
            f"SET n.project_name = p.project_name, n.name = p.name, "
            f"n.qualified_name = p.qualified_name, n.file_path = p.file_path",
            {"pkgs": params},
        )
        await self.execute_write(
            f"UNWIND $pkgs AS p "
            f"MATCH (parent {{uid: p.parent_uid}}), (child {{uid: p.uid}}) "
            f"MERGE (parent)-[:{RelType.CONTAINS}]->(child)",
            {"pkgs": params},
        )

    async def delete_project_data(self, project_name: str) -> None:
        """Delete all nodes belonging to a project (for full reindex)."""
        await self.execute_write(
            "MATCH (n {project_name: $project_name}) DETACH DELETE n",
            {"project_name": project_name},
        )

    async def update_project_metadata(self, project_name: str, **metadata: Any) -> None:
        """Update properties on the Project node."""
        uid = project_name
        set_clause = ", ".join(f"n.{k} = ${k}" for k in metadata)
        if not set_clause:
            return
        await self.execute_write(
            f"MATCH (n:{NodeLabel.PROJECT} {{uid: $uid}}) SET {set_clause}",
            {"uid": uid, **metadata},
        )

    async def get_project_status(self, project_name: str | None = None) -> list[dict[str, Any]]:
        """Query Project nodes for status display."""
        if project_name:
            return await self.execute(
                f"MATCH (n:{NodeLabel.PROJECT} {{uid: $uid}}) RETURN n",
                {"uid": project_name},
            )
        return await self.execute(f"MATCH (n:{NodeLabel.PROJECT}) RETURN n")

    async def get_project_git_hash(self, project_name: str) -> str | None:
        """Read stored git_hash from the Project node."""
        records = await self.execute(
            f"MATCH (n:{NodeLabel.PROJECT} {{uid: $uid}}) RETURN n.git_hash AS git_hash",
            {"uid": project_name},
        )
        if not records or records[0]["git_hash"] is None:
            return None
        return records[0]["git_hash"]

    async def get_project_file_paths(self, project_name: str) -> set[str]:
        """Return all distinct file_paths indexed for a project.

        Includes Package nodes (from ``__init__.py``) so delta detection
        doesn't treat them as newly added on every re-index.
        """
        # Reference-counted stubs are excluded even though they now carry a file_path:
        # theirs names a file this project *mentions*, not one it indexed. Counting it
        # here would make every referenced data file look like a source file that had
        # since been deleted — inflating the delta ratio and publishing a `deleted`
        # FileChanged that DETACH DELETEs the node on every delta index.
        records = await self.execute(
            f"MATCH (n {{project_name: $p}}) "
            f"WHERE NOT n:{NodeLabel.PROJECT} AND NOT n:{NodeLabel.SCHEMA_VERSION} "
            f"AND NOT n:{NodeLabel.RESOURCE_FILE} AND NOT n:{NodeLabel.ENV_VAR} "
            "RETURN DISTINCT n.file_path AS fp",
            {"p": project_name},
        )
        return {r["fp"] for r in records if r["fp"]}

    async def count_entities(self, project_name: str) -> int:
        """Count all entity nodes (Module, TypeDef, Callable, Value, Package) for a project."""
        records = await self.execute(
            "MATCH (n {project_name: $project_name}) "
            f"WHERE n:{NodeLabel.MODULE} OR n:{NodeLabel.TYPE_DEF} OR n:{NodeLabel.CALLABLE} "
            f"OR n:{NodeLabel.VALUE} OR n:{NodeLabel.PACKAGE} "
            "RETURN count(n) AS cnt",
            {"project_name": project_name},
        )
        return records[0]["cnt"] if records else 0

    # -- Import resolution helpers ---------------------------------------------

    async def resolve_imports(  # noqa: PLR0912, PLR0915
        self,
        project_name: str,
        import_rels: list[ParsedRelationship],
    ) -> None:
        """Resolve IMPORTS relationships after all files in a batch have been upserted.

        Classifies each import as internal (target exists in graph) or external
        (no match → create ExternalPackage/ExternalSymbol stubs), then creates
        IMPORTS edges for both.
        """
        if not import_rels:
            return

        # 1. Query all internal entity qualified_name → uid.
        #    Every referenced-not-defined label is excluded: their
        #    qualified_names live in synthetic namespaces (ext/, res/) that an
        #    import must never resolve into.
        records = await self.execute(
            f"MATCH (n {{project_name: $p}}) "
            f"WHERE NOT n:{NodeLabel.EXTERNAL_PACKAGE} AND NOT n:{NodeLabel.EXTERNAL_SYMBOL} "
            f"AND NOT n:{NodeLabel.RESOURCE_FILE} AND NOT n:{NodeLabel.ENV_VAR} "
            f"AND NOT n:{NodeLabel.SCHEMA_VERSION} AND NOT n:{NodeLabel.PROJECT} "
            "RETURN n.qualified_name AS qn, n.uid AS uid, n.file_path AS fp",
            {"p": project_name},
        )
        internal_map: dict[str, str] = {}
        py_importers: set[str] = set()  # uids of Python-file entities (prefix fallback is Python-only)
        for r in records:
            internal_map[r["qn"]] = r["uid"]
            if (r["fp"] or "").endswith((".py", ".pyi")):
                py_importers.add(r["uid"])

        # 2. Classify imports as internal or external
        import_edges: list[dict[str, Any]] = []  # [{from_uid, to_uid, type_only?}]
        ext_packages: dict[str, dict[str, str]] = {}  # top_level → {uid, name, qn, project_name}
        ext_symbols: dict[str, dict[str, str]] = {}  # dotted_path → {uid, name, qn, package, project_name}

        for rel in import_rels:
            to_name = rel.to_name
            from_uid = rel.from_qualified_name  # already project-prefixed uid
            is_type_only = rel.properties.get("type_only", False)

            # Internal match — exact first; for Python importers, fall back to
            # progressively shorter dotted prefixes (imports of re-exported or
            # non-entity names resolve to the closest containing module/package).
            # The fallback is Python-only per S2: in other languages, dotted
            # import paths (java.util.List, System.Collections.Generic) live in
            # a different namespace than path-derived qualified_names, so a
            # prefix hit there would misclassify an external import as internal.
            target_uid = internal_map.get(to_name)
            if target_uid is None and from_uid in py_importers:
                prefix = to_name
                while target_uid is None and "." in prefix:
                    prefix = prefix.rsplit(".", 1)[0]
                    target_uid = internal_map.get(prefix)
            if target_uid is not None:
                edge: dict[str, Any] = {"from_uid": from_uid, "to_uid": target_uid}
                if is_type_only:
                    edge["type_only"] = True
                import_edges.append(edge)
                continue

            # External import — derive top-level package
            top_level = to_name.split(".")[0]
            if not top_level:
                logger.debug("Skipping malformed import name {!r} from {}", to_name, from_uid)
                continue
            pkg_uid = f"{project_name}:ext/{top_level}"

            if top_level not in ext_packages:
                ext_packages[top_level] = {
                    "uid": pkg_uid,
                    "project_name": project_name,
                    "name": top_level,
                    "qualified_name": f"ext/{top_level}",
                }

            if to_name == top_level:
                # Bare package import (e.g. `import os`) → point to ExternalPackage
                edge = {"from_uid": from_uid, "to_uid": pkg_uid}
                if is_type_only:
                    edge["type_only"] = True
                import_edges.append(edge)
            else:
                # Symbol import (e.g. `from loguru import logger`) → ExternalSymbol
                sym_uid = f"{project_name}:ext/{to_name}"
                sym_name = to_name.rsplit(".", 1)[-1]
                if to_name not in ext_symbols:
                    ext_symbols[to_name] = {
                        "uid": sym_uid,
                        "project_name": project_name,
                        "name": sym_name,
                        "qualified_name": f"ext/{to_name}",
                        "package": top_level,
                    }
                edge = {"from_uid": from_uid, "to_uid": sym_uid}
                if is_type_only:
                    edge["type_only"] = True
                import_edges.append(edge)

        # 3. MERGE ExternalPackage nodes
        if ext_packages:
            await self.execute_write(
                f"UNWIND $packages AS pkg "
                f"MERGE (n:{NodeLabel.EXTERNAL_PACKAGE} {{uid: pkg.uid}}) "
                f"ON CREATE SET n.project_name = pkg.project_name, n.name = pkg.name, "
                f"n.qualified_name = pkg.qualified_name",
                {"packages": list(ext_packages.values())},
            )

        # 4. MERGE ExternalSymbol nodes
        if ext_symbols:
            await self.execute_write(
                f"UNWIND $symbols AS sym "
                f"MERGE (n:{NodeLabel.EXTERNAL_SYMBOL} {{uid: sym.uid}}) "
                f"ON CREATE SET n.project_name = sym.project_name, n.name = sym.name, "
                f"n.qualified_name = sym.qualified_name, n.package = sym.package",
                {"symbols": list(ext_symbols.values())},
            )

        # 5. CONTAINS edges (ExternalPackage → ExternalSymbol)
        contains_edges = [
            {"pkg_uid": f"{project_name}:ext/{sym['package']}", "sym_uid": sym["uid"]} for sym in ext_symbols.values()
        ]
        if contains_edges:
            await self.execute_write(
                f"UNWIND $edges AS e "
                f"MATCH (p:{NodeLabel.EXTERNAL_PACKAGE} {{uid: e.pkg_uid}}), "
                f"(s:{NodeLabel.EXTERNAL_SYMBOL} {{uid: e.sym_uid}}) "
                f"MERGE (p)-[:{RelType.CONTAINS}]->(s)",
                {"edges": contains_edges},
            )

        # 6. IMPORTS edges (both internal and external targets)
        #    Split into type_only and normal for efficient batching
        normal_edges = [e for e in import_edges if not e.get("type_only")]
        type_only_edges = [e for e in import_edges if e.get("type_only")]

        if normal_edges:
            await self.execute_write(
                f"UNWIND $rels AS r "
                f"MATCH (a {{uid: r.from_uid}}), (b {{uid: r.to_uid}}) "
                f"MERGE (a)-[:{RelType.IMPORTS}]->(b)",
                {"rels": normal_edges},
            )
        if type_only_edges:
            await self.execute_write(
                f"UNWIND $rels AS r "
                f"MATCH (a {{uid: r.from_uid}}), (b {{uid: r.to_uid}}) "
                f"MERGE (a)-[e:{RelType.IMPORTS}]->(b) ON CREATE SET e.type_only = true",
                {"rels": type_only_edges},
            )

        logger.debug(
            "Resolved {} imports ({} packages, {} symbols created)",
            len(import_rels),
            len(ext_packages),
            len(ext_symbols),
        )

    async def resolve_config_refs(self, project_name: str, ref_rels: list[ParsedRelationship]) -> None:
        """MERGE EnvVar/ResourceFile nodes and their READS_ENV/REFERENCES_FILE edges.

        Runs post-batch for the same reason ``resolve_imports`` does: the target
        node does not exist until this call creates it.

        Node properties come from :func:`_plan_config_refs`'s four-key allowlist
        and edges are written bare — see the module-level "capture NAMES, never
        VALUES" invariant above.  ``MERGE ... ON CREATE SET`` (never a bare
        ``SET``) means a re-resolve of an existing node cannot overwrite it
        either.
        """
        if not ref_rels:
            return

        plan = _plan_config_refs(project_name, ref_rels)

        for label, nodes in (
            (NodeLabel.ENV_VAR, plan.env_nodes),
            (NodeLabel.RESOURCE_FILE, plan.file_nodes),
        ):
            if not nodes:
                continue
            await self.execute_write(
                f"UNWIND $nodes AS n "
                f"MERGE (x:{label} {{uid: n.uid}}) "
                f"ON CREATE SET x.project_name = n.project_name, x.name = n.name, "
                f"x.qualified_name = n.qualified_name "
                # Unconditional, unlike the rest, so nodes created before file_path was
                # planned self-heal on the next resolve instead of needing a migration.
                # Safe for EnvVar, whose node dicts carry no such key: a missing map key
                # is null in Cypher and SET to null is a no-op.
                f"SET x.file_path = n.file_path",
                {"nodes": list(nodes.values())},
            )

        for rel_type in (RelType.READS_ENV, RelType.REFERENCES_FILE):
            edges = [{"from_uid": f, "to_uid": t} for f, t, rt in plan.edges if rt == rel_type.value]
            if not edges:
                continue
            await self.execute_write(
                f"UNWIND $rels AS r MATCH (a {{uid: r.from_uid}}), (b {{uid: r.to_uid}}) MERGE (a)-[:{rel_type}]->(b)",
                {"rels": edges},
            )

        logger.debug(
            "Resolved {} config refs ({} env vars, {} resource files)",
            len(ref_rels),
            len(plan.env_nodes),
            len(plan.file_nodes),
        )

    async def gc_orphaned_reference_nodes(self) -> int:
        """Delete EnvVar/ResourceFile nodes that nothing points at any more.

        These labels are reference-counted: they exist only because some entity
        referenced them, and they receive no structural edges, so incoming-edge
        count *is* the reference count (see schema._REFERENCE_COUNTED_LABELS).
        ``_recreate_batch_relationships`` drops every outgoing edge of a
        reparsed file's entities before recreating them, so the last reference
        vanishing from source is exactly the last incoming edge vanishing here.

        Cost is bounded by the two smallest labels in the graph, not by the
        graph: one label-index scan each, never a full scan.  Must run *after*
        the batch's ``resolve_config_refs``, never between the edge-delete and
        the recreate — in that window a still-referenced node has zero edges.
        """
        total = 0
        for label in sorted(_REFERENCE_COUNTED_LABELS, key=lambda lbl: lbl.value):
            records = await self.execute(f"MATCH (n:{label}) WHERE NOT ()-[]->(n) RETURN n.uid AS uid")
            uids = [r["uid"] for r in records if r["uid"]]
            if not uids:
                continue
            await self.execute_write(
                f"UNWIND $uids AS uid MATCH (n:{label} {{uid: uid}}) DETACH DELETE n",
                {"uids": uids},
            )
            total += len(uids)
        if total:
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
    ) -> None:
        """Resolve CALLS relationships after all files in a batch have been upserted.

        Each call rel has a bare name (e.g. ``"some_func"``) as ``to_name``.
        Resolution strategy (in priority order):
        1. **Import match** — caller's module imports something with that name.
        2. **Same-class sibling** — if caller is a method, check siblings in same TypeDef.
        3. **Same-file match** — any Callable with that name in the same file.
        4. **Project-wide match** — every Callable with that name.
        5. **Constructor call** — bare_name is a class name; resolves to its `__init__`(s).
        6. **Unresolved** — skip silently (builtins, dynamic calls).

        Every created edge is tagged ``confidence`` (``"resolved"`` for a single
        candidate, ``"ambiguous"`` when strategy 4/5 matched more than one — see
        ADR-0014) and ``strategy`` (``"import"|"sibling"|"same_file"|"project_unique"|
        "project_wide"|"constructor"``), plus three properties amending ADR-0014:
        ``candidate_count`` (how many targets the winning strategy returned),
        ``from_test`` (the caller lives in test code), and the numeric ``weight``
        derived from those two by :func:`_call_edge_weight`.

        *test_patterns* overrides the glob patterns behind ``from_test``;
        ``None`` uses ``SearchSettings``' defaults. A caller whose uid is absent
        from the lookup (not yet upserted, or a NULL file_path) has no path to
        match and is treated as non-test.
        """
        if not call_rels:
            return

        if lookup is None:
            lookup = await self._build_call_lookup(project_name)

        if name_to_typedefs is None:
            td_records = await self.execute(
                f"MATCH (n:{NodeLabel.TYPE_DEF} {{project_name: $p}}) "
                "RETURN n.name AS name, n.uid AS uid, n.file_path AS fp",
                {"p": project_name},
            )
            name_to_typedefs = {}
            for r in td_records:
                name_to_typedefs.setdefault(r["name"], []).append((r["uid"], r["fp"] or ""))

        # Resolve each call. Several call sites can produce the same (from,to)
        # pair; _combine_call_edge_facts decides what the single stored edge says.
        patterns = list(_DEFAULT_TEST_PATTERNS if test_patterns is None else test_patterns)
        caller_is_test: dict[str, bool] = {}
        edges: dict[tuple[str, str], _CallEdgeFacts] = {}
        resolved = 0
        ambiguous = 0
        unresolved = 0
        for rel in call_rels:
            result = _resolve_one_call(project_name, rel, lookup, name_to_typedefs)
            if result is None:
                unresolved += 1
                continue
            candidate_uids, strategy = result
            # A lone candidate is not enough on its own: an unverified receiver yields
            # exactly one name match and still cannot be trusted, so candidate_count 1
            # with confidence "ambiguous" is a real and informative combination.
            confidence = (
                "resolved" if len(candidate_uids) == 1 and strategy not in _UNVERIFIED_STRATEGIES else "ambiguous"
            )
            if confidence == "resolved":
                resolved += 1
            else:
                ambiguous += 1
            caller_uid = rel.from_qualified_name
            from_test = caller_is_test.get(caller_uid)
            if from_test is None:
                caller_name, caller_fp = lookup.uid_to_info.get(caller_uid, ("", ""))
                from_test = matches_test_pattern(caller_fp, caller_name, patterns)
                caller_is_test[caller_uid] = from_test
            observed = _CallEdgeFacts(confidence, strategy, len(candidate_uids), from_test)
            for target_uid in candidate_uids:
                key = (caller_uid, target_uid)
                prior = edges.get(key)
                edges[key] = observed if prior is None else _combine_call_edge_facts(prior, observed)

        # Batch-create CALLS edges
        if edges:
            edge_params = [
                {
                    "f": f,
                    "t": t,
                    "confidence": facts.confidence,
                    "strategy": facts.strategy,
                    "candidate_count": facts.candidate_count,
                    "from_test": facts.from_test,
                    "weight": _call_edge_weight(facts.candidate_count, facts.from_test, facts.strategy),
                }
                for (f, t), facts in edges.items()
            ]
            await self.execute_write(
                f"UNWIND $rels AS r "
                f"MATCH (a:{NodeLabel.CALLABLE} {{uid: r.f}}), (b:{NodeLabel.CALLABLE} {{uid: r.t}}) "
                f"MERGE (a)-[e:{RelType.CALLS}]->(b) "
                f"SET e.confidence = r.confidence, e.strategy = r.strategy, "
                f"e.candidate_count = r.candidate_count, e.from_test = r.from_test, e.weight = r.weight",
                {"rels": edge_params},
            )

        logger.debug("Resolved {} CALLS edges ({} ambiguous, {} unresolved)", resolved, ambiguous, unresolved)

    async def build_anchor_lookup(self) -> _AnchorLookup:
        """Build the cross-project lookup tables needed for anchor resolution."""
        file_records = await self.execute(
            f"MATCH (n) WHERE n:{NodeLabel.MODULE} OR n:{NodeLabel.DOC_FILE} OR n:{NodeLabel.NOTE} "
            "RETURN n.project_name AS project, n.file_path AS fp, n.uid AS uid, n.content_hash AS hash"
        )
        file_by_path: dict[str, dict[str, list[tuple[str, str]]]] = {}
        for r in file_records:
            file_by_path.setdefault(r["project"], {}).setdefault(r["fp"], []).append((r["uid"], r["hash"] or ""))

        symbol_records = await self.execute(
            f"MATCH (n) WHERE n:{NodeLabel.CALLABLE} OR n:{NodeLabel.TYPE_DEF} OR n:{NodeLabel.VALUE} "
            "RETURN n.project_name AS project, n.file_path AS fp, n.name AS name, n.uid AS uid, n.content_hash AS hash"
        )
        symbols_by_path: dict[str, dict[str, dict[str, list[tuple[str, str]]]]] = {}
        for r in symbol_records:
            proj_map = symbols_by_path.setdefault(r["project"], {})
            proj_map.setdefault(r["fp"] or "", {}).setdefault(r["name"], []).append((r["uid"], r["hash"] or ""))

        project_records = await self.execute(
            f"MATCH (p:{NodeLabel.PROJECT}) WHERE p.root_path IS NOT NULL "
            "RETURN p.project_name AS project, p.root_path AS root"
        )
        project_roots = {r["project"]: r["root"] for r in project_records if r["root"]}

        return _AnchorLookup(file_by_path=file_by_path, symbols_by_path=symbols_by_path, project_roots=project_roots)

    async def resolve_anchors(
        self,
        anchor_rels: list[ParsedRelationship],
        *,
        lookup: _AnchorLookup | None = None,
    ) -> None:
        """Resolve explicit ``anchors:`` frontmatter into DOCUMENTS edges after batch upsert.

        Anchors may cross project boundaries (uid/project-prefixed/absolute
        path forms), so resolution runs against a cross-project lookup
        rather than one project's own entities. Never multi-links: an
        ambiguous or missing target is recorded on the note's
        ``unresolved_anchors`` instead of guessing. ``anchor_hash`` captures
        the target's content_hash at link time — ``invalidate_stale_anchors``
        later compares against it to detect drift.
        """
        if not anchor_rels:
            return
        if lookup is None:
            lookup = await self.build_anchor_lookup()

        resolved: list[dict[str, str]] = []
        unresolved_by_note: dict[str, list[str]] = {}
        uid_form: list[ParsedRelationship] = []

        for rel in anchor_rels:
            if rel.properties.get("anchor_form", "") == "uid":
                uid_form.append(rel)
                continue

            target = _resolve_one_path_anchor(rel, lookup)
            if target is None:
                raw = rel.properties.get("anchor_raw", rel.to_name)
                unresolved_by_note.setdefault(rel.from_qualified_name, []).append(raw)
                continue

            file_uid, file_hash = target
            resolved.append({"from_uid": rel.from_qualified_name, "to_uid": file_uid, "to_hash": file_hash})

        if uid_form:
            records = await self.execute(
                "UNWIND $uids AS uid MATCH (b {uid: uid}) RETURN b.uid AS uid, b.content_hash AS hash",
                {"uids": list({r.to_name for r in uid_form})},
            )
            hash_by_uid = {r["uid"]: r["hash"] or "" for r in records}
            for rel in uid_form:
                found = hash_by_uid.get(rel.to_name)
                if found is None:
                    raw = rel.properties.get("anchor_raw", rel.to_name)
                    unresolved_by_note.setdefault(rel.from_qualified_name, []).append(raw)
                else:
                    resolved.append({"from_uid": rel.from_qualified_name, "to_uid": rel.to_name, "to_hash": found})

        if resolved:
            await self.execute_write(
                f"UNWIND $rels AS r "
                f"MATCH (a:{NodeLabel.NOTE} {{uid: r.from_uid}}) "
                f"MATCH (b {{uid: r.to_uid}}) "
                f"MERGE (a)-[e:{RelType.DOCUMENTS} {{link_type: 'anchor'}}]->(b) "
                f"SET e.confidence = 1.0, e.anchor_hash = r.to_hash, e.stale = false",
                {"rels": resolved},
            )

        # Every note with anchors this batch gets its unresolved list recomputed
        # from scratch — a note whose anchors now all resolve must be cleared,
        # not left with a stale failure list from a prior parse.
        all_notes = {rel.from_qualified_name for rel in anchor_rels}
        note_updates = [{"uid": note_uid, "unresolved": unresolved_by_note.get(note_uid, [])} for note_uid in all_notes]
        await self.execute_write(
            f"UNWIND $notes AS item MATCH (n:{NodeLabel.NOTE} {{uid: item.uid}}) "
            "SET n.unresolved_anchors = item.unresolved",
            {"notes": note_updates},
        )

        total_unresolved = sum(len(v) for v in unresolved_by_note.values())
        logger.debug("Resolved {} anchor edges ({} unresolved)", len(resolved), total_unresolved)

    async def invalidate_stale_anchors(self, changed_uids: set[str]) -> int:
        """Mark anchor DOCUMENTS edges stale whose target's content_hash has drifted.

        Runs after every upsert batch (not gated by significance) — the
        moment an anchored function's content changes, the note documenting
        it is flagged stale in retrieval within seconds.
        """
        if not changed_uids:
            return 0
        records = await self.execute(
            f"UNWIND $uids AS uid "
            f"MATCH (n:{NodeLabel.NOTE})-[r:{RelType.DOCUMENTS} {{link_type: 'anchor'}}]->(e {{uid: uid}}) "
            "WHERE r.anchor_hash <> e.content_hash "
            "SET r.stale = true "
            "RETURN count(r) AS cnt",
            {"uids": list(changed_uids)},
        )
        count = records[0]["cnt"] if records else 0
        if count:
            logger.debug("Marked {} anchor edge(s) stale", count)
        return count

    async def build_citation_lookup(self, project_name: str) -> _CitationLookup:
        """Build *project_name*'s canonical-key → document-node index for citation resolution."""
        records = await self.execute(
            f"MATCH (n {{project_name: $p}}) "
            f"WHERE n:{NodeLabel.DOC_FILE} OR n:{NodeLabel.DOC_SECTION} OR n:{NodeLabel.NOTE} "
            "RETURN labels(n)[0] AS label, n.uid AS uid, n.name AS name, n.file_path AS fp, "
            "n.header_level AS lvl",
            {"p": project_name},
        )
        by_key: dict[tuple[str, int], list[tuple[int, str]]] = {}
        for r in records:
            keys = _document_citation_keys(r["label"] or "", r["name"] or "", r["fp"] or "", r["lvl"])
            for key, rank in keys:
                by_key.setdefault(key, []).append((rank, r["uid"]))
        return _CitationLookup(by_key=by_key)

    async def resolve_citations(
        self,
        project_name: str,
        citations_by_uid: dict[str, list[str]],
        *,
        file_paths: Collection[str] | None = None,
        lookup: _CitationLookup | None = None,
        retry_unresolved: bool = False,
    ) -> None:
        """Turn recorded ``citations`` strings into DOCUMENTS edges after batch upsert.

        *citations_by_uid* maps a citing entity's uid to the raw citation
        strings ``extract_rationale`` found in its comments. Each resolves to
        at most one document node in the same project via the canonical
        ``(scheme, number)`` key (see ``_citation_key``), producing

            (document) -[:DOCUMENTS {link_type: 'citation'}]-> (citing entity)

        — doc → code, like every other DOCUMENTS edge (see the DIRECTION note
        in this module's citation section). ``link_type`` distinguishes these
        from ``'anchor'`` edges and from the heuristic ``'explicit'``/
        ``'symbol_mention'``/``'file_ref'`` links ``_create_doc_links`` emits,
        and ``confidence`` reflects how the document was identified — 1.0 only
        for a numbered file in a scheme-named directory.

        The edge is written from the document's node but *owned* by the citing
        file's parse, which is why ``_recreate_batch_relationships`` and
        ``_recreate_file_relationships`` exclude it from the delete phase (the
        same carve-out cross-file DEFINES gets): re-parsing the ADR must not
        drop the citations pointing out of it, because nothing in that parse
        could put them back.

        *file_paths* is the other half of that ownership: the citing files this
        call is (re)parsing. Their INBOUND citation edges are deleted before the
        MERGE below, which is what gives citations the delete-then-recreate
        lifecycle every other parsed relationship has. Those two ``_recreate_*``
        delete phases only ever sweep edges *leaving* the file being parsed, so
        they structurally cannot reach an inbound citation — without this pass a
        citation whose comment was deleted would survive forever. The scope is
        file paths rather than uids precisely so the removal case works: a file
        whose last citation is gone contributes no entry to *citations_by_uid*
        at all, and it stays bounded by the batch's file count either way.

        Omit *file_paths* (``None``) for any pass that resolves without
        reparsing the citing side — above all a ``retry_unresolved`` sweep,
        which is project-wide and must never delete.

        Resolution is project-scoped: every repo has an ADR-0001, so a
        cross-project lookup (what anchors do, because anchors carry explicit
        project/uid forms) would collide by construction.

        Anything that does not resolve — a typo'd ADR number, an ADR not
        indexed yet, or an inherently external scheme like RFC, which has no
        local document by definition — is recorded on the citing node's
        ``unresolved_citations``, never silently dropped. The list is
        recomputed from scratch for every uid passed in, so a citation that
        starts resolving clears itself.

        *retry_unresolved* additionally re-attempts entities already carrying a
        non-empty ``unresolved_citations``, re-reading their ``citations``
        property as the source of truth. Without it a first full index leaves
        every ADR reference broken: code files are almost always published
        before the ``wiki/``/``docs/`` tree, so the document node does not
        exist yet when the citing file's batch resolves. Callers run it
        whenever a batch adds or changes document nodes, and once more at the
        end of a run (see ``ASTConsumer._flush_deferred_resolution``).
        """
        if not citations_by_uid and not retry_unresolved and not file_paths:
            return

        if file_paths:
            # Revoke phase, scoped to the citing files being reparsed. Runs
            # before the MERGE below (and before the early return for an empty
            # payload) so a file whose last citation was deleted still clears.
            await self.execute_write(
                f"MATCH (entity {{project_name: $p}})<-[r:{RelType.DOCUMENTS} {{link_type: 'citation'}}]-() "
                "WHERE entity.file_path IN $fps DELETE r",
                {"p": project_name, "fps": list(file_paths)},
            )

        pending: dict[str, list[str]] = {uid: list(raws) for uid, raws in citations_by_uid.items()}
        if retry_unresolved:
            records = await self.execute(
                "MATCH (n {project_name: $p}) "
                "WHERE n.unresolved_citations IS NOT NULL AND size(n.unresolved_citations) > 0 "
                "RETURN n.uid AS uid, n.citations AS citations",
                {"p": project_name},
            )
            for r in records:
                # ``citations`` is the evidence; ``unresolved_citations`` is
                # only bookkeeping. Re-reading the former means an entity whose
                # citation comment was deleted gets its stale bookkeeping
                # cleared here rather than lingering forever.
                pending.setdefault(r["uid"], list(r["citations"] or []))

        if not pending:
            return
        if lookup is None:
            lookup = await self.build_citation_lookup(project_name)

        resolved: list[dict[str, Any]] = []
        unresolved_by_uid: dict[str, list[str]] = {}
        for entity_uid, raws in pending.items():
            for raw in raws:
                key = _citation_key(raw)
                target = _pick_citation_target(key, lookup) if key is not None else None
                if key is None or target is None or target[0] == entity_uid:
                    unresolved_by_uid.setdefault(entity_uid, []).append(raw)
                    continue
                doc_uid, confidence = target
                resolved.append(
                    {
                        "doc_uid": doc_uid,
                        "entity_uid": entity_uid,
                        "citation": _render_citation_key(key),
                        "confidence": confidence,
                    }
                )

        if resolved:
            await self.execute_write(
                f"UNWIND $rels AS r "
                f"MATCH (doc {{uid: r.doc_uid}}) "
                f"MATCH (entity {{uid: r.entity_uid}}) "
                f"MERGE (doc)-[e:{RelType.DOCUMENTS} {{link_type: 'citation'}}]->(entity) "
                f"SET e.confidence = r.confidence, e.citation = r.citation",
                {"rels": resolved},
            )

        entity_updates = [{"uid": uid, "unresolved": unresolved_by_uid.get(uid, [])} for uid in pending]
        await self.execute_write(
            "UNWIND $items AS item MATCH (n {uid: item.uid}) SET n.unresolved_citations = item.unresolved",
            {"items": entity_updates},
        )

        total_unresolved = sum(len(v) for v in unresolved_by_uid.values())
        logger.debug(
            "Resolved {} citation edge(s) for project {} ({} unresolved)",
            len(resolved),
            project_name,
            total_unresolved,
        )

    async def resolve_inherits(self, project_name: str, inherit_rels: list[ParsedRelationship]) -> None:
        """Link a class to its base, whether that base is in this project or imported.

        Runs post-batch rather than at create time because most bases are external. The
        parser has always emitted ``INHERITS -> StrEnum`` / ``-> ABC`` / ``-> Protocol``;
        the write path required the target to be an in-project ``TypeDef``, so the MATCH
        returned nothing and the edge was discarded without an error. Measured before this
        ran: 43 of 45 classes with a base had no inheritance edge.

        An in-project ``TypeDef`` wins over an ``ExternalSymbol`` of the same name — a
        local class shadowing an imported one is the class the code actually subclasses.
        A base that is neither (a builtin like ``Exception``, which is never imported and
        so has no node) stays unresolved rather than inventing a target.
        """
        if not inherit_rels:
            return

        params = [
            {"from_uid": r.from_qualified_name, "to_name": r.to_name, "project": project_name} for r in inherit_rels
        ]
        # Two passes rather than one OR-matched query: an OR over two labels makes
        # Memgraph scan both and would fan out to BOTH when a name exists in each.
        await self.execute_write(
            f"UNWIND $rels AS r "
            f"MATCH (a:{NodeLabel.TYPE_DEF} {{uid: r.from_uid}}), "
            f"(b:{NodeLabel.TYPE_DEF} {{project_name: r.project, name: r.to_name}}) "
            f"MERGE (a)-[:{RelType.INHERITS}]->(b)",
            {"rels": params},
        )
        await self.execute_write(
            f"UNWIND $rels AS r "
            f"MATCH (a:{NodeLabel.TYPE_DEF} {{uid: r.from_uid}}) "
            f"WHERE NOT (a)-[:{RelType.INHERITS}]->(:{NodeLabel.TYPE_DEF} {{name: r.to_name}}) "
            f"MATCH (b:{NodeLabel.EXTERNAL_SYMBOL} {{project_name: r.project, name: r.to_name}}) "
            f"MERGE (a)-[:{RelType.INHERITS}]->(b)",
            {"rels": params},
        )

        # Builtin bases have no node to point at because they are never imported —
        # `class StorageError(Exception)` names something that appears in no import
        # statement anywhere. So "show me every exception type" answered nothing while
        # StrEnum and ABC answered fine. The node is created here rather than at parse
        # time, and only for names Python actually defines, so a typo or a generic
        # (`Generic[T]`) still resolves to nothing instead of minting a node for itself.
        builtin_params = [p for p in params if hasattr(builtins, p["to_name"])]
        if builtin_params:
            await self.execute_write(
                f"UNWIND $rels AS r "
                f"MATCH (a:{NodeLabel.TYPE_DEF} {{uid: r.from_uid}}) "
                f"WHERE NOT (a)-[:{RelType.INHERITS}]->({{name: r.to_name}}) "
                f"MERGE (b:{NodeLabel.EXTERNAL_SYMBOL} {{uid: r.project + ':ext/builtins.' + r.to_name}}) "
                f"ON CREATE SET b.project_name = r.project, b.name = r.to_name, "
                f"b.qualified_name = 'builtins.' + r.to_name "
                f"MERGE (a)-[:{RelType.INHERITS}]->(b)",
                {"rels": builtin_params},
            )

    async def resolve_value_references(self, project_name: str, ref_rels: list[ParsedRelationship]) -> None:
        """Link a callable named as a value to the callable it names.

        Import-scope and same-file only, deliberately. A project-wide bare-name match is
        exactly what ADR-0022 removed from call resolution: `foo(bar)` where `bar` is a
        local variable that happens to share a name with some function elsewhere would
        manufacture an edge, and a wrong REFERENCES edge is worse than none because
        find_dead_code now treats it as proof of life.

        Only Callables are linked. Most identifier arguments are ordinary values and
        resolve to nothing, which is the correct outcome rather than a miss.
        """
        if not ref_rels:
            return

        by_type: dict[str, list[dict[str, str]]] = {}
        for r in ref_rels:
            by_type.setdefault(str(r.rel_type), []).append(
                {"from_uid": r.from_qualified_name, "to_name": r.to_name, "project": project_name}
            )
        for rel_type, params in by_type.items():
            if rel_type == RelType.EXPORTS:
                # A re-export resolves through the module's own IMPORTS edge, which is
                # proof it can see the name — no label constraint, since __all__ lists
                # functions, classes and constants alike.
                await self.execute_write(
                    f"UNWIND $rels AS r "
                    f"MATCH (m {{uid: r.from_uid}})-[:{RelType.IMPORTS}]->(t {{name: r.to_name}}) "
                    f"MERGE (m)-[:{RelType.EXPORTS}]->(t)",
                    {"rels": params},
                )
                continue
            # A field's declared type resolves to a TypeDef; a referenced or registering
            # name resolves to a Callable. Same scope rules either way.
            target = NodeLabel.TYPE_DEF if rel_type == RelType.USES_TYPE else NodeLabel.CALLABLE
            await self._link_named_callable(rel_type, params, target)

    async def _link_named_callable(
        self, rel_type: str, params: list[dict[str, str]], target: NodeLabel = NodeLabel.CALLABLE
    ) -> None:
        """Link a bare name to an entity, in the referrer's file or its import scope.

        Shared by REFERENCES, REGISTERED_BY and field-level USES_TYPE: all three name their
        target rather than uid it, and all must resolve in a scope the name provably
        reaches. A project-wide match is what ADR-0022 removed from call resolution.
        """
        await self.execute_write(
            f"UNWIND $rels AS r "
            f"MATCH (a {{uid: r.from_uid}}), (b:{target} {{project_name: r.project, name: r.to_name}}) "
            f"WHERE b.file_path = a.file_path AND b.uid <> a.uid "
            f"MERGE (a)-[:{rel_type}]->(b)",
            {"rels": params},
        )
        await self.execute_write(
            f"UNWIND $rels AS r "
            f"MATCH (a {{uid: r.from_uid}}), (b:{target} {{project_name: r.project, name: r.to_name}}) "
            f"WHERE b.uid <> a.uid AND NOT (a)-[:{rel_type}]->(b) "
            f"AND EXISTS {{ MATCH (m:{NodeLabel.MODULE} {{file_path: a.file_path, project_name: r.project}})"
            f"-[:{RelType.IMPORTS}]->(b) }} "
            f"MERGE (a)-[:{rel_type}]->(b)",
            {"rels": params},
        )

    async def resolve_protocol_conformance(self, project_name: str) -> int:
        """Link a class to every self-declared Protocol whose method set it satisfies.

        Python Protocol conformance is structural — `GraphClient` and `SqliteGraphClient`
        both satisfy `GraphBackend` and neither names it — so the graph held nothing
        implementing this codebase's central abstraction. 88 of its 102 `...`-bodied stub
        methods had no inbound edge at all.

        ADR-0023 rejected method-set containment and this is deliberately NOT that. There,
        containment had to INFER which class in a candidate set was the interface, and it
        elected small test doubles (`RecordingBus`, `FakeDrainBus`) for having the fewest
        methods. Here the interface identifies itself by inheriting `Protocol`; containment
        only answers "does this class satisfy it". Measured on this repo, that difference
        is 90/98 precision versus 20 of 20.

        Guards, each earning its place:
        - The Protocol must declare at least one non-dunder method. A zero-method Protocol
          is satisfied by everything.
        - A Protocol is never recorded as an implementation of another Protocol. Two such
          pairs showed up here (GraphBackend satisfies SearchGraph and GraphExecutor);
          true, but not what "implements" should return.
        - `inferred: true` on the edge, so a consumer can tell a structural match from a
          declared one. `_fetch_community_inputs` reads CALLS only, so this stays out of
          the Leiden weight space without needing a decision.
        """
        rows = await self.execute(
            f"MATCH (p:{NodeLabel.TYPE_DEF} {{project_name: $project}})"
            f"-[:{RelType.INHERITS}]->({{name: 'Protocol'}}) "
            f"MATCH (p)-[:{RelType.DEFINES}]->(pm:{NodeLabel.CALLABLE}) "
            "WHERE NOT pm.name STARTS WITH '__' "
            "WITH p, collect(DISTINCT pm.name) AS pms "
            "WHERE size(pms) > 0 "
            f"MATCH (c:{NodeLabel.TYPE_DEF} {{project_name: $project}}) "
            "WHERE c.kind = 'class' AND c.uid <> p.uid "
            f"AND NOT (c)-[:{RelType.INHERITS}]->({{name: 'Protocol'}}) "
            f"MATCH (c)-[:{RelType.DEFINES}]->(cm:{NodeLabel.CALLABLE}) "
            "WITH p, pms, c, collect(DISTINCT cm.name) AS cms "
            "WHERE all(x IN pms WHERE x IN cms) "
            f"MERGE (c)-[e:{RelType.IMPLEMENTS}]->(p) "
            "SET e.inferred = true "
            "RETURN count(e) AS c",
            {"project": project_name},
        )

        # Method level, derived from the class level rather than matched independently:
        # once LogNotifier is known to satisfy Notifier, its `notify` is the thing that
        # satisfies `Notifier.notify`. Deriving it means the two answers can never
        # disagree, and "which methods implement this stub?" stops being unanswerable for
        # the 88 GraphBackend methods that had no inbound edge at all.
        await self.execute_write(
            f"MATCH (c:{NodeLabel.TYPE_DEF})-[:{RelType.IMPLEMENTS}]->(p:{NodeLabel.TYPE_DEF}) "
            "WHERE c.project_name = $project "
            f"MATCH (p)-[:{RelType.DEFINES}]->(pm:{NodeLabel.CALLABLE}) "
            f"MATCH (c)-[:{RelType.DEFINES}]->(cm:{NodeLabel.CALLABLE}) "
            "WHERE cm.name = pm.name AND NOT pm.name STARTS WITH '__' "
            f"MERGE (cm)-[e:{RelType.IMPLEMENTS}]->(pm) "
            "SET e.inferred = true",
            {"project": project_name},
        )
        return rows[0]["c"] if rows else 0

    async def resolve_type_refs(  # noqa: PLR0912
        self,
        project_name: str,
        type_rels: list[ParsedRelationship],
        *,
        lookup: _CallLookup | None = None,
        name_to_typedefs: dict[str, list[tuple[str, str]]] | None = None,
    ) -> None:
        """Resolve USES_TYPE relationships after all files in a batch have been upserted.

        Each type rel has a bare name (e.g. ``"MyClass"``) as ``to_name``.
        Resolution strategy:
        1. **Import match** — caller's module imports something with that name.
        2. **Same-file TypeDef** — a TypeDef with that name in the same file.
        3. **Project-wide TypeDef** — any TypeDef with that name (unique only).
        4. **Unresolved** — skip silently (builtins, generic types).
        """
        if not type_rels:
            return

        if lookup is None:
            lookup = await self._build_call_lookup(project_name)

        if name_to_typedefs is None:
            td_records = await self.execute(
                f"MATCH (n:{NodeLabel.TYPE_DEF} {{project_name: $p}}) "
                "RETURN n.name AS name, n.uid AS uid, n.file_path AS fp",
                {"p": project_name},
            )
            name_to_typedefs = {}
            for r in td_records:
                name_to_typedefs.setdefault(r["name"], []).append((r["uid"], r["fp"] or ""))

        edges: set[tuple[str, str]] = set()
        resolved = 0
        for rel in type_rels:
            from_uid = rel.from_qualified_name
            type_name = rel.to_name

            # Derive caller's file_path for same-file matching
            caller_info = lookup.uid_to_info.get(from_uid)
            caller_fp = caller_info[1] if caller_info else ""

            # Derive module uid for import matching
            caller_qn = from_uid.split(":", 1)[1] if ":" in from_uid else from_uid
            parts = caller_qn.split(".")
            module_uid: str | None = None
            for i in range(len(parts) - 1, 0, -1):
                candidate = f"{project_name}:{'.'.join(parts[:i])}"
                if candidate in lookup.import_map:
                    module_uid = candidate
                    break

            target_uid: str | None = None

            # Strategy 1: Import match
            if module_uid and type_name in lookup.import_map.get(module_uid, {}):
                target_uid = lookup.import_map[module_uid][type_name]

            # Strategy 2: Same-file TypeDef
            if target_uid is None and caller_fp:
                for uid, fp in name_to_typedefs.get(type_name, []):
                    if fp == caller_fp:
                        target_uid = uid
                        break

            # Strategy 3: Project-wide unique TypeDef
            if target_uid is None:
                candidates = name_to_typedefs.get(type_name, [])
                if len(candidates) == 1:
                    target_uid = candidates[0][0]

            if target_uid is not None:
                edges.add((from_uid, target_uid))
                resolved += 1

        if edges:
            edge_params = [{"f": f, "t": t} for f, t in edges]
            await self.execute_write(
                f"UNWIND $rels AS r "
                f"MATCH (a {{uid: r.f}}), (b:{NodeLabel.TYPE_DEF} {{uid: r.t}}) "
                f"MERGE (a)-[:{RelType.USES_TYPE}]->(b)",
                {"rels": edge_params},
            )

        logger.debug("Resolved {} USES_TYPE edges", resolved)

    async def resolve_member_defines(
        self,
        project_name: str,
        member_rels: list[ParsedRelationship],
        *,
        lookup: _CallLookup | None = None,
        name_to_typedefs: dict[str, list[tuple[str, str]]] | None = None,
    ) -> None:
        """Resolve DEFINES edges from a parent TYPE NAME to a member Callable.

        Emitted by parsers for members whose parent type may live in another
        file (Go methods on package types, C++ out-of-line definitions).  Each
        rel carries the member uid in ``to_name``, the declaring module uid in
        ``from_qualified_name`` (fallback parent), and the parent type's bare
        name in ``properties["parent_type_name"]``.  ``parent_scope="package"``
        restricts matching to the member's directory (Go package rule).

        Ladder (first non-empty rung wins; >1 candidates in a rung => module
        fallback — never guess, no false edges):
        1. TypeDef with that name in the member's file.
        2. TypeDef with that name in the member's directory.
        3. Project-wide unique TypeDef (skipped when parent_scope="package").
        4. Fallback: DEFINES from the member's own Module.
        """
        if not member_rels:
            return
        if lookup is None:
            lookup = await self._build_call_lookup(project_name)
        if name_to_typedefs is None:
            td_records = await self.execute(
                f"MATCH (n:{NodeLabel.TYPE_DEF} {{project_name: $p}}) "
                "RETURN n.name AS name, n.uid AS uid, n.file_path AS fp",
                {"p": project_name},
            )
            name_to_typedefs = {}
            for r in td_records:
                name_to_typedefs.setdefault(r["name"], []).append((r["uid"], r["fp"] or ""))

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

        # Re-resolution is authoritative: drop previously-resolved parent edges
        # for these members first. The file-scoped rel delete preserves
        # cross-file DEFINES edges (see _recreate_file_relationships), so a
        # stale type edge would otherwise linger next to the new resolution.
        await self.execute_write(
            f"UNWIND $uids AS uid MATCH (a)-[r:{RelType.DEFINES}]->(b {{uid: uid}}) "
            f"WHERE a:{NodeLabel.TYPE_DEF} OR a:{NodeLabel.MODULE} DELETE r",
            {"uids": sorted({rel.to_name for rel in member_rels})},
        )

        if type_edges:
            await self.execute_write(
                f"UNWIND $rels AS r MATCH (a:{NodeLabel.TYPE_DEF} {{uid: r.f}}), (b {{uid: r.t}}) "
                f"MERGE (a)-[:{RelType.DEFINES}]->(b)",
                {"rels": [{"f": f, "t": t} for f, t in type_edges]},
            )
        if module_edges:
            await self.execute_write(
                f"UNWIND $rels AS r MATCH (a:{NodeLabel.MODULE} {{uid: r.f}}), (b {{uid: r.t}}) "
                f"MERGE (a)-[:{RelType.DEFINES}]->(b)",
                {"rels": [{"f": f, "t": t} for f, t in module_edges]},
            )
        logger.debug("Resolved {} member DEFINES edges ({} fell back to module)", len(type_edges), len(module_edges))

    async def _build_call_lookup(self, project_name: str) -> _CallLookup:
        """Build lookup tables needed for CALLS resolution."""
        # name → [(uid, file_path, visibility)]
        name_records = await self.execute(
            f"MATCH (n:{NodeLabel.CALLABLE} {{project_name: $p}}) "
            "RETURN n.name AS name, n.uid AS uid, n.file_path AS fp, n.visibility AS vis",
            {"p": project_name},
        )
        name_to_callables: dict[str, list[tuple[str, str, str]]] = {}
        uid_to_info: dict[str, tuple[str, str]] = {}
        for r in name_records:
            name_to_callables.setdefault(r["name"], []).append((r["uid"], r["fp"] or "", r["vis"] or "public"))
            uid_to_info[r["uid"]] = (r["name"], r["fp"] or "")

        # module/package uid → {imported_name: target_uid}
        import_map: dict[str, dict[str, str]] = {}
        for lbl in (NodeLabel.MODULE, NodeLabel.PACKAGE):
            import_records = await self.execute(
                f"MATCH (m:{lbl} {{project_name: $p}})-[:{RelType.IMPORTS}]->(t) "
                "RETURN m.uid AS mod_uid, t.name AS name, t.uid AS uid",
                {"p": project_name},
            )
            for r in import_records:
                import_map.setdefault(r["mod_uid"], {})[r["name"]] = r["uid"]

        # caller_uid → parent TypeDef uid, parent → children
        parent_records = await self.execute(
            f"MATCH (td:{NodeLabel.TYPE_DEF} {{project_name: $p}})-[:{RelType.DEFINES}]->(c:{NodeLabel.CALLABLE}) "
            "RETURN td.uid AS td_uid, td.name AS td_name, c.uid AS c_uid, c.is_stub AS c_stub",
            {"p": project_name},
        )
        caller_to_parent: dict[str, str] = {}
        parent_children: dict[str, list[str]] = {}
        stub_callables: set[str] = set()
        typedef_names: set[str] = set()
        for r in parent_records:
            if r["td_name"]:
                typedef_names.add(r["td_name"])
            caller_to_parent[r["c_uid"]] = r["td_uid"]
            parent_children.setdefault(r["td_uid"], []).append(r["c_uid"])
            if r["c_stub"]:
                stub_callables.add(r["c_uid"])

        return _CallLookup(
            name_to_callables=name_to_callables,
            import_map=import_map,
            caller_to_parent=caller_to_parent,
            parent_children=parent_children,
            uid_to_info=uid_to_info,
            stub_callables=frozenset(stub_callables),
            typedef_names=frozenset(typedef_names),
        )

    async def build_resolution_lookup(self, project_name: str) -> tuple[_CallLookup, dict[str, list[tuple[str, str]]]]:
        """Build shared lookup tables for both CALLS and USES_TYPE resolution.

        Returns ``(call_lookup, name_to_typedefs)`` where *name_to_typedefs*
        maps ``name → [(uid, file_path)]`` for TypeDef nodes.  Building both
        in a single call saves 3 redundant graph queries vs. calling
        ``_build_call_lookup`` twice.
        """
        lookup = await self._build_call_lookup(project_name)

        td_records = await self.execute(
            f"MATCH (n:{NodeLabel.TYPE_DEF} {{project_name: $p}}) "
            "RETURN n.name AS name, n.uid AS uid, n.file_path AS fp",
            {"p": project_name},
        )
        name_to_typedefs: dict[str, list[tuple[str, str]]] = {}
        for r in td_records:
            name_to_typedefs.setdefault(r["name"], []).append((r["uid"], r["fp"] or ""))

        return lookup, name_to_typedefs

    async def update_external_package_versions(
        self,
        project_name: str,
        versions: dict[str, str],
    ) -> None:
        """Set version properties on ExternalPackage nodes from dependency metadata."""
        if not versions:
            return
        params = [{"uid": f"{project_name}:ext/{pkg}", "version": ver} for pkg, ver in versions.items()]
        await self.execute_write(
            f"UNWIND $items AS item "
            f"MATCH (n:{NodeLabel.EXTERNAL_PACKAGE} {{uid: item.uid}}) "
            "SET n.version = item.version",
            {"items": params},
        )

    # -- Cross-project import resolution helpers --------------------------------

    async def _resolve_cross_project_read_phase(
        self,
        project_names: list[str],
        pkg_to_project: dict[str, str],
    ) -> tuple[list[dict[str, str]], dict[str, str], dict[str, str]]:
        """Batch-read stubs and resolve real entities for cross-project imports.

        Returns ``(matched_eps, sym_to_real, ep_to_real_pkg)`` where
        *matched_eps* are the ExternalPackage stubs that matched a sibling
        package, *sym_to_real* maps ExternalSymbol uid → real entity uid, and
        *ep_to_real_pkg* maps ExternalPackage uid → real Package uid.
        """
        # Fetch ALL ExternalPackage stubs across all projects in one query
        all_ext_pkgs = await self.execute(
            f"MATCH (ep:{NodeLabel.EXTERNAL_PACKAGE}) "
            "WHERE ep.project_name IN $projects "
            "RETURN ep.name AS name, ep.uid AS uid, ep.project_name AS proj",
            {"projects": project_names},
        )

        # Filter to those matching a sibling package (not self)
        matched_eps: list[dict[str, str]] = [
            {"name": ep["name"], "uid": ep["uid"], "proj": ep["proj"], "target": target}
            for ep in all_ext_pkgs
            if (target := pkg_to_project.get(ep["name"])) is not None and target != ep["proj"]
        ]
        if not matched_eps:
            return [], {}, {}

        # Build fast lookup: (proj, pkg_name) → target_project
        ep_target_map: dict[tuple[str, str], str] = {(ep["proj"], ep["name"]): ep["target"] for ep in matched_eps}

        # Fetch ALL ExternalSymbol stubs for matched packages in one query
        ep_keys = [{"proj": ep["proj"], "pkg": ep["name"]} for ep in matched_eps]
        all_ext_syms = await self.execute(
            f"UNWIND $keys AS k "
            f"MATCH (es:{NodeLabel.EXTERNAL_SYMBOL} {{project_name: k.proj, package: k.pkg}}) "
            "RETURN es.name AS name, es.uid AS uid, "
            "es.project_name AS proj, es.package AS pkg",
            {"keys": ep_keys},
        )

        # Build lookup pairs for bulk entity resolution
        lookup_pairs = [
            {"name": es["name"], "target_project": target, "es_uid": es["uid"]}
            for es in all_ext_syms
            if (target := ep_target_map.get((es["proj"], es["pkg"])))
        ]

        pkg_rewire = [
            {"ep_uid": ep["uid"], "pkg_name": ep["name"], "target_project": ep["target"]} for ep in matched_eps
        ]

        # Bulk-resolve real entities for ExternalSymbols.  No LIMIT — Cypher
        # LIMIT is global (one row for the whole UNWIND, not per pair); the
        # dict comprehension dedups multiple name matches per stub instead.
        sym_to_real: dict[str, str] = {}
        if lookup_pairs:
            real_matches = await self.execute(
                "UNWIND $pairs AS p "
                "MATCH (n {project_name: p.target_project, name: p.name}) "
                f"WHERE NOT n:{NodeLabel.EXTERNAL_PACKAGE} AND NOT n:{NodeLabel.EXTERNAL_SYMBOL} "
                f"AND NOT n:{NodeLabel.RESOURCE_FILE} AND NOT n:{NodeLabel.ENV_VAR} "
                f"AND NOT n:{NodeLabel.PROJECT} AND NOT n:{NodeLabel.SCHEMA_VERSION} "
                "RETURN p.es_uid AS es_uid, n.uid AS real_uid",
                {"pairs": lookup_pairs},
            )
            sym_to_real = {m["es_uid"]: m["real_uid"] for m in real_matches}

        # Bulk-resolve real Package nodes for bare package imports
        ep_to_real_pkg: dict[str, str] = {}
        if pkg_rewire:
            real_pkgs = await self.execute(
                "UNWIND $pairs AS p "
                f"MATCH (pkg:{NodeLabel.PACKAGE} {{project_name: p.target_project, name: p.pkg_name}}) "
                "RETURN p.ep_uid AS ep_uid, pkg.uid AS real_uid",
                {"pairs": pkg_rewire},
            )
            ep_to_real_pkg = {m["ep_uid"]: m["real_uid"] for m in real_pkgs}

        return matched_eps, sym_to_real, ep_to_real_pkg

    async def resolve_cross_project_imports(self, project_names: list[str]) -> int:
        """Rewire ExternalPackage/ExternalSymbol stubs that match real entities in sibling projects.

        For each project, finds ExternalPackage stubs whose name matches a Package
        in a sibling project, then rewires IMPORTS edges from ExternalSymbol stubs
        to the real entity. Orphaned stubs (no remaining inbound edges) are deleted.

        Returns the total number of imports rewired.
        """
        if len(project_names) < 2:
            return 0

        # Build map: package_name → project_name for all projects
        records = await self.execute(
            f"MATCH (pkg:{NodeLabel.PACKAGE}) "
            "WHERE pkg.project_name IN $projects "
            "RETURN pkg.name AS name, pkg.project_name AS project, pkg.qualified_name AS qn",
            {"projects": project_names},
        )
        pkg_to_project: dict[str, str] = {}
        for r in records:
            top_name = r["qn"].split(".")[0] if r["qn"] else r["name"]
            if top_name not in pkg_to_project:
                pkg_to_project[top_name] = r["project"]

        if not pkg_to_project:
            return 0

        # Batch-read stubs and resolve real entities
        matched_eps, sym_to_real, ep_to_real_pkg = await self._resolve_cross_project_read_phase(
            project_names, pkg_to_project
        )
        if not matched_eps:
            return 0

        # Writes — per-stub for correctness
        rewired = 0
        for es_uid, real_uid in sym_to_real.items():
            await self.execute_write(
                f"MATCH (src:{NodeLabel.MODULE})-[r:{RelType.IMPORTS}]->"
                f"(es:{NodeLabel.EXTERNAL_SYMBOL} {{uid: $es_uid}}) "
                f"MATCH (real {{uid: $real_uid}}) "
                f"CREATE (src)-[:{RelType.IMPORTS}]->(real) "
                "DELETE r",
                {"es_uid": es_uid, "real_uid": real_uid},
            )
            rewired += 1

        for ep_uid, real_uid in ep_to_real_pkg.items():
            await self.execute_write(
                f"MATCH (src:{NodeLabel.MODULE})-[r:{RelType.IMPORTS}]->"
                f"(ep:{NodeLabel.EXTERNAL_PACKAGE} {{uid: $ep_uid}}) "
                f"MATCH (real {{uid: $real_uid}}) "
                f"CREATE (src)-[:{RelType.IMPORTS}]->(real) "
                "DELETE r",
                {"ep_uid": ep_uid, "real_uid": real_uid},
            )

        # Delete orphaned stubs
        for ep in matched_eps:
            await self.execute_write(
                f"MATCH (es:{NodeLabel.EXTERNAL_SYMBOL} {{project_name: $proj, package: $pkg}}) "
                f"WHERE NOT ()-[:{RelType.IMPORTS}]->(es) "
                "DETACH DELETE es",
                {"proj": ep["proj"], "pkg": ep["name"]},
            )
            await self.execute_write(
                f"MATCH (ep:{NodeLabel.EXTERNAL_PACKAGE} {{uid: $uid}}) "
                f"WHERE NOT ()-[:{RelType.IMPORTS}]->(ep) AND NOT (ep)-[:{RelType.CONTAINS}]->() "
                "DETACH DELETE ep",
                {"uid": ep["uid"]},
            )

        logger.debug(
            "Cross-project import resolution: {} imports rewired across {} projects", rewired, len(project_names)
        )
        return rewired

    async def create_depends_on_edges(self, project_names: list[str]) -> int:
        """Create DEPENDS_ON edges between Project nodes based on cross-project IMPORTS.

        Queries all IMPORTS edges where source and target have different project_names,
        then creates DEPENDS_ON between the corresponding Project nodes.

        Returns the count of DEPENDS_ON edges created.
        """
        if len(project_names) < 2:
            return 0

        # Delete existing DEPENDS_ON edges between these projects
        await self.execute_write(
            f"MATCH (a:{NodeLabel.PROJECT})-[r:{RelType.DEPENDS_ON}]->(b:{NodeLabel.PROJECT}) "
            "WHERE a.name IN $projects AND b.name IN $projects "
            "DELETE r",
            {"projects": project_names},
        )

        # Find all cross-project import pairs
        records = await self.execute(
            f"MATCH (src)-[:{RelType.IMPORTS}]->(tgt) "
            "WHERE src.project_name IN $projects AND tgt.project_name IN $projects "
            "AND src.project_name <> tgt.project_name "
            "RETURN DISTINCT src.project_name AS from_proj, tgt.project_name AS to_proj",
            {"projects": project_names},
        )

        if not records:
            return 0

        # Create DEPENDS_ON edges
        edges = [{"from_proj": r["from_proj"], "to_proj": r["to_proj"]} for r in records]
        await self.execute_write(
            f"UNWIND $edges AS e "
            f"MATCH (a:{NodeLabel.PROJECT} {{name: e.from_proj}}), "
            f"(b:{NodeLabel.PROJECT} {{name: e.to_proj}}) "
            f"CREATE (a)-[:{RelType.DEPENDS_ON}]->(b)",
            {"edges": edges},
        )

        logger.debug("Created {} DEPENDS_ON edge(s) between projects", len(edges))
        return len(edges)

    # -- Detector enrichment helpers -------------------------------------------

    async def apply_property_enrichments(self, enrichments: list[PropertyEnrichment]) -> None:
        """Apply property enrichments from detectors to existing entity nodes.

        Batches all enrichments into a single UNWIND query.
        Uses ``+=`` (map merge) so existing properties are preserved.
        """
        items = [{"uid": e.qualified_name, "props": e.properties} for e in enrichments if e.properties]
        if items:
            await self.execute_write(
                "UNWIND $items AS item MATCH (n {uid: item.uid}) SET n += item.props",
                {"items": items},
            )

    # -- Detector lookups (parsing/languages/*.py) -----------------------------

    async def find_entity_uid(self, project_name: str, label: str, name: str) -> str | None:
        """Exact ``(project_name, name)`` -> uid lookup scoped to *label*.

        Used by parsing/languages detectors (test_mapping, di_injection) to
        cross-reference an entity by name during indexing — a plain point
        lookup, distinct from ``get_node_exact_matches`` (which also matches
        by uid and searches every label for the interactive ``get_node`` tool).
        """
        records = await self.execute(
            f"MATCH (n:{label} {{project_name: $p, name: $n}}) RETURN n.uid AS uid LIMIT 1",
            {"p": project_name, "n": name},
        )
        return records[0]["uid"] if records else None

    async def find_overridden_method(
        self, project_name: str, bases: list[str], method_name: str
    ) -> tuple[str, list[str]] | None:
        """First same-named method defined on any of *bases* — parsing/languages/
        python.py's ``ClassOverridesDetector`` (OVERRIDES/IMPLEMENTS detection).

        Returns ``(uid, tags)``; *tags* carries decorator info (e.g.
        ``decorator:abstractmethod``) the detector uses to distinguish
        IMPLEMENTS from OVERRIDES.
        """
        records = await self.execute(
            "MATCH (base:TypeDef {project_name: $p})-[:DEFINES]->(m:Callable) "
            "WHERE base.name IN $bases AND m.name = $method "
            "RETURN m.uid AS uid, m.tags AS tags LIMIT 1",
            {"p": project_name, "bases": bases, "method": method_name},
        )
        if not records:
            return None
        return records[0]["uid"], records[0].get("tags") or []

    # -- Embedding helpers -----------------------------------------------------

    async def get_embedding_config(self) -> tuple[str, int] | None:
        """Read embedding model and dimension from the SchemaVersion node.

        Returns ``(model, dimension)`` or ``None`` if not yet configured.
        Deterministic against duplicate SchemaVersion nodes (pre-fix damage):
        reads the canonical node — highest version, config-bearing preferred —
        never an arbitrary duplicate.
        """
        records = await self.execute(
            f"MATCH (sv:{NodeLabel.SCHEMA_VERSION}) "
            "RETURN sv.embedding_model AS model, sv.embedding_dimension AS dim "
            "ORDER BY coalesce(sv.version, -1) DESC, "
            "(CASE WHEN sv.embedding_model IS NULL THEN 0 ELSE 1 END) DESC LIMIT 1"
        )
        if not records or records[0]["model"] is None:
            return None
        return (records[0]["model"], records[0]["dim"])

    async def set_embedding_config(self, model: str, dimension: int) -> None:
        """Write embedding model and dimension to the SchemaVersion node."""
        await self.execute_write(
            f"MATCH (sv:{NodeLabel.SCHEMA_VERSION}) SET sv.embedding_model = $model, sv.embedding_dimension = $dim",
            {"model": model, "dim": dimension},
        )

    async def read_entity_texts(
        self,
        uids: list[str],
        *,
        labels: list[str] | None = None,
        chunk_size: int = 200,
    ) -> list[dict[str, Any]]:
        """Batch-read entity properties needed for embedding.

        ``uids`` must be full uid strings (``project:qualified_name``).
        If *labels* is provided (parallel to *uids*), entities are grouped
        by label so each query uses a label-constrained MATCH, hitting the
        per-label property index on ``uid`` instead of scanning all nodes.

        Queries are chunked to *chunk_size* uids per round-trip to avoid
        query timeouts on large batches.

        Returns list of dicts with keys: ``uid``, ``qualified_name``, ``name``,
        ``signature``, ``docstring``, ``source``, ``tags``, ``kind``, ``_label``,
        ``embed_hash``, ``has_embedding``.
        """
        ret = (
            "RETURN n.uid AS uid, n.qualified_name AS qualified_name, n.name AS name, "
            "n.signature AS signature, n.docstring AS docstring, "
            "n.source AS source, n.tags AS tags, "
            "n.kind AS kind, labels(n)[0] AS _label, "
            "n.embed_hash AS embed_hash, n.embedding IS NOT NULL AS has_embedding"
        )

        if labels is None or len(labels) != len(uids):
            # Fallback: label-free scan (slow on large graphs), chunked
            results: list[dict[str, Any]] = []
            for i in range(0, len(uids), chunk_size):
                chunk = uids[i : i + chunk_size]
                rows = await self.execute(
                    f"UNWIND $uids AS u MATCH (n) WHERE n.uid = u {ret}",
                    {"uids": chunk},
                )
                results.extend(rows)
            return results

        # Group uids by label → one indexed query per label, chunked
        by_label: dict[str, list[str]] = defaultdict(list)
        for uid, lbl in zip(uids, labels, strict=True):
            by_label[lbl].append(uid)

        async def _read_label(lbl: str, group_uids: list[str]) -> list[dict[str, Any]]:
            _assert_valid_label(lbl)
            rows: list[dict[str, Any]] = []
            for i in range(0, len(group_uids), chunk_size):
                chunk = group_uids[i : i + chunk_size]
                rows.extend(
                    await self.execute(
                        f"UNWIND $uids AS u MATCH (n:{lbl}) WHERE n.uid = u {ret}",
                        {"uids": chunk},
                    )
                )
            return rows

        label_results = await asyncio.gather(*[_read_label(lbl, guids) for lbl, guids in by_label.items()])
        return [row for rows in label_results for row in rows]

    async def read_embed_hashes(
        self,
        uids: list[str],
        *,
        labels: list[str] | None = None,
    ) -> dict[str, tuple[str | None, bool]]:
        """Batch-read embed_hash and embedding existence for entities.

        Returns ``{uid: (embed_hash, has_embedding)}`` for each matched node.
        Uses concurrent per-label reads when *labels* is provided.
        """
        ret = "RETURN n.uid AS uid, n.embed_hash AS embed_hash, n.embedding IS NOT NULL AS has_embedding"

        if labels is None or len(labels) != len(uids):
            rows = await self.execute(
                f"UNWIND $uids AS u MATCH (n) WHERE n.uid = u {ret}",
                {"uids": uids},
            )
            return {r["uid"]: (r["embed_hash"], r["has_embedding"]) for r in rows}

        by_label: dict[str, list[str]] = defaultdict(list)
        for uid, lbl in zip(uids, labels, strict=True):
            by_label[lbl].append(uid)

        async def _read_label(lbl: str, group_uids: list[str]) -> list[dict[str, Any]]:
            _assert_valid_label(lbl)
            return await self.execute(
                f"UNWIND $uids AS u MATCH (n:{lbl}) WHERE n.uid = u {ret}",
                {"uids": group_uids},
            )

        label_results = await asyncio.gather(*[_read_label(lbl, guids) for lbl, guids in by_label.items()])
        result: dict[str, tuple[str | None, bool]] = {}
        for rows in label_results:
            for r in rows:
                result[r["uid"]] = (r["embed_hash"], r["has_embedding"])
        return result

    async def find_unembedded_entities(self, project_name: str, *, limit: int = 5000) -> list[tuple[str, str, str]]:
        """``(uid, label, file_path)`` for entities that should carry a vector but do not.

        Only :data:`~code_atlas.schema._EMBEDDABLE_LABELS` are considered, because that is
        exactly the set the vector indices serve. ``DocFile``/``Package`` do get embedded by
        the AST stage but have no vector index, so re-embedding them would cost API calls
        for a vector nothing can search.
        """
        labels = "|".join(sorted(lbl.value for lbl in _EMBEDDABLE_LABELS))
        rows = await self.execute(
            f"MATCH (n:{labels}) WHERE n.project_name = $project AND n.embedding IS NULL AND n.uid IS NOT NULL "
            "RETURN n.uid AS uid, labels(n)[0] AS label, n.file_path AS file_path LIMIT $limit",
            {"project": project_name, "limit": limit},
        )
        return [(r["uid"], r["label"], r["file_path"] or "") for r in rows]

    async def write_embeddings(
        self,
        items: list[tuple[str, list[float]]],
        chunk_size: int = 50,
        *,
        labels: list[str] | None = None,
    ) -> None:
        """Batch-write embedding vectors to nodes by uid.

        If *labels* is provided (parallel to *items*), writes are grouped by
        label for index-backed matching.
        """
        if not items:
            return

        if labels is not None and len(labels) == len(items):
            by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for (uid, vec), lbl in zip(items, labels, strict=True):
                by_label[lbl].append({"uid": uid, "vector": vec})
            for lbl, group in by_label.items():
                _assert_valid_label(lbl)
                for i in range(0, len(group), chunk_size):
                    chunk = group[i : i + chunk_size]
                    await self.execute_write(
                        f"UNWIND $items AS item MATCH (n:{lbl}) WHERE n.uid = item.uid SET n.embedding = item.vector",
                        {"items": chunk},
                    )
        else:
            for i in range(0, len(items), chunk_size):
                chunk_items = items[i : i + chunk_size]
                params = [{"uid": uid, "vector": vec} for uid, vec in chunk_items]
                await self.execute_write(
                    "UNWIND $items AS item MATCH (n) WHERE n.uid = item.uid SET n.embedding = item.vector",
                    {"items": params},
                )

    async def write_embed_hashes(self, items: list[tuple[str, str]], *, labels: list[str] | None = None) -> None:
        """Batch-write embed_hash values to nodes by uid.

        If *labels* is provided (parallel to *items*), writes are grouped by
        label for index-backed matching.
        """
        if not items:
            return

        if labels is not None and len(labels) == len(items):
            by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for (uid, h), lbl in zip(items, labels, strict=True):
                by_label[lbl].append({"uid": uid, "hash": h})
            for lbl, group in by_label.items():
                _assert_valid_label(lbl)
                await self.execute_write(
                    f"UNWIND $items AS item MATCH (n:{lbl}) WHERE n.uid = item.uid SET n.embed_hash = item.hash",
                    {"items": group},
                )
        else:
            params = [{"uid": uid, "hash": h} for uid, h in items]
            await self.execute_write(
                "UNWIND $items AS item MATCH (n) WHERE n.uid = item.uid SET n.embed_hash = item.hash",
                {"items": params},
            )

    async def write_embeddings_and_hashes(
        self,
        items: list[tuple[str, list[float], str]],
        *,
        labels: list[str] | None = None,
    ) -> None:
        """Batch-write embedding vectors **and** embed_hashes in a single UNWIND.

        Each *item* is ``(uid, vector, embed_hash)``.  When *labels* is
        provided (parallel to *items*), writes are grouped by label so the
        ``MATCH`` can use label-scoped uid indices.
        """
        if not items:
            return

        if labels is not None:
            by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for (uid, vec, h), lbl in zip(items, labels, strict=True):
                by_label[lbl].append({"uid": uid, "vector": vec, "hash": h})
            for lbl, group in by_label.items():
                _assert_valid_label(lbl)
                await self.execute_write(
                    f"UNWIND $items AS item "
                    f"MATCH (n:{lbl}) WHERE n.uid = item.uid "
                    "SET n.embedding = item.vector, n.embed_hash = item.hash",
                    {"items": group},
                )
        else:
            params = [{"uid": uid, "vector": vec, "hash": h} for uid, vec, h in items]
            await self.execute_write(
                "UNWIND $items AS item "
                "MATCH (n) WHERE n.uid = item.uid "
                "SET n.embedding = item.vector, n.embed_hash = item.hash",
                {"items": params},
            )

    async def run_in_write_transaction(self, fn: Callable[[], Awaitable[_T]]) -> _T:
        """Run *fn* inside a managed write transaction (single session, auto-retry)."""

        async def _tx(tx: Any) -> _T:
            token = _active_tx_var.set(tx)
            try:
                return await fn()
            finally:
                _active_tx_var.reset(token)

        async with self._driver.session() as session:
            return await session.execute_write(_tx)

    async def clear_all_embeddings(self) -> None:
        """Remove embedding vectors and content hashes from all nodes."""
        await self.execute_write(
            "MATCH (n) WHERE n.embedding IS NOT NULL OR n.embed_hash IS NOT NULL REMOVE n.embedding, n.embed_hash"
        )

    async def rebuild_vector_indices(self, dimension: int) -> None:
        """Drop and recreate vector indices at the specified dimension."""
        for stmt in generate_drop_vector_index_ddl():
            await self._exec_ddl(stmt)
        if self._embeddings_enabled:
            for stmt in generate_vector_index_ddl(dimension):
                await self._exec_ddl(stmt)
        self._dimension = dimension

    async def get_vector_index_info(self) -> list[dict[str, Any]]:
        """Query Memgraph for vector index metadata.

        Returns a list of dicts with keys like ``index_name``, ``label``,
        ``property``, ``dimension``, ``size``, etc.
        """
        try:
            return await self.execute("CALL vector_search.show_index_info() YIELD * RETURN *")
        except Exception as exc:
            logger.debug("Could not fetch vector index info: {}", exc)
            return []

    # -- Text (BM25) search helpers -------------------------------------------

    async def text_search(
        self,
        query: str,
        label: str = "",
        limit: int = 20,
        project: str = "",
        projects: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 text search across text indices.

        Queries one or all text indices, optionally post-filters by project(s),
        and fuses per-index result lists by reciprocal rank — raw BM25 scores
        are not comparable across indices with different corpus statistics.
        """
        with _tracer.start_as_current_span("graph.text_search", attributes={"query": query[:100], "limit": limit}):
            # Backward compat: single project → projects list
            filter_projects = projects if projects is not None else ([project] if project else None)

            indices = (
                [f"text_{label.lower()}"] if label else [f"text_{lbl.value.lower()}" for lbl in _TEXT_SEARCHABLE_LABELS]
            )
            fetch_limit = limit * 3 if filter_projects else limit
            safe_query = _sanitize_bm25_query(query)

            async def _ts_one(idx: str) -> list[dict[str, Any]]:
                # Memgraph 3.11 changed the third parameter from a bare integer limit to a
                # config MAP; passing the old form is a hard ClientError, not a warning.
                # `text_search.search` is still broken (Tantivy "Unable to create search
                # query"), so search_all remains the only working entry point.
                cypher = (
                    f"CALL text_search.search_all('{idx}', $query, {{limit: {fetch_limit}}}) "
                    "YIELD node, score "
                    "RETURN node, score "
                    f"ORDER BY score DESC LIMIT {fetch_limit}"
                )
                try:
                    return await self.execute(cypher, {"query": safe_query})
                except Exception as exc:
                    logger.warning("Text search on {} failed: {}", idx, exc)
                    return []

            results_per_index = await asyncio.gather(*(_ts_one(idx) for idx in indices))
            all_results = _fuse_bm25_results(results_per_index)

            # Post-filter by project scope. GLOBAL_PROJECT always passes: a
            # shared node (EnvVar) belongs to every project, so scoping a
            # search to one repo must not hide the env vars that repo reads.
            if filter_projects:
                project_set = {*filter_projects, GLOBAL_PROJECT}
                all_results = [r for r in all_results if _node_project_name(r) in project_set]

            return all_results[:limit]

    async def get_text_index_info(self) -> list[dict[str, Any]]:
        """Query Memgraph for text index metadata via SHOW INDEX INFO (Memgraph 3.7+ DDL).

        Filters the generic index listing to text indices (type starts with 'label_text').
        Returns a list of dicts with index_type, label, and name keys.
        """
        try:
            rows = await self.execute("SHOW INDEX INFO")
            return [
                {
                    "index_type": r["index type"],
                    "label": r["label"],
                    "name": r["index type"].split("name: ")[-1].rstrip(")") if "name:" in r["index type"] else "",
                }
                for r in rows
                if str(r.get("index type", "")).startswith("label_text")
            ]
        except Exception as exc:
            logger.debug("Could not fetch text index info: {}", exc)
            return []

    # -- Vector search helpers -------------------------------------------------

    async def vector_search(
        self,
        vector: list[float],
        label: str = "",
        limit: int = 20,
        project: str = "",
        threshold: float = 0.0,
        projects: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic similarity search using pre-computed vector.

        Queries one or all vector indices, optionally post-filters by project(s)
        and similarity threshold, and returns results sorted by similarity
        descending.  Returns ``[{"node": Node, "similarity": float}, ...]``.
        """
        with _tracer.start_as_current_span("graph.vector_search", attributes={"limit": limit}):
            filter_projects = projects if projects is not None else ([project] if project else None)

            indices = [f"vec_{label.lower()}"] if label else [f"vec_{lbl.value.lower()}" for lbl in _EMBEDDABLE_LABELS]
            filtering = bool(filter_projects) or threshold > 0.0
            fetch_limit = limit * 3 if filtering else limit

            async def _vs_one(idx: str) -> list[dict[str, Any]]:
                # Memgraph 3.12's vector index is not purged synchronously on delete, so a
                # search can hand back nodes that are already gone. Touching one is fatal to
                # the whole query — `node.uid` raises "Trying to get a property from a
                # deleted object", and `node:Label` raises the same for labels — which after
                # a full re-index would take out semantic search entirely rather than drop a
                # row. Re-matching on id() is the guard: id() is the one thing still legal to
                # read off a dead node, and the MATCH yields nothing when it no longer exists.
                cypher = (
                    f"CALL vector_search.search('{idx}', {fetch_limit}, $vector) "
                    "YIELD node, similarity "
                    "WITH node, similarity "
                    "MATCH (live) WHERE id(live) = id(node) "
                    "RETURN live AS node, similarity "
                    f"ORDER BY similarity DESC LIMIT {fetch_limit}"
                )
                try:
                    return await self.execute(cypher, {"vector": vector})
                except Exception as exc:
                    logger.warning("Vector search on {} failed: {}", idx, exc)
                    return []

            results_per_index = await asyncio.gather(*(_vs_one(idx) for idx in indices))
            all_results: list[dict[str, Any]] = [r for batch in results_per_index for r in batch]

            if threshold > 0.0:
                all_results = [r for r in all_results if r.get("similarity", 0) >= threshold]
            if filter_projects:
                project_set = set(filter_projects)
                all_results = [r for r in all_results if _node_project_name(r) in project_set]

            all_results.sort(key=lambda rec: rec.get("similarity", 0), reverse=True)
            return all_results[:limit]

    # -- Graph (name-based) search helpers ------------------------------------

    async def graph_search(
        self,
        query: str,
        label: str = "",
        limit: int = 20,
        project: str = "",
        projects: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Name-based graph search with scored matching.

        Three-stage matching with decreasing scores:
        - Exact name match: score 3.0
        - Suffix match (qualified_name ends with .query): score 2.0
        - Contains match (name or qualified_name contains query): score 1.0

        Deduplicates by uid, keeping highest score.
        Returns ``[{"node": Node, "score": float}, ...]``.
        """
        with _tracer.start_as_current_span("graph.graph_search", attributes={"query": query[:100], "limit": limit}):
            return await self._graph_search_inner(query, label=label, limit=limit, project=project, projects=projects)

    async def _graph_search_inner(
        self,
        query: str,
        label: str = "",
        limit: int = 20,
        project: str = "",
        projects: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Inner implementation of graph_search.

        Uses a single UNION ALL query (one round-trip) instead of 3
        sequential execute() calls.
        """
        filter_projects = projects if projects is not None else ([project] if project else None)

        # GLOBAL_PROJECT rides along in the scope list — see text_search.
        project_clause = " AND n.project_name IN $projects" if filter_projects else ""
        params: dict[str, Any] = {
            "query": query,
            "suffix": f".{query}",
            "projects": [*filter_projects, GLOBAL_PROJECT] if filter_projects else [],
        }
        fetch_limit = limit * 3

        query_str = _build_graph_search_query(label, project_clause, fetch_limit)
        records = await self.execute(query_str, params)

        # Deduplicate by uid, keeping highest score
        scored: dict[str, tuple[Any, float]] = {}
        for r in records:
            node = r["node"]
            score: float = r["score"]
            uid = node.get("uid", "") if hasattr(node, "get") else ""
            if uid and (uid not in scored or scored[uid][1] < score):
                scored[uid] = (node, score)

        # Build result list sorted by score descending
        results = [{"node": node, "score": score} for node, score in scored.values()]
        results.sort(key=lambda rec: rec["score"], reverse=True)
        return results[:limit]

    async def batch_call_stats(self, uids: list[str], *, top_n: int = 5) -> dict[str, CallStats]:
        """Return caller/callee counts and top-N names per uid in a single round-trip."""
        with _tracer.start_as_current_span("graph.batch_call_stats", attributes={"count": len(uids)}):
            if not uids:
                return {}
            records = await self.execute(
                "UNWIND $uids AS uid "
                "MATCH (n {uid: uid}) "
                f"OPTIONAL MATCH (caller)-[:{RelType.CALLS}]->(n) "
                "WITH uid, n, count(DISTINCT caller) AS cc, collect(DISTINCT caller.name)[0..$top_n] AS cn "
                f"OPTIONAL MATCH (n)-[:{RelType.CALLS}]->(callee) "
                "RETURN uid, cc, cn, count(DISTINCT callee) AS ec, collect(DISTINCT callee.name)[0..$top_n] AS en",
                {"uids": uids, "top_n": top_n},
            )
            return {
                r["uid"]: CallStats(
                    caller_count=r["cc"],
                    callee_count=r["ec"],
                    caller_names=r["cn"] or [],
                    callee_names=r["en"] or [],
                )
                for r in records
            }

    # -- Analysis / diagram queries (server/analysis.py) ----------------------
    #
    # Query construction for analyze_repo's sub-analyses, generate_diagram's
    # diagram types, and trace_path/blast_radius — moved here from
    # server/analysis.py so callers never build Cypher directly (see
    # GraphBackend in graph/protocol.py). Each method returns plain dicts;
    # the Python-side shaping/aggregation stays in analysis.py, unchanged.

    async def node_exists(self, uid: str) -> bool:
        """Single-node existence check, used by ``server.analysis.blast_radius``."""
        exist_raw = await self.execute("OPTIONAL MATCH (n {uid: $uid}) RETURN n IS NOT NULL AS exists", {"uid": uid})
        return bool(exist_raw and exist_raw[0]["exists"])

    async def trace_path_between(
        self, from_uid: str, to_uid: str, max_depth: int, edge_types: tuple[str, ...]
    ) -> dict[str, Any]:
        """Existence check + shortest-path traversal for ``server.analysis.trace_path``.

        Returns ``from_exists``/``to_exists`` booleans plus (when both exist
        and a path is found) the shortest path's ``hop_count``, its
        ``path_weight``, and formatted ``hops`` (endpoint uid/name, edge type,
        and CALLS confidence/strategy/weight/from_test when present — ADR-0014
        and its weighting amendment).

        Shortest-path-first semantics are unchanged; among paths of *equal* hop
        count the one with the highest product of edge weights wins, so an
        all-resolved production path beats an equally short path through
        ambiguous or test-provenance edges. Edges with no ``weight`` property
        (IMPORTS, USES_TYPE, CALLS written before the amendment) count as
        ``_DEFAULT_EDGE_WEIGHT``, i.e. they neither help nor hurt.
        """
        params: dict[str, Any] = {"from_uid": from_uid, "to_uid": to_uid}
        exist_raw = await self.execute(
            "OPTIONAL MATCH (a {uid: $from_uid}) OPTIONAL MATCH (b {uid: $to_uid}) "
            "RETURN a IS NOT NULL AS from_exists, b IS NOT NULL AS to_exists",
            params,
        )
        exists = exist_raw[0] if exist_raw else {"from_exists": False, "to_exists": False}
        if not exists["from_exists"] or not exists["to_exists"]:
            return {
                "from_exists": exists["from_exists"],
                "to_exists": exists["to_exists"],
                "found": False,
                "hop_count": None,
                "hops": [],
                "path_weight": None,
            }

        rel_pattern = "|".join(edge_types)
        records = await self.execute(
            f"MATCH p=(a {{uid: $from_uid}})-[:{rel_pattern}*1..{max_depth}]->(b {{uid: $to_uid}}) "
            "RETURN nodes(p) AS path_nodes, relationships(p) AS path_rels, length(p) AS hops, "
            f"reduce(w = 1.0, r IN relationships(p) | w * coalesce(r.weight, {_DEFAULT_EDGE_WEIGHT})) AS path_weight "
            "ORDER BY hops, path_weight DESC LIMIT 1",
            params,
        )
        if not records:
            return {
                "from_exists": True,
                "to_exists": True,
                "found": False,
                "hop_count": None,
                "hops": [],
                "path_weight": None,
            }

        record = records[0]
        return {
            "from_exists": True,
            "to_exists": True,
            "found": True,
            "hop_count": record["hops"],
            "hops": _format_path_hops(record["path_nodes"], record["path_rels"]),
            "path_weight": record["path_weight"],
        }

    async def compute_blast_radius(
        self, uid: str, direction_kind: str, edge_types: tuple[str, ...], max_depth: int
    ) -> list[dict[str, Any]]:
        """One directional (``"out"``/``"in"``) traversal for ``server.analysis.blast_radius``.

        Returns affected-entity dicts with:

        - ``min_depth`` — shortest hop count to *uid*.
        - ``ambiguous_only`` — True unless some path made entirely of
          ``confidence: "resolved"`` edges reaches the entity (ADR-0014).
        - ``confidence_score`` — the *best* path's product of edge ``weight``
          properties. A product (rather than the minimum hop weight) because
          each hop's weight reads as an independent "is this edge real"
          factor, so uncertainty compounds along a chain the way it does in
          practice: two ambiguous hops are a weaker claim than one. Missing
          weights count as ``_DEFAULT_EDGE_WEIGHT``, so an all-resolved
          production chain scores 1.0 at any depth and ranking stays driven by
          evidence quality rather than distance (``min_depth`` already carries
          distance). Note the best-scoring path need not be the shortest one.
        - ``test_only`` — True when *no* path free of ``from_test`` edges
          reaches the entity, i.e. only test code gets there. Computed by the
          same second-traversal pattern as ``ambiguous_only`` rather than from
          the entity's own file path, so it reflects how the entity is reached
          rather than where it happens to live.
        """
        rel_pattern = "|".join(edge_types)
        pattern = (
            f"-[:{rel_pattern}*1..{max_depth}]->" if direction_kind == "out" else f"<-[:{rel_pattern}*1..{max_depth}]-"
        )
        all_raw = await self.execute(
            f"MATCH p=(start {{uid: $uid}}){pattern}(affected) "
            "WHERE affected.uid <> $uid "
            "RETURN affected.uid AS uid, affected.name AS name, affected.qualified_name AS qn, "
            "labels(affected)[0] AS label, affected.file_path AS file_path, "
            "min(length(p)) AS min_depth, "
            f"max(reduce(w = 1.0, r IN relationships(p) | w * coalesce(r.weight, {_DEFAULT_EDGE_WEIGHT}))) "
            "AS confidence_score",
            {"uid": uid},
        )
        resolved_raw = await self.execute(
            f"MATCH p=(start {{uid: $uid}}){pattern}(affected) "
            "WHERE affected.uid <> $uid AND all(r IN relationships(p) WHERE r.confidence = 'resolved') "
            "RETURN DISTINCT affected.uid AS uid",
            {"uid": uid},
        )
        # Same shape as resolved_raw (all(...) rather than none(...) so both
        # passes use the one predicate form already proven against Memgraph).
        production_raw = await self.execute(
            f"MATCH p=(start {{uid: $uid}}){pattern}(affected) "
            "WHERE affected.uid <> $uid AND all(r IN relationships(p) WHERE NOT coalesce(r.from_test, false)) "
            "RETURN DISTINCT affected.uid AS uid",
            {"uid": uid},
        )
        resolved_uids = {r["uid"] for r in resolved_raw}
        production_uids = {r["uid"] for r in production_raw}
        return [
            {
                "uid": r["uid"],
                "name": r["name"],
                "qualified_name": r["qn"],
                "label": r["label"],
                "file_path": r["file_path"],
                "min_depth": r["min_depth"],
                "direction": direction_kind,
                "ambiguous_only": r["uid"] not in resolved_uids,
                "confidence_score": r["confidence_score"],
                "test_only": r["uid"] not in production_uids,
            }
            for r in all_raw
        ]

    async def get_structure_overview(self, project: str, path: str, limit: int) -> dict[str, list[dict[str, Any]]]:
        """Entity counts, package breakdown, largest modules, external deps —
        ``analyze_repo(analysis="structure")``.
        """
        params: dict[str, Any] = {"project": project, "path": path}
        pa = " AND n.file_path STARTS WITH $path" if path else ""
        counts_raw = await self.execute(
            f"MATCH (n {{project_name: $project}}) "
            f"WHERE NOT n:Project AND NOT n:SchemaVersion{pa} "
            "RETURN labels(n)[0] AS label, n.kind AS kind, count(n) AS cnt "
            "ORDER BY cnt DESC",
            params,
        )
        pa_m = " WHERE m.file_path STARTS WITH $path" if path else ""
        pkg_raw = await self.execute(
            "MATCH (pkg:Package {project_name: $project})-[:CONTAINS]->(m:Module)"
            f"{pa_m} "
            "RETURN pkg.name AS package, pkg.qualified_name AS qn, count(m) AS modules "
            f"ORDER BY modules DESC LIMIT {limit}",
            params,
        )
        lm_w = " WHERE m.file_path STARTS WITH $path" if path else ""
        largest_raw = await self.execute(
            "MATCH (m:Module {project_name: $project})-[:DEFINES]->(e)"
            f"{lm_w} "
            "RETURN m.name AS module, m.qualified_name AS qn, m.file_path AS file_path, "
            f"count(e) AS entities ORDER BY entities DESC LIMIT {limit}",
            params,
        )
        ext_w = " WHERE src IS NULL OR src.file_path STARTS WITH $path" if path else ""
        ext_raw = await self.execute(
            "MATCH (ep:ExternalPackage {project_name: $project}) "
            "OPTIONAL MATCH (ep)<-[:IMPORTS]-(src) "
            f"{ext_w} "
            "RETURN ep.name AS package, ep.version AS version, count(src) AS imported_by "
            f"ORDER BY imported_by DESC LIMIT {limit}",
            params,
        )
        return {"counts": counts_raw, "packages": pkg_raw, "largest_modules": largest_raw, "external_deps": ext_raw}

    async def get_centrality_data(self, project: str, path: str, limit: int) -> dict[str, list[dict[str, Any]]]:
        """Hub entities, hub modules, and leaf entities — ``analyze_repo(analysis="centrality")``."""
        params: dict[str, Any] = {"project": project, "path": path}
        pa = " AND n.file_path STARTS WITH $path" if path else ""
        hubs_raw = await self.execute(
            "MATCH (n {project_name: $project})<-[r:IMPORTS|INHERITS|CALLS]-(src) "
            f"WHERE NOT n:ExternalPackage AND NOT n:ExternalSymbol{pa} "
            "RETURN n.name AS name, n.qualified_name AS qn, labels(n)[0] AS label, "
            "n.kind AS kind, n.file_path AS file_path, "
            "count(r) AS in_degree, "
            "sum(CASE WHEN type(r) = 'IMPORTS' THEN 1 ELSE 0 END) AS imported_by, "
            "sum(CASE WHEN type(r) = 'INHERITS' THEN 1 ELSE 0 END) AS inherited_by, "
            "sum(CASE WHEN type(r) = 'CALLS' THEN 1 ELSE 0 END) AS called_by "
            f"ORDER BY in_degree DESC LIMIT {limit}",
            params,
        )
        pa_m = " AND m.file_path STARTS WITH $path" if path else ""
        hub_modules_raw = await self.execute(
            "MATCH (m:Module {project_name: $project})<-[:IMPORTS]-(src) "
            f"WHERE true{pa_m} "
            "RETURN m.name AS name, m.qualified_name AS qn, m.file_path AS file_path, "
            f"count(src) AS imported_by ORDER BY imported_by DESC LIMIT {limit}",
            params,
        )
        pa_leaf = " AND n.file_path STARTS WITH $path" if path else ""
        leaf_raw = await self.execute(
            "MATCH (n {project_name: $project}) "
            "WHERE NOT n:Project AND NOT n:SchemaVersion AND NOT n:Package "
            f"AND NOT n:ExternalPackage AND NOT n:ExternalSymbol{pa_leaf} "
            # EnvVar/ResourceFile can never receive IMPORTS/INHERITS/CALLS, so
            # a label-only filter would report every one of them as a leaf.
            "AND NOT n:EnvVar AND NOT n:ResourceFile "
            "AND NOT ()-[:IMPORTS|INHERITS|CALLS]->(n) "
            "RETURN n.name AS name, n.qualified_name AS qn, labels(n)[0] AS label, "
            f"n.kind AS kind, n.file_path AS file_path LIMIT {limit}",
            params,
        )
        return {"hubs": hubs_raw, "hub_modules": hub_modules_raw, "leaves": leaf_raw}

    async def get_module_import_edges(self, project: str, path: str) -> dict[str, list[dict[str, Any]]]:
        """Direct module-to-module and entity-level import edges.

        Shared by ``analyze_repo(analysis="dependencies")`` and
        ``generate_diagram(diagram_type="imports")`` — identical query, only
        the downstream aggregation/rendering differs.
        """
        params: dict[str, Any] = {"project": project, "path": path}
        pa_m1 = " AND m1.file_path STARTS WITH $path" if path else ""
        direct_raw = await self.execute(
            "MATCH (m1:Module {project_name: $project})-[:IMPORTS]->"
            "(m2:Module {project_name: $project}) "
            f"WHERE m1 <> m2{pa_m1} "
            "RETURN m1.qualified_name AS from_mod, m2.qualified_name AS to_mod, "
            "m1.file_path AS from_path, m2.file_path AS to_path",
            params,
        )
        indirect_raw = await self.execute(
            "MATCH (m1:Module {project_name: $project})-[:IMPORTS]->(e)"
            "<-[:DEFINES]-(m2:Module {project_name: $project}) "
            f"WHERE m1 <> m2 AND NOT e:Module{pa_m1} "
            "RETURN m1.qualified_name AS from_mod, m2.qualified_name AS to_mod, "
            "m1.file_path AS from_path, m2.file_path AS to_path",
            params,
        )
        return {"direct": direct_raw, "indirect": indirect_raw}

    async def get_dependency_external_counts(self, project: str, path: str) -> dict[str, list[dict[str, Any]]]:
        """External package/symbol import counts — ``analyze_repo(analysis="dependencies")``."""
        params: dict[str, Any] = {"project": project, "path": path}
        pa_src = " AND src.file_path STARTS WITH $path" if path else ""
        ext_pkg_raw = await self.execute(
            "MATCH (src {project_name: $project})-[:IMPORTS]->(ep:ExternalPackage) "
            f"WHERE true{pa_src} "
            "RETURN ep.name AS package, count(src) AS cnt",
            params,
        )
        ext_sym_raw = await self.execute(
            "MATCH (src {project_name: $project})-[:IMPORTS]->(es:ExternalSymbol) "
            f"WHERE true{pa_src} "
            "RETURN es.package AS package, count(src) AS cnt",
            params,
        )
        return {"ext_packages": ext_pkg_raw, "ext_symbols": ext_sym_raw}

    async def get_quality_data(self, project: str, path: str) -> dict[str, list[dict[str, Any]]]:
        """Per-module entity counts and fan-in-inclusive import edges —
        ``analyze_repo(analysis="quality")``.
        """
        params: dict[str, Any] = {"project": project, "path": path}
        pa_m = " AND m.file_path STARTS WITH $path" if path else ""
        # Match on either side so a scoped module's fan-in from out-of-scope
        # importers (and fan-out to out-of-scope targets) are both captured —
        # see server.analysis._analyze_quality's docstring-level comment.
        pa_edge = " AND (m1.file_path STARTS WITH $path OR m2.file_path STARTS WITH $path)" if path else ""
        entity_raw = await self.execute(
            "MATCH (m:Module {project_name: $project})-[:DEFINES]->(e) "
            f"WHERE NOT e:Module{pa_m} "
            "RETURN m.qualified_name AS module, m.file_path AS file_path, count(e) AS entity_count "
            "ORDER BY entity_count DESC",
            params,
        )
        direct_raw = await self.execute(
            "MATCH (m1:Module {project_name: $project})-[:IMPORTS]->"
            "(m2:Module {project_name: $project}) "
            f"WHERE m1 <> m2{pa_edge} "
            "RETURN m1.qualified_name AS from_mod, m2.qualified_name AS to_mod",
            params,
        )
        indirect_raw = await self.execute(
            "MATCH (m1:Module {project_name: $project})-[:IMPORTS]->(e)"
            "<-[:DEFINES]-(m2:Module {project_name: $project}) "
            f"WHERE m1 <> m2 AND NOT e:Module{pa_edge} "
            "RETURN m1.qualified_name AS from_mod, m2.qualified_name AS to_mod",
            params,
        )
        return {"entities": entity_raw, "direct": direct_raw, "indirect": indirect_raw}

    async def get_patterns_data(self, project: str, path: str, limit: int) -> dict[str, list[dict[str, Any]]]:
        """Inheritance, enums, visibility distribution, docstring coverage, and
        detected patterns — ``analyze_repo(analysis="patterns")``.
        """
        params: dict[str, Any] = {"project": project, "path": path}
        pa = " AND child.file_path STARTS WITH $path" if path else ""
        inherit_raw = await self.execute(
            "MATCH (child:TypeDef {project_name: $project})-[:INHERITS]->(parent) "
            f"WHERE true{pa} "
            "RETURN child.name AS child, child.qualified_name AS child_qn, "
            f"parent.name AS parent, parent.qualified_name AS parent_qn LIMIT {limit}",
            params,
        )
        pa_n = " AND n.file_path STARTS WITH $path" if path else ""
        enum_raw = await self.execute(
            "MATCH (n:TypeDef {project_name: $project, kind: 'enum'})"
            f" WHERE true{pa_n} "
            "OPTIONAL MATCH (n)-[:DEFINES]->(m:Value) "
            "RETURN n.name AS name, n.qualified_name AS qn, n.file_path AS file_path, "
            f"count(m) AS members ORDER BY name LIMIT {limit}",
            params,
        )
        vis_raw = await self.execute(
            "MATCH (n {project_name: $project}) "
            f"WHERE n.visibility IS NOT NULL{pa_n} "
            "RETURN n.visibility AS visibility, count(n) AS cnt "
            "ORDER BY cnt DESC",
            params,
        )
        doc_raw = await self.execute(
            "MATCH (n {project_name: $project}) "
            f"WHERE (n:Callable OR n:TypeDef OR n:Value){pa_n} "
            "WITH count(n) AS total, "
            "sum(CASE WHEN n.docstring IS NOT NULL AND n.docstring <> '' THEN 1 ELSE 0 END) AS documented "
            "RETURN total, documented",
            params,
        )
        pattern_raw = await self.execute(
            "MATCH (n {project_name: $project})-[r:HANDLES_COMMAND|HANDLES_ROUTE|HANDLES_EVENT]->(target) "
            f"WHERE true{pa_n} "
            "RETURN type(r) AS pattern_type, n.name AS name, n.qualified_name AS qn, "
            f"target.name AS target_name ORDER BY pattern_type, name LIMIT {limit}",
            params,
        )
        return {
            "inheritance": inherit_raw,
            "enums": enum_raw,
            "visibility": vis_raw,
            "docstring": doc_raw,
            "detected_patterns": pattern_raw,
        }

    async def get_dead_code_candidates(self, project: str, path: str) -> list[dict[str, Any]]:
        """Invocable Callables/TypeDefs with zero incoming CALLS edges — ``analyze_repo(analysis="dead_code")``.

        Restricted to ``_CODE_ENTITY_KINDS`` so that config/infra declarations
        (Terraform resources, k8s objects, SQL tables, Dockerfile stages, CI
        jobs, ...) do not swamp the result: they carry the same Callable/TypeDef
        labels as real code but can never receive a CALLS edge, so a label-only
        filter reports every one of them as dead.
        """
        params: dict[str, Any] = {"project": project, "path": path, "code_kinds": sorted(_CODE_ENTITY_KINDS)}
        pa = " AND n.file_path STARTS WITH $path" if path else ""
        # "Unused" cannot mean "no CALLS edge". A class is used by being annotated,
        # subclassed or imported, and instantiating it calls its __init__ rather than the
        # class — so a CALLS-only test called 29 of 30 live entities in one package dead,
        # while the graph itself held the disproof (AppContext: 34 incoming edges,
        # including USES_TYPE). Acting on that output deletes working code, which makes a
        # false positive here far more expensive than a miss.
        refs = (
            f"{RelType.CALLS}|{RelType.USES_TYPE}|{RelType.IMPORTS}"
            f"|{RelType.INHERITS}|{RelType.IMPLEMENTS}|{RelType.OVERRIDES}"
            # Handed to a registry or a callback slot counts as used, even though the
            # call that eventually runs it is a framework's, not this codebase's.
            f"|{RelType.REFERENCES}"
        )
        return await self.execute(
            "MATCH (n {project_name: $project}) "
            f"WHERE (n:Callable OR n:TypeDef) AND n.kind IN $code_kinds "
            f"AND NOT n.name STARTS WITH '__'{pa} "
            f"AND NOT ()-[:{refs}]->(n) "
            # Constructing a class produces an edge to its __init__, not to the class.
            f"AND NOT (n)-[:{RelType.DEFINES}]->()<-[:{RelType.CALLS}]-() "
            # A function defined INSIDE another function is reached through its enclosing
            # scope, not by name: a decorator registers it, a closure returns it, a
            # callback receives it. Nothing calls it directly and nothing ever will, so a
            # by-name test reports every one of them dead. Indexing nested functions added
            # 39 such false positives here in one commit — 20 of them the @mcp.tool()
            # handlers that ARE this server's public surface. Its liveness is the
            # enclosing function's liveness, which this predicate already judges separately.
            f"AND NOT (:{NodeLabel.CALLABLE})-[:{RelType.DEFINES}]->(n) "
            # A registered handler is reached by whatever owns the registry. The edge runs
            # FROM the handler ("registered by"), so liveness is an outbound test here.
            f"AND NOT (n)-[:{RelType.REGISTERED_BY}]->() "
            "RETURN n.name AS name, n.qualified_name AS qn, labels(n)[0] AS label, "
            "n.kind AS kind, n.file_path AS file_path, n.line_start AS line_start "
            "ORDER BY n.file_path, n.line_start",
            params,
        )

    async def get_complexity_hotspots(self, project: str, path: str, limit: int) -> list[dict[str, Any]]:
        """Top-N Callables by LOC span — ``analyze_repo(analysis="complexity")``."""
        params: dict[str, Any] = {"project": project, "path": path}
        pa = " AND n.file_path STARTS WITH $path" if path else ""
        return await self.execute(
            "MATCH (n:Callable {project_name: $project}) "
            f"WHERE n.line_start IS NOT NULL AND n.line_end IS NOT NULL{pa} "
            "RETURN n.name AS name, n.qualified_name AS qn, n.kind AS kind, n.file_path AS file_path, "
            "n.line_start AS line_start, n.line_end AS line_end, (n.line_end - n.line_start) AS loc_span "
            f"ORDER BY loc_span DESC LIMIT {limit}",
            params,
        )

    async def get_git_signals_data(
        self, project: str, path: str, limit: int, bus_factor_threshold: int
    ) -> dict[str, list[dict[str, Any]]]:
        """Commit-count hotspots, bus-factor risks, and co-change pairs —
        ``analyze_repo(analysis="git_signals")``.
        """
        params: dict[str, Any] = {"project": project, "path": path, "max_authors": bus_factor_threshold}
        pa = " AND n.file_path STARTS WITH $path" if path else ""
        hotspots_raw = await self.execute(
            "MATCH (n {project_name: $project}) "
            f"WHERE n.git_commit_count IS NOT NULL{pa} "
            "RETURN n.name AS name, n.qualified_name AS qn, n.file_path AS file_path, "
            "n.git_commit_count AS commit_count, n.git_author_count AS author_count, "
            "n.git_days_since_last_commit AS days_since_last_commit "
            f"ORDER BY commit_count DESC LIMIT {limit}",
            params,
        )
        bus_factor_raw = await self.execute(
            "MATCH (n {project_name: $project}) "
            f"WHERE n.git_commit_count IS NOT NULL AND n.git_author_count <= $max_authors{pa} "
            "RETURN n.name AS name, n.qualified_name AS qn, n.file_path AS file_path, "
            "n.git_commit_count AS commit_count, n.git_author_count AS author_count "
            f"ORDER BY commit_count DESC LIMIT {limit}",
            params,
        )
        pa_edge = " AND (a.file_path STARTS WITH $path OR b.file_path STARTS WITH $path)" if path else ""
        co_change_raw = await self.execute(
            f"MATCH (a {{project_name: $project}})-[r:{RelType.CO_CHANGES_WITH}]->(b {{project_name: $project}}) "
            f"WHERE true{pa_edge} "
            "RETURN a.qualified_name AS a_qn, a.file_path AS a_path, "
            "b.qualified_name AS b_qn, b.file_path AS b_path, r.count AS count "
            f"ORDER BY count DESC LIMIT {limit}",
            params,
        )
        return {"hotspots": hotspots_raw, "bus_factor": bus_factor_raw, "co_change": co_change_raw}

    async def get_diagram_packages(self, project: str, path: str, max_nodes: int) -> list[dict[str, Any]]:
        """Package→child (Package|Module) CONTAINS edges — ``generate_diagram(diagram_type="packages")``."""
        params: dict[str, Any] = {"project": project, "path": path, "limit": max_nodes}
        pa = " AND child.file_path STARTS WITH $path" if path else ""
        return await self.execute(
            "MATCH (pkg:Package {project_name: $project})-[:CONTAINS]->(child) "
            f"WHERE (child:Package OR child:Module){pa} "
            "RETURN pkg.qualified_name AS parent_qn, pkg.name AS parent_name, "
            "labels(child)[0] AS child_label, child.qualified_name AS child_qn, child.name AS child_name "
            "ORDER BY parent_qn, child_qn LIMIT $limit",
            params,
        )

    async def get_diagram_inheritance(self, project: str, path: str, max_nodes: int) -> list[dict[str, Any]]:
        """Child→parent TypeDef INHERITS edges — ``generate_diagram(diagram_type="inheritance")``."""
        params: dict[str, Any] = {"project": project, "path": path, "limit": max_nodes}
        pa = " AND child.file_path STARTS WITH $path" if path else ""
        return await self.execute(
            "MATCH (child:TypeDef {project_name: $project})-[:INHERITS]->(parent) "
            f"WHERE true{pa} "
            "RETURN child.name AS child_name, child.qualified_name AS child_qn, "
            "child.kind AS child_kind, "
            "parent.name AS parent_name, parent.qualified_name AS parent_qn "
            "ORDER BY parent_qn, child_qn LIMIT $limit",
            params,
        )

    async def get_diagram_module_detail(self, project: str, path: str, max_nodes: int) -> dict[str, Any] | None:
        """Module lookup + its top-level entities/methods/inheritance —
        ``generate_diagram(diagram_type="module_detail")``. Returns ``None``
        when no module matches *path*.
        """
        params: dict[str, Any] = {"project": project, "path": path}
        modules = await self.execute(
            "MATCH (m:Module {project_name: $project}) "
            "WHERE m.file_path STARTS WITH $path "
            "RETURN m.name AS name, m.qualified_name AS qn, m.uid AS uid "
            "ORDER BY m.qualified_name LIMIT 1",
            params,
        )
        if not modules:
            return None
        mod = modules[0]
        entities = await self.execute(
            "MATCH (m {uid: $uid})-[:DEFINES]->(e) "
            "RETURN e.name AS name, e.qualified_name AS qn, labels(e)[0] AS label, "
            f"e.kind AS kind, e.visibility AS vis, e.signature AS sig ORDER BY e.line_start LIMIT {max_nodes}",
            {"uid": mod["uid"]},
        )
        methods = await self.execute(
            "MATCH (m {uid: $uid})-[:DEFINES]->(td:TypeDef)-[:DEFINES]->(method:Callable) "
            "RETURN td.qualified_name AS class_qn, td.name AS class_name, "
            "method.name AS name, method.visibility AS vis, method.kind AS kind "
            f"ORDER BY td.name, method.line_start LIMIT {max_nodes}",
            {"uid": mod["uid"]},
        )
        inherits = await self.execute(
            "MATCH (m {uid: $uid})-[:DEFINES]->(td:TypeDef)-[:INHERITS]->(parent) "
            "RETURN td.qualified_name AS child_qn, td.name AS child_name, "
            "parent.qualified_name AS parent_qn, parent.name AS parent_name "
            f"LIMIT {max_nodes}",
            {"uid": mod["uid"]},
        )
        return {"module": mod, "entities": entities, "methods": methods, "inherits": inherits}

    async def get_module_summary(
        self, project: str, path: str, limit: int, edge_limit: int
    ) -> dict[str, list[dict[str, Any]]]:
        """Whole-scope skeleton + boundary — ``analyze_repo(analysis="module_summary")``.

        *path* is a file or directory prefix; every Module/TypeDef/Callable/Value
        whose ``file_path`` starts with it is "in scope". Six record lists come
        back: ``modules`` and ``entities`` (the skeleton — signature, docstring,
        visibility, line span, DEFINES parent), ``internal_edges`` (both endpoints
        in scope), ``fan_in``/``fan_out`` (exactly one endpoint in scope — the
        boundary an agent needs to judge whether a change is safe), and ``docs``
        (inbound DOCUMENTS links).

        Edge rows return ``properties(r)`` wholesale rather than named
        confidence/strategy columns so any relationship property that exists
        (ADR-0014's ``confidence``/``strategy`` plus later additions) reaches
        the formatter without another query change.
        """
        params: dict[str, Any] = {"project": project, "path": path, "limit": limit, "edge_limit": edge_limit}
        modules = await self.execute(
            f"MATCH (m:{NodeLabel.MODULE} {{project_name: $project}}) WHERE m.file_path STARTS WITH $path "
            "RETURN m.qualified_name AS qn, m.name AS name, m.file_path AS file_path, "
            "m.docstring AS docstring ORDER BY m.file_path LIMIT $limit",
            params,
        )
        entities = await self.execute(
            "MATCH (e {project_name: $project}) "
            f"WHERE (e:{NodeLabel.TYPE_DEF} OR e:{NodeLabel.CALLABLE} OR e:{NodeLabel.VALUE}) "
            "AND e.file_path STARTS WITH $path "
            f"OPTIONAL MATCH (p)-[:{RelType.DEFINES}]->(e) "
            "RETURN e.uid AS uid, e.name AS name, e.qualified_name AS qn, labels(e)[0] AS label, "
            "e.kind AS kind, e.visibility AS vis, e.signature AS sig, e.docstring AS docstring, "
            "e.line_start AS line_start, e.line_end AS line_end, e.file_path AS file_path, "
            "p.qualified_name AS parent_qn ORDER BY e.file_path, e.line_start LIMIT $limit",
            params,
        )
        structural = f"{RelType.CALLS}|{RelType.INHERITS}|{RelType.IMPLEMENTS}|{RelType.USES_TYPE}|{RelType.OVERRIDES}"
        boundary = f"{structural}|{RelType.IMPORTS}"
        internal_edges = await self.execute(
            f"MATCH (a {{project_name: $project}})-[r:{structural}]->(b {{project_name: $project}}) "
            "WHERE a.file_path STARTS WITH $path AND b.file_path STARTS WITH $path AND a.uid <> b.uid "
            "RETURN a.qualified_name AS from_qn, b.qualified_name AS to_qn, type(r) AS rel_type, "
            "properties(r) AS props ORDER BY rel_type, from_qn, to_qn LIMIT $edge_limit",
            params,
        )
        fan_in = await self.execute(
            f"MATCH (a {{project_name: $project}})-[r:{boundary}]->(b {{project_name: $project}}) "
            "WHERE b.file_path STARTS WITH $path "
            "AND (a.file_path IS NULL OR NOT a.file_path STARTS WITH $path) "
            "RETURN a.qualified_name AS from_qn, a.name AS from_name, a.file_path AS from_path, "
            "labels(a)[0] AS from_label, b.qualified_name AS to_qn, type(r) AS rel_type, "
            "properties(r) AS props ORDER BY rel_type, to_qn, from_qn LIMIT $edge_limit",
            params,
        )
        fan_out = await self.execute(
            f"MATCH (a {{project_name: $project}})-[r:{boundary}]->(b {{project_name: $project}}) "
            "WHERE a.file_path STARTS WITH $path "
            "AND (b.file_path IS NULL OR NOT b.file_path STARTS WITH $path) "
            "RETURN a.qualified_name AS from_qn, b.qualified_name AS to_qn, b.name AS to_name, "
            "b.file_path AS to_path, labels(b)[0] AS to_label, type(r) AS rel_type, "
            "properties(r) AS props ORDER BY rel_type, from_qn, to_qn LIMIT $edge_limit",
            params,
        )
        docs = await self.execute(
            f"MATCH (d)-[r:{RelType.DOCUMENTS}]->(e {{project_name: $project}}) "
            "WHERE e.file_path STARTS WITH $path "
            "RETURN d.qualified_name AS doc_qn, d.name AS doc_name, labels(d)[0] AS doc_label, "
            "e.qualified_name AS to_qn, r.link_type AS link_type ORDER BY to_qn, doc_qn LIMIT $limit",
            params,
        )
        return {
            "modules": modules,
            "entities": entities,
            "internal_edges": internal_edges,
            "fan_in": fan_in,
            "fan_out": fan_out,
            "docs": docs,
        }

    # -- Context expansion / navigation (search/engine.py's expand_context) ---

    async def get_entity_by_uid(self, uid: str, label: str = "") -> dict[str, Any] | None:
        """Single node fetch by uid (optionally label-scoped) — ``expand_context``'s target lookup."""
        label_clause = f":{label}" if label else ""
        records = await self.execute(f"MATCH (n{label_clause} {{uid: $uid}}) RETURN n", {"uid": uid})
        return records[0].get("n") if records else None

    async def get_defining_parent(self, uid: str) -> dict[str, Any] | None:
        """The entity that DEFINES *uid* (its enclosing class/module) — ``expand_context``'s parent."""
        records = await self.execute(
            f"MATCH (p)-[:{RelType.DEFINES}]->(n {{uid: $uid}}) RETURN p AS n LIMIT 1", {"uid": uid}
        )
        return records[0].get("n") if records else None

    async def get_sibling_entities(self, uid: str, limit: int) -> list[dict[str, Any]]:
        """Other entities DEFINEd by *uid*'s same parent — ``expand_context``'s siblings."""
        records = await self.execute(
            f"MATCH (p)-[:{RelType.DEFINES}]->(n {{uid: $uid}}), (p)-[:{RelType.DEFINES}]->(s) "
            f"WHERE s.uid <> $uid RETURN s AS n LIMIT {limit}",
            {"uid": uid},
        )
        return [r["n"] for r in records]

    async def get_package_docstring(self, uid: str) -> str | None:
        """Docstring of the nearest enclosing Module (1-3 DEFINES hops) — ``expand_context``'s package context."""
        records = await self.execute(
            f"MATCH (pkg:{NodeLabel.MODULE})-[:{RelType.DEFINES}*1..3]->(target {{uid: $uid}}) "
            "RETURN pkg.docstring AS docstring LIMIT 1",
            {"uid": uid},
        )
        return records[0].get("docstring") if records else None

    async def get_callers(self, uid: str, label: str, call_depth: int, limit: int) -> list[dict[str, Any]]:
        """Callables reaching *uid* via 1..call_depth CALLS hops — ``expand_context``'s callers."""
        label_clause = f":{label}" if label else ""
        records = await self.execute(
            f"MATCH (caller:Callable)-[:{RelType.CALLS}*1..{call_depth}]->"
            f"(n{label_clause} {{uid: $uid}}) "
            f"RETURN DISTINCT caller AS n LIMIT {limit}",
            {"uid": uid},
        )
        return [r["n"] for r in records]

    async def get_callees(self, uid: str, label: str, call_depth: int, limit: int) -> list[dict[str, Any]]:
        """Callables reached from *uid* via 1..call_depth CALLS hops — ``expand_context``'s callees."""
        label_clause = f":{label}" if label else ""
        records = await self.execute(
            f"MATCH (n{label_clause} {{uid: $uid}})-[:{RelType.CALLS}*1..{call_depth}]->"
            f"(callee:Callable) RETURN DISTINCT callee AS n LIMIT {limit}",
            {"uid": uid},
        )
        return [r["n"] for r in records]

    async def get_linked_docs(self, uid: str, label: str, limit: int) -> list[dict[str, Any]]:
        """DocFile/DocSection/Note entities documenting *uid* — ``expand_context``'s docs.

        Each item is ``{"node": ..., "link_type": ..., "stale": ..., "anchor_hash": ...}``;
        ``stale``/``anchor_hash`` are only populated for explicit ``anchors:`` links (§3.6).

        DocFile is in the doc-side filter for citations: ``resolve_citations``
        resolves ``see ADR-14`` to the whole document, so the cited node is a
        DocFile far more often than a section. No other DOCUMENTS route can
        originate from one, so admitting the label adds nothing else.
        """
        label_clause = f":{label}" if label else ""
        records = await self.execute(
            f"MATCH (doc)-[r:{RelType.DOCUMENTS}]->(n{label_clause} {{uid: $uid}}) "
            f"WHERE doc:{NodeLabel.DOC_SECTION} OR doc:{NodeLabel.NOTE} OR doc:{NodeLabel.DOC_FILE} "
            "RETURN doc AS n, r.link_type AS link_type, r.stale AS stale, r.anchor_hash AS anchor_hash "
            f"LIMIT {limit}",
            {"uid": uid},
        )
        return [
            {
                "node": r["n"],
                "link_type": r.get("link_type"),
                "stale": r.get("stale"),
                "anchor_hash": r.get("anchor_hash"),
            }
            for r in records
        ]

    # -- get_node cascade / status queries (server/mcp.py, cli.py) ------------

    async def get_node_exact_matches(self, name: str, label: str, limit: int) -> list[dict[str, Any]]:
        """Exact match cascade (uid + exact name) — ``get_node`` stage A."""
        label_filter = f":{label}" if label else ""
        return await self.execute(
            f"MATCH (n{label_filter} {{uid: $name}}) RETURN n LIMIT {limit} "
            f"UNION ALL "
            f"MATCH (n{label_filter}) WHERE n.name = $name RETURN n LIMIT {limit}",
            {"name": name},
        )

    async def get_node_partial_matches(self, name: str, label: str, limit: int) -> list[dict[str, Any]]:
        """Partial match cascade (suffix > prefix > contains) — ``get_node`` stage B."""
        label_filter = f":{label}" if label else ""
        return await self.execute(
            f"MATCH (n{label_filter}) WHERE n.qualified_name ENDS WITH $suffix "
            f"RETURN n, 3 AS _match_score LIMIT {limit} "
            f"UNION ALL "
            f"MATCH (n{label_filter}) WHERE n.qualified_name STARTS WITH $prefix "
            f"RETURN n, 2 AS _match_score LIMIT {limit} "
            f"UNION ALL "
            f"MATCH (n{label_filter}) WHERE n.qualified_name CONTAINS $name OR n.name CONTAINS $name "
            f"RETURN n, 1 AS _match_score LIMIT {limit}",
            {"name": name, "suffix": f".{name}", "prefix": f"{name}."},
        )

    async def get_label_counts(self) -> dict[str, int]:
        """Per-label node counts across the whole graph — ``index_status``."""
        records = await self.execute("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY count DESC")
        return {r["label"]: r["count"] for r in records}

    async def get_project_dependency_edges(self) -> list[dict[str, Any]]:
        """Project-to-project DEPENDS_ON edges — ``atlas status`` and ``list_projects``."""
        return await self.execute(
            f"MATCH (a:{NodeLabel.PROJECT})-[:{RelType.DEPENDS_ON}]->(b:{NodeLabel.PROJECT}) "
            "RETURN a.name AS from_proj, b.name AS to_proj"
        )

    # -- Dream-mode lint queries (dream.py) ------------------------------------

    async def get_existing_uids(self, uids: list[str]) -> set[str]:
        """Which of *uids* exist in the graph — dream-mode dangling-link check."""
        if not uids:
            return set()
        records = await self.execute("UNWIND $uids AS uid MATCH (n {uid: uid}) RETURN uid", {"uids": uids})
        return {r["uid"] for r in records}

    async def get_orphan_notes(self) -> list[dict[str, Any]]:
        """Notes with no LINKS_TO edges in or out — dream-mode orphan check."""
        return await self.execute(
            f"MATCH (n:{NodeLabel.NOTE}) WHERE NOT (n)-[:{RelType.LINKS_TO}]-() "
            "RETURN n.uid AS uid, n.name AS name, n.project_name AS project_name, n.file_path AS file_path"
        )

    async def get_broken_anchor_notes(self) -> list[dict[str, Any]]:
        """Notes with broken/unresolved explicit ``anchors:`` — dream-mode lint check."""
        return await self.execute(
            f"MATCH (n:{NodeLabel.NOTE}) "
            "WHERE n.has_broken_anchors = true "
            "OR (n.unresolved_anchors IS NOT NULL AND size(n.unresolved_anchors) > 0) "
            "RETURN n.uid AS uid, n.name AS name, n.project_name AS project_name, n.file_path AS file_path, "
            "n.unresolved_anchors AS unresolved_anchors"
        )

    async def get_inbox_note_paths(self) -> list[str]:
        """Draft/inbox-path Note file paths, sorted — dream-mode inbox digest."""
        records = await self.execute(
            f"MATCH (n:{NodeLabel.NOTE}) WHERE n.kind = $draft OR n.file_path CONTAINS '/inbox/' "
            "RETURN n.file_path AS file_path ORDER BY file_path",
            {"draft": NoteKind.DRAFT.value},
        )
        return [r["file_path"] for r in records]

    async def get_note_embeddings(self) -> list[dict[str, Any]]:
        """uid/project_name/embedding for every Note with a stored vector — dream-mode similarity scan."""
        return await self.execute(
            f"MATCH (n:{NodeLabel.NOTE}) WHERE n.embedding IS NOT NULL "
            "RETURN n.uid AS uid, n.project_name AS project_name, n.embedding AS embedding"
        )

    async def write_git_file_signals(self, project_name: str, label: str, items: list[dict[str, Any]]) -> int:
        """Write commit-count/author-count/days-since-last-commit onto Module/DocFile nodes.

        Matched per-label (inline on the node pattern), not via a post-MATCH
        ``WHERE n:Module OR n:DocFile`` filter: Memgraph 3.7.2's planner
        mishandles ``UNWIND ... MATCH (n {prop: unwind_var}) WHERE ...`` once
        an earlier UNWIND row fails to match anything. Matching the label
        inline avoids the bug. ``atlas mine-git-history``.
        """
        if not items:
            return 0
        await self.execute_write(
            "UNWIND $items AS item "
            f"MATCH (n:{label} {{project_name: $p, file_path: item.fp}}) "
            "SET n.git_commit_count = item.cc, n.git_author_count = item.ac, "
            "n.git_days_since_last_commit = item.days",
            {"p": project_name, "items": items},
        )
        matched_rows = await self.execute(
            "UNWIND $items AS item "
            f"MATCH (n:{label} {{project_name: $p, file_path: item.fp}}) "
            "RETURN count(n) AS matched",
            {"p": project_name, "items": items},
        )
        return matched_rows[0]["matched"] if matched_rows else 0

    async def write_co_change_edges(self, project_name: str, pairs: list[dict[str, Any]]) -> int:
        """Create/update CO_CHANGES_WITH edges between co-changed Module files.

        ``file_a < file_b`` always (the caller sorts before pairing) — a
        single directed edge per pair is enough; readers treat it as
        symmetric. ``atlas mine-git-history``.
        """
        if not pairs:
            return 0
        await self.execute_write(
            "UNWIND $pairs AS pair "
            f"MATCH (a:{NodeLabel.MODULE} {{project_name: $p, file_path: pair.a}}) "
            f"MATCH (b:{NodeLabel.MODULE} {{project_name: $p, file_path: pair.b}}) "
            f"MERGE (a)-[r:{RelType.CO_CHANGES_WITH}]->(b) SET r.count = pair.cnt",
            {"p": project_name, "pairs": pairs},
        )
        edge_rows = await self.execute(
            "UNWIND $pairs AS pair "
            f"MATCH (a:{NodeLabel.MODULE} {{project_name: $p, file_path: pair.a}})"
            f"-[r:{RelType.CO_CHANGES_WITH}]->(b:{NodeLabel.MODULE} {{project_name: $p, file_path: pair.b}}) "
            "RETURN count(r) AS created",
            {"p": project_name, "pairs": pairs},
        )
        return edge_rows[0]["created"] if edge_rows else 0

    async def close(self) -> None:
        """Close the driver and release connections."""
        await self._driver.close()

    # -- Private helpers -----------------------------------------------------

    async def _batch_create_entities(self, project_name: str, entities: list[ParsedEntity]) -> None:
        """Batch-create entity nodes grouped by label."""
        sorted_entities = sorted(entities, key=attrgetter("label"))
        for label, group in groupby(sorted_entities, key=attrgetter("label")):
            entity_list = list(group)
            params = [
                {
                    "uid": e.qualified_name,
                    "project_name": project_name,
                    "name": e.name,
                    "qualified_name": (
                        e.qualified_name.split(":", 1)[1] if ":" in e.qualified_name else e.qualified_name
                    ),
                    "file_path": e.file_path,
                    "kind": e.kind,
                    "line_start": e.line_start,
                    "line_end": e.line_end,
                    "visibility": e.visibility,
                    "docstring": e.docstring,
                    "signature": e.signature,
                    "source": e.source,
                    "tags": e.tags,
                    "header_path": e.header_path,
                    "header_level": e.header_level,
                    "content_hash": e.content_hash,
                    "rationale": e.rationale,
                    # Empty -> null so the property is absent rather than an empty
                    # list on every node in the graph.
                    "citations": e.citations or None,
                    "extra_properties": e.extra_properties,
                }
                for e in entity_list
            ]
            query = (
                f"UNWIND $entities AS e "
                f"MERGE (n:{label.value} {{uid: e.uid}}) "
                f"ON CREATE SET "
                f"n.project_name = e.project_name, n.name = e.name, "
                f"n.qualified_name = e.qualified_name, n.file_path = e.file_path, "
                f"n.kind = e.kind, n.line_start = e.line_start, n.line_end = e.line_end, "
                f"n.visibility = e.visibility, n.docstring = e.docstring, "
                f"n.signature = e.signature, n.source = e.source, n.tags = e.tags, "
                f"n.header_path = e.header_path, n.header_level = e.header_level, "
                f"n.rationale = e.rationale, n.citations = e.citations, "
                f"n.content_hash = e.content_hash "
                f"ON MATCH SET "
                f"n.project_name = e.project_name, n.name = e.name, "
                f"n.qualified_name = e.qualified_name, n.file_path = e.file_path, "
                f"n.kind = e.kind, n.line_start = e.line_start, n.line_end = e.line_end, "
                f"n.visibility = e.visibility, n.docstring = e.docstring, "
                f"n.signature = e.signature, n.source = e.source, n.tags = e.tags, "
                f"n.header_path = e.header_path, n.header_level = e.header_level, "
                f"n.rationale = e.rationale, n.citations = e.citations, "
                f"n.content_hash = e.content_hash "
                f"SET n += e.extra_properties"
            )
            await self.execute_write(query, {"entities": params})

    async def _batch_update_entities(self, entities: list[ParsedEntity]) -> None:
        """Batch-update modified entity nodes by uid, grouped by label."""
        sorted_entities = sorted(entities, key=attrgetter("label"))
        for label, group in groupby(sorted_entities, key=attrgetter("label")):
            params = [
                {
                    "uid": e.qualified_name,
                    "name": e.name,
                    "kind": e.kind,
                    "line_start": e.line_start,
                    "line_end": e.line_end,
                    "visibility": e.visibility,
                    "docstring": e.docstring,
                    "signature": e.signature,
                    "source": e.source,
                    "tags": e.tags,
                    "header_path": e.header_path,
                    "header_level": e.header_level,
                    "content_hash": e.content_hash,
                    "rationale": e.rationale,
                    "citations": e.citations or None,
                    "extra_properties": e.extra_properties,
                }
                for e in group
            ]
            await self.execute_write(
                f"UNWIND $entities AS e "
                f"MATCH (n:{label.value} {{uid: e.uid}}) "
                "SET n.name = e.name, n.kind = e.kind, "
                "n.line_start = e.line_start, n.line_end = e.line_end, "
                "n.visibility = e.visibility, n.docstring = e.docstring, "
                "n.signature = e.signature, n.source = e.source, n.tags = e.tags, "
                "n.header_path = e.header_path, n.header_level = e.header_level, "
                "n.rationale = e.rationale, n.citations = e.citations, "
                "n.content_hash = e.content_hash, "
                "n += e.extra_properties",
                {"entities": params},
            )

    async def _batch_update_positions(self, entities: list[ParsedEntity]) -> None:
        """Update only line_start/line_end for entities whose content didn't change."""
        sorted_entities = sorted(entities, key=attrgetter("label"))
        for label, group in groupby(sorted_entities, key=attrgetter("label")):
            params = [{"uid": e.qualified_name, "ls": e.line_start, "le": e.line_end} for e in group]
            await self.execute_write(
                f"UNWIND $entities AS e MATCH (n:{label.value} {{uid: e.uid}}) "
                "SET n.line_start = e.ls, n.line_end = e.le",
                {"entities": params},
            )

    async def _batch_delete_entities(self, uids_by_label: dict[str, list[str]]) -> None:
        """Delete entity nodes by uid, grouped by label for index-backed matching.

        A node that still has inbound edges from a DIFFERENT file is preserved
        (only its own outgoing edges are stripped) instead of being fully
        DETACH DELETEd — those cross-file edges (e.g. CALLS/INHERITS/DOCUMENTS)
        come from files that aren't being re-parsed right now, so nothing could
        recreate them if the entity reappears later (e.g. a brief comment-out
        edit). Nodes with no such foreign inbound edges are fully removed,
        matching prior behavior.

        Cross-file DEFINES edges are excluded from this check: they are
        (re-)created by resolve_member_defines keyed off the MEMBER's own
        file (Go receiver methods, C++ out-of-line methods — see S5), so they
        are always re-resolved whenever the member's file is reprocessed,
        including on reappearance. Treating them as "foreign" here would make
        a genuinely deleted cross-file member an undeletable zombie.

        Anchor- and citation-type DOCUMENTS edges are also excluded: unlike
        CALLS/INHERITS/heuristic-DOCUMENTS edges (which preserve the zombie
        because nothing could recreate them), an explicit anchors: reference is
        meant to go stale/broken and be surfaced to the user (§3.6), not keep
        an otherwise-dead entity alive forever. Deletion proceeds normally, and
        the affected Note's ``has_broken_anchors`` is set in the same
        statement as the DETACH DELETE below. Citations are excluded for the
        same reason plus a stronger one: they *are* recreatable — the citing
        file's next parse re-runs ``resolve_citations`` — so an inbound
        citation from some ADR must not keep a genuinely deleted function
        alive as a zombie.
        """
        # Pass 1: read which uids have a foreign inbound edge, across all
        # labels, before any writes — so an earlier label's delete in this
        # same call can't affect a later label's read. Referrers that are
        # themselves being deleted in this same call don't count: whether they
        # end up DETACH DELETEd or edge-stripped, their edge to `n` is gone by
        # the time this call completes either way — so if that's `n`'s only
        # foreign referrer, `n` must be fully removed too, not preserved as an
        # edge-stripped zombie node still visible to direct uid/search lookups.
        all_uids = [uid for uids in uids_by_label.values() for uid in uids]
        preserve_by_label: dict[str, list[str]] = {}
        remove_by_label: dict[str, list[str]] = {}
        for label, uids in uids_by_label.items():
            if not uids:
                continue
            if label:
                _assert_valid_label(label)
            match_n = f"MATCH (n:{label} {{uid: uid}})" if label else "MATCH (n {uid: uid})"
            referenced = await self.execute(
                f"UNWIND $uids AS uid {match_n} "
                "MATCH (other)-[r]->(n) WHERE (other.file_path IS NULL OR other.file_path <> n.file_path) "
                f"AND NOT type(r) = '{RelType.DEFINES}' "
                f"AND NOT (type(r) = '{RelType.DOCUMENTS}' AND r.link_type IN ['anchor', 'citation']) "
                "AND NOT other.uid IN $all_uids "
                "RETURN DISTINCT uid",
                {"uids": uids, "all_uids": all_uids},
            )
            referenced_uids = {r["uid"] for r in referenced}
            preserve_by_label[label] = [u for u in uids if u in referenced_uids]
            remove_by_label[label] = [u for u in uids if u not in referenced_uids]

        # Pass 2: apply deletes/strips.
        for label, uids in remove_by_label.items():
            if not uids:
                continue
            match_n = f"MATCH (n:{label} {{uid: uid}})" if label else "MATCH (n {uid: uid})"
            # Deletion marking is folded into the same statement as the DETACH
            # DELETE: a node about to be removed may be an explicit anchor
            # target, so any Note anchoring into it gets has_broken_anchors
            # set before the node (and its inbound DOCUMENTS edge) disappears.
            await self.execute_write(
                f"UNWIND $uids AS uid {match_n} "
                f"OPTIONAL MATCH (note:{NodeLabel.NOTE})-[:{RelType.DOCUMENTS} {{link_type: 'anchor'}}]->(n) "
                "FOREACH (_ IN CASE WHEN note IS NOT NULL THEN [1] ELSE [] END | SET note.has_broken_anchors = true) "
                "WITH DISTINCT n "
                "DETACH DELETE n",
                {"uids": uids},
            )
        for label, uids in preserve_by_label.items():
            if not uids:
                continue
            match_n = f"MATCH (n:{label} {{uid: uid}})" if label else "MATCH (n {uid: uid})"
            await self.execute_write(
                f"UNWIND $uids AS uid {match_n} OPTIONAL MATCH (n)-[out]->() DELETE out",
                {"uids": uids},
            )

    async def _recreate_batch_relationships(
        self,
        project_name: str,
        file_rels: dict[str, list[ParsedRelationship]],
        new_file_paths: set[str],
    ) -> None:
        """Delete then recreate relationships for multiple files in batched queries.

        *new_file_paths* are files with no prior data — their old rels are skipped
        in the delete phase.  All rels across files are pooled for creation.
        """
        # --- Delete phase: single label-free query for all file entities ---
        # Cross-file DEFINES edges are preserved: they are created by
        # resolve_member_defines from the MEMBER file's parse, so this file's
        # recreation would never restore them. Citation DOCUMENTS edges are
        # preserved for the same reason: they leave the cited document's node
        # but are created by resolve_citations from the CITING file's parse —
        # which is also what deletes them, in its own file_paths-scoped revoke
        # pass, since they are inbound to the citing file and this query only
        # ever sweeps outbound edges.
        delete_fps = [fp for fp in file_rels if fp not in new_file_paths]
        if delete_fps:
            await self.execute_write(
                f"MATCH (n {{project_name: $p}})-[r]->(m) "
                f"WHERE n.file_path IN $fps AND NOT n:{NodeLabel.PACKAGE} AND NOT n:{NodeLabel.PROJECT} "
                f"AND NOT (type(r) = '{RelType.DEFINES}' AND coalesce(m.file_path, n.file_path) <> n.file_path) "
                f"AND NOT (type(r) = '{RelType.DOCUMENTS}' AND r.link_type = 'citation') "
                f"DELETE r",
                {"p": project_name, "fps": delete_fps},
            )

        # --- Create phase: pool all rels across files ---
        all_rels: list[ParsedRelationship] = []
        for rels in file_rels.values():
            all_rels.extend(rels)

        await self._create_relationships(project_name, all_rels)

    async def _recreate_file_relationships(
        self,
        project_name: str,
        file_path: str,
        relationships: list[ParsedRelationship],
        *,
        skip_delete: bool = False,
    ) -> None:
        """Delete all relationships originating from this file's entities, then recreate them.

        Cross-file DEFINES edges (resolve_member_defines output, owned by the
        member file's parse) and citation DOCUMENTS edges (resolve_citations
        output, owned by the citing file's parse) are preserved — recreation
        would never restore them. Citations are revoked instead by
        ``resolve_citations``'s own ``file_paths``-scoped pass, which is the
        only phase that can see them: they run INTO the citing file's entities,
        and this query only deletes edges running out of them.
        """
        if not skip_delete:
            await self.execute_write(
                f"MATCH (n {{project_name: $p, file_path: $f}})-[r]->(m) "
                f"WHERE NOT n:{NodeLabel.PACKAGE} AND NOT n:{NodeLabel.PROJECT} "
                f"AND NOT (type(r) = '{RelType.DEFINES}' AND coalesce(m.file_path, $f) <> $f) "
                f"AND NOT (type(r) = '{RelType.DOCUMENTS}' AND r.link_type = 'citation') DELETE r",
                {"p": project_name, "f": file_path},
            )
        await self._create_relationships(project_name, relationships)

    async def _create_relationships(
        self,
        project_name: str,
        relationships: list[ParsedRelationship],
    ) -> None:
        """Create relationships from a flat list of ParsedRelationship.

        Shared by both single-file and batched upsert paths.
        IMPORTS, CALLS, and USES_TYPE are excluded — they're resolved post-batch.
        DEFINES rels carrying a ``parent_type_name`` property are also excluded —
        resolved post-batch via ``resolve_member_defines``.
        IMPLEMENTS arrives in two shapes: detector-emitted target uids (always
        contain ``:``) follow the uid path; parser-emitted bare type names
        (TS/Java/C#/PHP/Rust) resolve by name like INHERITS.
        """
        # IMPLEMENTS arrives in two shapes: detector-emitted target uids
        # (always contain ':') and parser-emitted bare type names
        # (TS/Java/C#/PHP/Rust) — bare names resolve like INHERITS.
        uid_rels: list[ParsedRelationship] = []
        other_rels: list[ParsedRelationship] = []
        for r in relationships:
            if r.rel_type == RelType.IMPLEMENTS:
                (uid_rels if ":" in r.to_name else other_rels).append(r)
            elif r.rel_type in _UID_ROUTED_REL_TYPES:
                if "parent_type_name" not in r.properties:
                    uid_rels.append(r)
            else:
                other_rels.append(r)

        # uid-based rels: one UNWIND per rel_type
        for rel_type, group in groupby(sorted(uid_rels, key=attrgetter("rel_type")), key=attrgetter("rel_type")):
            rel_params = [
                {"from_uid": r.from_qualified_name, "to_uid": r.to_name, "props": r.properties or {}} for r in group
            ]
            await self.execute_write(
                f"UNWIND $rels AS r MATCH (a {{uid: r.from_uid}}), (b {{uid: r.to_uid}}) "
                f"MERGE (a)-[e:{rel_type}]->(b) SET e += r.props",
                {"rels": rel_params},
            )

        # Name-matched rels. INHERITS is deliberately absent — see _POST_BATCH_REL_TYPES.
        implements_rels = [r for r in other_rels if r.rel_type == RelType.IMPLEMENTS]
        doc_rels = [r for r in other_rels if r.rel_type == RelType.DOCUMENTS]

        for name_rel_type, name_rels in ((RelType.IMPLEMENTS, implements_rels),):
            if not name_rels:
                continue
            params = [
                {"from_uid": r.from_qualified_name, "project": project_name, "to_name": r.to_name} for r in name_rels
            ]
            await self.execute_write(
                f"UNWIND $rels AS r "
                f"MATCH (a:{NodeLabel.TYPE_DEF} {{uid: r.from_uid}}), "
                f"(b:{NodeLabel.TYPE_DEF} {{project_name: r.project, name: r.to_name}}) "
                f"CREATE (a)-[:{name_rel_type}]->(b)",
                {"rels": params},
            )

        if doc_rels:
            await self._create_doc_links(project_name, doc_rels)

    async def _create_doc_links(self, project_name: str, doc_rels: list[ParsedRelationship]) -> None:
        """Create DOCUMENTS edges via batched name/path matching.

        Two batched queries: one for symbol-based links (exact name match),
        one for file-path-based links (suffix match on file_path). The
        from-side may be a DocSection (heading-level docs) or a Note
        (frontmatter-triggered vault/memory files) — both emit heuristic
        DOCUMENTS refs the same way. The to-side excludes Note/DocSection
        (heuristic mentions should only land on genuine code/doc-file
        entities, not other notes or subsections) and never multi-links —
        an ambiguous match (more than one candidate) is left unresolved
        rather than fanning out one edge per candidate.
        """
        symbol_params = []
        file_params = []
        for rel in doc_rels:
            props = rel.properties
            entry = {
                "from_uid": rel.from_qualified_name,
                "to_name": rel.to_name,
                "link_type": props.get("link_type", ""),
                "confidence": props.get("confidence", 0.0),
            }
            if props.get("is_file_ref"):
                file_params.append(entry)
            else:
                symbol_params.append(entry)

        created = 0
        if symbol_params:
            records = await self.execute(
                f"UNWIND $rels AS r "
                f"MATCH (a {{uid: r.from_uid}}) WHERE a:{NodeLabel.DOC_SECTION} OR a:{NodeLabel.NOTE} "
                f"MATCH (b {{project_name: $project, name: r.to_name}}) "
                f"WHERE NOT b:{NodeLabel.NOTE} AND NOT b:{NodeLabel.DOC_SECTION} "
                f"WITH r, a, collect(b) AS candidates WHERE size(candidates) = 1 "
                f"WITH r, a, candidates[0] AS b "
                f"CREATE (a)-[e:{RelType.DOCUMENTS} {{link_type: r.link_type, confidence: r.confidence}}]->(b) "
                f"RETURN count(e) AS cnt",
                {"rels": symbol_params, "project": project_name},
            )
            created += records[0]["cnt"] if records else 0

        if file_params:
            records = await self.execute(
                f"UNWIND $rels AS r "
                f"MATCH (a {{uid: r.from_uid}}) WHERE a:{NodeLabel.DOC_SECTION} OR a:{NodeLabel.NOTE} "
                f"MATCH (b {{project_name: $project}}) WHERE b.file_path ENDS WITH r.to_name "
                f"AND NOT b:{NodeLabel.NOTE} AND NOT b:{NodeLabel.DOC_SECTION} "
                f"WITH r, a, collect(b) AS candidates WHERE size(candidates) = 1 "
                f"WITH r, a, candidates[0] AS b "
                f"CREATE (a)-[e:{RelType.DOCUMENTS} {{link_type: r.link_type, confidence: r.confidence}}]->(b) "
                f"RETURN count(e) AS cnt",
                {"rels": file_params, "project": project_name},
            )
            created += records[0]["cnt"] if records else 0

        attempted = len(doc_rels)
        if created < attempted:
            logger.debug(
                "DOCUMENTS links: {}/{} resolved for project {}",
                created,
                attempted,
                project_name,
            )

    async def _apply_full_schema(self) -> None:
        """Apply all constraints, indices, vector indices, and text indices.

        On a fresh database (no SchemaVersion node), vector/text indices from
        a previous session may still exist with stale internal state.  Drop
        them first so they are cleanly recreated at the current dimension.
        """
        # Drop stale search indices left over from a wiped database
        await self._drop_all_vector_indices()
        for stmt in generate_drop_text_index_ddl():
            await self._exec_ddl(stmt)

        for stmt in generate_unique_constraint_ddl():
            await self._exec_ddl(stmt)
        for stmt in generate_existence_constraint_ddl():
            await self._exec_ddl(stmt)
        for stmt in generate_index_ddl():
            await self._exec_ddl(stmt)
        for stmt in generate_composite_index_ddl():
            await self._exec_ddl(stmt)
        if self._embeddings_enabled:
            await self._create_vector_indices()
        for stmt in generate_text_index_ddl():
            await self._exec_ddl(stmt)

    async def _migrate_v8_drop_unverified_calls(self) -> None:
        """v8: ``project_unique`` no longer resolves an attribute call on a receiver
        whose type is unknown.

        Those edges are the one class this migration must remove rather than leave to be
        overwritten. They were written as ``confidence: "resolved"`` with full weight, so
        nothing downstream distrusts them — ``ambiguous_only`` cannot flag them and the
        outline's annotation renders nothing, every property being at its neutral value.
        Re-parsing alone would not clear a stale one whose call site has since gone.

        Only ``project_unique`` edges are dropped. The other four strategies are
        lexically grounded (an import, a class sibling, the same file, a constructor) and
        are unaffected by the change, so purging them would cost a rebuild for nothing.
        """
        await self.execute_write(f"MATCH ()-[r:{RelType.CALLS}]->() WHERE r.strategy = 'project_unique' DELETE r")
        await self.execute_write(
            f"MATCH (n) WHERE (n:{NodeLabel.MODULE} OR n:{NodeLabel.PACKAGE}) AND n.file_hash IS NOT NULL "
            "REMOVE n.file_hash"
        )
        await self.execute_write(f"MATCH (p:{NodeLabel.PROJECT}) REMOVE p.git_hash")
        logger.info(
            "Schema v8: dropped project_unique CALLS edges and cleared file/git hashes — "
            "run 'atlas index' to re-resolve them against the call's receiver"
        )

    async def _migrate_v9_clear_for_abstract_bases(self) -> None:
        """v9: TypeDefs gained ``is_abstract``, and CALLS resolution now uses it.

        Only the parser can produce the flag and nothing in the graph can be used to
        derive it, so every file has to go through the parser again — the same reasoning
        as v7, and the file-hash gate would otherwise skip each unchanged file forever.

        Ambiguous CALLS edges are dropped because their candidate sets were computed
        without the flag: a set that included a Protocol stub was resolved across it, and
        re-parsing alone would leave the stale edge alongside the corrected one. Resolved
        edges from the lexically-grounded strategies are unaffected and are left in place
        rather than rebuilt for nothing.
        """
        await self.execute_write(f"MATCH ()-[r:{RelType.CALLS}]->() WHERE r.confidence = 'ambiguous' DELETE r")
        await self.execute_write(
            f"MATCH (n) WHERE (n:{NodeLabel.MODULE} OR n:{NodeLabel.PACKAGE}) AND n.file_hash IS NOT NULL "
            "REMOVE n.file_hash"
        )
        await self.execute_write(f"MATCH (p:{NodeLabel.PROJECT}) REMOVE p.git_hash")
        logger.info(
            "Schema v9: cleared ambiguous CALLS edges and file/git hashes — run 'atlas index' "
            "to re-resolve them against Protocol/ABC declarations"
        )

    async def _migrate_v10_stub_flag_moved_to_methods(self) -> None:
        """v10: "is this a stub" is now asked of the method, not of its class.

        v9 flagged the CLASS via its bases, which conflated Protocol (all stubs) with ABC
        (one abstractmethod among many real methods). That deleted true callees from
        candidate sets and left same-named siblings to be promoted to resolved edges.
        Every resolved CALLS edge decided under that rule has to be re-derived, and the
        stale class-level flag has to go so nothing reads it again.
        """
        await self.execute_write(f"MATCH (t:{NodeLabel.TYPE_DEF}) WHERE t.is_abstract IS NOT NULL REMOVE t.is_abstract")
        await self.execute_write(f"MATCH ()-[r:{RelType.CALLS}]->() DELETE r")
        await self.execute_write(
            f"MATCH (n) WHERE (n:{NodeLabel.MODULE} OR n:{NodeLabel.PACKAGE}) AND n.file_hash IS NOT NULL "
            "REMOVE n.file_hash"
        )
        await self.execute_write(f"MATCH (p:{NodeLabel.PROJECT}) REMOVE p.git_hash")
        logger.info("Schema v10: cleared CALLS edges and the class-level is_abstract flag — run 'atlas index'")

    async def _reconcile_search_indices(self) -> None:
        """Recreate vector/text indices that have gone missing at the current version.

        The version check answers "is the schema shape current?", not "do the indices
        still exist?" — and those come apart. An index can vanish while the version
        node still reads current: a restore from a snapshot predating it, a manual
        drop, a CREATE that failed after the version was written. Every subsequent
        startup then takes the no-op branch, so nothing ever puts it back.

        The failure is silent, which is what makes it worth a startup check. Node
        embeddings keep being written, ``health_check`` keeps reporting the embedding
        provider healthy, and ``hybrid_search`` keeps returning results — just BM25
        ones, with the vector channel contributing nothing. Observed in the field:
        a graph with 5481 embedded Callables and zero vector indices.

        Only missing indices are created; this is not a drop-and-rebuild.
        """
        try:
            rows = await self.execute("SHOW INDEX INFO")
        except Exception as exc:  # pragma: no cover - catalogue unavailable
            logger.debug("Could not read index catalogue to reconcile: {}", exc)
            return

        # Memgraph 3.12 reports vector-index labels as ":Callable"; 3.7 reported "Callable",
        # and label+property/label_text rows still do. Without the strip this set never
        # intersects the expected one, so every ensure_schema believes all six vector
        # indices are missing. The re-CREATE is harmless — a duplicate CREATE VECTOR INDEX
        # is a verified no-op that preserves the populated index — but it warns on every
        # startup and permanently blinds the detector to an index that is genuinely gone,
        # which is the exact failure this reconciliation was added to catch.
        # removeprefix, not lstrip: lstrip(":") would also eat the ":" in a ":A|:B" filter.
        present_vector = {
            str(r.get("label")).removeprefix(":") for r in rows if "vector" in str(r.get("index type", "")).lower()
        }
        present_text = {
            str(r["index type"]).split("name: ")[-1].rstrip(")")
            for r in rows
            if str(r.get("index type", "")).startswith("label_text")
        }

        expected_vector = {lbl.value for lbl in _EMBEDDABLE_LABELS} if self._embeddings_enabled else set()
        expected_text = {f"text_{lbl.value.lower()}" for lbl in _TEXT_SEARCHABLE_LABELS}

        missing_vector = expected_vector - present_vector
        missing_text = expected_text - present_text
        if not missing_vector and not missing_text:
            return

        logger.warning(
            "Schema v{} is current but search indices are missing (vector: {}, text: {}) — recreating. "
            "Semantic search returns nothing without them.",
            SCHEMA_VERSION,
            sorted(missing_vector) or "none",
            sorted(missing_text) or "none",
        )
        if missing_vector:
            await self._create_vector_indices()
        if missing_text:
            for stmt in generate_text_index_ddl():
                await self._exec_ddl(stmt)

    async def _migrate_indices(self) -> None:
        """Drop and recreate vector/text indices (dimension may have changed).

        Also applies property and composite indices (idempotent via _exec_ddl).
        """
        for stmt in generate_index_ddl():
            await self._exec_ddl(stmt)
        for stmt in generate_composite_index_ddl():
            await self._exec_ddl(stmt)
        await self._drop_all_vector_indices()
        for stmt in generate_drop_text_index_ddl():
            await self._exec_ddl(stmt)
        if self._embeddings_enabled:
            await self._create_vector_indices()
        for stmt in generate_text_index_ddl():
            await self._exec_ddl(stmt)

    async def _migrate_v3_clear_freshness_markers(self) -> None:
        """v3: content_hash now covers entity source. Clear file/git hashes so the
        next index run re-parses every file and heals entities whose body-only
        edits were invisible under the v2 hash formula.
        """
        await self.execute_write(
            f"MATCH (n) WHERE (n:{NodeLabel.MODULE} OR n:{NodeLabel.PACKAGE}) AND n.file_hash IS NOT NULL "
            "REMOVE n.file_hash"
        )
        await self.execute_write(f"MATCH (p:{NodeLabel.PROJECT}) REMOVE p.git_hash")
        logger.info(
            "Schema v3: cleared stored file/git hashes — run 'atlas index' to refresh entities indexed before v3"
        )

    async def _migrate_v4_clear_freshness_markers(self) -> None:
        """v4: content_hash now folds in extra_properties (frontmatter), changing every entity's
        hash value even though most entities' extra_properties is empty (the extra empty list
        element still shifts the \\0-joined hash input). Clear file/git hashes so the next index
        run re-parses every file — cheap, since AST diffing then finds no real content changes for
        anything but the new Note entities.
        """
        await self.execute_write(
            f"MATCH (n) WHERE (n:{NodeLabel.MODULE} OR n:{NodeLabel.PACKAGE}) AND n.file_hash IS NOT NULL "
            "REMOVE n.file_hash"
        )
        await self.execute_write(f"MATCH (p:{NodeLabel.PROJECT}) REMOVE p.git_hash")
        logger.info(
            "Schema v4: cleared stored file/git hashes — run 'atlas index' to refresh entities indexed before v4"
        )

    async def _migrate_v5_clear_freshness_markers(self) -> None:
        """v5: markdown.py now excludes ``anchors`` from Note.extra_properties and instead
        emits DOCUMENTS(link_type='anchor') relationships, changing content_hash for any note
        with an ``anchors:`` key even though the file's bytes are unchanged. Clear file/git
        hashes so the next index run re-parses every file and resolves those anchors.
        """
        await self.execute_write(
            f"MATCH (n) WHERE (n:{NodeLabel.MODULE} OR n:{NodeLabel.PACKAGE}) AND n.file_hash IS NOT NULL "
            "REMOVE n.file_hash"
        )
        await self.execute_write(f"MATCH (p:{NodeLabel.PROJECT}) REMOVE p.git_hash")
        logger.info(
            "Schema v5: cleared stored file/git hashes — run 'atlas index' to refresh entities indexed before v5"
        )

    async def _migrate_v6_clear_freshness_markers(self) -> None:
        """v6: CALLS edges gained numeric ``weight``/``candidate_count``/``from_test``, and
        entities gained ``rationale``/``citations``. Both are produced during indexing, so an
        existing graph keeps neither until its files are re-parsed — and the file-hash gate
        skips unchanged files, so nothing would ever re-run. Clear file/git hashes to force a
        full re-parse, and drop the pre-v6 CALLS edges so they are rebuilt with weights rather
        than surviving unweighted (a missing weight silently reads as 1.0 in MAGE's Leiden).
        """
        await self.execute_write(
            f"MATCH (n) WHERE (n:{NodeLabel.MODULE} OR n:{NodeLabel.PACKAGE}) AND n.file_hash IS NOT NULL "
            "REMOVE n.file_hash"
        )
        await self.execute_write(f"MATCH (p:{NodeLabel.PROJECT}) REMOVE p.git_hash")
        await self.execute_write(f"MATCH ()-[r:{RelType.CALLS}]->() WHERE r.weight IS NULL DELETE r")
        logger.info(
            "Schema v6: cleared stored file/git hashes and unweighted CALLS edges — "
            "run 'atlas index' to rebuild with edge weights and rationale"
        )

    async def _migrate_v7_clear_freshness_markers(self) -> None:
        """v7: EnvVar/ResourceFile nodes and their READS_ENV/REFERENCES_FILE edges.

        A full re-parse *is* required here, and not by analogy with v3-v6 — this
        one has its own reason. Unlike those migrations, no existing entity's
        ``content_hash`` changes: the new data is a whole class of node that
        only the parser can produce, and it is produced from source text nobody
        has ever looked at before. There is nothing in the graph to derive it
        from, so every file has to go through the parser again. The file-hash
        gate would otherwise skip every unchanged file forever and the two new
        labels would stay permanently empty on any pre-v7 index.

        Nothing is deleted, unlike v6's unweighted-CALLS purge: there are no
        pre-v7 EnvVar/ResourceFile nodes to be stale.
        """
        await self.execute_write(
            f"MATCH (n) WHERE (n:{NodeLabel.MODULE} OR n:{NodeLabel.PACKAGE}) AND n.file_hash IS NOT NULL "
            "REMOVE n.file_hash"
        )
        await self.execute_write(f"MATCH (p:{NodeLabel.PROJECT}) REMOVE p.git_hash")
        logger.info(
            "Schema v7: cleared stored file/git hashes — run 'atlas index' to extract "
            "environment-variable and referenced-file nodes"
        )

    async def _set_schema_version(self, version: int) -> None:
        """Create or update the SchemaVersion singleton node.

        Collapses duplicate SchemaVersion nodes (left behind by the pre-fix
        migration MERGE) into one canonical node first: the highest-version
        node wins, back-filling missing embedding-config fields from the
        duplicates being removed so ``get_embedding_config`` never loses the
        stored config to the collapse.
        """
        await self.execute_write(
            f"MATCH (sv:{NodeLabel.SCHEMA_VERSION}) "
            "WITH sv ORDER BY coalesce(sv.version, -1) DESC, "
            "(CASE WHEN sv.embedding_model IS NULL THEN 0 ELSE 1 END) DESC "
            "WITH collect(sv) AS nodes WHERE size(nodes) > 1 "
            "WITH head(nodes) AS keep, tail(nodes) AS dupes "
            "SET keep.embedding_model = coalesce(keep.embedding_model, "
            "        head([d IN dupes WHERE d.embedding_model IS NOT NULL | d.embedding_model])), "
            "    keep.embedding_dimension = coalesce(keep.embedding_dimension, "
            "        head([d IN dupes WHERE d.embedding_dimension IS NOT NULL | d.embedding_dimension])) "
            "WITH dupes UNWIND dupes AS d DETACH DELETE d"
        )
        await self.execute_write(
            f"MERGE (sv:{NodeLabel.SCHEMA_VERSION}) SET sv.version = $version",
            {"version": version},
        )

    async def _exec_ddl(self, stmt: str) -> None:
        """Execute a DDL statement, ignoring 'already exists' / 'doesn't exist' errors."""
        try:
            await self.execute_write(stmt)
        except Exception as exc:
            msg = str(exc).lower()
            # Memgraph raises errors for duplicate constraints/indices and missing drops
            if "already exists" in msg or "doesn't exist" in msg or "not found" in msg:
                logger.debug("DDL skipped (idempotent): {}", stmt.rstrip(";"))
            else:
                raise

    async def _drop_all_vector_indices(self) -> None:
        """Drop every vector index, waiting until Memgraph confirms removal.

        Memgraph's ``DROP VECTOR INDEX`` returns success before internal
        state is fully cleaned up.  A subsequent ``CREATE VECTOR INDEX`` at
        a different dimension will fail unless we verify that the catalogue
        is actually empty.  Poll with short delays to handle this.
        """
        for drop_stmt in generate_drop_vector_index_ddl():
            try:
                await self.execute_write(drop_stmt)
                logger.debug("Dropped: {}", drop_stmt.rstrip(";"))
            except Exception as exc:
                logger.debug("Drop skipped ({}): {}", exc, drop_stmt.rstrip(";"))

        # Poll the catalogue until all vector indices are gone
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                rows = await self.execute("CALL vector_search.show_index_info() YIELD index_name RETURN index_name")
            except Exception:
                # If the procedure itself fails, assume no indices
                break

            if not rows:
                break

            names = [r["index_name"] for r in rows if r.get("index_name")]
            logger.debug(
                "Vector indices still in catalogue (attempt {}/{}): {}",
                attempt + 1,
                max_attempts,
                names,
            )

            # Try dropping each remaining index
            for name in names:
                with contextlib.suppress(Exception):
                    await self.execute_write(f"DROP VECTOR INDEX {name};")

            await asyncio.sleep(0.3)

    async def _create_vector_indices(self) -> None:
        """Create all vector indices, retrying if Memgraph's async DROP hasn't settled.

        Memgraph's ``DROP VECTOR INDEX`` returns success before internal state
        is fully cleaned.  A ``CREATE`` at a different dimension will fail
        with a 'dimensions' error.  This method retries the entire batch of
        CREATE statements with backoff until Memgraph accepts them.
        """
        stmts = generate_vector_index_ddl(self._dimension)
        max_retries = 10
        for attempt in range(max_retries):
            failed = False
            for stmt in stmts:
                try:
                    await self.execute_write(stmt)
                except Exception as exc:
                    msg = str(exc).lower()
                    if "already exists" in msg:
                        continue
                    if "dimensions" in msg:
                        failed = True
                        break
                    raise
            if not failed:
                return
            # Dimension mismatch — old internal state not yet cleared
            logger.debug(
                "Vector index CREATE blocked by stale dimension (attempt {}/{}), waiting…",
                attempt + 1,
                max_retries,
            )
            await self._drop_all_vector_indices()
            await asyncio.sleep(0.5 * (attempt + 1))
        # Final attempt — let exceptions propagate
        for stmt in stmts:
            await self._exec_ddl(stmt)
