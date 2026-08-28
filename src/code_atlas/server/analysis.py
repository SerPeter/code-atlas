"""Repository analysis and diagram generation for Code Atlas MCP server.

Pure Python formatting/aggregation over backend-provided records — no raw
Cypher/SQL here (see ``GraphBackend`` in ``graph/protocol.py``), no LLM
calls, no file reads, no new dependencies.
"""

from __future__ import annotations

import hashlib
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from code_atlas.backends.sqlite_graph import SqliteGraphClient
from code_atlas.schema import primary_label_expr
from code_atlas.search.engine import matches_test_pattern

if TYPE_CHECKING:
    from code_atlas.graph.protocol import GraphBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MERMAID_UNSAFE = re.compile(r"[^a-zA-Z0-9_]")

_VALID_ANALYSES = frozenset(
    {
        "structure",
        "centrality",
        "dependencies",
        "patterns",
        "quality",
        "dead_code",
        "complexity",
        "communities",
        "git_signals",
        "module_summary",
    }
)
_VALID_DIAGRAM_TYPES = frozenset({"packages", "imports", "inheritance", "module_detail"})

# trace_path / blast_radius (information-retrieval family, see ADR-0013) default
# edge sets — trace_path follows any structural/call relationship, blast_radius
# is specifically about call impact so it defaults to CALLS only.
_DEFAULT_TRACE_EDGE_TYPES = ("CALLS", "IMPORTS", "USES_TYPE")
# What "affected" means. CALLS alone answered "who EXECUTES this", which is a different
# and much smaller question than the one this tool exists for — and a class is never
# called, it is annotated, constructed and implemented. Measured on this repo, CALLS-only
# left 289 of 1,568 src entities (18%) with zero visible dependents, including GraphClient,
# whose 239 dependents it reported as none.
#
# DEFINES and CONTAINS are deliberately absent. They are containment, not dependency:
# including DEFINES would make changing one method "affect" its class and transitively
# everything the class touches, which is how a blast radius stops meaning anything.
_DEFAULT_BLAST_EDGE_TYPES = (
    "CALLS",
    "USES_TYPE",
    "IMPLEMENTS",
    "OVERRIDES",
    "INHERITS",
    "REFERENCES",
    "REGISTERED_BY",
    "IMPORTS",
)

# Quality analysis thresholds (v1 — hardcoded for medium Python projects)
_GOD_MODULE_ENTITY_THRESHOLD = 30
_TANGLED_FAN_THRESHOLD = 8
_INSTABILITY_LOW = 0.1
_INSTABILITY_HIGH = 0.9

# Communities noise threshold — singleton/near-singleton clusters aren't
# actionable groupings, drop them before ranking by size.
_COMMUNITY_NOISE_THRESHOLD = 2

# Communities clustering (see _analyze_communities). Greedy modularity is subject
# to the well-known resolution limit (Fortunato & Barthelemy 2007): on a weighted
# graph it fuses genuinely distinct blocks whose internal weight is small relative
# to the whole. The standard remedy is to re-run the maximizer on each community's
# *induced* subgraph and keep the split when the sub-partition is itself
# well-structured. _COMMUNITY_SPLIT_MIN_MODULARITY is that "well-structured" bar,
# expressed as the sub-partition's modularity inside its own subgraph — roughly
# half of Newman & Girvan's 0.3 "significant community structure" rule of thumb,
# because the subgraph being split is already a cohesive block. Measured on this
# repo's own module graph the partition is flat across 0.08-0.17 (7 communities);
# 0.12 sits in the middle of that plateau. Below ~0.06 it starts carving cohesive
# packages apart, above ~0.18 it stops splitting the giant blob at all.
_COMMUNITY_SPLIT_MIN_MODULARITY = 0.12
_COMMUNITY_MAX_SPLIT_DEPTH = 6
# Modularity gains are floats; treat anything under this as "no gain" so the
# agglomeration terminates instead of chasing rounding noise.
_MODULARITY_EPSILON = 1e-12

# Bus-factor risk: a file with only this many distinct authors (and at least
# one commit) is a single/double-owner risk worth flagging.
_BUS_FACTOR_AUTHOR_THRESHOLD = 1


def _sid(name: str) -> str:
    """Convert a qualified name to a safe, collision-resistant Mermaid node ID.

    Sanitization alone (replacing non-alphanumeric chars with '_') is not
    injective — e.g. 'pkg.data_utils' and 'pkg.data.utils' both collapse to
    'pkg_data_utils'. Appending a short hash of the original name keeps IDs
    deterministic (same name always maps to the same ID) while making
    distinct names map to distinct IDs.
    """
    sanitized = _MERMAID_UNSAFE.sub("_", name)
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{sanitized}_{digest}"


def _slabel(text: str, max_len: int = 40) -> str:
    """Truncate and escape a label for Mermaid display."""
    text = text.replace('"', "'").replace("<", "&lt;").replace(">", "&gt;")
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


def _module_package(qn: str) -> str:
    """Derive the parent package from a module qualified name."""
    return qn.rsplit(".", 1)[0] if "." in qn else ""


def _matches_test_qn(qn: str, patterns: list[str]) -> bool:
    """Like ``matches_test_pattern``, for records that only carry a qualified name (no file_path).

    Converts dots to slashes so directory-style patterns (``tests/``) still match
    (e.g. ``tests.unit.foo`` -> ``tests/unit/foo``).
    """
    return matches_test_pattern(qn.replace(".", "/"), qn.rsplit(".", 1)[-1], patterns)


_TEST_FILTER_FETCH_CAP = 200


def _padded_limit(limit: int, test_patterns: tuple[str, ...]) -> int:
    """Query-level fetch size when test_patterns filtering is active.

    Several graph methods apply LIMIT at the Cypher/SQL level, before Python-side
    test_patterns filtering runs. Fetching only *limit* rows and filtering afterward
    would silently return fewer than *limit* results whenever a filtered-out entity
    occupied one of those slots, instead of backfilling from real candidates beyond
    the original cutoff. Padding the query-level fetch (then truncating to *limit*
    after filtering) avoids that under-delivery.
    """
    return min(limit * 5, _TEST_FILTER_FETCH_CAP) if test_patterns else limit


def truncation_notice(shown: int, total: int, remedy: str) -> Literal[False] | dict[str, Any]:
    """What a capped result withheld, or ``False`` when nothing was cut.

    A bare ``truncated: true`` says something is missing but not whether it is one
    row or three hundred, and hands the caller no lever. That gap has a specific
    failure mode: an agent reads a capped list as a complete one and reports "X has
    no callers" — silence read as absence. Reporting *cut* alongside a concrete
    *remedy* is what makes the omission actionable rather than merely flagged.

    Returns ``False`` rather than an empty dict when nothing was cut, so the
    existing ``if result["truncated"]:`` contract keeps working unchanged and only
    the informative case grows a payload.

    *total* must be a real count. Where the caller cannot afford one — a search that
    fetched ``limit + 1`` rows knows more exist but not how many — use
    :func:`more_available_notice` instead of passing a bounded stand-in.
    """
    cut = total - shown
    if cut <= 0:
        return False
    return {"shown": shown, "total": total, "cut": cut, "remedy": remedy}


def more_available_notice(shown: int, remedy: str) -> dict[str, Any]:
    """Rows were withheld and the true count was never computed.

    ``total`` and ``cut`` are ``None`` rather than absent, so a caller reading
    ``truncated["cut"]`` gets "unknown" instead of a number — which is the whole
    point. The search tools fetch ``limit + 1`` rows to learn *whether* more exist;
    reporting that as ``total`` made ``cut`` arithmetically incapable of exceeding 1,
    so a repo with 5,000 matches answered ``{"shown": 20, "total": 21, "cut": 1}``
    while the MCP instruction told agents to "read `cut` before concluding a short
    list is a complete one" (ATL-111).

    A real count would mean running every channel unbounded on every query. Saying
    "more, count unknown" is cheap and true; a bounded number is neither.
    """
    return {"shown": shown, "total": None, "cut": None, "has_more": True, "remedy": remedy}


# ---------------------------------------------------------------------------
# Public dispatchers
# ---------------------------------------------------------------------------


async def analyze_repo(
    graph: GraphBackend,
    analysis: str,
    project: str,
    path: str = "",
    limit: int = 20,
    test_patterns: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Dispatch to the requested sub-analysis.

    *test_patterns*, when non-empty, drops entities/modules matching those glob
    patterns from every sub-analysis's ranked/listed output (hub entities, largest
    modules, complexity hotspots, community members, git-signal hotspots, etc.) —
    test scaffolding otherwise dominates these rankings purely by volume (see
    ADR-0016). Whole-repo aggregate counts that don't rank individual entities
    (structure's label_counts/kind_counts, patterns' visibility_distribution/
    docstring_coverage) are intentionally left unfiltered — those describe total
    repo composition, not "notable" entities. ``module_summary`` draws the same
    line one level down: its boundary lists (fan-in/fan-out) are filtered, its
    in-scope entity listing is not, because the caller named that path.
    """
    if analysis not in _VALID_ANALYSES:
        return {
            "error": f"Unknown analysis '{analysis}'. Valid: {sorted(_VALID_ANALYSES)}",
            "code": "INVALID_ANALYSIS",
        }
    dispatch = {
        "structure": _analyze_structure,
        "centrality": _analyze_centrality,
        "dependencies": _analyze_dependencies,
        "patterns": _analyze_patterns,
        "quality": _analyze_quality,
        "dead_code": _analyze_dead_code,
        "complexity": _analyze_complexity,
        "communities": _analyze_communities,
        "git_signals": _analyze_git_signals,
        "module_summary": _analyze_module_summary,
    }
    return await dispatch[analysis](graph, project, path, limit, test_patterns)


async def generate_diagram(
    graph: GraphBackend,
    diagram_type: str,
    project: str,
    path: str = "",
    max_nodes: int = 30,
    test_patterns: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Dispatch to the requested diagram generator."""
    if diagram_type not in _VALID_DIAGRAM_TYPES:
        return {
            "error": f"Unknown diagram type '{diagram_type}'. Valid: {sorted(_VALID_DIAGRAM_TYPES)}",
            "code": "INVALID_DIAGRAM_TYPE",
        }
    if diagram_type == "imports":
        return await _diagram_imports(graph, project, path, max_nodes, test_patterns)
    dispatch = {
        "packages": _diagram_packages,
        "inheritance": _diagram_inheritance,
        "module_detail": _diagram_module_detail,
    }
    return await dispatch[diagram_type](graph, project, path, max_nodes)


# ---------------------------------------------------------------------------
# trace_path / blast_radius (information-retrieval family, ADR-0013)
# ---------------------------------------------------------------------------


async def trace_path(
    graph: GraphBackend,
    from_uid: str,
    to_uid: str,
    max_depth: int = 6,
    edge_types: tuple[str, ...] = _DEFAULT_TRACE_EDGE_TYPES,
) -> dict[str, Any]:
    """Find the shortest path between two entities, bounded by ``max_depth`` hops.

    Traverses *edge_types* (default CALLS|IMPORTS|USES_TYPE). Returns the
    hop-by-hop path — edge type, endpoint uid/name, and CALLS confidence/
    strategy/weight/from_test when present (ADR-0014 and its weighting
    amendment) — or a ``found: false`` result when no path exists within
    ``max_depth``.

    Shortest-path-first is unchanged; ties between paths of equal hop count go
    to the higher ``path_weight`` (the product of the path's edge weights), so
    an all-resolved production route wins over an equally short route through
    ambiguous or test-provenance calls.
    """
    t0 = time.monotonic()
    result = await graph.trace_path_between(from_uid, to_uid, max_depth, edge_types)
    elapsed = (time.monotonic() - t0) * 1000

    if not result["from_exists"]:
        return {"error": f"Node not found: {from_uid}", "code": "NOT_FOUND"}
    if not result["to_exists"]:
        return {"error": f"Node not found: {to_uid}", "code": "NOT_FOUND"}

    if not result["found"]:
        return {
            "found": False,
            "from_uid": from_uid,
            "to_uid": to_uid,
            "max_depth": max_depth,
            "message": f"No path found within {max_depth} hops",
            "query_ms": round(elapsed, 1),
        }

    path_weight = result.get("path_weight")
    return {
        "found": True,
        "from_uid": from_uid,
        "to_uid": to_uid,
        "hop_count": result["hop_count"],
        "hops": result["hops"],
        "path_weight": round(path_weight, 6) if isinstance(path_weight, int | float) else None,
        "query_ms": round(elapsed, 1),
    }


_BLAST_DIRECTIONS = {"callers": ("in",), "callees": ("out",), "both": ("out", "in")}


async def blast_radius(
    graph: GraphBackend,
    uid: str,
    direction: str = "callers",
    max_depth: int = 3,
    edge_types: tuple[str, ...] = _DEFAULT_BLAST_EDGE_TYPES,
    limit: int = 20,
    test_patterns: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Depth-limited transitive closure of callers/callees/both from *uid*.

    "callers" traverses incoming edges (who transitively depends on *uid*),
    "callees" traverses outgoing edges (what *uid* transitively depends on).
    Each entry carries ``via`` — the edge types incident to *uid* on the paths
    that reach it — so a dependent found through REFERENCES or USES_TYPE is not
    read as a caller. That distinction is ADR-0099\'s, and it belongs in the
    output rather than in an omission that hid every type in the codebase.
    Each affected entity is flagged ``ambiguous_only: true`` when no path made
    entirely of ``confidence: "resolved"`` CALLS edges (ADR-0014) reaches it
    within ``max_depth`` — a heuristic signal, not a guarantee (e.g. an
    out-of-scope edge_types override without a confidence property always
    counts as not-resolved). Two further per-entity signals come from the
    weighting amendment to ADR-0014: ``test_only`` (no test-free path reaches
    it) and ``confidence_score`` (the best path's product of edge weights).

    Results are ordered nearest-first, then production impact before test-only
    impact, then by descending ``confidence_score`` — so what a change most
    likely breaks in production surfaces above heuristic and test-only hits.
    Backends that predate these fields (or hand-built test doubles) simply
    fall back to the neutral "production, fully-weighted" defaults.
    """
    t0 = time.monotonic()

    if not await graph.node_exists(uid):
        return {"error": f"Node not found: {uid}", "code": "NOT_FOUND"}

    dir_kinds = _BLAST_DIRECTIONS.get(direction)
    if dir_kinds is None:
        return {
            "error": f"Invalid direction '{direction}'. Valid: callers, callees, both",
            "code": "INVALID_DIRECTION",
        }

    affected: dict[str, dict[str, Any]] = {}
    for dir_kind in dir_kinds:
        entries = await graph.compute_blast_radius(uid, dir_kind, edge_types, max_depth)
        for entry in entries:
            existing = affected.get(entry["uid"])
            if existing is None or entry["min_depth"] < existing["min_depth"]:
                affected[entry["uid"]] = entry

    if test_patterns:
        # Filter on the entity's own path/name, NOT on its ``test_only`` flag. That flag
        # answers a different question — "no test-free CALL path reaches this" — is baked
        # in at index time from the *caller* side, and so cannot honour query-time
        # patterns. The two disagree on 131 of the entities across a sampled traversal.
        patterns = list(test_patterns)
        affected = {
            k: v
            for k, v in affected.items()
            if not matches_test_pattern(v["file_path"] or "", v["name"] or "", patterns)
        }

    elapsed = (time.monotonic() - t0) * 1000
    results = sorted(
        affected.values(),
        key=lambda x: (
            x["min_depth"],
            x.get("test_only", False),
            -x.get("confidence_score", 1.0),
            x["qualified_name"] or "",
        ),
    )
    total = len(results)
    return {
        "uid": uid,
        "direction": direction,
        "max_depth": max_depth,
        "affected_count": total,
        "affected": results[:limit],
        "truncated": truncation_notice(
            min(limit, total),
            total,
            "raise `limit`, or narrow with `edge_types` / a smaller `max_depth`. "
            "The cut entries are the lowest-impact ones; `affected_count` is the true total.",
        ),
        "query_ms": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


async def _analyze_structure(
    graph: GraphBackend, project: str, path: str, limit: int, test_patterns: tuple[str, ...] = ()
) -> dict[str, Any]:
    t0 = time.monotonic()
    data = await graph.get_structure_overview(project, path, _padded_limit(limit, test_patterns))

    # Entity counts by label + kind
    label_counts: dict[str, int] = {}
    kind_counts: dict[str, dict[str, int]] = {}
    for r in data["counts"]:
        lbl = r["label"]
        label_counts[lbl] = label_counts.get(lbl, 0) + r["cnt"]
        if r["kind"]:
            kind_counts.setdefault(lbl, {})[r["kind"]] = r["cnt"]

    largest_modules_raw = data["largest_modules"]
    if test_patterns:
        patterns = list(test_patterns)
        largest_modules_raw = [
            r for r in largest_modules_raw if not matches_test_pattern(r["file_path"] or "", r["module"], patterns)
        ]
    # Query was padded above for filtering headroom — re-truncate every list (filtered
    # or not) back to the caller-requested limit.
    largest_modules_raw = largest_modules_raw[:limit]
    packages_raw = data["packages"][:limit]
    external_deps_raw = data["external_deps"][:limit]

    elapsed = (time.monotonic() - t0) * 1000
    return {
        "analysis": "structure",
        "project": project,
        "label_counts": label_counts,
        "kind_breakdown": kind_counts,
        "packages": [
            {"name": r["package"], "qualified_name": r["qn"], "module_count": r["modules"]} for r in packages_raw
        ],
        "largest_modules": [
            {
                "name": r["module"],
                "qualified_name": r["qn"],
                "file_path": r["file_path"],
                "entity_count": r["entities"],
            }
            for r in largest_modules_raw
        ],
        "external_dependencies": [
            {"package": r["package"], "version": r["version"], "imported_by": r["imported_by"]}
            for r in external_deps_raw
        ],
        "query_ms": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Centrality
# ---------------------------------------------------------------------------


async def _analyze_centrality(
    graph: GraphBackend, project: str, path: str, limit: int, test_patterns: tuple[str, ...] = ()
) -> dict[str, Any]:
    t0 = time.monotonic()
    data = await graph.get_centrality_data(project, path, _padded_limit(limit, test_patterns))

    hubs, hub_modules, leaves = data["hubs"], data["hub_modules"], data["leaves"]
    if test_patterns:
        patterns = list(test_patterns)
        hubs = [r for r in hubs if not matches_test_pattern(r["file_path"] or "", r["name"], patterns)]
        hub_modules = [r for r in hub_modules if not matches_test_pattern(r["file_path"] or "", r["name"], patterns)]
        leaves = [r for r in leaves if not matches_test_pattern(r["file_path"] or "", r["name"], patterns)]
    # Query was padded above for filtering headroom — re-truncate back to the
    # caller-requested limit.
    hubs, hub_modules, leaves = hubs[:limit], hub_modules[:limit], leaves[:limit]

    elapsed = (time.monotonic() - t0) * 1000
    return {
        "analysis": "centrality",
        "project": project,
        "hub_entities": [
            {
                "name": r["name"],
                "qualified_name": r["qn"],
                "label": r["label"],
                "kind": r["kind"],
                "file_path": r["file_path"],
                "in_degree": r["in_degree"],
                "imported_by": r["imported_by"],
                "inherited_by": r["inherited_by"],
                "called_by": r["called_by"],
            }
            for r in hubs
        ],
        "hub_modules": [
            {
                "name": r["name"],
                "qualified_name": r["qn"],
                "file_path": r["file_path"],
                "imported_by": r["imported_by"],
            }
            for r in hub_modules
        ],
        "leaf_entities": [
            {
                "name": r["name"],
                "qualified_name": r["qn"],
                "label": r["label"],
                "kind": r["kind"],
                "file_path": r["file_path"],
            }
            for r in leaves
        ],
        "query_ms": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def _module_imports_from_records(
    direct: list[dict[str, Any]], indirect: list[dict[str, Any]]
) -> dict[tuple[str, str], int]:
    """Merge direct and entity-level import records into module-pair weights."""
    edges: dict[tuple[str, str], int] = {}
    for r in direct + indirect:
        key = (r["from_mod"], r["to_mod"])
        edges[key] = edges.get(key, 0) + 1
    return edges


async def _analyze_dependencies(
    graph: GraphBackend, project: str, path: str, limit: int, test_patterns: tuple[str, ...] = ()
) -> dict[str, Any]:
    t0 = time.monotonic()

    module_edges = await graph.get_module_import_edges(project, path)
    edge_weights = _module_imports_from_records(module_edges["direct"], module_edges["indirect"])

    # Drop edges touching a test module before deriving internal/cross-package/circular
    # views — same "filter the shared graph once" approach as _analyze_quality. No
    # per-edge file_path here (module-qualified-name keys only), so match on qn.
    if test_patterns:
        patterns = list(test_patterns)
        edge_weights = {
            (f, t): w
            for (f, t), w in edge_weights.items()
            if not _matches_test_qn(f, patterns) and not _matches_test_qn(t, patterns)
        }

    # Sort by weight descending
    internal_imports = sorted(
        [{"from": k[0], "to": k[1], "weight": v} for k, v in edge_weights.items()],
        key=lambda x: x["weight"],
        reverse=True,
    )[:limit]

    # Cross-package coupling (derive from module imports)
    pkg_edges: dict[tuple[str, str], int] = {}
    for (from_mod, to_mod), weight in edge_weights.items():
        from_pkg = _module_package(from_mod)
        to_pkg = _module_package(to_mod)
        if from_pkg != to_pkg:
            key = (from_pkg, to_pkg)
            pkg_edges[key] = pkg_edges.get(key, 0) + weight
    cross_package = sorted(
        [{"from": k[0], "to": k[1], "weight": v} for k, v in pkg_edges.items()],
        key=lambda x: x["weight"],
        reverse=True,
    )[:limit]

    # Circular dependencies (any cycle length, via strongly-connected components)
    circular = _detect_circular(edge_weights)[:10]

    # External package import counts — not test_patterns-filtered: aggregated purely
    # by imported package name with no per-importer identity to filter on.
    ext_data = await graph.get_dependency_external_counts(project, path)
    ext_counts: dict[str, int] = {}
    for r in ext_data["ext_packages"]:
        ext_counts[r["package"]] = ext_counts.get(r["package"], 0) + r["cnt"]
    for r in ext_data["ext_symbols"]:
        if r["package"]:
            ext_counts[r["package"]] = ext_counts.get(r["package"], 0) + r["cnt"]
    external_imports = sorted(
        [{"package": k, "import_count": v} for k, v in ext_counts.items()],
        key=lambda x: x["import_count"],
        reverse=True,
    )[:limit]

    elapsed = (time.monotonic() - t0) * 1000
    return {
        "analysis": "dependencies",
        "project": project,
        "internal_imports": internal_imports,
        "cross_package_coupling": cross_package,
        "circular_dependencies": circular,
        "external_imports": external_imports,
        "query_ms": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------


async def _analyze_patterns(
    graph: GraphBackend, project: str, path: str, limit: int, test_patterns: tuple[str, ...] = ()
) -> dict[str, Any]:
    t0 = time.monotonic()
    data = await graph.get_patterns_data(project, path, _padded_limit(limit, test_patterns))
    doc_raw = data["docstring"]
    doc_stats = doc_raw[0] if doc_raw else {"total": 0, "documented": 0}

    inheritance_raw, enums_raw, detected_raw = data["inheritance"], data["enums"], data["detected_patterns"]
    if test_patterns:
        patterns = list(test_patterns)
        # No file_path on inheritance/detected_patterns records — match on qualified name.
        inheritance_raw = [
            r
            for r in inheritance_raw
            if not _matches_test_qn(r["child_qn"], patterns) and not _matches_test_qn(r["parent_qn"], patterns)
        ]
        enums_raw = [r for r in enums_raw if not matches_test_pattern(r["file_path"] or "", r["name"], patterns)]
        detected_raw = [r for r in detected_raw if not _matches_test_qn(r["qn"], patterns)]
    # Query was padded above for filtering headroom — re-truncate back to the
    # caller-requested limit.
    inheritance_raw, enums_raw, detected_raw = inheritance_raw[:limit], enums_raw[:limit], detected_raw[:limit]

    elapsed = (time.monotonic() - t0) * 1000
    return {
        "analysis": "patterns",
        "project": project,
        "inheritance": [
            {
                "child": r["child"],
                "child_qualified_name": r["child_qn"],
                "parent": r["parent"],
                "parent_qualified_name": r["parent_qn"],
            }
            for r in inheritance_raw
        ],
        "enums": [
            {"name": r["name"], "qualified_name": r["qn"], "file_path": r["file_path"], "members": r["members"]}
            for r in enums_raw
        ],
        # visibility_distribution/docstring_coverage are whole-repo aggregates (not
        # per-entity lists) — intentionally not test_patterns-filtered, see analyze_repo's
        # docstring.
        "visibility_distribution": {r["visibility"]: r["cnt"] for r in data["visibility"]},
        "docstring_coverage": {
            "total": doc_stats["total"],
            "documented": doc_stats["documented"],
            "percentage": round(doc_stats["documented"] / doc_stats["total"] * 100, 1) if doc_stats["total"] else 0,
        },
        "detected_patterns": [
            {"type": r["pattern_type"], "name": r["name"], "qualified_name": r["qn"], "target": r["target_name"]}
            for r in detected_raw
        ],
        "query_ms": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Quality (health score + spaghettification detection)
# ---------------------------------------------------------------------------


def _health_score(
    modularity_ratio: float,
    god_count: int,
    circular_count: int,
    tangled_count: int,
    max_fan_in: int,
    max_fan_out: int,
    extreme_count: int,
) -> tuple[int, dict[str, int]]:
    """Compute a 0-100 composite health score from quality metrics."""
    modularity_s = round(modularity_ratio * 100)
    god_s = max(0, 100 - god_count * 20)
    circular_s = max(0, 100 - circular_count * 15)
    tangled_s = max(0, 100 - tangled_count * 20)
    coupling_s = max(0, 100 - max(0, max_fan_in + max_fan_out - 10) * 5)
    instability_s = max(0, 100 - extreme_count * 15)

    score = round(
        modularity_s * 0.25
        + god_s * 0.20
        + circular_s * 0.20
        + tangled_s * 0.15
        + coupling_s * 0.10
        + instability_s * 0.10
    )
    breakdown = {
        "modularity": modularity_s,
        "god_modules": god_s,
        "circular_deps": circular_s,
        "tangled": tangled_s,
        "coupling": coupling_s,
        "instability": instability_s,
    }
    return score, breakdown


def _compute_quality_flags(
    all_modules: set[str],
    entity_counts: dict[str, int],
    file_paths: dict[str, str],
    edge_weights: dict[tuple[str, str], int],
    limit: int,
) -> dict[str, Any]:
    """Compute all quality metrics from pre-fetched graph data.

    Returns the full result dict (minus ``analysis``, ``project``, ``query_ms``).
    """
    # Fan-in / fan-out per module
    fan_out: dict[str, set[str]] = {}
    fan_in: dict[str, set[str]] = {}
    for from_mod, to_mod in edge_weights:
        fan_out.setdefault(from_mod, set()).add(to_mod)
        fan_in.setdefault(to_mod, set()).add(from_mod)

    fo_counts = {m: len(fan_out.get(m, set())) for m in all_modules}
    fi_counts = {m: len(fan_in.get(m, set())) for m in all_modules}

    # Modularity ratio. Only counts edges sourced from an in-scope module —
    # when path-scoped, edge_weights also carries inbound edges from
    # out-of-scope importers (needed for accurate fan-in below), which must
    # not dilute this module's own intra/cross-package import ratio.
    intra, total = 0, 0
    for (from_mod, to_mod), weight in edge_weights.items():
        if from_mod not in all_modules:
            continue
        total += weight
        if _module_package(from_mod) == _module_package(to_mod):
            intra += weight
    modularity_ratio = intra / total if total > 0 else 1.0

    # God modules (full list for scoring, sliced for output)
    god_modules_all = sorted(
        [
            {"module": m, "file_path": file_paths.get(m, ""), "entity_count": ec}
            for m, ec in entity_counts.items()
            if ec > _GOD_MODULE_ENTITY_THRESHOLD
        ],
        key=lambda x: x["entity_count"],
        reverse=True,
    )

    # Circular dependencies. Restricted to cycles fully contained in
    # all_modules — a path-scoped edge whose other endpoint lies outside the
    # analyzed set must not manufacture a false "cycle" with an out-of-scope
    # module.
    circular = [pair for pair in _detect_circular(edge_weights) if all(m in all_modules for m in pair["cycle"])]
    circular_modules: set[str] = set()
    for pair in circular:
        circular_modules.update(pair["cycle"])

    # Tangled modules (full list for scoring, sliced for output)
    tangled_all = sorted(
        [
            {"module": m, "file_path": file_paths.get(m, ""), "fan_in": fi_counts[m], "fan_out": fo_counts[m]}
            for m in all_modules
            if fi_counts[m] > _TANGLED_FAN_THRESHOLD and fo_counts[m] > _TANGLED_FAN_THRESHOLD
        ],
        key=lambda x: x["fan_in"] + x["fan_out"],
        reverse=True,
    )

    # Coupling stats
    fo_values = list(fo_counts.values()) if all_modules else []
    fi_values = list(fi_counts.values()) if all_modules else []
    max_fi = max(fi_values) if fi_values else 0
    max_fo = max(fo_values) if fo_values else 0
    coupling_stats = {
        "avg_fan_in": round(sum(fi_values) / len(fi_values), 2) if fi_values else 0.0,
        "max_fan_in": max_fi,
        "avg_fan_out": round(sum(fo_values) / len(fo_values), 2) if fo_values else 0.0,
        "max_fan_out": max_fo,
    }

    # Instability + extremes (full lists for scoring, sliced for output)
    instabilities = _compute_instabilities(all_modules, fo_counts, fi_counts)
    rigid_all = sorted(
        [{"module": m, "instability": round(i, 3)} for m, i in instabilities.items() if i < _INSTABILITY_LOW],
        key=lambda x: x["instability"],
    )
    unstable_all = sorted(
        [{"module": m, "instability": round(i, 3)} for m, i in instabilities.items() if i > _INSTABILITY_HIGH],
        key=lambda x: x["instability"],
        reverse=True,
    )

    # Worst modules (aggregate flags)
    worst_modules = _aggregate_worst_modules(
        all_modules, entity_counts, file_paths, fo_counts, fi_counts, instabilities, circular_modules, limit
    )

    # Score uses full counts (not truncated lists) to avoid inflating the score
    score, breakdown = _health_score(
        modularity_ratio=modularity_ratio,
        god_count=len(god_modules_all),
        circular_count=len(circular),
        tangled_count=len(tangled_all),
        max_fan_in=max_fi,
        max_fan_out=max_fo,
        extreme_count=len(rigid_all) + len(unstable_all),
    )

    return {
        "health_score": score,
        "modularity_ratio": round(modularity_ratio, 3),
        "god_modules": god_modules_all[:limit],
        "circular_dependency_count": len(circular),
        "circular_dependencies": circular[:limit],
        "tangled_modules": tangled_all[:limit],
        "coupling_stats": coupling_stats,
        "instability": {"rigid": rigid_all[:limit], "unstable": unstable_all[:limit]},
        "worst_modules": worst_modules,
        "score_breakdown": breakdown,
    }


def _find_sccs(adjacency: dict[str, set[str]]) -> list[list[str]]:
    """Find strongly-connected components via an iterative Tarjan's algorithm.

    Iterative (explicit stack) to avoid Python's recursion limit on large
    import graphs. Returns components of every size; a component with more
    than one member is exactly a set of modules mutually reachable from one
    another — i.e. an import cycle of any length.
    """
    index_of: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: set[str] = set()
    scc_stack: list[str] = []
    components: list[list[str]] = []
    counter = 0

    for start in adjacency:
        if start in index_of:
            continue
        work: list[tuple[str, Any, str | None]] = [(start, iter(adjacency.get(start, ())), None)]
        while work:
            node, neighbors, parent = work[-1]
            if node not in index_of:
                index_of[node] = counter
                lowlink[node] = counter
                counter += 1
                scc_stack.append(node)
                on_stack.add(node)

            pushed = False
            for neighbor in neighbors:
                if neighbor not in index_of:
                    work.append((neighbor, iter(adjacency.get(neighbor, ())), node))
                    pushed = True
                    break
                if neighbor in on_stack:
                    lowlink[node] = min(lowlink[node], index_of[neighbor])
            if pushed:
                continue

            work.pop()
            if parent is not None:
                lowlink[parent] = min(lowlink[parent], lowlink[node])
            if lowlink[node] == index_of[node]:
                component: list[str] = []
                while True:
                    w = scc_stack.pop()
                    on_stack.discard(w)
                    component.append(w)
                    if w == node:
                        break
                components.append(component)

    return components


def _detect_circular(edge_weights: dict[tuple[str, str], int]) -> list[dict[str, Any]]:
    """Find modules involved in an import cycle of any length.

    Uses strongly-connected components so cycles longer than a mutual A<->B
    pair (e.g. A->B->C->A) are detected, not just 2-cycles.
    """
    adjacency: dict[str, set[str]] = {}
    for from_mod, to_mod in edge_weights:
        adjacency.setdefault(from_mod, set()).add(to_mod)

    circular: list[dict[str, Any]] = []
    for component in _find_sccs(adjacency):
        if len(component) < 2:
            continue
        cycle = sorted(component)
        circular.append({"module_a": cycle[0], "module_b": cycle[1], "cycle": cycle})
    return sorted(circular, key=lambda c: c["cycle"])


def _compute_instabilities(modules: set[str], fo_counts: dict[str, int], fi_counts: dict[str, int]) -> dict[str, float]:
    """Compute Martin's instability metric Ce/(Ca+Ce) per module."""
    result: dict[str, float] = {}
    for m in modules:
        ce, ca = fo_counts[m], fi_counts[m]
        result[m] = ce / (ca + ce) if (ca + ce) > 0 else 0.5
    return result


def _aggregate_worst_modules(
    all_modules: set[str],
    entity_counts: dict[str, int],
    file_paths: dict[str, str],
    fo_counts: dict[str, int],
    fi_counts: dict[str, int],
    instabilities: dict[str, float],
    circular_modules: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    """Build the worst-modules list by aggregating per-module issue flags."""
    issues: dict[str, list[str]] = {}
    for m, ec in entity_counts.items():
        if ec > _GOD_MODULE_ENTITY_THRESHOLD:
            issues.setdefault(m, []).append("god_module")
    for m in circular_modules:
        issues.setdefault(m, []).append("circular_dependency")
    for m in all_modules:
        if fi_counts[m] > _TANGLED_FAN_THRESHOLD and fo_counts[m] > _TANGLED_FAN_THRESHOLD:
            issues.setdefault(m, []).append("tangled")
        if instabilities.get(m, 0.5) < _INSTABILITY_LOW:
            issues.setdefault(m, []).append("rigid")
        if instabilities.get(m, 0.5) > _INSTABILITY_HIGH:
            issues.setdefault(m, []).append("unstable")
    return sorted(
        [
            {
                "module": m,
                "file_path": file_paths.get(m, ""),
                "issues": flags,
                "fan_in": fi_counts.get(m, 0),
                "fan_out": fo_counts.get(m, 0),
                "entity_count": entity_counts.get(m, 0),
                "instability": round(instabilities.get(m, 0.5), 3),
            }
            for m, flags in issues.items()
        ],
        key=lambda x: (len(x["issues"]), x["entity_count"]),
        reverse=True,
    )[:limit]


async def _analyze_quality(
    graph: GraphBackend, project: str, path: str, limit: int, test_patterns: tuple[str, ...] = ()
) -> dict[str, Any]:
    t0 = time.monotonic()
    data = await graph.get_quality_data(project, path)

    entity_counts: dict[str, int] = {}
    file_paths: dict[str, str] = {}
    for r in data["entities"]:
        entity_counts[r["module"]] = r["entity_count"]
        file_paths[r["module"]] = r["file_path"]

    edge_weights = _module_imports_from_records(data["direct"], data["indirect"])

    # Collect all modules (including those with no edges). When scoped to a
    # path, restrict to modules the entity query actually matched — edge
    # endpoints outside the path (now included above for accurate fan-in)
    # must not be scored as if they were in-scope modules.
    all_modules: set[str] = set(entity_counts.keys())
    if not path:
        for from_mod, to_mod in edge_weights:
            all_modules.add(from_mod)
            all_modules.add(to_mod)

    # Test modules are architecture-quality noise (god/circular/tangled/rigid/unstable
    # flags on test files aren't actionable) — drop them before scoring, not just from
    # the god-module list, so they don't skew fan-in/fan-out or instability either.
    if test_patterns:
        patterns = list(test_patterns)
        test_modules = {
            m for m in all_modules if matches_test_pattern(file_paths.get(m, ""), m.rsplit(".", 1)[-1], patterns)
        }
        if test_modules:
            all_modules -= test_modules
            entity_counts = {m: c for m, c in entity_counts.items() if m not in test_modules}
            file_paths = {m: p for m, p in file_paths.items() if m not in test_modules}
            edge_weights = {
                (f, t): w for (f, t), w in edge_weights.items() if f not in test_modules and t not in test_modules
            }

    metrics = _compute_quality_flags(all_modules, entity_counts, file_paths, edge_weights, limit)

    elapsed = (time.monotonic() - t0) * 1000
    return {"analysis": "quality", "project": project, **metrics, "query_ms": round(elapsed, 1)}


# ---------------------------------------------------------------------------
# Dead code
# ---------------------------------------------------------------------------


async def _analyze_dead_code(
    graph: GraphBackend, project: str, path: str, limit: int, test_patterns: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Callables/TypeDefs with zero incoming CALLS edges (any confidence).

    Scoped to *invocable* entities — the backend gates on
    ``graph.client._CODE_ENTITY_KINDS``, so config/infra declarations parsed out
    of Terraform, SQL, Kubernetes/Compose/CI YAML, XML and Dockerfiles are not
    reported. They share the Callable/TypeDef labels with real code but can
    never be the target of a CALLS edge, so listing them says nothing.

    Excludes dunder methods (``__init__``, ``__new__``, etc. — name STARTS WITH
    '__') and entries matching *test_patterns*. An entity reached only by an
    ``confidence: "ambiguous"`` CALLS edge (ADR-0014) still counts as "not
    dead" here — an ambiguous edge is still some evidence of a call site, even
    if the exact target is uncertain, and treating it as dead would be a worse
    false positive than leaving it out.

    Caveat (same style as the "quality" analysis): dynamic dispatch,
    reflection, and framework entry points (CLI commands, route handlers,
    test fixtures invoked by name) are invisible to static CALLS resolution
    and can still false-positive as dead code.

    One shape is permanently unresolvable rather than merely unresolved: a
    method overriding a base this project does not contain. Its only caller is
    the library — pydantic calls ``AtlasSettings.settings_customise_sources``
    on every construction — and the base is an ExternalSymbol with no members,
    so no OVERRIDES edge is even producible. Left visible on purpose. The
    exclusion that would silence it (a class inheriting an ExternalSymbol)
    covers 46 TypeDefs here including ``TierConsumer(abc.ABC)``, and would hide
    real rot in all of them to remove one known row.
    """
    t0 = time.monotonic()
    raw = await graph.get_dead_code_candidates(project, path)

    candidates = [
        {
            "name": r["name"],
            "qualified_name": r["qn"],
            "label": r["label"],
            "kind": r["kind"],
            "file_path": r["file_path"],
            "line_start": r["line_start"],
        }
        for r in raw
    ]
    if test_patterns:
        patterns = list(test_patterns)
        candidates = [c for c in candidates if not matches_test_pattern(c["file_path"] or "", c["name"], patterns)]

    elapsed = (time.monotonic() - t0) * 1000
    total = len(candidates)
    return {
        "analysis": "dead_code",
        "project": project,
        "dead_code_count": total,
        "dead_code": candidates[:limit],
        "truncated": truncation_notice(
            min(limit, total),
            total,
            "raise `limit`, or scope with `path`. `dead_code_count` is the true total.",
        ),
        "query_ms": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Complexity (LOC-span proxy)
# ---------------------------------------------------------------------------


async def _analyze_complexity(
    graph: GraphBackend, project: str, path: str, limit: int, test_patterns: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Top-N Callables by ``line_end - line_start``.

    This is a crude LOC-span proxy, not true cyclomatic complexity (branch-
    node counting) — no new parsing work was needed since line_start/line_end
    already persist on every Callable.
    """
    t0 = time.monotonic()
    raw = await graph.get_complexity_hotspots(project, path, _padded_limit(limit, test_patterns))
    if test_patterns:
        patterns = list(test_patterns)
        raw = [r for r in raw if not matches_test_pattern(r["file_path"] or "", r["name"], patterns)]
    # Query was padded above for filtering headroom — re-truncate back to the
    # caller-requested limit.
    raw = raw[:limit]

    elapsed = (time.monotonic() - t0) * 1000
    return {
        "analysis": "complexity",
        "project": project,
        "hotspots": [
            {
                "name": r["name"],
                "qualified_name": r["qn"],
                "kind": r["kind"],
                "file_path": r["file_path"],
                "line_start": r["line_start"],
                "line_end": r["line_end"],
                "loc_span": r["loc_span"],
            }
            for r in raw
        ],
        "query_ms": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Communities (module-granularity greedy modularity over the aggregated
# CALLS+IMPORTS graph — ADR-0013's find_communities)
# ---------------------------------------------------------------------------

# Undirected edge key: always (min, max) so a->b and b->a fold into one entry.
type _ModuleEdges = dict[tuple[str, str], float]


def _undirected_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _greedy_modularity(nodes: set[str], edges: _ModuleEdges) -> list[list[str]]:
    """Agglomerative greedy modularity maximization (Clauset-Newman-Moore).

    Every node starts alone; the connected community pair with the largest
    modularity gain is merged repeatedly until no merge improves modularity.
    For weighted undirected graphs the gain of merging communities *i* and *j*
    is ``2 * (w_ij/2m - tot_i*tot_j/(2m)^2)`` where ``w_ij`` is the weight
    between them, ``tot`` their summed node degrees and ``m`` the total edge
    weight.

    **Deterministic by construction** — ties are broken on the lexicographically
    smallest community-key pair, and a community's key is the smallest member
    name it has absorbed. Identical input always yields byte-identical output,
    which is the whole reason this exists instead of MAGE's Leiden (documented
    non-deterministic, so consecutive identical calls could disagree and no two
    runs were diffable).

    Isolated nodes (no incident edge) come back as their own single-member
    community.
    """
    all_nodes = sorted(nodes | {n for key in edges for n in key})
    members: dict[str, list[str]] = {n: [n] for n in all_nodes}
    if not edges:
        return [[n] for n in all_nodes]

    degree: dict[str, float] = dict.fromkeys(members, 0.0)
    between: _ModuleEdges = {}
    total_weight = 0.0
    for (u, v), w in edges.items():
        degree[u] += w
        degree[v] += w
        total_weight += w
        key = _undirected_key(u, v)
        between[key] = between.get(key, 0.0) + w
    two_m = 2.0 * total_weight

    while between:
        gains = {
            key: 2.0 * (w / two_m - degree[key[0]] * degree[key[1]] / (two_m * two_m)) for key, w in between.items()
        }
        best_gain = max(gains.values())
        if best_gain <= _MODULARITY_EPSILON:
            break
        keep, absorbed = min(key for key, gain in gains.items() if gain >= best_gain - _MODULARITY_EPSILON)

        members[keep].extend(members.pop(absorbed))
        degree[keep] += degree.pop(absorbed)
        merged: _ModuleEdges = {}
        for (a, b), w in between.items():
            left = keep if a == absorbed else a
            right = keep if b == absorbed else b
            if left == right:
                continue
            key = _undirected_key(left, right)
            merged[key] = merged.get(key, 0.0) + w
        between = merged

    return [sorted(group) for group in members.values()]


def _modularity(partition: list[list[str]], edges: _ModuleEdges) -> float:
    """Newman modularity Q of *partition* over the weighted undirected *edges*."""
    total_weight = sum(edges.values())
    if total_weight <= 0:
        return 0.0
    community_of = {node: idx for idx, group in enumerate(partition) for node in group}
    internal = [0.0] * len(partition)
    degree = [0.0] * len(partition)
    for (u, v), w in edges.items():
        cu, cv = community_of[u], community_of[v]
        degree[cu] += w
        degree[cv] += w
        if cu == cv:
            internal[cu] += w
    two_m = 2.0 * total_weight
    return sum(inner / total_weight - (deg / two_m) ** 2 for inner, deg in zip(internal, degree, strict=True))


def _detect_module_communities(nodes: set[str], edges: _ModuleEdges, depth: int = 0) -> list[list[str]]:
    """Greedy modularity plus recursive refinement of the resolution limit.

    Plain modularity maximization systematically under-splits: on this repo it
    fuses everything below the parsing package into three blobs. Re-running the
    same maximizer on each community's induced subgraph and accepting the split
    only when that sub-partition scores at least
    ``_COMMUNITY_SPLIT_MIN_MODULARITY`` inside its own subgraph recovers the
    real subsystems without introducing a hand-tuned resolution parameter that
    would need re-tuning per repository.
    """
    partition = _greedy_modularity(nodes, edges)
    if depth >= _COMMUNITY_MAX_SPLIT_DEPTH:
        return partition

    refined: list[list[str]] = []
    for group in partition:
        if len(group) < _COMMUNITY_NOISE_THRESHOLD + 1:
            refined.append(group)
            continue
        inside = set(group)
        sub_edges = {key: w for key, w in edges.items() if key[0] in inside and key[1] in inside}
        sub_partition = _greedy_modularity(inside, sub_edges)
        if len(sub_partition) > 1 and _modularity(sub_partition, sub_edges) >= _COMMUNITY_SPLIT_MIN_MODULARITY:
            refined.extend(_detect_module_communities(inside, sub_edges, depth + 1))
        else:
            refined.append(group)
    return refined


async def _fetch_community_inputs(
    graph: GraphBackend, project: str, path: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Module inventory + module-pair CALLS weights, both read-only.

    Two raw-Cypher reads rather than ``GraphBackend`` methods, for the same
    reason ``cypher_query``/``validate_cypher`` bypass the contract: there is no
    portable method for either shape yet, and adding one touches
    ``graph/protocol.py``/``graph/client.py``/``backends/sqlite_graph.py``. That
    is the *only* thing standing between this analysis and the SQLite backend —
    the clustering itself is pure Python now (see ``_analyze_communities``).

    The CALLS read aggregates in the database (``sum`` grouped by the endpoint
    file paths) so the ~10k callable-level edges never cross the wire; what comes
    back is one row per ordered file pair. ``coalesce(r.weight, 1.0)`` matches
    ``_CALL_WEIGHT_BASE`` — edges written before the weighting amendment to
    ADR-0014 count as one fully-resolved production call.
    """
    params: dict[str, Any] = {"project": project, "path": path}
    module_scope = " AND m.file_path STARTS WITH $path" if path else ""
    modules = await graph.execute(
        "MATCH (m:Module {project_name: $project}) "
        f"WHERE m.file_path IS NOT NULL{module_scope} "
        "RETURN m.uid AS uid, m.name AS name, m.qualified_name AS qn, m.file_path AS file_path",
        params,
    )
    call_scope = " AND a.file_path STARTS WITH $path AND b.file_path STARTS WITH $path" if path else ""
    # The three CASE sums split each pair's calls by the ADR-0028 evidence states:
    # ambiguous (a guess), no-confidence (never looked up), and the resolved rest.
    # Aggregated here so the map can label a module edge without a second query.
    call_edges = await graph.execute(
        "MATCH (a {project_name: $project})-[r:CALLS]->(b {project_name: $project}) "
        "WHERE a.file_path IS NOT NULL AND b.file_path IS NOT NULL "
        f"AND a.file_path <> b.file_path{call_scope} "
        "RETURN a.file_path AS from_path, b.file_path AS to_path, "
        "sum(coalesce(r.weight, 1.0)) AS weight, "
        "count(r) AS calls, "
        "sum(CASE WHEN r.confidence = 'ambiguous' THEN 1 ELSE 0 END) AS ambiguous_calls, "
        "sum(CASE WHEN r.confidence IS NULL AND r.strategy IS NULL THEN 1 ELSE 0 END) AS unlooked_calls",
        params,
    )
    return modules, call_edges


async def fetch_entity_graph(graph: GraphBackend, project: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Every entity in *project* and every dependency edge between them.

    The full-scope entity map's data access — the entity-level analogue of
    ``_fetch_community_inputs``. Dependency edges only (``_DEFAULT_BLAST_EDGE_TYPES``,
    ADR-0029): DEFINES/CONTAINS are containment, which the map synthesises from the
    qualified names as faint scaffolding rather than fetching row by row.

    Raw Cypher for the same reason the community inputs are: no portable backend
    method serves either shape, and the map is already Memgraph-gated.
    """
    params: dict[str, Any] = {"project": project}
    entities = await graph.execute(
        "MATCH (n {project_name: $project}) "
        "WHERE n.qualified_name IS NOT NULL AND NOT n:Project "
        f"RETURN n.uid AS uid, n.qualified_name AS qn, n.name AS name, {primary_label_expr('n')} AS label, "
        "n.kind AS kind, n.file_path AS file_path, n.line_start AS line_start, n.line_end AS line_end",
        params,
    )
    # CONTAINS rides along for the externals: a library's package CONTAINS its
    # symbols, which is the structure that keeps a stub library recognisable as one
    # shape instead of a scatter of loose names. The map draws it as scaffolding.
    rel_types = ", ".join(f"'{t}'" for t in (*_DEFAULT_BLAST_EDGE_TYPES, "CONTAINS"))
    edges = await graph.execute(
        "MATCH (a {project_name: $project})-[r]->(b {project_name: $project}) "
        f"WHERE type(r) IN [{rel_types}] "
        "AND a.qualified_name IS NOT NULL AND b.qualified_name IS NOT NULL "
        "RETURN a.qualified_name AS from_qn, b.qualified_name AS to_qn, type(r) AS rel_type, "
        "r.weight AS weight, r.confidence AS confidence, r.strategy AS strategy",
        params,
    )
    return entities, edges


async def fetch_external_imports(
    graph: GraphBackend, project: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """External packages and every edge reaching out to them, for the module map.

    The first-party/third-party boundary: 690 module-level IMPORTS land on
    ExternalPackage/ExternalSymbol nodes in this repo's graph, and the module map
    never showed one. Links come back keyed by the *source file*, so a Callable's
    import or a TypeDef's INHERITS is attributed to the module that owns it.
    """
    params: dict[str, Any] = {"project": project}
    packages = await graph.execute(
        "MATCH (x:ExternalPackage {project_name: $project}) "
        "RETURN x.uid AS uid, x.qualified_name AS qn, x.name AS name",
        params,
    )
    links = await graph.execute(
        "MATCH (s {project_name: $project})-[r:IMPORTS|INHERITS|USES_TYPE|REFERENCES]->(x) "
        "WHERE (x:ExternalPackage OR x:ExternalSymbol) AND s.file_path IS NOT NULL "
        "RETURN s.file_path AS from_path, x.qualified_name AS to_qn, count(r) AS links",
        params,
    )
    return packages, links


async def fetch_doc_modules(graph: GraphBackend, project: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Documentation files and where they point, for the module map.

    Markdown docs and knowledge notes are DocFile/Note nodes, not Modules, so the
    module inventory never sees them — the map draws them as non-code files behind the
    same toggle as the CI YAML. Links aggregate per file pair: a DOCUMENTS edge lands
    on an entity, and the entity's ``file_path`` names the module that owns it.
    """
    params: dict[str, Any] = {"project": project}
    docs = await graph.execute(
        "MATCH (d {project_name: $project}) "
        "WHERE (d:DocFile OR d:Note) AND d.file_path IS NOT NULL "
        "RETURN d.uid AS uid, d.qualified_name AS qn, d.name AS name, d.file_path AS file_path",
        params,
    )
    links = await graph.execute(
        "MATCH (d {project_name: $project})-[r:DOCUMENTS|LINKS_TO]->(e {project_name: $project}) "
        "WHERE (d:DocFile OR d:DocSection OR d:Note) "
        "AND d.file_path IS NOT NULL AND e.file_path IS NOT NULL "
        "RETURN d.file_path AS from_path, e.file_path AS to_path, count(r) AS links",
        params,
    )
    return docs, links


async def fetch_first_hop_external(graph: GraphBackend, project: str) -> list[dict[str, Any]]:
    """Modules in *other* indexed projects that this project's modules import.

    First hop only. The map's default scope is one project plus its immediate
    neighbours; following the dependency chain outward turns a readable picture into the
    whole graph, which is the "show everything" toggle rather than the default.

    Mirrors ``get_module_import_edges``'s direct and indirect shapes — an IMPORTS edge
    lands on a Module or on the individual symbol a Module DEFINES — so a cross-project
    dependency is not missed just because it named a function rather than a package.
    """
    params: dict[str, Any] = {"project": project}
    direct = await graph.execute(
        "MATCH (m1:Module {project_name: $project})-[:IMPORTS]->(m2:Module) "
        "WHERE m2.project_name <> $project AND m2.project_name IS NOT NULL "
        "RETURN m1.qualified_name AS from_mod, m2.qualified_name AS to_mod, "
        "m2.project_name AS to_project",
        params,
    )
    indirect = await graph.execute(
        "MATCH (m1:Module {project_name: $project})-[:IMPORTS]->(e)<-[:DEFINES]-(m2:Module) "
        "WHERE NOT e:Module AND m2.project_name <> $project AND m2.project_name IS NOT NULL "
        "RETURN m1.qualified_name AS from_mod, m2.qualified_name AS to_mod, "
        "m2.project_name AS to_project",
        params,
    )
    return [*direct, *indirect]


@dataclass(frozen=True)
class ModuleGraph:
    """The weighted module graph and its community partition, in both orientations.

    Shared rather than private because two callers need the *same* clustering:
    ``find_communities`` reports it, and the web map view draws it. Rebuilding the graph
    in the view would let the picture drift from the tool's answer for the same project,
    which is the one divergence the UI must not have.

    Two edge views, because the two consumers genuinely need different things:

    * ``edges`` — keyed by :func:`_undirected_key`, so a pair appears once with its
      summed weight (CALLS weight per ADR-0017, plus import counts). Modularity is
      defined on undirected graphs, so this is what the clustering must see.
    * ``directed`` — keyed ``(depender, dependency)`` as the sources actually record it.
      A dependency map that cannot say which way a dependency points is missing the thing
      a reader opens it for. Collapsing both into one dict meant the map's ``source`` and
      ``target`` were simply the alphabetically-first and -second name: on the real
      graph, 615 of 615 edges had ``source < target``, so an arrowhead would have pointed
      in dictionary order.

    A mutual dependency appears once in ``edges`` and twice in ``directed``.
    """

    modules: dict[str, dict[str, Any]]
    edges: dict[tuple[str, str], float]
    partition: list[list[str]]
    # Required, not defaulted: a ModuleGraph built without it would render a map with no
    # edges at all, and silently. Direction is part of the contract now.
    directed: dict[tuple[str, str], float]
    # Per directed pair, the strongest claim among the entity edges it aggregates:
    # structural (an import is a fact) > resolved > guessed > unknown (never looked up).
    # ADR-0028's states, carried up to module level so the map does not have to invent
    # one — "unknown is not zero" only works if unknown actually reaches the client.
    evidence: dict[tuple[str, str], str] = field(default_factory=dict)


@dataclass(frozen=True)
class ArchitecturePairs:
    """The architecture metrics' edge source, with its exclusions stated.

    One provider for the page, the JSON endpoint and the index-time history
    snapshot — they measured three different graphs under one metric name until
    2026-08 (the page read CALLS+IMPORTS, the endpoint and the history read the
    near-empty direct Module->Module import edges, ~10x apart on this repo).

    ``pairs`` is the filtered population the metrics run on. ``all_pairs`` is
    the same population before the evidence filter (tests already excluded), so
    a consumer can state what the guesses would add rather than hiding them.
    """

    pairs: dict[tuple[str, str], str]
    all_pairs: dict[tuple[str, str], str]
    module_paths: dict[str, str]
    excluded_test_modules: int
    excluded_guessed_pairs: int


async def fetch_architecture_pairs(
    graph: GraphBackend,
    project: str,
    *,
    include_tests: bool = False,
    include_guessed: bool = False,
    test_patterns: tuple[str, ...] = (),
) -> ArchitecturePairs:
    """Directed module pairs with evidence, filtered to what architecture means.

    Tests are excluded by default: nothing imports a test module, so every test
    adds an unreachable target to every module's denominator — the shipped
    propagation cost *improved* whenever tests were written. Guessed and
    never-looked-up pairs are excluded by default because the transitive closure
    is maximally sensitive to a false edge: one wrong guess bridges two whole
    subsystems (measured here: 19% hard-evidence vs 78% with guesses).

    On SQLite the CALLS aggregation is unavailable and the pairs are the direct
    import edges, all structural — the same source the page already used there.
    """
    from code_atlas.settings import SearchSettings  # noqa: PLC0415

    patterns = list(test_patterns) if test_patterns else list(SearchSettings().test_patterns)

    if isinstance(graph, SqliteGraphClient):
        raw = await graph.get_module_import_edges(project, "")
        sqlite_pairs = {
            (str(r.get("from_mod")), str(r.get("to_mod"))): "structural"
            for r in raw.get("direct", [])
            if r.get("from_mod") and r.get("to_mod") and r.get("from_mod") != r.get("to_mod")
        }
        paths: dict[str, str] = {}
        excluded = 0
        if not include_tests:
            names = {n for pair in sqlite_pairs for n in pair}
            test_names = {n for n in names if matches_test_pattern("", n.rsplit(".", 1)[-1], patterns)}
            excluded = len(test_names)
            sqlite_pairs = {
                p: ev for p, ev in sqlite_pairs.items() if p[0] not in test_names and p[1] not in test_names
            }
        return ArchitecturePairs(
            pairs=sqlite_pairs,
            all_pairs=dict(sqlite_pairs),
            module_paths=paths,
            excluded_test_modules=excluded,
            excluded_guessed_pairs=0,
        )

    module_graph = await build_module_graph(graph, project, "")
    paths = {qn: str(info.get("file_path") or "") for qn, info in module_graph.modules.items()}
    pair_ev = {pair: module_graph.evidence.get(pair, "unknown") for pair in module_graph.directed}

    excluded_tests = 0
    if not include_tests:
        modules = {n for pair in pair_ev for n in pair}
        test_modules = {n for n in modules if matches_test_pattern(paths.get(n, ""), n.rsplit(".", 1)[-1], patterns)}
        excluded_tests = len(test_modules)
        pair_ev = {p: ev for p, ev in pair_ev.items() if p[0] not in test_modules and p[1] not in test_modules}

    all_pairs = dict(pair_ev)
    excluded_guessed = 0
    if not include_guessed:
        hard = {p: ev for p, ev in pair_ev.items() if ev in ("structural", "resolved")}
        excluded_guessed = len(pair_ev) - len(hard)
        pair_ev = hard

    return ArchitecturePairs(
        pairs=pair_ev,
        all_pairs=all_pairs,
        module_paths=paths,
        excluded_test_modules=excluded_tests,
        excluded_guessed_pairs=excluded_guessed,
    )


async def build_module_graph(
    graph: GraphBackend, project: str, path: str, *, test_patterns: tuple[str, ...] = ()
) -> ModuleGraph:
    """Aggregate the callable-level graph up to modules and cluster it.

    Module granularity is the point, not a simplification: at callable granularity the
    CALLS+IMPORTS subgraph puts ~95% of production code in one community at every usable
    resolution — see :func:`_analyze_communities` for why no resolution parameter fixes
    that.
    """
    module_rows, call_rows = await _fetch_community_inputs(graph, project, path)

    if test_patterns:
        patterns = list(test_patterns)
        module_rows = [r for r in module_rows if not matches_test_pattern(r["file_path"] or "", r["name"], patterns)]

    modules_by_qn = {r["qn"]: r for r in module_rows if r["qn"]}
    qn_by_path = {r["file_path"]: r["qn"] for r in module_rows if r["file_path"] and r["qn"]}

    # Accumulated in one pass. Both inputs are directed at source — CALLS runs
    # caller->callee, IMPORTS runs importer->imported — so the orientation is there to
    # keep; the undirected key is where it used to be discarded.
    edges: dict[tuple[str, str], float] = {}
    directed: dict[tuple[str, str], float] = {}

    def _add(depender: str, dependency: str, weight: float) -> None:
        key = _undirected_key(depender, dependency)
        edges[key] = edges.get(key, 0.0) + weight
        pair = (depender, dependency)
        directed[pair] = directed.get(pair, 0.0) + weight

    # The strongest claim seen per directed pair. Rank order is ADR-0028's:
    # a structural fact beats a resolved match beats a guess beats "never looked up".
    evidence_rank = {"structural": 3, "resolved": 2, "guessed": 1, "unknown": 0}
    evidence: dict[tuple[str, str], str] = {}

    def _claim(depender: str, dependency: str, state: str) -> None:
        pair = (depender, dependency)
        if evidence_rank[state] > evidence_rank.get(evidence.get(pair, ""), -1):
            evidence[pair] = state

    for row in call_rows:
        from_qn = qn_by_path.get(row["from_path"])
        to_qn = qn_by_path.get(row["to_path"])
        if from_qn is None or to_qn is None or from_qn == to_qn:
            continue
        _add(from_qn, to_qn, float(row["weight"]))
        calls = int(row.get("calls") or 0)
        ambiguous = int(row.get("ambiguous_calls") or 0)
        unlooked = int(row.get("unlooked_calls") or 0)
        if calls - ambiguous - unlooked > 0:
            _claim(from_qn, to_qn, "resolved")
        elif ambiguous:
            _claim(from_qn, to_qn, "guessed")
        elif calls:
            _claim(from_qn, to_qn, "unknown")

    import_records = await graph.get_module_import_edges(project, path)
    import_pairs = _module_imports_from_records(import_records["direct"], import_records["indirect"])
    for (from_mod, to_mod), count in import_pairs.items():
        if from_mod == to_mod or from_mod not in modules_by_qn or to_mod not in modules_by_qn:
            continue
        _add(from_mod, to_mod, float(count))
        _claim(from_mod, to_mod, "structural")

    partition = _detect_module_communities(set(modules_by_qn), edges)
    partition.sort(key=lambda group: (-len(group), group[0] if group else ""))
    return ModuleGraph(modules=modules_by_qn, edges=edges, partition=partition, directed=directed, evidence=evidence)


async def _analyze_communities(
    graph: GraphBackend, project: str, path: str, limit: int, test_patterns: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Cluster the project's **modules** into subsystems by greedy modularity.

    Answers "what subsystems does this codebase have?". That is a question about
    modules (order 10^2 here), not about individual callables (order 10^3-10^4),
    and getting the granularity wrong is what made the previous MAGE-Leiden
    implementation useless: projected at callable granularity the CALLS+IMPORTS
    subgraph put ~95% of production code into one community at every usable
    resolution, because (a) a real call graph is densely connected through shared
    helpers — CALLS alone gives a 98% giant component here — and (b) IMPORTS
    almost never joins two Modules, it joins a Module to the individual *symbol*
    it imports, so every module importing a shared symbol hubs through that one
    node. No ``resolution_parameter`` fixes that; only aggregating does.

    So the graph actually clustered is built in Python, one node per Module:

    * **CALLS**, aggregated from callable level. Each callable-to-callable edge
      is attributed to the modules owning its endpoints (via ``file_path``) and
      the per-pair weights are **summed** — a module pair joined by many
      confident production calls outranks one joined by a single ambiguous or
      test-provenance call, which is exactly what the numeric ``weight``
      property from the ADR-0014 weighting amendment encodes (``1/candidate_count``
      for ambiguous edges, discounted again for ``from_test``). Summing, rather
      than averaging or counting, is what makes volume *and* confidence both
      count; averaging would let one high-confidence call outrank fifty.
    * **IMPORTS**, via ``get_module_import_edges`` — both the rare direct
      Module->Module edges and the far more common Module->symbol edges resolved
      back through ``DEFINES`` to the module that owns the symbol. That
      resolution is what turns IMPORTS into genuine module-to-module structure;
      it is the same aggregation the ``dependencies`` analysis already does, and
      it reuses the same ``_module_imports_from_records`` helper. Each import
      record contributes 1.0, matching ``_CALL_WEIGHT_BASE``: one import is worth
      exactly one fully-resolved non-test call.

    Intra-module calls (both endpoints in the same file) and self-imports are
    dropped — they say how cohesive a module is internally, not which modules
    belong together. Reciprocal pairs fold into a single undirected edge whose
    weight is the sum of both directions.

    ExternalPackage/ExternalSymbol can no longer act as false hubs (the bug that
    motivated the old projection's exclusion clause): only ``:Module`` nodes are
    ever clustered, and a CALLS endpoint only enters the graph if its
    ``file_path`` maps to one of them, so an external node has nowhere to appear.

    Clustering is **greedy modularity (Clauset-Newman-Moore)** — see
    ``_greedy_modularity`` — with recursive refinement for the resolution limit
    (``_detect_module_communities``). Deliberately deterministic: MAGE's Leiden
    is documented non-deterministic, so its output could differ between two
    identical calls and could not be diffed across runs. The aggregated graph is
    small (order 10^2 nodes) so an exact Python maximizer is instant, and it
    needs no query module, no materialized helper edges and no write access —
    the whole analysis path stays read-only.

    Communities of size < ``_COMMUNITY_NOISE_THRESHOLD`` (isolated modules —
    config/manifest pseudo-modules, leaf scripts) are dropped as noise; the rest
    are returned largest-first, capped at *limit* (which also caps members shown
    per community). ``modularity`` in the result is Q for the full partition
    before that noise cut, so it stays comparable across calls.

    *test_patterns*, when non-empty, drops test modules from the module
    inventory **before** the graph is built, so test connectivity cannot bridge
    two production subsystems (the ``exclude test entities from the input graph,
    not just its output`` rule). Matching uses the canonical
    ``matches_test_pattern`` on the module's own file_path/name.

    Still guarded off on the embedded SQLite backend, but no longer because of
    MAGE: the clustering is portable now and only the two read queries in
    ``_fetch_community_inputs`` are Memgraph-specific. Making it work on SQLite
    is a mechanical follow-up — add a portable inventory + module-pair-CALLS
    method to ``GraphBackend`` and both implementations — after which this guard
    and ``mcp.py``'s ``remove_tool("find_communities")`` both come out.
    """
    if isinstance(graph, SqliteGraphClient):
        return {
            "analysis": "communities",
            "error": (
                "unsupported on the sqlite backend — the module-pair CALLS aggregation and module "
                "inventory it clusters are still raw Cypher reads (see _fetch_community_inputs)"
            ),
        }

    t0 = time.monotonic()
    module_graph = await build_module_graph(graph, project, path, test_patterns=test_patterns)
    modules_by_qn, edges, partition = module_graph.modules, module_graph.edges, module_graph.partition

    sized: list[dict[str, Any]] = [
        {
            "community_id": idx,
            "size": len(group),
            "members": [
                {
                    "uid": modules_by_qn[qn]["uid"],
                    "name": modules_by_qn[qn]["name"],
                    "qualified_name": qn,
                    "label": "Module",
                    "file_path": modules_by_qn[qn]["file_path"],
                }
                for qn in group
            ],
        }
        for idx, group in enumerate(partition)
        if len(group) >= _COMMUNITY_NOISE_THRESHOLD
    ]

    result: dict[str, Any] = {
        "analysis": "communities",
        "project": project,
        "granularity": "module",
        "module_count": len(modules_by_qn),
        "edge_count": len(edges),
        "modularity": round(_modularity(partition, edges), 4),
        "community_count": len(sized),
        "communities": [{**c, "members": c["members"][:limit]} for c in sized[:limit]],
        "noise_threshold": _COMMUNITY_NOISE_THRESHOLD,
        "query_ms": round((time.monotonic() - t0) * 1000, 1),
    }
    if not sized:
        result["note"] = (
            "No communities detected — no two in-scope modules are joined by a call or import "
            "(try a broader `path`, or check the project is fully indexed)."
        )
    return result


# ---------------------------------------------------------------------------
# Git signals (hotspots, bus-factor risks, co-change pairs — ADR-0013 find_hotspots)
# ---------------------------------------------------------------------------


async def _analyze_git_signals(
    graph: GraphBackend, project: str, path: str, limit: int, test_patterns: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Surface git-mined signals: commit-count hotspots, bus-factor risks, top co-change pairs.

    Reads properties/edges written by ``indexing/git_signals.py``'s
    ``write_git_signals`` — populated by the one-shot ``atlas mine-git-history``
    CLI command, not the continuous indexing pipeline. If that command has
    never been run for this project, all three lists come back empty (not an
    error) — ``mined`` is ``False`` in that case so callers can tell "no signal"
    apart from "ran, found nothing".
    """
    t0 = time.monotonic()
    data = await graph.get_git_signals_data(
        project, path, _padded_limit(limit, test_patterns), _BUS_FACTOR_AUTHOR_THRESHOLD
    )
    hotspots_raw = data["hotspots"]
    # mined reflects whether mine-git-history has ever run at all, independent of
    # test_patterns filtering — compute it from the pre-filter list.
    mined = bool(hotspots_raw)

    bus_factor_raw, co_change_raw = data["bus_factor"], data["co_change"]
    if test_patterns:
        patterns = list(test_patterns)
        hotspots_raw = [r for r in hotspots_raw if not matches_test_pattern(r["file_path"] or "", r["name"], patterns)]
        bus_factor_raw = [
            r for r in bus_factor_raw if not matches_test_pattern(r["file_path"] or "", r["name"], patterns)
        ]
        co_change_raw = [
            r
            for r in co_change_raw
            if not matches_test_pattern(r["a_path"] or "", r["a_qn"], patterns)
            and not matches_test_pattern(r["b_path"] or "", r["b_qn"], patterns)
        ]
    # Query was padded above for filtering headroom — re-truncate back to the
    # caller-requested limit.
    hotspots_raw, bus_factor_raw, co_change_raw = hotspots_raw[:limit], bus_factor_raw[:limit], co_change_raw[:limit]

    elapsed = (time.monotonic() - t0) * 1000
    return {
        "analysis": "git_signals",
        "project": project,
        "mined": mined,
        "hotspots": [
            {
                "name": r["name"],
                "qualified_name": r["qn"],
                "file_path": r["file_path"],
                "commit_count": r["commit_count"],
                "author_count": r["author_count"],
                "days_since_last_commit": r["days_since_last_commit"],
            }
            for r in hotspots_raw
        ],
        "bus_factor_risks": [
            {
                "name": r["name"],
                "qualified_name": r["qn"],
                "file_path": r["file_path"],
                "commit_count": r["commit_count"],
                "author_count": r["author_count"],
            }
            for r in bus_factor_raw
        ],
        "co_change_pairs": [
            {
                "a": r["a_qn"],
                "a_file_path": r["a_path"],
                "b": r["b_qn"],
                "b_file_path": r["b_path"],
                "count": r["count"],
            }
            for r in co_change_raw
        ],
        "query_ms": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Diagram: packages (containment tree)
# ---------------------------------------------------------------------------


async def _diagram_packages(graph: GraphBackend, project: str, path: str, max_nodes: int) -> dict[str, Any]:
    t0 = time.monotonic()
    records = await graph.get_diagram_packages(project, path, max_nodes)

    if not records:
        elapsed = (time.monotonic() - t0) * 1000
        mermaid = 'graph TD\n    empty["No packages found"]'
        return {"type": "packages", "mermaid": mermaid, "node_count": 0, "query_ms": round(elapsed, 1)}

    lines = ["graph TD"]
    nodes: set[str] = set()
    for r in records:
        p_id = _sid(r["parent_qn"])
        c_id = _sid(r["child_qn"])
        c_icon = "📦" if r["child_label"] == "Package" else "📄"
        if p_id not in nodes:
            lines.append(f'    {p_id}["{_slabel(r["parent_name"])}"]')
            nodes.add(p_id)
        if c_id not in nodes:
            lines.append(f'    {c_id}["{c_icon} {_slabel(r["child_name"])}"]')
            nodes.add(c_id)
        lines.append(f"    {p_id} --> {c_id}")

    elapsed = (time.monotonic() - t0) * 1000
    return {
        "type": "packages",
        "mermaid": "\n".join(lines),
        "node_count": len(nodes),
        "query_ms": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Diagram: imports (module dependency graph)
# ---------------------------------------------------------------------------

# Mermaid is a rendering format, and measurement says it never pays for itself as text:
# it costs 3.2-3.6x a plain adjacency list at EVERY size from 2 nodes to 122, because the
# overhead is per-edge (_sid()'s sha1 suffix is re-emitted at both endpoints of every edge
# line — 36.8% of a 57-node document). There is no crossover to find. So this threshold is
# not a token inflection; it is the point below which the absolute cost is small enough
# that keeping a picture a human can actually render stays worth it, and above which the
# rendered picture is a hairball nobody looks at anyway.
_DIAGRAM_MERMAID_MAX_NODES = 25


def _render_grouped_adjacency(
    nodes: set[str], edges: list[tuple[tuple[str, str], int]], paths: dict[str, str] | None = None
) -> str:
    """Community-grouped adjacency — the cheapest lossless rendering measured (0.19x
    Mermaid at 57 nodes, 0.20x at 122) and also the most local: a node's neighbourhood
    spans a median of 12 lines against Mermaid's 270.

    Same-community targets render as a bare leaf name; cross-community ones are tagged
    ``Cn:leaf`` and stay inline on the source's own line. Deferring them to a trailing
    section measured both larger and less local, so inlining wins twice.
    """
    undirected: _ModuleEdges = {}
    for (a, b), w in edges:
        undirected[_undirected_key(a, b)] = undirected.get(_undirected_key(a, b), 0.0) + float(w)
    communities = _detect_module_communities(set(nodes), undirected)

    community_of = {qn: i for i, members in enumerate(communities) for qn in members}
    # Leaf names collide once test modules are in scope (conftest, test_client, ...), so
    # fall back to the shortest suffix that is unique across the rendered node set.
    leaf_counts = Counter(qn.rsplit(".", 1)[-1] for qn in nodes)
    label = {qn: (qn.rsplit(".", 1)[-1] if leaf_counts[qn.rsplit(".", 1)[-1]] == 1 else qn) for qn in nodes}

    def target(src: str, dst: str, weight: int) -> str:
        c_src, c_dst = community_of.get(src), community_of.get(dst)
        tag = f"C{c_dst}:" if c_dst is not None and c_dst != c_src else ""
        return f"{tag}{label[dst]}" + (f"*{weight}" if weight > 1 else "")

    out_edges: dict[str, list[str]] = {}
    for (a, b), w in edges:
        out_edges.setdefault(a, []).append(target(a, b, w))

    # In-degree is stated per node because every line is out-edges only, which makes the
    # most-depended-on modules render as the emptiest lines — a blind reader called this
    # the format's biggest blind spot, and it is one integer per node to fix.
    in_deg = Counter(b for (_a, b), _w in edges)
    lines = [
        f"IMPORTS {len(nodes)} modules, {len(edges)} edges, {len(communities)} clusters",
        "LEGEND 'a > b, c': a imports b and c | Cn: target is in cluster n | *N: N import "
        "statements | '(<-N)': imported by N of these modules | a bare name imports none of them",
        "CLUSTERS are computed by greedy modularity over this import graph, not by directory.",
        # Silence reads as "there are none", which is the wrong conclusion to invite from a
        # document someone is using to reason about a dependency surface.
        "SCOPE first-party modules only — external and stdlib imports are excluded from this graph.",
    ]

    # Cycles as one block, never as a per-edge annotation: O(cycles) instead of O(edges),
    # and a structural pathology belongs where it can be seen rather than dispersed across
    # the lines it implicates. Reuses the same SCC finder as analyze_repo(quality).
    adjacency: dict[str, set[str]] = {}
    for (a, b), _w in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set())
    cycles = [c for c in _find_sccs(adjacency) if len(c) > 1]
    if cycles:
        lines.append(f"CYCLES {len(cycles)} import cycle(s) — each set imports itself transitively:")
        lines.extend(f"  {' <-> '.join(sorted(label[q] for q in c if q in label))}" for c in cycles)

    # Identity, paid once per node rather than per mention. A node appears once as a
    # source but many times as a target, so inlining its path would be O(edge-endpoints);
    # the shared directory is stated once so each entry carries only what distinguishes it.
    known = {qn: str(p).replace("\\", "/") for qn, p in (paths or {}).items() if qn in nodes and p}
    if known:
        by_dir: dict[str, list[str]] = {}
        for p in known.values():
            head, _, base = p.rpartition("/")
            by_dir.setdefault(head + "/" if head else "", []).append(base)
        lines.append("FILES by directory — every module above lives in exactly one of these")
        lines.extend(f"  {d or '(repo root)'} {' '.join(sorted(files))}" for d, files in sorted(by_dir.items()))

    def node_line(qn: str) -> str:
        head = label[qn] + (f"(<-{in_deg[qn]})" if in_deg.get(qn) else "")
        targets = out_edges.get(qn)
        return f"  {head} > {', '.join(sorted(targets))}" if targets else f"  {head}"

    # Most-depended-on first. Alphabetical ordering carried no information — a reader
    # called it out as following "no rule I can identify" — and this costs nothing while
    # putting the load-bearing modules at the top of each cluster.
    def by_load(qn: str) -> tuple[int, str]:
        return (-in_deg.get(qn, 0), label[qn])

    for i, members in enumerate(communities):
        lines.extend(["", f"[C{i}] {len(members)} modules"])
        lines.extend(node_line(qn) for qn in sorted(members, key=by_load))
    loose = sorted((n for n in nodes if n not in community_of), key=by_load)
    if loose:
        lines.extend(["", f"[unclustered] {len(loose)} modules"])
        lines.extend(node_line(qn) for qn in loose)
    return "\n".join(lines)


async def _diagram_imports(
    graph: GraphBackend, project: str, path: str, max_nodes: int, test_patterns: tuple[str, ...] = ()
) -> dict[str, Any]:
    t0 = time.monotonic()
    module_edges = await graph.get_module_import_edges(project, path)
    edge_weights = _module_imports_from_records(module_edges["direct"], module_edges["indirect"])
    mod_paths: dict[str, str] = {}
    for r in module_edges["direct"] + module_edges["indirect"]:
        for qn_key, path_key in (("from_mod", "from_path"), ("to_mod", "to_path")):
            if r.get(qn_key) and r.get(path_key):
                mod_paths[r[qn_key]] = r[path_key]
    if test_patterns:
        # _analyze_dependencies and _analyze_quality already filter; this one never did,
        # so 60 of the 100 nodes at the cap were test modules — 60% of the node budget
        # spent on things nobody asks a dependency diagram about.
        patterns = list(test_patterns)
        edge_weights = {
            (a, b): w
            for (a, b), w in edge_weights.items()
            if not _matches_test_qn(a, patterns) and not _matches_test_qn(b, patterns)
        }

    if not edge_weights:
        elapsed = (time.monotonic() - t0) * 1000
        mermaid = 'graph LR\n    empty["No imports found"]'
        return {"type": "imports", "mermaid": mermaid, "node_count": 0, "query_ms": round(elapsed, 1)}

    # Collect nodes and cap at max_nodes
    all_nodes: set[str] = set()
    for from_mod, to_mod in edge_weights:
        all_nodes.add(from_mod)
        all_nodes.add(to_mod)

    # If too many nodes, keep only those in highest-weight edges
    # Keep scanning past the cap instead of stopping outright: a lower-weight
    # edge whose two endpoints are already kept adds no new nodes and must
    # still be included, not dropped just because the cap was hit earlier.
    sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1], reverse=True)
    kept_nodes: set[str] = set()
    kept_edges: list[tuple[tuple[str, str], int]] = []
    for (from_mod, to_mod), weight in sorted_edges:
        new_nodes = {n for n in (from_mod, to_mod) if n not in kept_nodes}
        if len(kept_nodes) + len(new_nodes) > max_nodes:
            continue
        kept_nodes.update(new_nodes)
        kept_edges.append(((from_mod, to_mod), weight))

    if len(kept_nodes) > _DIAGRAM_MERMAID_MAX_NODES:
        elapsed = (time.monotonic() - t0) * 1000
        return {
            "type": "imports",
            "format": "outline",
            "outline": _render_grouped_adjacency(kept_nodes, kept_edges, mod_paths),
            "node_count": len(kept_nodes),
            "query_ms": round(elapsed, 1),
        }

    lines = ["graph LR"]
    lines.extend(f'    {_sid(qn)}["{_slabel(qn)}"]' for qn in sorted(kept_nodes))
    for (from_mod, to_mod), weight in kept_edges:
        label_part = f"|{weight}|" if weight > 1 else ""
        lines.append(f"    {_sid(from_mod)} -->{label_part} {_sid(to_mod)}")

    elapsed = (time.monotonic() - t0) * 1000
    return {
        "type": "imports",
        "format": "mermaid",
        "mermaid": "\n".join(lines),
        "node_count": len(kept_nodes),
        "query_ms": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Diagram: inheritance (class hierarchy)
# ---------------------------------------------------------------------------


async def _diagram_inheritance(graph: GraphBackend, project: str, path: str, max_nodes: int) -> dict[str, Any]:
    t0 = time.monotonic()
    records = await graph.get_diagram_inheritance(project, path, max_nodes)

    if not records:
        elapsed = (time.monotonic() - t0) * 1000
        mermaid = 'classDiagram\n    class Empty\n    note "No inheritance found"'
        return {"type": "inheritance", "mermaid": mermaid, "node_count": 0, "query_ms": round(elapsed, 1)}

    lines = ["classDiagram"]
    nodes: set[str] = set()
    for r in records:
        parent_id = _sid(r["parent_qn"])
        child_id = _sid(r["child_qn"])
        if parent_id not in nodes:
            lines.append(f'    class {parent_id}["{_slabel(r["parent_name"])}"]')
            nodes.add(parent_id)
        if child_id not in nodes:
            kind_label = r["child_kind"] or "class"
            lines.append(f'    class {child_id}["{_slabel(r["child_name"])}"]')
            lines.append(f"    <<{kind_label}>> {child_id}")
            nodes.add(child_id)
        lines.append(f"    {parent_id} <|-- {child_id}")

    elapsed = (time.monotonic() - t0) * 1000
    return {
        "type": "inheritance",
        "mermaid": "\n".join(lines),
        "node_count": len(nodes),
        "query_ms": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Diagram: module_detail (single module's classes + methods)
# ---------------------------------------------------------------------------


def _module_detail_entity_lines(
    e: dict[str, Any],
    eid: str,
    class_methods: dict[str, list[dict[str, Any]]],
    vis_prefix: dict[str, str],
) -> list[str]:
    """Render one entity's Mermaid classDiagram lines (class declaration + members)."""
    lines = [f'    class {eid}["{_slabel(e["name"])}"]']
    if e["label"] == "TypeDef":
        lines.append(f"    <<{e['kind'] or 'class'}>> {eid}")
        for meth in class_methods.get(e["qn"], []):
            prefix = vis_prefix.get(meth["vis"] or "public", "+")
            lines.append(f"    {eid} : {prefix}{meth['name']}()")
    elif e["label"] == "Callable":
        lines.append(f"    <<{e['kind'] or 'function'}>> {eid}")
    elif e["label"] == "Value":
        lines.append(f"    <<{e['kind'] or 'value'}>> {eid}")
    return lines


async def _diagram_module_detail(graph: GraphBackend, project: str, path: str, max_nodes: int) -> dict[str, Any]:
    t0 = time.monotonic()

    if not path:
        return {
            "error": "path parameter required for module_detail diagram (file path prefix of the module)",
            "code": "PATH_REQUIRED",
        }

    detail = await graph.get_diagram_module_detail(project, path, max_nodes)
    if detail is None:
        return {"error": f"No module found matching path '{path}'", "code": "NOT_FOUND"}

    mod = detail["module"]
    entities = detail["entities"]
    methods = detail["methods"]
    inherits = detail["inherits"]

    if not entities:
        elapsed = (time.monotonic() - t0) * 1000
        return {
            "type": "module_detail",
            "module": mod["qn"],
            "mermaid": f'classDiagram\n    note "Module {_slabel(mod["qn"])} has no entities"',
            "node_count": 0,
            "query_ms": round(elapsed, 1),
        }

    # Build method lookup: class_qn → [methods]
    class_methods: dict[str, list[dict[str, Any]]] = {}
    for m in methods:
        class_methods.setdefault(m["class_qn"], []).append(m)

    lines = ["classDiagram"]
    nodes: set[str] = set()
    vis_prefix = {"public": "+", "private": "-", "protected": "#", "internal": "~"}

    for e in entities:
        eid = _sid(e["qn"])
        if eid in nodes:
            continue
        nodes.add(eid)
        lines.extend(_module_detail_entity_lines(e, eid, class_methods, vis_prefix))

    # Add inheritance edges. Skip children truncated out of the declared
    # node set by the max_nodes cap on entities — otherwise Mermaid silently
    # renders an unlabeled node for the dangling reference.
    for inh in inherits:
        child_id = _sid(inh["child_qn"])
        if child_id not in nodes:
            continue
        parent_id = _sid(inh["parent_qn"])
        if parent_id not in nodes:
            lines.append(f'    class {parent_id}["{_slabel(inh["parent_name"])}"]')
            nodes.add(parent_id)
        lines.append(f"    {parent_id} <|-- {child_id}")

    elapsed = (time.monotonic() - t0) * 1000
    return {
        "type": "module_detail",
        "module": mod["qn"],
        "mermaid": "\n".join(lines),
        "node_count": len(nodes),
        "query_ms": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Module summary — token-dense whole-scope skeleton (analyze_repo sub-case)
# ---------------------------------------------------------------------------
#
# The competing shape is "the agent just reads the files". So the output is a
# rendered text skeleton, not a JSON record set: repeating {"qualified_name":
# ..., "signature": ..., "docstring": ...} keys for every entity costs more
# tokens than the information they label. Precedent for a rendered string in
# this module is generate_diagram's `mermaid`.

# analyze_repo's shared `limit` means "max items per sub-section" and is clamped
# to 100 by the tool layer — far too small to budget a whole package's entities.
# module_summary scales it instead of overloading it: `limit` * these factors is
# the real cap. Default limit 20 -> 200 entities, 600 edges per boundary list.
_MODULE_SUMMARY_ENTITY_FACTOR = 10
_MODULE_SUMMARY_EDGE_FACTOR = 3

# Detail tiers, widest-scope-first. The tier is chosen by the size of the rendered
# result, never by the shape of `path`: one module can hold 316 entities while a
# package holds 12, so selecting on "is this a package?" gets it backwards in both
# directions. Strictly nested (T0 ⊂ T1 ⊂ T2) so drilling down is additive.
_TIER_MAP = "T0"
_TIER_SKELETON = "T1"
_TIER_DETAIL = "T2"
_MODULE_SUMMARY_TIERS = (_TIER_DETAIL, _TIER_SKELETON, _TIER_MAP)

_MODULE_SUMMARY_VIS = {"public": "+", "private": "-", "protected": "#", "internal": "~"}
_MODULE_SUMMARY_FALLBACK_KIND = {"TypeDef": "class", "Callable": "def", "Value": "var"}
_MODULE_SUMMARY_EXTERNAL_LABELS = frozenset({"ExternalPackage", "ExternalSymbol"})
_MODULE_SUMMARY_SIG_MAX = 160
_MODULE_SUMMARY_DOC_MAX = 100
# Hard ceiling on the rendered outline. ~60k chars is roughly 15k tokens: large enough
# that a normal package is never cut, small enough that a pathological one cannot blow
# the context this tool exists to conserve.
_MODULE_SUMMARY_OUTLINE_MAX = 60_000
_MODULE_SUMMARY_LEGEND = (
    "LEGEND visibility marker (-private #protected ~internal) only when the name or signature "
    "does not already say it | L<start>[-<end> when >=20 lines] | "
    "'# ' first docstring line | a > b: a uses b | a < b: a used by b | 'ext/' outside this project | "
    "[k=v] non-default edge props"
)
# T1 and T0 render different bodies, so they get different legends — an entry describing
# a form that does not appear is the same failure as a count that disagrees with the text.
_MODULE_SUMMARY_LEGEND_SKELETON = (
    "LEGEND public entities only, no signatures | L<start> | '# ' first docstring line | "
    "edges are module-level with (count): a > b: a uses b | a < b: a used by b"
)
_MODULE_SUMMARY_LEGEND_MAP = (
    "LEGEND modules only, no entities | 'name (N)': N entities in scope | '# ' first docstring line | "
    "edges are module-level with (count): a > b: a uses b | a < b: a used by b"
)

_WHITESPACE_RUN = re.compile(r"\s+")
_RST_LITERAL = re.compile(r"``([^`]+)``")

# Above this many ambiguous edges from one source, list the count instead of the targets.
# Small sets are usually right and worth reading; large ones are a name collision fanning
# out across the project and say nothing.
_ADJACENCY_AMBIGUOUS_MAX = 3

_STDLIB_MODULES = frozenset(sys.stdlib_module_names)
_EXTERNAL_PREFIX = "ext/"
_STDLIB_PREFIX = "std/"


def _mark_external(target: str) -> str:
    """Re-prefix a stdlib import ``std/`` so it is distinguishable from a real dependency.

    ``ext/`` means only "outside this project", so ``ext/hashlib`` and ``ext/litellm``
    render identically and the outline cannot answer "what does this depend on".

    Python's stdlib list is the only one available in-process, so this is a Python-only
    refinement; anything unrecognised keeps ``ext/`` rather than being guessed at. A
    package named after a Python stdlib module in another language would be mislabelled
    — the honest fix is recording the language on the node at parse time, which needs a
    re-index (see the spec for the wider external-dependency work).
    """
    if not target.startswith(_EXTERNAL_PREFIX):
        return target
    root = target[len(_EXTERNAL_PREFIX) :].split(".", 1)[0]
    return _STDLIB_PREFIX + target[len(_EXTERNAL_PREFIX) :] if root in _STDLIB_MODULES else target


# Below this many lines a span's end adds nothing an agent acts on — it can read the
# entity in one go either way. " L40-67" costs 4 tokens against " L40"'s 2, so the end
# is spent only where the entity is big enough for its size to be the point.
_MODULE_SUMMARY_SPAN_RANGE_MIN = 20


def _first_doc_line(text: str | None, max_len: int = _MODULE_SUMMARY_DOC_MAX) -> str:
    """First non-empty docstring line, whitespace-collapsed and truncated.

    First line only is deliberate: it is the summary sentence by convention in
    every docstring style this indexer sees, and full docstrings would dominate
    the outline's token budget.

    RST inline literals are unwrapped because the backticks are pure markup here
    — nothing renders this string — and they tokenize as their own tokens.
    """
    if not text:
        return ""
    lines = text.splitlines()
    for i, raw in enumerate(lines):
        line = _WHITESPACE_RUN.sub(" ", raw).strip()
        if not line:
            continue
        line = _RST_LITERAL.sub(r"\1", line)
        if len(line) > max_len:
            return line[: max_len - 3] + "..."
        # A first line that wraps mid-sentence reads as mangled rather than clipped —
        # "...on transient provider errors (rate limits," with nothing to say more was
        # dropped. Signature truncation already marks itself; match it.
        return line + "..." if any(ln.strip() for ln in lines[i + 1 :]) else line
    return ""


def _compact_signature(sig: str | None) -> str:
    """Collapse a stored (possibly multi-line) signature onto one bounded line."""
    if not sig:
        return ""
    text = _WHITESPACE_RUN.sub(" ", sig).strip()
    return text if len(text) <= _MODULE_SUMMARY_SIG_MAX else text[: _MODULE_SUMMARY_SIG_MAX - 3] + "..."


def _common_dotted_prefix(qns: list[str]) -> str:
    """Longest shared dotted namespace of *qns* (``""`` when there is none)."""
    if not qns:
        return ""
    parts = qns[0].split(".")
    for qn in qns[1:]:
        other = qn.split(".")
        keep = 0
        for a, b in zip(parts, other, strict=False):
            if a != b:
                break
            keep += 1
        parts = parts[:keep]
        if not parts:
            return ""
    return ".".join(parts)


def _rel_name(qn: str | None, prefix: str) -> str:
    """Strip the scope's shared dotted prefix — the single biggest token win.

    Qualified names in a code graph are long and share almost all of their
    text within one package; the prefix is stated once in the outline header
    instead of on every line.
    """
    if not qn:
        return "?"
    if prefix and qn.startswith(prefix + "."):
        return qn[len(prefix) + 1 :]
    return qn


def _line_span(line_start: Any, line_end: Any) -> str:
    if line_start is None:
        return ""
    if line_end is None or line_end == line_start:
        return f"L{line_start}"
    if line_end - line_start < _MODULE_SUMMARY_SPAN_RANGE_MIN:
        return f"L{line_start}"
    return f"L{line_start}-{line_end}"


def _quiet_edge_prop(value: Any) -> bool:
    """Whether an edge property value is the neutral one and not worth tokens.

    Value-based, not key-based, on purpose: any property whose value is absent/
    empty, false, 1 (the neutral weight/count), or ``"resolved"`` is dropped and
    everything else is rendered. New CALLS edge properties (ADR-0014's
    confidence/strategy and anything added later) therefore surface in the
    outline without this module knowing their names.
    """
    if value is None or value == "":
        return True
    if isinstance(value, bool):
        return not value
    if isinstance(value, int | float):
        return value == 1
    return value == "resolved"


def _edge_annotation(props: dict[str, Any] | None) -> str:
    """``[k=v k2=v2]`` for the informative subset of an edge's properties."""
    if not props:
        return ""
    # `strategy` is never neutral by value (it is always a non-empty, non-"resolved"
    # string), so the value-based rule alone renders it on every CALLS edge. How a
    # confidently-resolved edge was resolved is not worth tokens here — keep it only
    # when the edge is not confidently resolved, which is when it explains something.
    quiet_keys = {"strategy"} if props.get("confidence", "resolved") == "resolved" else set()
    parts = [
        f"{k}={round(v, 3) if isinstance(v, float) else v}"
        for k, v in sorted(props.items())
        if k not in quiet_keys and not _quiet_edge_prop(v)
    ]
    return f"[{' '.join(parts)}]" if parts else ""


def _top_callers(fan_in: list[dict[str, Any]], limit: int = 5) -> list[str]:
    """One line ranking the outside FILES that reach into this scope.

    "Who uses this most" is the first question a reader brings to a boundary section and
    the one it could not answer: the counts were spread across dozens of comma-separated
    rows, so both blind readers had to tally 123 names by hand to get there.
    """
    counts = Counter(str(r.get("from_path") or r.get("from_qn") or "?") for r in fan_in)
    if len(counts) < 2:
        return []
    top = ", ".join(f"{path}({n})" for path, n in counts.most_common(limit))
    more = f", +{len(counts) - limit} more" if len(counts) > limit else ""
    return [f"  TOP CALLERS by file: {top}{more}"]


def _adjacency_lines(
    rows: list[dict[str, Any]],
    prefix: str,
    src_key: str,
    dst_key: str,
    arrow: str,
) -> list[str]:
    """Collapse edge rows to one line per (rel_type, source): ``src > a, b, c``.

    One row per edge would repeat the source name once per neighbour; grouping
    pays for it once.
    """
    # Per (rel_type, src): confidently-resolved targets, and ambiguous ones held apart.
    firm: dict[str, dict[str, set[str]]] = {}
    vague: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        src = _rel_name(row[src_key], prefix)
        props = row.get("props") or {}
        # No external marker: every external node's qualified name already begins
        # "ext/" (verified across all 585 in the index), so a trailing * was exactly
        # redundant with the prefix. A blind reader of this format called it out as
        # carrying no information.
        target = _mark_external(_rel_name(row[dst_key], prefix))
        bucket = vague if props.get("confidence") == "ambiguous" else firm
        bucket.setdefault(row["rel_type"], {}).setdefault(src, set()).add(target + _edge_annotation(props))

    lines: list[str] = []
    for rel_type in sorted(set(firm) | set(vague)):
        lines.append(f"  {rel_type}")
        srcs = sorted(set(firm.get(rel_type, {})) | set(vague.get(rel_type, {})))
        for src in srcs:
            shown = sorted(firm.get(rel_type, {}).get(src, set()))
            amb = vague.get(rel_type, {}).get(src, set())
            # Enumerating ambiguous edges is what makes these rows unusable: one entity
            # in this index has 295 ambiguous callers — that is every same-named method
            # in the project, not a fact about the code. Small sets still list, since
            # they are often right; past the threshold only the count is honest.
            if len(amb) > _ADJACENCY_AMBIGUOUS_MAX:
                shown.append(f"+{len(amb)} ambiguous (use blast_radius)")
            else:
                # Below the threshold they are enumerated in full, annotations included:
                # candidate_count and strategy are what make an ambiguous edge auditable.
                shown.extend(sorted(amb))
            if shown:
                lines.append(f"    {src} {arrow} {', '.join(shown)}")
    return lines


def _visibility_is_evident(vis: str, name: str, rendered: str) -> bool:
    """Whether the rendered line already tells you the visibility without a marker.

    A marker that repeats what the line already says is a second notation channel to
    keep in sync, and the ``#`` chosen for *protected* collides with the ``#`` that
    introduces a docstring. Emitting one only where it adds information removes the
    collision wherever the language expresses visibility itself:

    - Python and friends put it in the name — every one of this index's 1,580 non-public
      entities is recoverable from a leading underscore.
    - Java, C#, C++ and PHP put the keyword in the signature.

    What is left is field-like entities in keyword languages, which store no signature.
    Those lines have no signature to host a stray hash, so the collision cannot arise.
    """
    if vis == "public":
        return True
    if name.startswith("_"):
        return True
    return vis.lower() in rendered.lower()


def _tier_entities(entities: list[dict[str, Any]], tier: str) -> list[dict[str, Any]]:
    """The entity subset a tier renders. Strictly nested: T0 ⊂ T1 ⊂ T2.

    Nesting is the point — drilling down is additive, so an agent never has to
    re-read what it already holds.
    """
    if tier == _TIER_DETAIL:
        return entities
    if tier == _TIER_MAP:
        return []
    # T1 drops Value (42% of entities here, and a module constant is rarely what
    # someone widening their scope is looking for) and non-public members.
    return [e for e in entities if e["label"] != "Value" and (e["vis"] or "public") == "public"]


def _skeleton_lines(modules: list[dict[str, Any]], entities: list[dict[str, Any]], tier: str) -> list[str]:
    """Per-file entity outline: signature, visibility, line span, first doc line.

    Members of an in-scope TypeDef are indented under it (the DEFINES parent),
    so the containment structure is carried by layout instead of an edge list.
    The 2-space indent costs exactly one token per entity (measured) and is the
    cheapest encoding of that containment; deeper nesting is free.

    At T1 the signature is dropped and only ``kind name`` is kept — signatures are
    36% of the rendered outline, by far the largest single component, and the
    first docstring line carries more meaning per token than the parameter list.
    """
    mod_by_path = {m["file_path"]: m for m in modules}
    typedef_qns = {e["qn"] for e in entities if e["label"] == "TypeDef"}
    by_path: dict[str, list[dict[str, Any]]] = {}
    for e in entities:
        by_path.setdefault(e["file_path"] or "", []).append(e)

    def parent_of(e: dict[str, Any]) -> str | None:
        """Owning TypeDef, falling back to the qualified name.

        Dataclass fields and enum members arrive with ``parent_qn`` unset — they have
        no DEFINES edge, unlike methods — so indenting on that alone left ~50 of this
        scope's entities looking like module-level globals. Their qualified name still
        carries the owner, so derive it rather than requiring a re-index.
        """
        if e["parent_qn"]:
            return str(e["parent_qn"])
        qn = e["qn"] or ""
        head = qn.rsplit(".", 1)[0] if "." in qn else ""
        return head if head in typedef_qns else None

    lines: list[str] = []
    for file_path in sorted(set(mod_by_path) | set(by_path)):
        mod = mod_by_path.get(file_path)
        # Both the qualified name and the path, despite the redundancy: dropping the
        # path measured at only -399 tokens (0.6%) and a dotted name does not tell you
        # where the file is — nothing in `code_atlas.parsing.ast` implies `src/`.
        lines.append("")
        lines.append(f"{mod['qn']} ({file_path})" if mod else f"({file_path})")
        mod_doc = _first_doc_line(mod.get("docstring")) if mod else ""
        if mod_doc:
            lines.append(f" # {mod_doc}")
        for e in by_path.get(file_path, []):
            indent = "    " if parent_of(e) in typedef_qns else "  "
            vis = e["vis"] or "public"
            kind_name = f"{e['kind'] or _MODULE_SUMMARY_FALLBACK_KIND.get(e['label'], '')} {e['name']}".strip()
            header = (_compact_signature(e["sig"]) or kind_name) if tier == _TIER_DETAIL else kind_name
            marker = "" if _visibility_is_evident(vis, e["name"], header) else _MODULE_SUMMARY_VIS.get(vis, "") + " "
            span = _line_span(e["line_start"], e["line_end"])
            doc = _first_doc_line(e["docstring"])
            lines.append(f"{indent}{marker}{header}{' ' + span if span else ''}{' # ' + doc if doc else ''}")
    return lines


def _map_lines(modules: list[dict[str, Any]], entities: list[dict[str, Any]]) -> list[str]:
    """T0: one line per module — name, entity count, first docstring line.

    No entities at all. At repo scope this is the only tier that fits, and a
    complete list of modules beats a detailed view of an arbitrary 40% of them.
    """
    counts: dict[str, int] = {}
    for e in entities:
        counts[e["file_path"] or ""] = counts.get(e["file_path"] or "", 0) + 1
    lines: list[str] = [""]
    for m in sorted(modules, key=lambda m: m["file_path"] or ""):
        doc = _first_doc_line(m.get("docstring"))
        n = counts.get(m["file_path"], 0)
        lines.append(f"{m['qn']} ({n}){' # ' + doc if doc else ''}")
    return lines


def _doc_link_lines(docs: list[dict[str, Any]], prefix: str) -> list[str]:
    """``entity < note(link_type), ...`` for inbound DOCUMENTS edges."""
    grouped: dict[str, set[str]] = {}
    for d in docs:
        label = d["doc_qn"] or d["doc_name"] or "?"
        link = d.get("link_type")
        grouped.setdefault(_rel_name(d["to_qn"], prefix), set()).add(f"{label}({link})" if link else str(label))
    return [f"  {target} < {', '.join(sorted(refs))}" for target, refs in sorted(grouped.items())]


@dataclass(frozen=True)
class _OutlineInputs:
    """Everything ``_render_outline`` needs, so the tier cascade can re-render
    from memory without re-querying."""

    path: str
    modules: list[dict[str, Any]]
    entities: list[dict[str, Any]]
    internal_edges: list[dict[str, Any]]
    fan_in: list[dict[str, Any]]
    fan_out: list[dict[str, Any]]
    docs: list[dict[str, Any]]
    prefix: str
    qn_to_module: dict[str, str]
    truncated: bool


def _render_outline(src: _OutlineInputs, tier: str) -> str:
    """Render the whole outline at one detail tier."""
    shown = _tier_entities(src.entities, tier)
    # "N of M" whenever the tier is showing less than everything: the outline is what the
    # agent reads, so the shortfall has to be visible there, not only in the JSON field.
    n_shown, n_total = len(shown), len(src.entities)
    counted = f"{n_shown} entities" if n_shown == n_total else f"{n_shown} of {n_total} entities"
    # Count the file blocks actually rendered, not the Module rows. A package __init__ is
    # a Package node with no Module row, so "3 module(s)" sat above four file blocks —
    # every blind reader of this format flagged the contradiction.
    n_files = len({m["file_path"] for m in src.modules} | {e["file_path"] or "" for e in src.entities})
    header = f"SCOPE {src.path} | {n_files} file(s) | {counted} | DETAIL {tier}"
    lines = [header + ("  |  TRUNCATED (raise limit for more)" if src.truncated else "")]
    if src.prefix:
        lines.append(f"NAMES below are relative to {src.prefix} unless fully qualified")
    legends = {
        _TIER_DETAIL: _MODULE_SUMMARY_LEGEND,
        _TIER_SKELETON: _MODULE_SUMMARY_LEGEND_SKELETON,
        _TIER_MAP: _MODULE_SUMMARY_LEGEND_MAP,
    }
    lines.append(legends[tier])
    lines.extend(
        _map_lines(src.modules, src.entities) if tier == _TIER_MAP else _skeleton_lines(src.modules, shown, tier)
    )

    p, q = src.prefix, src.qn_to_module
    if tier == _TIER_DETAIL:
        internal = _adjacency_lines(src.internal_edges, p, "from_qn", "to_qn", ">")
        inbound = _adjacency_lines(src.fan_in, p, "to_qn", "from_qn", "<")
        outbound = _adjacency_lines(src.fan_out, p, "from_qn", "to_qn", ">")
    else:
        internal = _aggregated_boundary_lines(src.internal_edges, p, "from_qn", "", "to_qn", ">", q)
        inbound = _aggregated_boundary_lines(src.fan_in, p, "to_qn", "from_path", "from_qn", "<", q)
        outbound = _aggregated_boundary_lines(src.fan_out, p, "from_qn", "to_path", "to_qn", ">", q)
    if internal:
        lines.extend(["", f"EDGES within scope ({len(src.internal_edges)})", *internal])
    if inbound:
        lines.extend(
            ["", f"FAN-IN — callers/importers outside this scope ({len(src.fan_in)})", *_top_callers(src.fan_in)]
        )
        lines.extend(inbound)
    if outbound:
        lines.extend(["", f"FAN-OUT — what this scope depends on ({len(src.fan_out)})", *outbound])
    doc_lines = _doc_link_lines(src.docs, p)
    if doc_lines:
        lines.extend(["", f"DOCS — linked notes/docs ({len(src.docs)})", *doc_lines])
    return "\n".join(lines)


def _aggregated_boundary_lines(
    rows: list[dict[str, Any]],
    prefix: str,
    scope_key: str,
    other_path_key: str,
    other_qn_key: str,
    arrow: str,
    qn_to_module: dict[str, str],
) -> list[str]:
    """Boundary edges collapsed to module granularity with per-target counts.

    Enumerating the boundary per entity is what actually blows the budget: measured
    on ``src/code_atlas``, FAN-IN alone is 58.5% of the rendered outline against the
    entity skeleton's 19%. Below T2 the question a boundary answers is "which modules
    reach into this scope", not "which of 3839 individual edges" — so both endpoints
    collapse to their module and identical pairs become a count.
    """

    def group_of(qn: str, path: str) -> str:
        """Module for an in-scope endpoint; the file path for one outside the scope,
        whose module qualified name this query never returns."""
        known = qn_to_module.get(qn)
        if known:
            return _rel_name(known, prefix)
        return path or _rel_name(qn, prefix) or "?"

    grouped: dict[str, Counter[str]] = {}
    for r in rows:
        scope = group_of(r.get(scope_key) or "", "")
        other = group_of(r.get(other_qn_key) or "", str(r.get(other_path_key) or "") if other_path_key else "")
        if other == scope:
            continue  # collapsed to a self-edge by the aggregation; says nothing
        grouped.setdefault(scope, Counter())[other] += 1
    return [
        f"  {mod} {arrow} {', '.join(f'{o}({n})' if n > 1 else o for o, n in sorted(others.items()))}"
        for mod, others in sorted(grouped.items())
    ]


def _dedupe_entities(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop duplicate rows produced by the OPTIONAL MATCH / LEFT JOIN on DEFINES."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        uid = row["uid"]
        if uid in seen:
            continue
        seen.add(uid)
        out.append(row)
    return out


async def _analyze_module_summary(
    graph: GraphBackend, project: str, path: str, limit: int, test_patterns: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Token-dense skeleton of everything under *path*, plus its scope boundary.

    Emits, as one rendered ``outline`` string: every in-scope entity's
    signature/visibility/line-span/first-docstring-line grouped by file with
    class members indented under their class; the intra-scope CALLS/INHERITS/
    IMPLEMENTS/USES_TYPE/OVERRIDES adjacency; the boundary (``FAN-IN`` — who
    outside calls in, ``FAN-OUT`` — what this scope depends on, including
    external packages); and inbound DOCUMENTS links. No entity bodies, no full
    docstrings — that is the whole compression argument versus reading the
    files.

    *path* is required. Fan-in/fan-out are defined by "exactly one endpoint has
    a file_path under *path*", so with no path everything is in scope and the
    boundary — the most valuable part — would be empty by construction.

    *test_patterns* filtering applies to in-scope entities as well as the
    boundary, **unless *path* itself names a test location** — then the caller
    plainly wants the tests and filtering them would return nothing. Test bodies
    are detail nobody widening their scope asked for; the earlier rule (boundary
    only, because "the caller named *path* explicitly") held that against them
    even at repo scope, where it is exactly the noise the tier system exists to
    shed.

    Detail is chosen by rendering: T2 first, and if the result exceeds the size
    ceiling, again at T1, then T0. A complete outline at a coarser tier beats a
    detailed one cut off mid-section — the previous behaviour truncated the
    joined string from the end, which silently deleted EDGES/FAN-IN/FAN-OUT
    while leaving their counts in the response claiming otherwise.
    """
    t0 = time.monotonic()
    if not path:
        return {
            "analysis": "module_summary",
            "project": project,
            "error": "path parameter required for module_summary (a file or package path prefix)",
            "code": "PATH_REQUIRED",
        }

    entity_limit = limit * _MODULE_SUMMARY_ENTITY_FACTOR
    edge_limit = entity_limit * _MODULE_SUMMARY_EDGE_FACTOR
    raw = await graph.get_module_summary(project, path, entity_limit, edge_limit)

    modules = raw["modules"]
    entities = _dedupe_entities(raw["entities"])
    fan_in, fan_out = raw["fan_in"], raw["fan_out"]
    internal_edges = raw["internal_edges"]
    scope_is_tests = bool(test_patterns) and matches_test_pattern(path, "", list(test_patterns))
    if test_patterns:
        patterns = list(test_patterns)
        fan_in = [r for r in fan_in if not matches_test_pattern(r["from_path"] or "", r["from_name"] or "", patterns)]
        fan_out = [r for r in fan_out if not matches_test_pattern(r["to_path"] or "", r["to_name"] or "", patterns)]
        if not scope_is_tests:
            modules = [m for m in modules if not matches_test_pattern(m["file_path"] or "", "", patterns)]
            entities = [e for e in entities if not matches_test_pattern(e["file_path"] or "", "", patterns)]
            kept_qns = {e["qn"] for e in entities}
            internal_edges = [r for r in internal_edges if r["from_qn"] in kept_qns and r["to_qn"] in kept_qns]

    if not modules and not entities:
        return {
            "analysis": "module_summary",
            "project": project,
            "path": path,
            "error": f"No indexed modules or entities found under path '{path}'",
            "code": "NOT_FOUND",
        }

    prefix = _common_dotted_prefix([m["qn"] for m in modules if m["qn"]])
    # Unlike blast_radius/dead_code, these lists were cut by the QUERY's own cap, so the
    # true total was never fetched and cannot be reported. Naming which caps were hit is
    # the honest equivalent — a fabricated `cut` count would be worse than the bare
    # boolean it replaces.
    caps_hit = [
        name
        for name, hit in (
            ("entities", len(raw["entities"]) >= entity_limit),
            ("internal_edges", len(raw["internal_edges"]) >= edge_limit),
            ("fan_in", len(raw["fan_in"]) >= edge_limit),
            ("fan_out", len(raw["fan_out"]) >= edge_limit),
        )
        if hit
    ]
    truncated = bool(caps_hit)
    # An entity in a package __init__ has no Module row (those are Package nodes), so
    # fall back to its file path rather than to its own qualified name — the latter
    # would leave it ungrouped and defeat the aggregation.
    mod_qn_by_path = {m["file_path"]: m["qn"] for m in modules}
    qn_to_module = {e["qn"]: mod_qn_by_path.get(e["file_path"]) or e["file_path"] or e["qn"] for e in entities}
    qn_to_module.update({m["qn"]: m["qn"] for m in modules})

    src = _OutlineInputs(
        path=path,
        modules=modules,
        entities=entities,
        internal_edges=internal_edges,
        fan_in=fan_in,
        fan_out=fan_out,
        docs=raw["docs"],
        prefix=prefix,
        qn_to_module=qn_to_module,
        truncated=truncated,
    )
    # Drop a tier at a time until the whole thing fits. Rendering is pure string work
    # over rows already in memory — no extra query — so at most three passes.
    outline = ""
    tier = _TIER_MAP
    for candidate in _MODULE_SUMMARY_TIERS:
        tier, outline = candidate, _render_outline(src, candidate)
        if len(outline) <= _MODULE_SUMMARY_OUTLINE_MAX:
            break
    # Only reachable if even T0 overflows — thousands of modules under one path.
    size_capped = len(outline) > _MODULE_SUMMARY_OUTLINE_MAX
    if size_capped:
        keep = outline[:_MODULE_SUMMARY_OUTLINE_MAX].rsplit("\n", 1)[0]
        outline = keep + "\n... OUTLINE TRUNCATED (size cap) — narrow `path` for a complete view"

    elapsed = (time.monotonic() - t0) * 1000
    return {
        "analysis": "module_summary",
        "project": project,
        "path": path,
        "modules": [m["qn"] for m in modules],
        # Both, deliberately: a lone count that disagrees with the outline is how the
        # old size cap misled callers into thinking they had the whole picture.
        "entity_count": len(entities),
        "entities_rendered": len(_tier_entities(entities, tier)),
        "internal_edge_count": len(internal_edges),
        "fan_in_count": len(fan_in),
        "fan_out_count": len(fan_out),
        "detail_tier": tier,
        "detail_tiers": {
            _TIER_DETAIL: "every entity, full signature, per-entity boundary",
            _TIER_SKELETON: "public non-Value entities, no signatures, module-level boundary",
            _TIER_MAP: "modules only",
        },
        "next_step": (
            ""
            if tier == _TIER_DETAIL
            else f"Detail was reduced to {tier} to fit. Narrow `path` to a sub-package or single file for more."
        ),
        "truncated": (
            {
                "caps_hit": caps_hit,
                "size_capped": size_capped,
                "remedy": (
                    "narrow `path` to a sub-package or single file. The listed collections hit their "
                    "query cap, so their true totals are HIGHER than the counts reported above."
                ),
            }
            if truncated or size_capped
            else False
        ),
        "outline": outline,
        "query_ms": round(elapsed, 1),
    }
