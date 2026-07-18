"""Repository analysis and diagram generation for Code Atlas MCP server.

Pure Cypher queries + Python formatting — no LLM calls, no file reads,
no new dependencies.
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import TYPE_CHECKING, Any

from code_atlas.schema import RelType
from code_atlas.search.engine import matches_test_pattern

if TYPE_CHECKING:
    from code_atlas.graph.client import GraphClient

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
    }
)
_VALID_DIAGRAM_TYPES = frozenset({"packages", "imports", "inheritance", "module_detail"})

# trace_path / blast_radius (information-retrieval family, see ADR-0013) default
# edge sets — trace_path follows any structural/call relationship, blast_radius
# is specifically about call impact so it defaults to CALLS only.
_DEFAULT_TRACE_EDGE_TYPES = ("CALLS", "IMPORTS", "USES_TYPE")
_DEFAULT_BLAST_EDGE_TYPES = ("CALLS",)

# Quality analysis thresholds (v1 — hardcoded for medium Python projects)
_GOD_MODULE_ENTITY_THRESHOLD = 30
_TANGLED_FAN_THRESHOLD = 8
_INSTABILITY_LOW = 0.1
_INSTABILITY_HIGH = 0.9

# Communities noise threshold — singleton/near-singleton clusters aren't
# actionable groupings, drop them before ranking by size.
_COMMUNITY_NOISE_THRESHOLD = 2

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


# ---------------------------------------------------------------------------
# Public dispatchers
# ---------------------------------------------------------------------------


async def analyze_repo(
    graph: GraphClient,
    analysis: str,
    project: str,
    path: str = "",
    limit: int = 20,
    test_patterns: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Dispatch to the requested sub-analysis.

    *test_patterns* excludes test modules from quality scoring (god-module,
    circular, tangled, rigid, unstable) — unused by the other sub-analyses, so
    only the ``quality`` dispatch receives it.
    """
    if analysis not in _VALID_ANALYSES:
        return {
            "error": f"Unknown analysis '{analysis}'. Valid: {sorted(_VALID_ANALYSES)}",
            "code": "INVALID_ANALYSIS",
        }
    if analysis == "quality":
        return await _analyze_quality(graph, project, path, limit, test_patterns)
    if analysis == "dead_code":
        return await _analyze_dead_code(graph, project, path, limit, test_patterns)
    dispatch = {
        "structure": _analyze_structure,
        "centrality": _analyze_centrality,
        "dependencies": _analyze_dependencies,
        "patterns": _analyze_patterns,
        "complexity": _analyze_complexity,
        "communities": _analyze_communities,
        "git_signals": _analyze_git_signals,
    }
    return await dispatch[analysis](graph, project, path, limit)


async def generate_diagram(
    graph: GraphClient,
    diagram_type: str,
    project: str,
    path: str = "",
    max_nodes: int = 30,
) -> dict[str, Any]:
    """Dispatch to the requested diagram generator."""
    if diagram_type not in _VALID_DIAGRAM_TYPES:
        return {
            "error": f"Unknown diagram type '{diagram_type}'. Valid: {sorted(_VALID_DIAGRAM_TYPES)}",
            "code": "INVALID_DIAGRAM_TYPE",
        }
    dispatch = {
        "packages": _diagram_packages,
        "imports": _diagram_imports,
        "inheritance": _diagram_inheritance,
        "module_detail": _diagram_module_detail,
    }
    return await dispatch[diagram_type](graph, project, path, max_nodes)


# ---------------------------------------------------------------------------
# trace_path / blast_radius (information-retrieval family, ADR-0013)
# ---------------------------------------------------------------------------


def _format_path_hops(path_nodes: list[Any], path_rels: list[Any]) -> list[dict[str, Any]]:
    """Render a Cypher path's nodes/relationships into per-hop dicts.

    Includes CALLS ``confidence``/``strategy`` edge properties (ADR-0014) when
    present on the traversed edge.
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
        hops.append(hop)
    return hops


async def trace_path(
    graph: GraphClient,
    from_uid: str,
    to_uid: str,
    max_depth: int = 6,
    edge_types: tuple[str, ...] = _DEFAULT_TRACE_EDGE_TYPES,
) -> dict[str, Any]:
    """Find the shortest path between two entities, bounded by ``max_depth`` hops.

    Traverses *edge_types* (default CALLS|IMPORTS|USES_TYPE). Returns the
    hop-by-hop path — edge type, endpoint uid/name, and CALLS confidence/
    strategy when present (ADR-0014) — or a ``found: false`` result when no
    path exists within ``max_depth``.
    """
    t0 = time.monotonic()
    params: dict[str, Any] = {"from_uid": from_uid, "to_uid": to_uid}

    exist_raw = await graph.execute(
        "OPTIONAL MATCH (a {uid: $from_uid}) OPTIONAL MATCH (b {uid: $to_uid}) "
        "RETURN a IS NOT NULL AS from_exists, b IS NOT NULL AS to_exists",
        params,
    )
    exists = exist_raw[0] if exist_raw else {"from_exists": False, "to_exists": False}
    if not exists["from_exists"]:
        return {"error": f"Node not found: {from_uid}", "code": "NOT_FOUND"}
    if not exists["to_exists"]:
        return {"error": f"Node not found: {to_uid}", "code": "NOT_FOUND"}

    rel_pattern = "|".join(edge_types)
    records = await graph.execute(
        f"MATCH p=(a {{uid: $from_uid}})-[:{rel_pattern}*1..{max_depth}]->(b {{uid: $to_uid}}) "
        "RETURN nodes(p) AS path_nodes, relationships(p) AS path_rels, length(p) AS hops "
        "ORDER BY hops LIMIT 1",
        params,
    )
    elapsed = (time.monotonic() - t0) * 1000

    if not records:
        return {
            "found": False,
            "from_uid": from_uid,
            "to_uid": to_uid,
            "max_depth": max_depth,
            "message": f"No path found within {max_depth} hops",
            "query_ms": round(elapsed, 1),
        }

    record = records[0]
    return {
        "found": True,
        "from_uid": from_uid,
        "to_uid": to_uid,
        "hop_count": record["hops"],
        "hops": _format_path_hops(record["path_nodes"], record["path_rels"]),
        "query_ms": round(elapsed, 1),
    }


_BLAST_DIRECTIONS = {"callers": ("in",), "callees": ("out",), "both": ("out", "in")}


async def blast_radius(
    graph: GraphClient,
    uid: str,
    direction: str = "callers",
    max_depth: int = 3,
    edge_types: tuple[str, ...] = _DEFAULT_BLAST_EDGE_TYPES,
    limit: int = 20,
) -> dict[str, Any]:
    """Depth-limited transitive closure of callers/callees/both from *uid*.

    "callers" traverses incoming edges (who transitively depends on *uid*),
    "callees" traverses outgoing edges (what *uid* transitively depends on).
    Each affected entity is flagged ``ambiguous_only: true`` when no path made
    entirely of ``confidence: "resolved"`` CALLS edges (ADR-0014) reaches it
    within ``max_depth`` — a heuristic signal, not a guarantee (e.g. an
    out-of-scope edge_types override without a confidence property always
    counts as not-resolved).
    """
    t0 = time.monotonic()

    exist_raw = await graph.execute("OPTIONAL MATCH (n {uid: $uid}) RETURN n IS NOT NULL AS exists", {"uid": uid})
    if not exist_raw or not exist_raw[0]["exists"]:
        return {"error": f"Node not found: {uid}", "code": "NOT_FOUND"}

    dir_kinds = _BLAST_DIRECTIONS.get(direction)
    if dir_kinds is None:
        return {
            "error": f"Invalid direction '{direction}'. Valid: callers, callees, both",
            "code": "INVALID_DIRECTION",
        }

    rel_pattern = "|".join(edge_types)
    affected: dict[str, dict[str, Any]] = {}
    for dir_kind in dir_kinds:
        pattern = f"-[:{rel_pattern}*1..{max_depth}]->" if dir_kind == "out" else f"<-[:{rel_pattern}*1..{max_depth}]-"
        all_raw = await graph.execute(
            f"MATCH p=(start {{uid: $uid}}){pattern}(affected) "
            "WHERE affected.uid <> $uid "
            "RETURN affected.uid AS uid, affected.name AS name, affected.qualified_name AS qn, "
            "labels(affected)[0] AS label, affected.file_path AS file_path, "
            "min(length(p)) AS min_depth",
            {"uid": uid},
        )
        resolved_raw = await graph.execute(
            f"MATCH p=(start {{uid: $uid}}){pattern}(affected) "
            "WHERE affected.uid <> $uid AND all(r IN relationships(p) WHERE r.confidence = 'resolved') "
            "RETURN DISTINCT affected.uid AS uid",
            {"uid": uid},
        )
        resolved_uids = {r["uid"] for r in resolved_raw}
        for r in all_raw:
            entry = affected.get(r["uid"])
            if entry is None or r["min_depth"] < entry["min_depth"]:
                affected[r["uid"]] = {
                    "uid": r["uid"],
                    "name": r["name"],
                    "qualified_name": r["qn"],
                    "label": r["label"],
                    "file_path": r["file_path"],
                    "min_depth": r["min_depth"],
                    "direction": dir_kind,
                    "ambiguous_only": r["uid"] not in resolved_uids,
                }

    elapsed = (time.monotonic() - t0) * 1000
    results = sorted(affected.values(), key=lambda x: (x["min_depth"], x["qualified_name"] or ""))
    total = len(results)
    return {
        "uid": uid,
        "direction": direction,
        "max_depth": max_depth,
        "affected_count": total,
        "affected": results[:limit],
        "truncated": total > limit,
        "query_ms": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


async def _analyze_structure(graph: GraphClient, project: str, path: str, limit: int) -> dict[str, Any]:
    t0 = time.monotonic()
    params: dict[str, Any] = {"project": project, "path": path}
    pa = " AND n.file_path STARTS WITH $path" if path else ""

    # Entity counts by label + kind
    counts_raw = await graph.execute(
        f"MATCH (n {{project_name: $project}}) "
        f"WHERE NOT n:Project AND NOT n:SchemaVersion{pa} "
        "RETURN labels(n)[0] AS label, n.kind AS kind, count(n) AS cnt "
        "ORDER BY cnt DESC",
        params,
    )
    label_counts: dict[str, int] = {}
    kind_counts: dict[str, dict[str, int]] = {}
    for r in counts_raw:
        lbl = r["label"]
        label_counts[lbl] = label_counts.get(lbl, 0) + r["cnt"]
        if r["kind"]:
            kind_counts.setdefault(lbl, {})[r["kind"]] = r["cnt"]

    # Package breakdown — modules per package
    pa_m = " WHERE m.file_path STARTS WITH $path" if path else ""
    pkg_raw = await graph.execute(
        "MATCH (pkg:Package {project_name: $project})-[:CONTAINS]->(m:Module)"
        f"{pa_m} "
        "RETURN pkg.name AS package, pkg.qualified_name AS qn, count(m) AS modules "
        f"ORDER BY modules DESC LIMIT {limit}",
        params,
    )

    # Largest modules by defined entity count
    lm_w = " WHERE m.file_path STARTS WITH $path" if path else ""
    largest_raw = await graph.execute(
        "MATCH (m:Module {project_name: $project})-[:DEFINES]->(e)"
        f"{lm_w} "
        "RETURN m.name AS module, m.qualified_name AS qn, m.file_path AS file_path, "
        f"count(e) AS entities ORDER BY entities DESC LIMIT {limit}",
        params,
    )

    # External dependencies
    ext_w = " WHERE src IS NULL OR src.file_path STARTS WITH $path" if path else ""
    ext_raw = await graph.execute(
        "MATCH (ep:ExternalPackage {project_name: $project}) "
        "OPTIONAL MATCH (ep)<-[:IMPORTS]-(src) "
        f"{ext_w} "
        "RETURN ep.name AS package, ep.version AS version, count(src) AS imported_by "
        f"ORDER BY imported_by DESC LIMIT {limit}",
        params,
    )

    elapsed = (time.monotonic() - t0) * 1000
    return {
        "analysis": "structure",
        "project": project,
        "label_counts": label_counts,
        "kind_breakdown": kind_counts,
        "packages": [{"name": r["package"], "qualified_name": r["qn"], "module_count": r["modules"]} for r in pkg_raw],
        "largest_modules": [
            {
                "name": r["module"],
                "qualified_name": r["qn"],
                "file_path": r["file_path"],
                "entity_count": r["entities"],
            }
            for r in largest_raw
        ],
        "external_dependencies": [
            {"package": r["package"], "version": r["version"], "imported_by": r["imported_by"]} for r in ext_raw
        ],
        "query_ms": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Centrality
# ---------------------------------------------------------------------------


async def _analyze_centrality(graph: GraphClient, project: str, path: str, limit: int) -> dict[str, Any]:
    t0 = time.monotonic()
    params: dict[str, Any] = {"project": project, "path": path}
    pa = " AND n.file_path STARTS WITH $path" if path else ""

    # Hub entities — most referenced (inbound IMPORTS|INHERITS|CALLS)
    hubs_raw = await graph.execute(
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

    # Hub modules — most imported
    pa_m = " AND m.file_path STARTS WITH $path" if path else ""
    hub_modules_raw = await graph.execute(
        "MATCH (m:Module {project_name: $project})<-[:IMPORTS]-(src) "
        f"WHERE true{pa_m} "
        "RETURN m.name AS name, m.qualified_name AS qn, m.file_path AS file_path, "
        f"count(src) AS imported_by ORDER BY imported_by DESC LIMIT {limit}",
        params,
    )

    # Leaf entities — no inbound IMPORTS|INHERITS|CALLS
    pa_leaf = " AND n.file_path STARTS WITH $path" if path else ""
    leaf_raw = await graph.execute(
        "MATCH (n {project_name: $project}) "
        "WHERE NOT n:Project AND NOT n:SchemaVersion AND NOT n:Package "
        f"AND NOT n:ExternalPackage AND NOT n:ExternalSymbol{pa_leaf} "
        "AND NOT ()-[:IMPORTS|INHERITS|CALLS]->(n) "
        "RETURN n.name AS name, n.qualified_name AS qn, labels(n)[0] AS label, "
        f"n.kind AS kind, n.file_path AS file_path LIMIT {limit}",
        params,
    )

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
            for r in hubs_raw
        ],
        "hub_modules": [
            {
                "name": r["name"],
                "qualified_name": r["qn"],
                "file_path": r["file_path"],
                "imported_by": r["imported_by"],
            }
            for r in hub_modules_raw
        ],
        "leaf_entities": [
            {
                "name": r["name"],
                "qualified_name": r["qn"],
                "label": r["label"],
                "kind": r["kind"],
                "file_path": r["file_path"],
            }
            for r in leaf_raw
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


async def _analyze_dependencies(graph: GraphClient, project: str, path: str, limit: int) -> dict[str, Any]:
    t0 = time.monotonic()
    params: dict[str, Any] = {"project": project, "path": path}
    pa_m1 = " AND m1.file_path STARTS WITH $path" if path else ""

    # Direct module-to-module imports
    direct_raw = await graph.execute(
        "MATCH (m1:Module {project_name: $project})-[:IMPORTS]->"
        "(m2:Module {project_name: $project}) "
        f"WHERE m1 <> m2{pa_m1} "
        "RETURN m1.qualified_name AS from_mod, m2.qualified_name AS to_mod",
        params,
    )

    # Entity imports → parent module
    indirect_raw = await graph.execute(
        "MATCH (m1:Module {project_name: $project})-[:IMPORTS]->(e)"
        "<-[:DEFINES]-(m2:Module {project_name: $project}) "
        f"WHERE m1 <> m2 AND NOT e:Module{pa_m1} "
        "RETURN m1.qualified_name AS from_mod, m2.qualified_name AS to_mod",
        params,
    )

    edge_weights = _module_imports_from_records(direct_raw, indirect_raw)

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

    # External package import counts
    pa_src = " AND src.file_path STARTS WITH $path" if path else ""
    ext_pkg_raw = await graph.execute(
        "MATCH (src {project_name: $project})-[:IMPORTS]->(ep:ExternalPackage) "
        f"WHERE true{pa_src} "
        "RETURN ep.name AS package, count(src) AS cnt",
        params,
    )
    ext_sym_raw = await graph.execute(
        "MATCH (src {project_name: $project})-[:IMPORTS]->(es:ExternalSymbol) "
        f"WHERE true{pa_src} "
        "RETURN es.package AS package, count(src) AS cnt",
        params,
    )
    ext_counts: dict[str, int] = {}
    for r in ext_pkg_raw:
        ext_counts[r["package"]] = ext_counts.get(r["package"], 0) + r["cnt"]
    for r in ext_sym_raw:
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


async def _analyze_patterns(graph: GraphClient, project: str, path: str, limit: int) -> dict[str, Any]:
    t0 = time.monotonic()
    params: dict[str, Any] = {"project": project, "path": path}
    pa = " AND child.file_path STARTS WITH $path" if path else ""

    # Inheritance hierarchies
    inherit_raw = await graph.execute(
        "MATCH (child:TypeDef {project_name: $project})-[:INHERITS]->(parent) "
        f"WHERE true{pa} "
        "RETURN child.name AS child, child.qualified_name AS child_qn, "
        f"parent.name AS parent, parent.qualified_name AS parent_qn LIMIT {limit}",
        params,
    )

    # Enums
    pa_n = " AND n.file_path STARTS WITH $path" if path else ""
    enum_raw = await graph.execute(
        "MATCH (n:TypeDef {project_name: $project, kind: 'enum'})"
        f" WHERE true{pa_n} "
        "OPTIONAL MATCH (n)-[:DEFINES]->(m:Value) "
        "RETURN n.name AS name, n.qualified_name AS qn, n.file_path AS file_path, "
        f"count(m) AS members ORDER BY name LIMIT {limit}",
        params,
    )

    # Visibility distribution
    vis_raw = await graph.execute(
        "MATCH (n {project_name: $project}) "
        f"WHERE n.visibility IS NOT NULL{pa_n} "
        "RETURN n.visibility AS visibility, count(n) AS cnt "
        "ORDER BY cnt DESC",
        params,
    )

    # Docstring coverage
    doc_raw = await graph.execute(
        "MATCH (n {project_name: $project}) "
        f"WHERE (n:Callable OR n:TypeDef OR n:Value){pa_n} "
        "WITH count(n) AS total, "
        "sum(CASE WHEN n.docstring IS NOT NULL AND n.docstring <> '' THEN 1 ELSE 0 END) AS documented "
        "RETURN total, documented",
        params,
    )
    doc_stats = doc_raw[0] if doc_raw else {"total": 0, "documented": 0}

    # Pattern-detected relationships (routes, events, commands)
    pattern_raw = await graph.execute(
        "MATCH (n {project_name: $project})-[r:HANDLES_COMMAND|HANDLES_ROUTE|HANDLES_EVENT]->(target) "
        f"WHERE true{pa_n} "
        "RETURN type(r) AS pattern_type, n.name AS name, n.qualified_name AS qn, "
        f"target.name AS target_name ORDER BY pattern_type, name LIMIT {limit}",
        params,
    )

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
            for r in inherit_raw
        ],
        "enums": [
            {"name": r["name"], "qualified_name": r["qn"], "file_path": r["file_path"], "members": r["members"]}
            for r in enum_raw
        ],
        "visibility_distribution": {r["visibility"]: r["cnt"] for r in vis_raw},
        "docstring_coverage": {
            "total": doc_stats["total"],
            "documented": doc_stats["documented"],
            "percentage": round(doc_stats["documented"] / doc_stats["total"] * 100, 1) if doc_stats["total"] else 0,
        },
        "detected_patterns": [
            {"type": r["pattern_type"], "name": r["name"], "qualified_name": r["qn"], "target": r["target_name"]}
            for r in pattern_raw
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
    graph: GraphClient, project: str, path: str, limit: int, test_patterns: tuple[str, ...] = ()
) -> dict[str, Any]:
    t0 = time.monotonic()
    params: dict[str, Any] = {"project": project, "path": path}
    pa_m = " AND m.file_path STARTS WITH $path" if path else ""
    # Match on either side so a scoped module's fan-in from out-of-scope
    # importers (and fan-out to out-of-scope targets) are both captured.
    # Filtering on m1 alone (the importer) would undercount fan-in for
    # in-scope modules imported from outside the path.
    pa_edge = " AND (m1.file_path STARTS WITH $path OR m2.file_path STARTS WITH $path)" if path else ""

    # Query 1: entity counts per module
    entity_raw = await graph.execute(
        "MATCH (m:Module {project_name: $project})-[:DEFINES]->(e) "
        f"WHERE NOT e:Module{pa_m} "
        "RETURN m.qualified_name AS module, m.file_path AS file_path, count(e) AS entity_count "
        "ORDER BY entity_count DESC",
        params,
    )
    entity_counts: dict[str, int] = {}
    file_paths: dict[str, str] = {}
    for r in entity_raw:
        entity_counts[r["module"]] = r["entity_count"]
        file_paths[r["module"]] = r["file_path"]

    # Query 2 & 3: module-level import edges (reuse existing pattern)
    direct_raw = await graph.execute(
        "MATCH (m1:Module {project_name: $project})-[:IMPORTS]->"
        "(m2:Module {project_name: $project}) "
        f"WHERE m1 <> m2{pa_edge} "
        "RETURN m1.qualified_name AS from_mod, m2.qualified_name AS to_mod",
        params,
    )
    indirect_raw = await graph.execute(
        "MATCH (m1:Module {project_name: $project})-[:IMPORTS]->(e)"
        "<-[:DEFINES]-(m2:Module {project_name: $project}) "
        f"WHERE m1 <> m2 AND NOT e:Module{pa_edge} "
        "RETURN m1.qualified_name AS from_mod, m2.qualified_name AS to_mod",
        params,
    )
    edge_weights = _module_imports_from_records(direct_raw, indirect_raw)

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
    graph: GraphClient, project: str, path: str, limit: int, test_patterns: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Callables/TypeDefs with zero incoming CALLS edges (any confidence).

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
    """
    t0 = time.monotonic()
    params: dict[str, Any] = {"project": project, "path": path}
    pa = " AND n.file_path STARTS WITH $path" if path else ""

    raw = await graph.execute(
        "MATCH (n {project_name: $project}) "
        f"WHERE (n:Callable OR n:TypeDef) AND NOT n.name STARTS WITH '__'{pa} "
        "AND NOT ()-[:CALLS]->(n) "
        "RETURN n.name AS name, n.qualified_name AS qn, labels(n)[0] AS label, "
        "n.kind AS kind, n.file_path AS file_path, n.line_start AS line_start "
        "ORDER BY n.file_path, n.line_start",
        params,
    )

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
        "truncated": total > limit,
        "query_ms": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Complexity (LOC-span proxy)
# ---------------------------------------------------------------------------


async def _analyze_complexity(graph: GraphClient, project: str, path: str, limit: int) -> dict[str, Any]:
    """Top-N Callables by ``line_end - line_start``.

    This is a crude LOC-span proxy, not true cyclomatic complexity (branch-
    node counting) — no new parsing work was needed since line_start/line_end
    already persist on every Callable.
    """
    t0 = time.monotonic()
    params: dict[str, Any] = {"project": project, "path": path}
    pa = " AND n.file_path STARTS WITH $path" if path else ""

    raw = await graph.execute(
        "MATCH (n:Callable {project_name: $project}) "
        f"WHERE n.line_start IS NOT NULL AND n.line_end IS NOT NULL{pa} "
        "RETURN n.name AS name, n.qualified_name AS qn, n.kind AS kind, n.file_path AS file_path, "
        "n.line_start AS line_start, n.line_end AS line_end, (n.line_end - n.line_start) AS loc_span "
        f"ORDER BY loc_span DESC LIMIT {limit}",
        params,
    )

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
# Communities (Leiden clustering over the CALLS+IMPORTS subgraph, ADR-0013/MAGE)
# ---------------------------------------------------------------------------


async def _analyze_communities(graph: GraphClient, project: str, path: str, limit: int) -> dict[str, Any]:
    """Cluster the project's CALLS+IMPORTS subgraph via Leiden community detection.

    Requires a MAGE-enabled Memgraph image (``memgraph/memgraph-mage``, not the
    plain community ``memgraph/memgraph`` image) — ``leiden_community_detection.get()``
    is a MAGE query module, not core Cypher (same procedure-call pattern already
    used for ``text_search.search_all``/``vector_search.search``). Returns a
    clear ``PROCEDURE_UNAVAILABLE`` error instead of raising when the module
    isn't installed.

    ExternalPackage/ExternalSymbol nodes are excluded from both edge endpoints —
    otherwise dozens of unrelated modules that merely reference the same external
    type (e.g. many modules' return-type annotations pointing at
    ``collections.abc.Coroutine``) turn that external node into a false hub that
    glues most of the project into one meaningless giant community. Communities
    are meant to reflect cohesive *project* subsystems, not "everything that
    happens to import the same stdlib/third-party symbol".

    Communities of size < ``_COMMUNITY_NOISE_THRESHOLD`` (isolated/near-isolated
    nodes) are dropped as noise; the remaining communities are returned
    largest-first, capped at *limit* (which also caps members shown per
    community — full membership can be large at scale).
    """
    t0 = time.monotonic()
    params: dict[str, Any] = {"project": project, "path": path}
    # Exclude ExternalPackage/ExternalSymbol from both endpoints — dozens of unrelated
    # modules referencing the same external type (e.g. collections.abc.Coroutine) would
    # otherwise act as false hub nodes gluing the whole project into one giant community.
    excl = "NOT a:ExternalPackage AND NOT a:ExternalSymbol AND NOT b:ExternalPackage AND NOT b:ExternalSymbol"
    pa = " AND a.file_path STARTS WITH $path AND b.file_path STARTS WITH $path" if path else ""

    query = (
        "MATCH p=(a {project_name: $project})-[:CALLS|IMPORTS]->(b {project_name: $project}) "
        f"WHERE {excl}{pa} "
        "WITH project(p) AS subgraph "
        "CALL leiden_community_detection.get(subgraph) YIELD node, community_id "
        "RETURN node.uid AS uid, node.name AS name, node.qualified_name AS qn, "
        "labels(node)[0] AS label, node.file_path AS file_path, community_id AS community_id"
    )
    try:
        raw = await graph.execute(query, params)
    except Exception as exc:
        return {
            "error": (
                "Community detection unavailable: leiden_community_detection.get() is a MAGE query "
                f"module — confirm Memgraph is running the memgraph-mage image, not memgraph. ({exc})"
            ),
            "code": "PROCEDURE_UNAVAILABLE",
        }

    # community_id < 0 means "unassigned" (per MAGE docs) — drop before grouping.
    groups: dict[int, list[dict[str, Any]]] = {}
    for r in raw:
        if r["community_id"] < 0:
            continue
        groups.setdefault(r["community_id"], []).append(
            {
                "uid": r["uid"],
                "name": r["name"],
                "qualified_name": r["qn"],
                "label": r["label"],
                "file_path": r["file_path"],
            }
        )

    sized: list[dict[str, Any]] = sorted(
        (
            {"community_id": cid, "size": len(members), "members": members}
            for cid, members in groups.items()
            if len(members) >= _COMMUNITY_NOISE_THRESHOLD
        ),
        key=lambda c: c["size"],
        reverse=True,
    )

    elapsed = (time.monotonic() - t0) * 1000
    return {
        "analysis": "communities",
        "project": project,
        "community_count": len(sized),
        "communities": [{**c, "members": c["members"][:limit]} for c in sized[:limit]],
        "noise_threshold": _COMMUNITY_NOISE_THRESHOLD,
        "query_ms": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Git signals (hotspots, bus-factor risks, co-change pairs — ADR-0013 find_hotspots)
# ---------------------------------------------------------------------------


async def _analyze_git_signals(graph: GraphClient, project: str, path: str, limit: int) -> dict[str, Any]:
    """Surface git-mined signals: commit-count hotspots, bus-factor risks, top co-change pairs.

    Reads properties/edges written by ``indexing/git_signals.py``'s
    ``write_git_signals`` — populated by the one-shot ``atlas mine-git-history``
    CLI command, not the continuous indexing pipeline. If that command has
    never been run for this project, all three lists come back empty (not an
    error) — ``mined`` is ``False`` in that case so callers can tell "no signal"
    apart from "ran, found nothing".
    """
    t0 = time.monotonic()
    params: dict[str, Any] = {"project": project, "path": path}
    pa = " AND n.file_path STARTS WITH $path" if path else ""

    hotspots_raw = await graph.execute(
        "MATCH (n {project_name: $project}) "
        f"WHERE n.git_commit_count IS NOT NULL{pa} "
        "RETURN n.name AS name, n.qualified_name AS qn, n.file_path AS file_path, "
        "n.git_commit_count AS commit_count, n.git_author_count AS author_count, "
        "n.git_days_since_last_commit AS days_since_last_commit "
        f"ORDER BY commit_count DESC LIMIT {limit}",
        params,
    )

    bus_factor_raw = await graph.execute(
        "MATCH (n {project_name: $project}) "
        f"WHERE n.git_commit_count IS NOT NULL AND n.git_author_count <= {_BUS_FACTOR_AUTHOR_THRESHOLD}{pa} "
        "RETURN n.name AS name, n.qualified_name AS qn, n.file_path AS file_path, "
        "n.git_commit_count AS commit_count, n.git_author_count AS author_count "
        f"ORDER BY commit_count DESC LIMIT {limit}",
        params,
    )

    pa_edge = " AND (a.file_path STARTS WITH $path OR b.file_path STARTS WITH $path)" if path else ""
    co_change_raw = await graph.execute(
        f"MATCH (a {{project_name: $project}})-[r:{RelType.CO_CHANGES_WITH}]->(b {{project_name: $project}}) "
        f"WHERE true{pa_edge} "
        "RETURN a.qualified_name AS a_qn, a.file_path AS a_path, "
        "b.qualified_name AS b_qn, b.file_path AS b_path, r.count AS count "
        f"ORDER BY count DESC LIMIT {limit}",
        params,
    )

    elapsed = (time.monotonic() - t0) * 1000
    mined = bool(hotspots_raw)
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


async def _diagram_packages(graph: GraphClient, project: str, path: str, max_nodes: int) -> dict[str, Any]:
    t0 = time.monotonic()
    params: dict[str, Any] = {"project": project, "path": path, "limit": max_nodes}
    pa = " AND child.file_path STARTS WITH $path" if path else ""

    # Packages and their contained modules
    records = await graph.execute(
        "MATCH (pkg:Package {project_name: $project})-[:CONTAINS]->(child) "
        f"WHERE (child:Package OR child:Module){pa} "
        "RETURN pkg.qualified_name AS parent_qn, pkg.name AS parent_name, "
        "labels(child)[0] AS child_label, child.qualified_name AS child_qn, child.name AS child_name "
        "ORDER BY parent_qn, child_qn LIMIT $limit",
        params,
    )

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


async def _diagram_imports(graph: GraphClient, project: str, path: str, max_nodes: int) -> dict[str, Any]:
    t0 = time.monotonic()
    params: dict[str, Any] = {"project": project, "path": path}
    pa_m1 = " AND m1.file_path STARTS WITH $path" if path else ""

    # Direct module imports
    direct_raw = await graph.execute(
        "MATCH (m1:Module {project_name: $project})-[:IMPORTS]->"
        "(m2:Module {project_name: $project}) "
        f"WHERE m1 <> m2{pa_m1} "
        "RETURN m1.qualified_name AS from_mod, m2.qualified_name AS to_mod",
        params,
    )
    # Entity imports → parent module
    indirect_raw = await graph.execute(
        "MATCH (m1:Module {project_name: $project})-[:IMPORTS]->(e)"
        "<-[:DEFINES]-(m2:Module {project_name: $project}) "
        f"WHERE m1 <> m2 AND NOT e:Module{pa_m1} "
        "RETURN m1.qualified_name AS from_mod, m2.qualified_name AS to_mod",
        params,
    )

    edge_weights = _module_imports_from_records(direct_raw, indirect_raw)

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

    lines = ["graph LR"]
    lines.extend(f'    {_sid(qn)}["{_slabel(qn)}"]' for qn in sorted(kept_nodes))
    for (from_mod, to_mod), weight in kept_edges:
        label_part = f"|{weight}|" if weight > 1 else ""
        lines.append(f"    {_sid(from_mod)} -->{label_part} {_sid(to_mod)}")

    elapsed = (time.monotonic() - t0) * 1000
    return {
        "type": "imports",
        "mermaid": "\n".join(lines),
        "node_count": len(kept_nodes),
        "query_ms": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Diagram: inheritance (class hierarchy)
# ---------------------------------------------------------------------------


async def _diagram_inheritance(graph: GraphClient, project: str, path: str, max_nodes: int) -> dict[str, Any]:
    t0 = time.monotonic()
    params: dict[str, Any] = {"project": project, "path": path, "limit": max_nodes}
    pa = " AND child.file_path STARTS WITH $path" if path else ""

    records = await graph.execute(
        "MATCH (child:TypeDef {project_name: $project})-[:INHERITS]->(parent) "
        f"WHERE true{pa} "
        "RETURN child.name AS child_name, child.qualified_name AS child_qn, "
        "child.kind AS child_kind, "
        "parent.name AS parent_name, parent.qualified_name AS parent_qn "
        "ORDER BY parent_qn, child_qn LIMIT $limit",
        params,
    )

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


async def _diagram_module_detail(graph: GraphClient, project: str, path: str, max_nodes: int) -> dict[str, Any]:
    t0 = time.monotonic()
    params: dict[str, Any] = {"project": project, "path": path}

    if not path:
        return {
            "error": "path parameter required for module_detail diagram (file path prefix of the module)",
            "code": "PATH_REQUIRED",
        }

    # Find the module
    modules = await graph.execute(
        "MATCH (m:Module {project_name: $project}) "
        "WHERE m.file_path STARTS WITH $path "
        "RETURN m.name AS name, m.qualified_name AS qn, m.uid AS uid "
        "ORDER BY m.qualified_name LIMIT 1",
        params,
    )
    if not modules:
        return {"error": f"No module found matching path '{path}'", "code": "NOT_FOUND"}

    mod = modules[0]

    # Top-level entities defined by this module
    entities = await graph.execute(
        "MATCH (m {uid: $uid})-[:DEFINES]->(e) "
        "RETURN e.name AS name, e.qualified_name AS qn, labels(e)[0] AS label, "
        f"e.kind AS kind, e.visibility AS vis, e.signature AS sig ORDER BY e.line_start LIMIT {max_nodes}",
        {"uid": mod["uid"]},
    )

    # Methods defined by TypeDefs in this module
    methods = await graph.execute(
        "MATCH (m {uid: $uid})-[:DEFINES]->(td:TypeDef)-[:DEFINES]->(method:Callable) "
        "RETURN td.qualified_name AS class_qn, td.name AS class_name, "
        "method.name AS name, method.visibility AS vis, method.kind AS kind "
        f"ORDER BY td.name, method.line_start LIMIT {max_nodes}",
        {"uid": mod["uid"]},
    )

    # Inheritance for TypeDefs in this module
    inherits = await graph.execute(
        "MATCH (m {uid: $uid})-[:DEFINES]->(td:TypeDef)-[:INHERITS]->(parent) "
        "RETURN td.qualified_name AS child_qn, td.name AS child_name, "
        "parent.qualified_name AS parent_qn, parent.name AS parent_name "
        f"LIMIT {max_nodes}",
        {"uid": mod["uid"]},
    )

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
