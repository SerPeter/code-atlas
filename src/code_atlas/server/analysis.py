"""Repository analysis and diagram generation for Code Atlas MCP server.

Pure Cypher queries + Python formatting — no LLM calls, no file reads,
no new dependencies.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from code_atlas.graph.client import GraphClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MERMAID_UNSAFE = re.compile(r"[^a-zA-Z0-9_]")

_VALID_ANALYSES = frozenset({"structure", "centrality", "dependencies", "patterns"})
_VALID_DIAGRAM_TYPES = frozenset({"packages", "imports", "inheritance", "module_detail"})


def _sid(name: str) -> str:
    """Convert a qualified name to a safe Mermaid node ID."""
    return _MERMAID_UNSAFE.sub("_", name)


def _slabel(text: str, max_len: int = 40) -> str:
    """Truncate and escape a label for Mermaid display."""
    text = text.replace('"', "'").replace("<", "&lt;").replace(">", "&gt;")
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


# ---------------------------------------------------------------------------
# Public dispatchers
# ---------------------------------------------------------------------------


async def analyze_repo(
    graph: GraphClient,
    analysis: str,
    project: str,
    path: str = "",
    limit: int = 20,
) -> dict[str, Any]:
    """Dispatch to the requested sub-analysis."""
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
    ext_raw = await graph.execute(
        "MATCH (ep:ExternalPackage {project_name: $project}) "
        "OPTIONAL MATCH (ep)<-[:IMPORTS]-(src) "
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
        from_pkg = from_mod.split(".")[0]
        to_pkg = to_mod.split(".")[0]
        if from_pkg != to_pkg:
            key = (from_pkg, to_pkg)
            pkg_edges[key] = pkg_edges.get(key, 0) + weight
    cross_package = sorted(
        [{"from": k[0], "to": k[1], "weight": v} for k, v in pkg_edges.items()],
        key=lambda x: x["weight"],
        reverse=True,
    )[:limit]

    # Circular dependencies (mutual imports at module level)
    seen: set[tuple[str, str]] = set()
    circular: list[dict[str, str]] = []
    for from_mod, to_mod in edge_weights:
        reverse = (to_mod, from_mod)
        if reverse in edge_weights and from_mod < to_mod and (from_mod, to_mod) not in seen:
            circular.append({"module_a": from_mod, "module_b": to_mod})
            seen.add((from_mod, to_mod))
            if len(circular) >= 10:
                break

    # External package import counts
    ext_pkg_raw = await graph.execute(
        "MATCH (src {project_name: $project})-[:IMPORTS]->(ep:ExternalPackage) "
        "RETURN ep.name AS package, count(src) AS cnt",
        params,
    )
    ext_sym_raw = await graph.execute(
        "MATCH (src {project_name: $project})-[:IMPORTS]->(es:ExternalSymbol) "
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
# Diagram: packages (containment tree)
# ---------------------------------------------------------------------------


async def _diagram_packages(graph: GraphClient, project: str, path: str, max_nodes: int) -> dict[str, Any]:
    t0 = time.monotonic()
    params: dict[str, Any] = {"project": project, "path": path, "limit": max_nodes}

    # Packages and their contained modules
    records = await graph.execute(
        "MATCH (pkg:Package {project_name: $project})-[:CONTAINS]->(child) "
        "WHERE child:Package OR child:Module "
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
    sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1], reverse=True)
    kept_nodes: set[str] = set()
    kept_edges: list[tuple[tuple[str, str], int]] = []
    for (from_mod, to_mod), weight in sorted_edges:
        if len(kept_nodes) >= max_nodes:
            break
        kept_nodes.add(from_mod)
        kept_nodes.add(to_mod)
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
        "ORDER BY td.name, method.line_start",
        {"uid": mod["uid"]},
    )

    # Inheritance for TypeDefs in this module
    inherits = await graph.execute(
        "MATCH (m {uid: $uid})-[:DEFINES]->(td:TypeDef)-[:INHERITS]->(parent) "
        "RETURN td.qualified_name AS child_qn, td.name AS child_name, "
        "parent.qualified_name AS parent_qn, parent.name AS parent_name",
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

        if e["label"] == "TypeDef":
            lines.append(f'    class {eid}["{_slabel(e["name"])}"]')
            kind = e["kind"] or "class"
            lines.append(f"    <<{kind}>> {eid}")
            # Add methods
            for meth in class_methods.get(e["qn"], []):
                prefix = vis_prefix.get(meth["vis"] or "public", "+")
                lines.append(f"    {eid} : {prefix}{meth['name']}()")
        elif e["label"] == "Callable":
            lines.append(f'    class {eid}["{_slabel(e["name"])}"]')
            lines.append(f"    <<{e['kind'] or 'function'}>> {eid}")
        elif e["label"] == "Value":
            lines.append(f'    class {eid}["{_slabel(e["name"])}"]')
            lines.append(f"    <<{e['kind'] or 'value'}>> {eid}")

    # Add inheritance edges
    for inh in inherits:
        child_id = _sid(inh["child_qn"])
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
