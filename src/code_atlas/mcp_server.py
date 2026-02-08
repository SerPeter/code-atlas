"""MCP server for Code Atlas.

Exposes the Memgraph graph database to AI coding agents via MCP tools.
Read-only query interface — connects directly to Memgraph, no daemon dependency.
"""

from __future__ import annotations

import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger
from mcp.server.fastmcp import Context, FastMCP

from code_atlas.embeddings import EmbedClient, EmbeddingError
from code_atlas.graph import GraphClient
from code_atlas.schema import (
    _CODE_LABELS,
    _DOC_LABELS,
    _EMBEDDABLE_LABELS,
    _EXTERNAL_LABELS,
    _TEXT_SEARCHABLE_LABELS,
    SCHEMA_VERSION,
    CallableKind,
    NodeLabel,
    RelType,
    TypeDefKind,
    ValueKind,
    Visibility,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from code_atlas.settings import AtlasSettings

# ---------------------------------------------------------------------------
# Application context
# ---------------------------------------------------------------------------

_WRITE_KEYWORDS = re.compile(
    r"\b(CREATE|DELETE|SET|MERGE|REMOVE|DROP|DETACH)\b",
    re.IGNORECASE,
)

_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)

_DEFAULT_LIMIT = 20
_MAX_LIMIT = 100
_DOCSTRING_TRUNCATE = 200


@dataclass
class AppContext:
    graph: GraphClient
    settings: AtlasSettings
    embed: EmbedClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_app_ctx(ctx: Context) -> AppContext:
    """Extract AppContext from the MCP request context."""
    return ctx.request_context.lifespan_context


def _serialize_node(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a neo4j record containing Node objects to plain dicts."""
    out: dict[str, Any] = {}
    for key, value in record.items():
        if hasattr(value, "items") and hasattr(value, "labels"):
            # neo4j Node object
            node_dict = dict(value.items())
            node_dict["_labels"] = sorted(value.labels)
            out[key] = node_dict
        else:
            out[key] = value
    return out


def _compact_node(record: dict[str, Any]) -> dict[str, Any]:
    """Extract compact metadata from a node record."""
    node = record.get("node") or record.get("n")
    if node is None:
        return _serialize_node(record)

    props = dict(node.items()) if hasattr(node, "items") else (node if isinstance(node, dict) else {})

    compact: dict[str, Any] = {}
    for field in (
        "uid",
        "name",
        "qualified_name",
        "kind",
        "file_path",
        "line_start",
        "line_end",
        "signature",
    ):
        if field in props:
            compact[field] = props[field]

    docstring = props.get("docstring")
    if docstring:
        compact["docstring"] = docstring[:_DOCSTRING_TRUNCATE] + ("..." if len(docstring) > _DOCSTRING_TRUNCATE else "")

    if hasattr(node, "labels"):
        compact["_labels"] = sorted(node.labels)

    # Preserve score/similarity from search results
    for score_key in ("score", "similarity"):
        if score_key in record:
            compact[score_key] = record[score_key]

    return compact


def _node_project_name(record: dict[str, Any]) -> str:
    """Extract project_name from a record containing a neo4j Node."""
    node = record.get("node") or record.get("n")
    if node is None:
        return ""
    if hasattr(node, "get"):
        return node.get("project_name", "")
    return ""


def _result(records: list[dict[str, Any]], *, limit: int, query_ms: float, total: int | None = None) -> dict[str, Any]:
    """Consistent result envelope."""
    return {
        "results": records,
        "count": len(records),
        "truncated": total is not None and total > limit,
        "query_ms": round(query_ms, 1),
    }


def _error(message: str, *, code: str) -> dict[str, Any]:
    """Error envelope."""
    return {"error": message, "code": code}


def _clamp_limit(limit: int | None) -> int:
    """Clamp limit to [1, 100], default 20."""
    if limit is None:
        return _DEFAULT_LIMIT
    return max(1, min(limit, _MAX_LIMIT))


# Visibility ranking: lower = more relevant (public entities preferred)
_VISIBILITY_RANK: dict[str, int] = {"public": 0, "protected": 1, "internal": 2, "private": 3}


def _rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank disambiguation results by relevance.

    Sorting criteria (applied in order):
    1. Source over test — entities whose file_path does NOT contain "test" first
    2. Visibility — public > protected > internal > private
    3. Shorter qualified_name — more canonical entities first
    """

    def _sort_key(node: dict[str, Any]) -> tuple[int, int, int]:
        fp = (node.get("file_path") or "").lower()
        is_test = 1 if ("test" in fp) else 0
        vis = _VISIBILITY_RANK.get(node.get("visibility", "public"), 0)
        qn_len = len(node.get("qualified_name", ""))
        return (is_test, vis, qn_len)

    return sorted(results, key=_sort_key)


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def create_mcp_server(settings: AtlasSettings) -> FastMCP:
    """Create and configure the Code Atlas MCP server."""

    @asynccontextmanager
    async def app_lifespan(_server: FastMCP) -> AsyncIterator[AppContext]:
        graph = GraphClient(settings)
        try:
            await graph.ping()
        except Exception as exc:
            logger.error("Cannot reach Memgraph at {}:{} — {}", settings.memgraph.host, settings.memgraph.port, exc)
            raise

        logger.info("MCP connected to Memgraph at {}:{}", settings.memgraph.host, settings.memgraph.port)
        embed = EmbedClient(settings.embeddings)
        app_ctx = AppContext(graph=graph, settings=settings, embed=embed)
        try:
            yield app_ctx
        finally:
            await graph.close()
            logger.info("MCP server shut down")

    mcp = FastMCP(
        name="code-atlas",
        instructions=(
            "Code Atlas — graph-powered code intelligence. "
            "Start with schema_info and index_status. "
            "Use get_node to find entities, get_context to expand. "
            "Use cypher_query for custom traversals. "
            "Use text_search for keywords, vector_search for semantics."
        ),
        lifespan=app_lifespan,
    )

    _register_query_tools(mcp)
    _register_search_tools(mcp)
    _register_info_tools(mcp)
    return mcp


# ---------------------------------------------------------------------------
# Tool registration — split to stay under statement limits
# ---------------------------------------------------------------------------


def _register_node_tools(mcp: FastMCP) -> None:
    """Register the get_node tool (separated for statement-count limits)."""

    @mcp.tool(
        description=(
            "Find code entities by name. "
            "Cascade: exact uid → exact name → suffix → prefix → contains. "
            "First stage with results wins. Results ranked by relevance. "
            "Returns compact metadata (name, kind, file, lines, signature). "
            "Use get_context to expand a result."
        ),
    )
    async def get_node(name: str, label: str = "", limit: int = 20, ctx: Context = None) -> dict[str, Any]:  # type: ignore[assignment]
        app = _get_app_ctx(ctx)
        clamped = _clamp_limit(limit)

        label_filter = f":{label}" if label else ""
        t0 = time.monotonic()

        # Stage 1: Exact uid match (if name contains ':')
        if ":" in name:
            records = await app.graph.execute(
                f"MATCH (n{label_filter} {{uid: $name}}) RETURN n LIMIT {clamped}",
                {"name": name},
            )
            if records:
                elapsed = (time.monotonic() - t0) * 1000
                ranked = _rank_results([_compact_node(r) for r in records])
                return _result(ranked, limit=clamped, query_ms=elapsed)

        # Stage 2: Exact name match
        records = await app.graph.execute(
            f"MATCH (n{label_filter}) WHERE n.name = $name RETURN n LIMIT {clamped}",
            {"name": name},
        )
        if records:
            elapsed = (time.monotonic() - t0) * 1000
            ranked = _rank_results([_compact_node(r) for r in records])
            return _result(ranked, limit=clamped, query_ms=elapsed)

        # Stage 3: ENDS WITH suffix match
        suffix = f".{name}"
        records = await app.graph.execute(
            f"MATCH (n{label_filter}) WHERE n.qualified_name ENDS WITH $suffix RETURN n LIMIT {clamped}",
            {"suffix": suffix},
        )
        if records:
            elapsed = (time.monotonic() - t0) * 1000
            ranked = _rank_results([_compact_node(r) for r in records])
            return _result(ranked, limit=clamped, query_ms=elapsed)

        # Stage 4: STARTS WITH prefix match
        prefix = f"{name}."
        records = await app.graph.execute(
            f"MATCH (n{label_filter}) WHERE n.qualified_name STARTS WITH $prefix RETURN n LIMIT {clamped}",
            {"prefix": prefix},
        )
        if records:
            elapsed = (time.monotonic() - t0) * 1000
            ranked = _rank_results([_compact_node(r) for r in records])
            return _result(ranked, limit=clamped, query_ms=elapsed)

        # Stage 5: CONTAINS match (qualified_name or name)
        records = await app.graph.execute(
            f"MATCH (n{label_filter}) WHERE n.qualified_name CONTAINS $name OR n.name CONTAINS $name "
            f"RETURN n LIMIT {clamped}",
            {"name": name},
        )
        elapsed = (time.monotonic() - t0) * 1000
        ranked = _rank_results([_compact_node(r) for r in records])
        return _result(ranked, limit=clamped, query_ms=elapsed)


def _register_query_tools(mcp: FastMCP) -> None:
    """Register cypher_query and get_context tools."""

    @mcp.tool(
        description=(
            "Execute read-only Cypher against the graph. "
            "LIMIT auto-applied (default 20, max 100). "
            "Write operations rejected. "
            "Call schema_info first to see available node types and relationships."
        ),
    )
    async def cypher_query(query: str, limit: int = 20, ctx: Context = None) -> dict[str, Any]:  # type: ignore[assignment]
        app = _get_app_ctx(ctx)
        clamped = _clamp_limit(limit)

        if _WRITE_KEYWORDS.search(query):
            return _error("Write operations are not allowed via MCP", code="WRITE_REJECTED")

        # Auto-append LIMIT if missing
        if not _LIMIT_RE.search(query):
            query = query.rstrip().rstrip(";") + f" LIMIT {clamped}"

        t0 = time.monotonic()
        try:
            records = await app.graph.execute(query)
        except Exception as exc:
            return _error(str(exc), code="QUERY_ERROR")
        elapsed = (time.monotonic() - t0) * 1000

        serialized = [_serialize_node(r) for r in records]
        return _result(serialized, limit=clamped, query_ms=elapsed)

    _register_node_tools(mcp)

    @mcp.tool(
        description=(
            "Expand a node into its neighborhood: parent, children, callers, callees, bases, subclasses. "
            "Pass the uid from a get_node result. "
            "Set include_source=true to include source code (file_path + line range). "
            "depth controls relationship traversal (default 1, max 3)."
        ),
    )
    async def get_context(
        uid: str,
        depth: int = 1,
        include_source: bool = False,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        app = _get_app_ctx(ctx)
        depth = max(1, min(depth, 3))
        sub_limit = 20

        t0 = time.monotonic()

        # Fetch target node
        records = await app.graph.execute("MATCH (n {uid: $uid}) RETURN n", {"uid": uid})
        if not records:
            return _error(f"Node not found: {uid}", code="NOT_FOUND")

        target = _compact_node(records[0])

        if include_source:
            node = records[0].get("n")
            if node and hasattr(node, "items"):
                props = dict(node.items())
                target["source_code"] = props.get("source_code")

        # Parent (who DEFINES this node)
        parent_records = await app.graph.execute(
            f"MATCH (p)-[:{RelType.DEFINES}]->(n {{uid: $uid}}) RETURN p AS n LIMIT {sub_limit}",
            {"uid": uid},
        )

        # Children (this node DEFINES)
        children_records = await app.graph.execute(
            f"MATCH (n {{uid: $uid}})-[:{RelType.DEFINES}]->(c) RETURN c AS n LIMIT {sub_limit}",
            {"uid": uid},
        )

        # Callers (who CALLS this node)
        caller_records = await app.graph.execute(
            f"MATCH (caller)-[:{RelType.CALLS}]->(n {{uid: $uid}}) RETURN caller AS n LIMIT {sub_limit}",
            {"uid": uid},
        )

        # Callees (this node CALLS)
        callee_records = await app.graph.execute(
            f"MATCH (n {{uid: $uid}})-[:{RelType.CALLS}]->(callee) RETURN callee AS n LIMIT {sub_limit}",
            {"uid": uid},
        )

        # Bases (this node INHERITS from)
        base_records = await app.graph.execute(
            f"MATCH (n {{uid: $uid}})-[:{RelType.INHERITS}]->(base) RETURN base AS n LIMIT {sub_limit}",
            {"uid": uid},
        )

        # Subclasses (who INHERITS from this node)
        subclass_records = await app.graph.execute(
            f"MATCH (sub)-[:{RelType.INHERITS}]->(n {{uid: $uid}}) RETURN sub AS n LIMIT {sub_limit}",
            {"uid": uid},
        )

        elapsed = (time.monotonic() - t0) * 1000

        return {
            "node": target,
            "parent": [_compact_node(r) for r in parent_records],
            "children": [_compact_node(r) for r in children_records],
            "callers": [_compact_node(r) for r in caller_records],
            "callees": [_compact_node(r) for r in callee_records],
            "bases": [_compact_node(r) for r in base_records],
            "subclasses": [_compact_node(r) for r in subclass_records],
            "query_ms": round(elapsed, 1),
        }


def _register_search_tools(mcp: FastMCP) -> None:
    """Register text_search and vector_search tools."""

    @mcp.tool(
        description=(
            "BM25 keyword search across code entities using Tantivy. "
            "Searches all text indices by default, or a single label if specified. "
            "Valid labels: Callable, Module, TypeDef, Value, DocSection. "
            "Optional: project (filter by project_name). "
            'Query syntax: quoted phrases ("exact phrase"), '
            "field-specific (name:UserService, docstring:authentication), "
            "wildcards (get*User), boolean (foo AND bar, foo OR bar). "
            "Returns compact metadata with relevance score."
        ),
    )
    async def text_search(
        query: str,
        label: str = "",
        limit: int = 20,
        project: str = "",
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        app = _get_app_ctx(ctx)
        clamped = _clamp_limit(limit)

        t0 = time.monotonic()
        all_results = await app.graph.text_search(query, label=label, limit=clamped, project=project)
        elapsed = (time.monotonic() - t0) * 1000

        compacted = [_compact_node(r) for r in all_results]
        return _result(compacted, limit=clamped, query_ms=elapsed)

    @mcp.tool(
        description=(
            "Semantic similarity search using vector embeddings. "
            "Embeds the query via TEI, then searches vector indices. "
            "Searches all embeddable labels by default, or a single label if specified. "
            "Valid labels: Callable, Module, TypeDef, Value, DocSection. "
            "Optional: project (filter by project_name), threshold (min similarity 0-1). "
            "Returns compact metadata with similarity score."
        ),
    )
    async def vector_search(
        query: str,
        label: str = "",
        limit: int = 20,
        project: str = "",
        threshold: float = 0.0,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        app = _get_app_ctx(ctx)
        clamped = _clamp_limit(limit)
        filtering = bool(project) or threshold > 0.0
        fetch_limit = clamped * 3 if filtering else clamped

        # Embed the query
        try:
            vector = await app.embed.embed_one(query)
        except EmbeddingError as exc:
            return _error(f"Embedding service unavailable: {exc}", code="EMBED_ERROR")

        indices = [f"vec_{label.lower()}"] if label else [f"vec_{lbl.value.lower()}" for lbl in _EMBEDDABLE_LABELS]

        t0 = time.monotonic()
        all_results: list[dict[str, Any]] = []

        for index_name in indices:
            cypher = (
                f"CALL vector_search.search('{index_name}', {fetch_limit}, $vector) "
                "YIELD node, similarity "
                "RETURN node, similarity "
                f"ORDER BY similarity DESC LIMIT {fetch_limit}"
            )
            try:
                records = await app.graph.execute(cypher, {"vector": vector})
                all_results.extend(records)
            except Exception as exc:
                logger.warning("Vector search on {} failed: {}", index_name, exc)

        # Post-filter by threshold
        if threshold > 0.0:
            all_results = [r for r in all_results if r.get("similarity", 0) >= threshold]

        # Post-filter by project scope
        if project:
            all_results = [r for r in all_results if _node_project_name(r) == project]

        # Sort by similarity descending and take top results
        all_results.sort(key=lambda rec: rec.get("similarity", 0), reverse=True)
        all_results = all_results[:clamped]
        elapsed = (time.monotonic() - t0) * 1000

        compacted = [_compact_node(r) for r in all_results]
        return _result(compacted, limit=clamped, query_ms=elapsed)


def _register_info_tools(mcp: FastMCP) -> None:
    """Register index_status and schema_info tools."""

    @mcp.tool(
        description=(
            "Show indexed projects, entity counts, and schema version. "
            "Use this to understand what data is available before querying."
        ),
    )
    async def index_status(ctx: Context = None) -> dict[str, Any]:  # type: ignore[assignment]
        app = _get_app_ctx(ctx)
        t0 = time.monotonic()

        projects_raw = await app.graph.get_project_status()
        projects = []
        for row in projects_raw:
            node = row.get("n")
            if node is None:
                continue
            props = dict(node.items()) if hasattr(node, "items") else node
            name = props.get("name", "?")
            entity_count = await app.graph.count_entities(name)
            projects.append(
                {
                    "name": name,
                    "file_count": props.get("file_count"),
                    "entity_count": entity_count,
                    "last_indexed_at": props.get("last_indexed_at"),
                    "git_hash": props.get("git_hash"),
                }
            )

        # Per-label counts
        label_counts_raw = await app.graph.execute(
            "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY count DESC"
        )
        label_counts = {r["label"]: r["count"] for r in label_counts_raw}

        # Vector and text index info
        vec_index_info = await app.graph.get_vector_index_info()
        text_index_info = await app.graph.get_text_index_info()

        elapsed = (time.monotonic() - t0) * 1000

        return {
            "projects": projects,
            "label_counts": label_counts,
            "vector_indices": vec_index_info,
            "text_indices": text_index_info,
            "schema_version": SCHEMA_VERSION,
            "query_ms": round(elapsed, 1),
        }

    @mcp.tool(
        description=(
            "Graph schema reference: node labels, relationship types, kind discriminators, properties. "
            "Call this first to understand what you can query. No database call needed."
        ),
    )
    async def schema_info() -> dict[str, Any]:
        return {
            "node_labels": {
                "code": sorted(lbl.value for lbl in _CODE_LABELS),
                "documentation": sorted(lbl.value for lbl in _DOC_LABELS),
                "external": sorted(lbl.value for lbl in _EXTERNAL_LABELS),
                "meta": [NodeLabel.SCHEMA_VERSION.value],
            },
            "relationship_types": sorted(r.value for r in RelType),
            "kind_discriminators": {
                "TypeDefKind": sorted(k.value for k in TypeDefKind),
                "CallableKind": sorted(k.value for k in CallableKind),
                "ValueKind": sorted(k.value for k in ValueKind),
                "Visibility": sorted(v.value for v in Visibility),
            },
            "common_properties": [
                "uid",
                "name",
                "qualified_name",
                "kind",
                "file_path",
                "line_start",
                "line_end",
                "signature",
                "docstring",
                "visibility",
                "tags",
                "project_name",
            ],
            "text_searchable_labels": sorted(lbl.value for lbl in _TEXT_SEARCHABLE_LABELS),
            "vector_searchable_labels": sorted(lbl.value for lbl in _EMBEDDABLE_LABELS),
            "uid_format": "{project_name}:{qualified_name}",
            "schema_version": SCHEMA_VERSION,
        }
