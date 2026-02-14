"""MCP server for Code Atlas.

Exposes the Memgraph graph database to AI coding agents via MCP tools.
Auto-starts file watcher + pipeline when Valkey is reachable; falls back
to query-only mode otherwise.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import tomllib
import urllib.parse
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from mcp.server.fastmcp import Context, FastMCP

from code_atlas.graph.client import GraphClient, QueryTimeoutError
from code_atlas.indexing.daemon import DaemonManager
from code_atlas.indexing.orchestrator import StalenessChecker
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
from code_atlas.search.embeddings import EmbedClient, EmbeddingError
from code_atlas.search.engine import CompactNode, SearchType, expand_context, expand_scope
from code_atlas.search.engine import hybrid_search as _hybrid_search
from code_atlas.search.guidance import (
    _RELATIONSHIP_SUMMARY,
    CYPHER_EXAMPLES,
    get_guide,
    plan_strategy,
    validate_cypher_explain,
    validate_cypher_static,
)
from code_atlas.server.health import run_health_checks
from code_atlas.settings import AtlasSettings
from code_atlas.telemetry import get_tracer, init_telemetry, shutdown_telemetry

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_tracer = get_tracer(__name__)

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
    staleness: StalenessChecker | None = None
    daemon: DaemonManager = field(default_factory=DaemonManager)
    resolved_root: Path | None = field(default=None, repr=False)
    roots_checked: bool = field(default=False, repr=False)
    vector_enabled: bool = field(default=True, repr=False)


# ---------------------------------------------------------------------------
# MCP Roots helpers
# ---------------------------------------------------------------------------

_ROOTS_TIMEOUT = 2.0  # seconds — fast fail for broken/missing clients


def _file_uri_to_path(uri: str) -> Path:
    """Convert a ``file://`` URI to a local :class:`Path` (cross-platform)."""
    parsed = urllib.parse.urlparse(uri)
    return Path(urllib.request.url2pathname(parsed.path))


async def _try_list_roots(ctx: Context) -> Path | None:
    """Attempt to get the first root from the MCP client, with timeout.

    Returns ``None`` on any failure (timeout, no roots, no session).
    """
    try:
        session = ctx.session
        result = await asyncio.wait_for(session.list_roots(), timeout=_ROOTS_TIMEOUT)
        roots = result.roots if hasattr(result, "roots") else result
        if roots:
            uri = str(roots[0].uri)
            if uri.startswith("file://"):
                return _file_uri_to_path(uri)
    except Exception:
        pass
    return None


async def _switch_root(app: AppContext, new_root: Path) -> None:
    """Stop daemon, re-create settings from *new_root*, restart daemon."""
    await app.daemon.stop()
    app.daemon = DaemonManager()

    # Re-read atlas.toml from new root and re-create settings.
    # Init kwargs have highest Pydantic precedence (init > env > toml > default)
    # so env vars still apply for fields not in the toml.
    overrides: dict[str, Any] = {"project_root": new_root}
    toml_path = new_root / "atlas.toml"
    if toml_path.is_file():
        with toml_path.open("rb") as fh:
            overrides.update(tomllib.load(fh))
        overrides["project_root"] = new_root  # ensure root wins over toml

    app.settings = AtlasSettings(**overrides)
    app.embed = EmbedClient(app.settings.embeddings)
    app.resolved_root = new_root
    app.staleness = StalenessChecker(new_root, project_name=new_root.name)

    # Re-check embedding model match for new root
    app.vector_enabled = True
    stored_config = await app.graph.get_embedding_config()
    if stored_config is not None:
        stored_model, _stored_dim = stored_config
        if stored_model != app.settings.embeddings.model:
            logger.warning(
                "Embedding model mismatch after root switch (stored='{}', current='{}'). Vector search disabled.",
                stored_model,
                app.settings.embeddings.model,
            )
            app.vector_enabled = False

    started = await app.daemon.start(app.settings, app.graph)
    if started:
        logger.info("Daemon restarted for new root: {}", new_root)
    else:
        logger.info("Query-only mode for new root: {} (no Valkey)", new_root)


async def _maybe_update_root(app: AppContext, ctx: Context) -> None:
    """On first tool call, try MCP roots.  Restart daemon if root changed.

    Short-circuits immediately after first check via ``roots_checked`` flag.
    """
    if app.roots_checked:
        return
    app.roots_checked = True

    root = await _try_list_roots(ctx)
    if root is None:
        logger.debug("MCP roots unavailable — keeping current root: {}", app.settings.project_root)
        return

    root = root.resolve()
    current = app.settings.project_root.resolve()
    if root == current:
        logger.debug("MCP root matches current root: {}", current)
        return

    logger.info("MCP root differs from current root ({} → {}), switching…", current, root)
    await _switch_root(app, root)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_app_ctx(ctx: Context) -> AppContext:
    """Extract AppContext from the MCP request context."""
    return ctx.request_context.lifespan_context


async def _ensure_root(ctx: Context) -> AppContext:
    """Extract AppContext and ensure MCP roots have been checked."""
    app: AppContext = ctx.request_context.lifespan_context
    await _maybe_update_root(app, ctx)
    return app


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
    for key in (
        "uid",
        "name",
        "qualified_name",
        "kind",
        "file_path",
        "line_start",
        "line_end",
        "signature",
        "visibility",
    ):
        if key in props:
            compact[key] = props[key]

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


def _compact_node_to_dict(node: CompactNode) -> dict[str, Any]:
    """Serialize a CompactNode dataclass to a plain dict for JSON output."""
    out: dict[str, Any] = {
        "uid": node.uid,
        "name": node.name,
        "qualified_name": node.qualified_name,
        "kind": node.kind,
        "file_path": node.file_path,
    }
    if node.line_start is not None:
        out["line_start"] = node.line_start
    if node.line_end is not None:
        out["line_end"] = node.line_end
    if node.signature:
        out["signature"] = node.signature
    if node.docstring:
        out["docstring"] = node.docstring[:_DOCSTRING_TRUNCATE] + (
            "..." if len(node.docstring) > _DOCSTRING_TRUNCATE else ""
        )
    if node.labels:
        out["_labels"] = node.labels
    return out


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


async def _with_staleness(app: AppContext, result: dict[str, Any], *, scope: str = "") -> dict[str, Any]:
    """Annotate a query result envelope with staleness info.

    - ``stale_mode == "ignore"``: return result unchanged.
    - ``stale_mode == "lock"`` and stale: return an error envelope.
    - ``stale_mode == "warn"``: add ``stale``, ``stale_since``, ``changed_files`` keys.

    When staleness is indeterminate (project never indexed or index in progress),
    ``stale`` is set to ``None`` rather than ``True``.
    """
    stale_mode = app.settings.index.stale_mode
    if stale_mode == "ignore" or app.staleness is None:
        return result

    checker = app.staleness
    # Only check matching project — scope may be comma-separated
    if scope:
        scope_names = {s.strip() for s in scope.split(",") if s.strip()}
        if checker.project_name not in scope_names:
            return result

    try:
        info = await asyncio.wait_for(
            checker.check(app.graph, include_changed=(stale_mode == "warn")),
            timeout=5.0,
        )
    except TimeoutError:
        logger.warning("Staleness check timed out — skipping annotation")
        return result

    if stale_mode == "lock" and info.stale:
        msg = "Index is stale"
        if info.last_indexed_commit:
            msg += f" (last indexed: {info.last_indexed_commit[:8]})"
        msg += ". Re-index before querying."
        return _error(msg, code="STALE_INDEX")

    # warn mode (default)
    # Indeterminate: stale=True but never indexed (no stored commit)
    if info.stale and info.last_indexed_commit is None:
        result["stale"] = None  # indeterminate — never indexed or index in progress
    else:
        result["stale"] = info.stale
    if info.stale:
        result["stale_since"] = info.last_indexed_commit
        if info.changed_files:
            result["changed_files"] = info.changed_files
    return result


# Visibility ranking: lower = more relevant (public entities preferred)
_VISIBILITY_RANK: dict[str, int] = {"public": 0, "protected": 1, "internal": 2, "private": 3}


def _rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank disambiguation results by relevance.

    Sorting criteria (applied in order):
    0. Internal over external — ExternalSymbol/ExternalPackage stubs last
    1. Source over test — entities whose file_path does NOT contain "test" first
    2. Visibility — public > protected > internal > private
    3. Shorter qualified_name — more canonical entities first
    """

    def _sort_key(node: dict[str, Any]) -> tuple[int, int, int, int]:
        labels = node.get("_labels") or []
        is_external = 1 if any(lbl in ("ExternalSymbol", "ExternalPackage") for lbl in labels) else 0
        fp = (node.get("file_path") or "").lower()
        is_test = 1 if ("test" in fp) else 0
        vis = _VISIBILITY_RANK.get(node.get("visibility", "public"), 0)
        qn_len = len(node.get("qualified_name", ""))
        return (is_external, is_test, vis, qn_len)

    return sorted(results, key=_sort_key)


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def create_mcp_server(settings: AtlasSettings, *, strict: bool = False) -> FastMCP:  # noqa: PLR0915
    """Create and configure the Code Atlas MCP server."""

    @asynccontextmanager
    async def app_lifespan(_server: FastMCP) -> AsyncIterator[AppContext]:
        init_telemetry(settings.observability)

        graph = GraphClient(settings)
        try:
            await graph.ping()
        except Exception as exc:
            logger.error("Cannot reach Memgraph at {}:{} — {}", settings.memgraph.host, settings.memgraph.port, exc)
            raise

        logger.info("MCP connected to Memgraph at {}:{}", settings.memgraph.host, settings.memgraph.port)

        # Check embedding model lock
        vector_enabled = True
        stored_config = await graph.get_embedding_config()
        if stored_config is not None:
            stored_model, _stored_dim = stored_config
            if stored_model != settings.embeddings.model:
                if strict:
                    await graph.close()
                    msg = (
                        f"Embedding model mismatch: stored='{stored_model}', "
                        f"configured='{settings.embeddings.model}'. "
                        "Refusing to start in strict mode. Run 'atlas index --full' to re-embed."
                    )
                    raise RuntimeError(msg)
                logger.warning(
                    "Embedding model mismatch (stored='{}', current='{}'). Vector search disabled.",
                    stored_model,
                    settings.embeddings.model,
                )
                vector_enabled = False

        embed = EmbedClient(settings.embeddings)
        staleness = StalenessChecker(settings.project_root, project_name=settings.project_root.name)
        daemon = DaemonManager()
        app_ctx = AppContext(
            graph=graph,
            settings=settings,
            embed=embed,
            staleness=staleness,
            daemon=daemon,
            resolved_root=settings.project_root,
            vector_enabled=vector_enabled,
        )

        # Register handler for roots/list_changed notification so we re-probe
        # on next tool call.  _mcp_server.notification_handlers is private API
        # in FastMCP — the only way to register notification handlers today.
        try:
            raw = _server._mcp_server  # noqa: SLF001

            async def _on_roots_changed(*_args: object, **_kwargs: object) -> None:
                app_ctx.roots_checked = False
                logger.debug("Received roots/list_changed — will re-probe on next tool call")

            raw.notification_handlers["notifications/roots/list_changed"] = _on_roots_changed  # type: ignore[index]
        except Exception:
            logger.debug("Could not register roots/list_changed handler — root updates via notification disabled")

        # Auto-start watcher + pipeline if Valkey is reachable
        daemon_running = await daemon.start(settings, graph)
        if daemon_running:
            logger.info("Auto-indexing active (watching {})", settings.project_root)
        else:
            logger.info("Query-only mode (no Valkey)")

        try:
            yield app_ctx
        finally:
            await app_ctx.daemon.stop()
            await graph.close()
            shutdown_telemetry()
            logger.info("MCP server shut down")

    mcp = FastMCP(
        name="code-atlas",
        instructions=(
            "Code Atlas — graph-powered code intelligence. "
            "Start with get_usage_guide for workflow guidance. "
            "Use hybrid_search as the primary search tool. "
            "Use get_node to find entities by name, get_context to expand neighborhoods. "
            "Use schema_info for Cypher examples, validate_cypher to check queries before running them."
        ),
        lifespan=app_lifespan,
    )

    _register_query_tools(mcp)
    _register_search_tools(mcp)
    _register_hybrid_tool(mcp)
    _register_info_tools(mcp)
    _register_subagent_tools(mcp)
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
        app = await _ensure_root(ctx)
        clamped = _clamp_limit(limit)

        label_filter = f":{label}" if label else ""
        t0 = time.monotonic()
        found: list[dict[str, Any]] | None = None

        try:
            # Stage 1: Exact uid match (if name contains ':')
            if ":" in name:
                records = await app.graph.execute(
                    f"MATCH (n{label_filter} {{uid: $name}}) RETURN n LIMIT {clamped}",
                    {"name": name},
                )
                if records:
                    found = records

            # Stage 2: Exact name match
            if found is None:
                records = await app.graph.execute(
                    f"MATCH (n{label_filter}) WHERE n.name = $name RETURN n LIMIT {clamped}",
                    {"name": name},
                )
                if records:
                    found = records

            # Stage 3: ENDS WITH suffix match
            if found is None:
                suffix = f".{name}"
                records = await app.graph.execute(
                    f"MATCH (n{label_filter}) WHERE n.qualified_name ENDS WITH $suffix RETURN n LIMIT {clamped}",
                    {"suffix": suffix},
                )
                if records:
                    found = records

            # Stage 4: STARTS WITH prefix match
            if found is None:
                prefix = f"{name}."
                records = await app.graph.execute(
                    f"MATCH (n{label_filter}) WHERE n.qualified_name STARTS WITH $prefix RETURN n LIMIT {clamped}",
                    {"prefix": prefix},
                )
                if records:
                    found = records

            # Stage 5: CONTAINS match (qualified_name or name)
            if found is None:
                found = await app.graph.execute(
                    f"MATCH (n{label_filter}) WHERE n.qualified_name CONTAINS $name OR n.name CONTAINS $name "
                    f"RETURN n LIMIT {clamped}",
                    {"name": name},
                )
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")

        elapsed = (time.monotonic() - t0) * 1000
        ranked = _rank_results([_compact_node(r) for r in found])
        return await _with_staleness(app, _result(ranked, limit=clamped, query_ms=elapsed))


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
        app = await _ensure_root(ctx)
        clamped = _clamp_limit(limit)

        if _WRITE_KEYWORDS.search(query):
            return _error("Write operations are not allowed via MCP", code="WRITE_REJECTED")

        # Auto-append LIMIT if missing
        if not _LIMIT_RE.search(query):
            query = query.rstrip().rstrip(";") + f" LIMIT {clamped}"

        t0 = time.monotonic()
        try:
            records = await app.graph.execute(query)
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")
        except Exception as exc:
            return _error(str(exc), code="QUERY_ERROR")
        elapsed = (time.monotonic() - t0) * 1000

        serialized = [_serialize_node(r) for r in records]
        return await _with_staleness(app, _result(serialized, limit=clamped, query_ms=elapsed))

    _register_node_tools(mcp)

    @mcp.tool(
        description=(
            "Expand a node into its neighborhood: parent, siblings, callers, callees, docs. "
            "Pass the uid from a get_node or hybrid_search result. "
            "Toggle sections with include_hierarchy, include_calls, include_docs. "
            "call_depth controls CALLS traversal hops (default 1, max 3)."
        ),
    )
    async def get_context(
        uid: str,
        include_hierarchy: bool = True,
        include_calls: bool = True,
        call_depth: int = 1,
        include_docs: bool = True,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        app = await _ensure_root(ctx)
        t0 = time.monotonic()

        try:
            expanded = await expand_context(
                app.graph,
                uid,
                include_hierarchy=include_hierarchy,
                include_calls=include_calls,
                call_depth=call_depth,
                include_docs=include_docs,
                max_siblings=app.settings.search.max_siblings,
                max_callers=app.settings.search.max_callers,
            )
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")

        if expanded is None:
            return _error(f"Node not found: {uid}", code="NOT_FOUND")

        elapsed = (time.monotonic() - t0) * 1000

        result = {
            "node": _compact_node_to_dict(expanded.target),
            "parent": _compact_node_to_dict(expanded.parent) if expanded.parent else None,
            "siblings": [_compact_node_to_dict(s) for s in expanded.siblings],
            "callers": [_compact_node_to_dict(c) for c in expanded.callers],
            "callees": [_compact_node_to_dict(c) for c in expanded.callees],
            "docs": [_compact_node_to_dict(d) for d in expanded.docs],
            "package_context": expanded.package_context,
            "query_ms": round(elapsed, 1),
        }
        return await _with_staleness(app, result)


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
        app = await _ensure_root(ctx)
        clamped = _clamp_limit(limit)

        t0 = time.monotonic()
        try:
            all_results = await app.graph.text_search(query, label=label, limit=clamped, project=project)
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")
        elapsed = (time.monotonic() - t0) * 1000

        compacted = [_compact_node(r) for r in all_results]
        return await _with_staleness(app, _result(compacted, limit=clamped, query_ms=elapsed), scope=project)

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
        app = await _ensure_root(ctx)
        if not app.vector_enabled:
            return _error(
                "Vector search disabled: embedding model mismatch. Run 'atlas index --full' to re-embed.",
                code="MODEL_MISMATCH",
            )
        clamped = _clamp_limit(limit)

        # Embed the query
        try:
            vector = await app.embed.embed_one(query)
        except EmbeddingError as exc:
            return _error(f"Embedding service unavailable: {exc}", code="EMBED_ERROR")

        t0 = time.monotonic()
        try:
            all_results = await app.graph.vector_search(
                vector, label=label, limit=clamped, project=project, threshold=threshold
            )
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")
        elapsed = (time.monotonic() - t0) * 1000

        compacted = [_compact_node(r) for r in all_results]
        return await _with_staleness(app, _result(compacted, limit=clamped, query_ms=elapsed), scope=project)


def _register_hybrid_tool(mcp: FastMCP) -> None:
    """Register the hybrid_search tool."""

    @mcp.tool(
        description=(
            "Primary search tool — fuses graph name-matching, BM25 keyword search, and vector "
            "semantic search via Reciprocal Rank Fusion (RRF). "
            "Automatically adjusts channel weights based on query shape: identifier-like queries "
            "boost graph+BM25, natural language boosts vector. "
            "By default, test entities, .pyi stubs, and generated code are excluded. "
            "Set exclude_tests/exclude_stubs/exclude_generated to false to include them "
            "(e.g. to find tests for a function). "
            "Optional: search_types (comma-separated: graph,vector,bm25), "
            'scope (project name filter), weights (JSON: {"graph": 2.0}). '
            "Returns results ranked by fused score with provenance (which channels found each result)."
        ),
    )
    async def hybrid_search(
        query: str,
        limit: int = 20,
        search_types: str = "",
        scope: str = "",
        weights: str = "",
        exclude_tests: bool | None = None,
        exclude_stubs: bool | None = None,
        exclude_generated: bool | None = None,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        app = await _ensure_root(ctx)
        clamped = _clamp_limit(limit)

        # Parse search_types
        types: list[SearchType] | None = None
        if search_types:
            types = [SearchType(s.strip()) for s in search_types.split(",") if s.strip()]

        # Parse weights
        weight_dict: dict[str, float] | None = None
        if weights:
            try:
                weight_dict = json.loads(weights)
            except json.JSONDecodeError:
                return _error("Invalid weights JSON", code="INVALID_WEIGHTS")

        # Resolve scope through expand_scope for monorepo glob/comma support
        resolved_scope = scope
        if scope and ("*" in scope or "," in scope):
            project_rows = await app.graph.get_project_status()
            all_project_names = []
            for row in project_rows:
                node = row.get("n")
                if node:
                    props = dict(node.items()) if hasattr(node, "items") else node
                    all_project_names.append(props.get("name", ""))
            expanded = expand_scope(scope, all_project_names, app.settings.monorepo.always_include)
            # Pass expanded projects directly — use empty scope and set projects on search calls
            resolved_scope = ",".join(expanded) if expanded else ""

        t0 = time.monotonic()
        try:
            results = await _hybrid_search(
                graph=app.graph,
                embed=app.embed if app.vector_enabled else None,
                settings=app.settings.search,
                query=query,
                search_types=types,
                limit=clamped,
                scope=resolved_scope,
                weights=weight_dict,
                exclude_tests=exclude_tests,
                exclude_stubs=exclude_stubs,
                exclude_generated=exclude_generated,
            )
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")
        elapsed = (time.monotonic() - t0) * 1000

        serialized = [
            {
                "uid": r.uid,
                "name": r.name,
                "qualified_name": r.qualified_name,
                "kind": r.kind,
                "file_path": r.file_path,
                "line_start": r.line_start,
                "line_end": r.line_end,
                "signature": r.signature,
                "docstring": (
                    r.docstring[:_DOCSTRING_TRUNCATE] + ("..." if len(r.docstring) > _DOCSTRING_TRUNCATE else "")
                    if r.docstring
                    else ""
                ),
                "visibility": r.visibility,
                "_labels": r.labels,
                "rrf_score": round(r.rrf_score, 6),
                "sources": r.sources,
            }
            for r in results
        ]
        return await _with_staleness(app, _result(serialized, limit=clamped, query_ms=elapsed), scope=scope)


def _register_info_tools(mcp: FastMCP) -> None:
    """Register index_status and schema_info tools."""

    @mcp.tool(
        description=(
            "Show indexed projects, entity counts, and schema version. "
            "Use this to understand what data is available before querying."
        ),
    )
    async def index_status(ctx: Context = None) -> dict[str, Any]:  # type: ignore[assignment]
        app = await _ensure_root(ctx)
        t0 = time.monotonic()

        try:
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
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")

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
            "List all indexed projects with metadata and dependencies. "
            "For monorepos, shows sub-projects with DEPENDS_ON relationships. "
            "Returns name, file_count, entity_count, depends_on, depended_by for each project."
        ),
    )
    async def list_projects(ctx: Context = None) -> dict[str, Any]:  # type: ignore[assignment]
        app = await _ensure_root(ctx)
        t0 = time.monotonic()

        try:
            projects_raw = await app.graph.get_project_status()
            if not projects_raw:
                return _result([], limit=0, query_ms=0)

            # Collect DEPENDS_ON relationships
            depends_records = await app.graph.execute(
                "MATCH (a:Project)-[:DEPENDS_ON]->(b:Project) RETURN a.name AS from_proj, b.name AS to_proj"
            )
            depends_on_map: dict[str, list[str]] = {}
            depended_by_map: dict[str, list[str]] = {}
            for r in depends_records:
                depends_on_map.setdefault(r["from_proj"], []).append(r["to_proj"])
                depended_by_map.setdefault(r["to_proj"], []).append(r["from_proj"])

            result_list = []
            for row in projects_raw:
                node = row.get("n")
                if node is None:
                    continue
                props = dict(node.items()) if hasattr(node, "items") else node
                name = props.get("name", "?")
                entity_count = await app.graph.count_entities(name)
                result_list.append(
                    {
                        "name": name,
                        "file_count": props.get("file_count"),
                        "entity_count": entity_count,
                        "last_indexed_at": props.get("last_indexed_at"),
                        "git_hash": props.get("git_hash"),
                        "depends_on": sorted(depends_on_map.get(name, [])),
                        "depended_by": sorted(depended_by_map.get(name, [])),
                    }
                )
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")

        elapsed = (time.monotonic() - t0) * 1000
        return _result(result_list, limit=len(result_list), query_ms=elapsed)

    @mcp.tool(
        description=(
            "Graph schema reference: node labels, relationship types, kind discriminators, "
            "properties, Cypher examples. Call this first to understand what you can query."
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
            "relationship_summary": dict(_RELATIONSHIP_SUMMARY),
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
            "cypher_examples": list(CYPHER_EXAMPLES),
            "uid_format": "{project_name}:{qualified_name}",
            "schema_version": SCHEMA_VERSION,
        }

    @mcp.tool(
        description=(
            "Check infrastructure health: Memgraph, TEI, Valkey, schema, config, index. "
            "Returns ok (bool), per-check status with messages and suggestions, and elapsed_ms."
        ),
    )
    async def health_check(ctx: Context = None) -> dict[str, Any]:  # type: ignore[assignment]
        app = await _ensure_root(ctx)
        report = await run_health_checks(app.settings, graph=app.graph, embed=app.embed)
        return {
            "ok": report.ok,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "detail": c.detail,
                    "suggestion": c.suggestion,
                }
                for c in report.checks
            ],
            "elapsed_ms": round(report.elapsed_ms, 1),
        }


def _register_subagent_tools(mcp: FastMCP) -> None:
    """Register subagent guidance tools: validate_cypher, get_usage_guide, plan_search_strategy."""

    @mcp.tool(
        description=(
            "Check Cypher for errors before running it. "
            "Catches write ops, invalid labels/relationships, missing RETURN/LIMIT, unbalanced syntax. "
            "Returns issues list (empty = valid)."
        ),
    )
    async def validate_cypher(query: str, ctx: Context = None) -> dict[str, Any]:  # type: ignore[assignment]
        issues = validate_cypher_static(query)

        # Try EXPLAIN against live DB if available
        try:
            app = await _ensure_root(ctx)
            explain_issue = await validate_cypher_explain(app.graph, query)
            if explain_issue is not None:
                issues.append(explain_issue)
        except Exception:
            pass  # No DB context — static checks only

        has_errors = any(i.level == "error" for i in issues)
        return {
            "valid": not has_errors,
            "issues": [{"level": i.level, "message": i.message} for i in issues],
        }

    @mcp.tool(
        description=(
            "How to use Code Atlas tools. "
            "Call with no args for a quick-start guide, or pass a topic: "
            "'searching', 'cypher', 'navigation', 'patterns'."
        ),
    )
    async def get_usage_guide(topic: str = "") -> dict[str, Any]:
        return get_guide(topic)

    @mcp.tool(
        description="Analyze a question and recommend which search tool to use with suggested parameters.",
    )
    async def plan_search_strategy(question: str) -> dict[str, Any]:
        return plan_strategy(question)
