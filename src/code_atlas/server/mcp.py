"""MCP server for Code Atlas.

Exposes the Memgraph graph database to AI coding agents via MCP tools.
Auto-starts file watcher + pipeline when Valkey is reachable; falls back
to query-only mode otherwise.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
import tomllib
import urllib.parse
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import orjson
from loguru import logger
from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

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
from code_atlas.server.analysis import analyze_repo as _analyze_repo
from code_atlas.server.analysis import generate_diagram as _generate_diagram
from code_atlas.server.health import run_health_checks
from code_atlas.settings import AtlasSettings, derive_project_name, find_git_root
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


def _strip_cypher_string_literals(query: str) -> str:
    """Blank out the contents of single/double-quoted string literals.

    Used before write-keyword scanning so a literal value equal to (or
    containing) a write keyword — e.g. ``WHERE n.name = 'set'`` — doesn't
    trigger a false-positive write rejection. Handles backslash-escaped
    quotes inside literals.
    """
    out: list[str] = []
    quote: str | None = None
    escaped = False
    for ch in query:
        if quote is None:
            out.append(ch)
            if ch in ("'", '"'):
                quote = ch
        elif escaped:
            out.append(" ")
            escaped = False
        elif ch == "\\":
            out.append(" ")
            escaped = True
        elif ch == quote:
            out.append(ch)
            quote = None
        else:
            out.append(" ")
    return "".join(out)


_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)

_DEFAULT_LIMIT = 20
_MAX_LIMIT = 100
_DOCSTRING_TRUNCATE = 200


@dataclass
class AppContext:
    graph: GraphClient
    settings: AtlasSettings
    embed: EmbedClient | None
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
    app.resolved_root = new_root
    app.staleness = StalenessChecker(new_root)

    # Re-check embedding model match for new root
    if not app.settings.embeddings.enabled:
        app.embed = None
        app.vector_enabled = False
    else:
        app.embed = EmbedClient(app.settings.embeddings)
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


def _is_project_root(path: Path) -> bool:
    """True when *path* is a plausible Atlas project root — it has its own
    ``atlas.toml`` or is a git root.

    Guards the roots probe from silently hijacking the served project namespace
    with an incidental subdirectory or unrelated folder the MCP client happens
    to advertise (which would orphan the configured/indexed project).
    """
    return (path / "atlas.toml").is_file() or find_git_root(path) == path


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

    if not _is_project_root(root):
        logger.warning(
            "MCP client root {} is not an Atlas project (no atlas.toml, not a git root) — "
            "keeping the configured project {}; queries stay scoped to the indexed project.",
            root,
            current,
        )
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


def _serialize_value(value: Any) -> Any:
    """Recursively convert neo4j graph objects to plain JSON-serializable values.

    Handles Node and Relationship objects, including when nested inside lists
    (e.g. ``collect(n)`` results) — not just top-level record values.
    """
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if hasattr(value, "items") and hasattr(value, "labels"):
        # neo4j Node object
        node_dict = dict(value.items())
        node_dict["_labels"] = sorted(value.labels)
        return {k: _serialize_value(v) for k, v in node_dict.items()}
    if hasattr(value, "items") and hasattr(value, "type") and hasattr(value, "nodes"):
        # neo4j Relationship object
        rel_dict = dict(value.items())
        rel_dict["_type"] = value.type
        return {k: _serialize_value(v) for k, v in rel_dict.items()}
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    return value


def _serialize_node(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a neo4j record containing Node/Relationship objects (including
    those nested inside lists from aggregations) to plain JSON-serializable dicts."""
    return {key: _serialize_value(value) for key, value in record.items()}


def _compact_node(record: dict[str, Any], *, detail: str = "summary") -> dict[str, Any]:
    """Extract compact metadata from a node record.

    *detail* controls output verbosity:
    - ``"summary"`` (default): truncated docstring, no source code.
    - ``"full"``: full docstring, includes source code.
    """
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
        if detail == "full":
            compact["docstring"] = docstring
        else:
            compact["docstring"] = docstring[:_DOCSTRING_TRUNCATE] + (
                "..." if len(docstring) > _DOCSTRING_TRUNCATE else ""
            )

    if detail == "full":
        source = props.get("source")
        if source:
            compact["source"] = source

    if hasattr(node, "labels"):
        compact["_labels"] = sorted(node.labels)

    # Preserve score/similarity from search results
    for score_key in ("score", "similarity"):
        if score_key in record:
            compact[score_key] = record[score_key]

    return compact


def _compact_node_to_dict(node: CompactNode, *, include_source: bool = True) -> dict[str, Any]:
    """Serialize a CompactNode dataclass to a plain dict for JSON output.

    *include_source*: when ``False``, omits the ``source`` field (used for
    neighborhood nodes in ``get_context`` to reduce payload).
    """
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
    if include_source and node.source:
        out["source"] = node.source
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
    except TimeoutError, QueryTimeoutError:
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


async def _enrich_with_calls(graph: GraphClient, results: list[dict[str, Any]], *, detail: str) -> None:
    """Inject caller/callee stats into *results* dicts in-place.

    In ``"full"`` mode, adds ``caller_count``, ``callee_count``, ``callers``
    (top-5 names), and ``callees`` (top-5 names).  In ``"summary"`` mode this
    is a no-op.
    """
    if detail != "full" or not results:
        return
    uids = [r["uid"] for r in results if "uid" in r]
    if not uids:
        return
    stats = await graph.batch_call_stats(uids)
    for r in results:
        uid = r.get("uid", "")
        st = stats.get(uid)
        if st:
            r["caller_count"] = st.caller_count
            r["callee_count"] = st.callee_count
            r["callers"] = st.caller_names
            r["callees"] = st.callee_names


async def _default_scope_projects(app: AppContext) -> list[str]:
    """Default project scope: the current project plus any monorepo sub-projects.

    Sub-project entities are stored with ``project_name = '{root}/{sub}'``
    (orchestrator.py, watcher.py). Without this expansion, a search with no
    explicit scope/project resolves to the bare root name and silently
    excludes all sub-project code.
    """
    root_name = derive_project_name(app.settings.project_root)
    try:
        project_rows = await app.graph.get_project_status()
    except Exception as exc:
        logger.debug("Could not resolve monorepo sub-projects for default scope: {}", exc)
        return [root_name]

    all_names: list[str] = []
    for row in project_rows:
        node = row.get("n")
        if node:
            props = dict(node.items()) if hasattr(node, "items") else node
            name = props.get("name", "")
            if name:
                all_names.append(name)

    siblings = [n for n in all_names if n == root_name or n.startswith(f"{root_name}/")]
    return siblings or [root_name]


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def create_mcp_server(  # noqa: PLR0915
    settings: AtlasSettings,
    *,
    strict: bool = False,
    host: str = "127.0.0.1",
    port: int = 8000,
    catchup: bool = True,
) -> FastMCP:
    """Create and configure the Code Atlas MCP server.

    *catchup* (default True) runs one blocking delta index pass at startup so
    edits made while the daemon was down are indexed before live consumption.
    Set False to skip it (faster startup at the cost of missing offline edits).
    """

    @asynccontextmanager
    async def app_lifespan(_server: FastMCP) -> AsyncIterator[AppContext]:  # noqa: PLR0915
        init_telemetry(settings.observability)

        graph = GraphClient(settings)
        try:
            await graph.ping()
        except Exception as exc:
            logger.error("Cannot reach Memgraph at {}:{} — {}", settings.memgraph.host, settings.memgraph.port, exc)
            raise

        logger.info("MCP connected to Memgraph at {}:{}", settings.memgraph.host, settings.memgraph.port)

        # Embedding setup — skipped entirely in lightweight mode
        vector_enabled = True
        embed: EmbedClient | None = None
        if not settings.embeddings.enabled:
            vector_enabled = False
            logger.info("Lightweight mode: embeddings disabled, vector search unavailable")
        else:
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
            # Implicit degradation: probe TEI, fall back to lightweight if unreachable
            tei_ok = False
            with contextlib.suppress(Exception):
                tei_ok = await embed.health_check()
            if not tei_ok:
                logger.warning("Embedding service unreachable — running in lightweight mode. Vector search disabled.")
                embed = None
                vector_enabled = False
        staleness = StalenessChecker(settings.project_root)
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

        # Auto-start watcher + pipeline if Valkey is reachable.
        # catchup runs one blocking delta index pass before consumers start.
        daemon_running = await daemon.start(settings, graph, catchup=catchup)
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
            "Use schema_info for Cypher examples, validate_cypher to check queries before running them. "
            "Call get_usage_guide('guidelines') for tips on structuring code for better search results."
        ),
        host=host,
        port=port,
        lifespan=app_lifespan,
    )

    _register_query_tools(mcp)
    _register_search_tools(mcp)
    _register_hybrid_tool(mcp)
    _register_info_tools(mcp)
    _register_subagent_tools(mcp)
    _register_analysis_tools(mcp)
    return mcp


# ---------------------------------------------------------------------------
# Tool registration — split to stay under statement limits
# ---------------------------------------------------------------------------


def _register_node_tools(mcp: FastMCP) -> None:
    """Register the get_node tool (separated for statement-count limits)."""

    @mcp.tool(
        description=(
            "Find code entities by name. "
            "Cascade: exact (uid + name) → partial (suffix > prefix > contains). "
            "First stage with results wins. Results ranked by relevance. "
            "Use get_context to expand a result. "
            "Returns: {results: [{uid, name, qualified_name, kind, file_path, "
            "line_start, line_end, signature, docstring}], count, truncated, query_ms}. "
            "Pass detail='full' to include source code, full docstrings, and caller/callee info."
        ),
    )
    async def get_node(
        name: Annotated[str, Field(description="Entity name, qualified name, or uid to look up.")],
        label: Annotated[
            str,
            Field(
                "",
                description=(
                    "Restrict to a node label: Callable, Module, TypeDef, Value, Package, "
                    "DocFile, DocSection, ExternalSymbol. Empty = all."
                ),
            ),
        ] = "",
        limit: Annotated[int, Field(20, description="Max results to return.", ge=1, le=100)] = 20,
        detail: Annotated[
            str,
            Field(
                "summary",
                description="'summary' (default) or 'full' (add source, full docstrings, call stats).",
            ),
        ] = "summary",
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        app = await _ensure_root(ctx)
        clamped = _clamp_limit(limit)

        if label and label not in NodeLabel:
            valid = ", ".join(sorted(lbl.value for lbl in NodeLabel))
            return {"error": f"Invalid label: {label!r}. Valid labels: {valid}"}
        label_filter = f":{label}" if label else ""
        t0 = time.monotonic()
        found: list[dict[str, Any]] | None = None
        total: int | None = None
        peek = clamped + 1  # fetch one extra row to detect truncation

        try:
            # Stage A: Exact matches (uid + exact name) — 1 RTT
            found = await app.graph.execute(
                f"MATCH (n{label_filter} {{uid: $name}}) RETURN n LIMIT {peek} "
                f"UNION ALL "
                f"MATCH (n{label_filter}) WHERE n.name = $name RETURN n LIMIT {peek}",
                {"name": name},
            )
            if found:
                total = len(found)
                found = found[:clamped]

            # Stage B: Partial matches (suffix > prefix > contains) — 1 RTT
            if not found:
                found = await app.graph.execute(
                    f"MATCH (n{label_filter}) WHERE n.qualified_name ENDS WITH $suffix "
                    f"RETURN n, 3 AS _match_score LIMIT {peek} "
                    f"UNION ALL "
                    f"MATCH (n{label_filter}) WHERE n.qualified_name STARTS WITH $prefix "
                    f"RETURN n, 2 AS _match_score LIMIT {peek} "
                    f"UNION ALL "
                    f"MATCH (n{label_filter}) WHERE n.qualified_name CONTAINS $name OR n.name CONTAINS $name "
                    f"RETURN n, 1 AS _match_score LIMIT {peek}",
                    {"name": name, "suffix": f".{name}", "prefix": f"{name}."},
                )
                # Deduplicate by uid, keeping highest _match_score
                seen: dict[str, dict[str, Any]] = {}
                for r in found:
                    uid = r["n"]["uid"]
                    if r.get("_match_score", 0) > seen.get(uid, {}).get("_match_score", -1):
                        seen[uid] = r
                total = len(seen)
                found = sorted(seen.values(), key=lambda rec: rec.get("_match_score", 0), reverse=True)[:clamped]
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")

        elapsed = (time.monotonic() - t0) * 1000
        ranked = _rank_results([_compact_node(r, detail=detail) for r in found])
        await _enrich_with_calls(app.graph, ranked, detail=detail)
        return await _with_staleness(app, _result(ranked, limit=clamped, query_ms=elapsed, total=total))


def _register_query_tools(mcp: FastMCP) -> None:
    """Register cypher_query and get_context tools."""

    @mcp.tool(
        description=(
            "Execute read-only Cypher against the graph. "
            "LIMIT auto-applied; write operations rejected. "
            "Call schema_info first for available labels and relationships. "
            "Returns: {results: [record, ...], count, truncated, query_ms}."
        ),
    )
    async def cypher_query(
        query: Annotated[str, Field(description="Read-only Cypher query. LIMIT is auto-appended if missing.")],
        limit: Annotated[int, Field(20, description="Max results (auto-appended as LIMIT clause).", ge=1, le=100)] = 20,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        app = await _ensure_root(ctx)
        clamped = _clamp_limit(limit)

        if _WRITE_KEYWORDS.search(_strip_cypher_string_literals(query)):
            return _error("Write operations are not allowed via MCP", code="WRITE_REJECTED")

        # Auto-append LIMIT if missing, requesting one extra row to detect truncation.
        # If the caller supplied their own LIMIT, honor it as-is (truncation unknown).
        auto_limited = not _LIMIT_RE.search(query)
        if auto_limited:
            query = query.rstrip().rstrip(";") + f" LIMIT {clamped + 1}"

        t0 = time.monotonic()
        try:
            records = await app.graph.execute(query)
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")
        except Exception as exc:
            return _error(str(exc), code="QUERY_ERROR")
        elapsed = (time.monotonic() - t0) * 1000

        total = len(records) if auto_limited else None
        if auto_limited:
            records = records[:clamped]

        serialized = [_serialize_node(r) for r in records]
        return await _with_staleness(app, _result(serialized, limit=clamped, query_ms=elapsed, total=total))

    _register_node_tools(mcp)

    @mcp.tool(
        description=(
            "Expand a node into its neighborhood: parent, siblings, callers, callees, docs. "
            "Pass a uid from get_node or hybrid_search results. "
            "Returns: {node, parent, siblings, callers, callees, docs, package_context, query_ms}."
        ),
    )
    async def get_context(
        uid: Annotated[str, Field(description="Unique identifier of the node to expand (from search results).")],
        include_hierarchy: Annotated[bool, Field(True, description="Include parent and sibling entities.")] = True,
        include_calls: Annotated[bool, Field(True, description="Include callers and callees.")] = True,
        call_depth: Annotated[
            int, Field(1, description="CALLS traversal hops (1 = direct callers/callees only).", ge=1, le=3)
        ] = 1,
        include_docs: Annotated[bool, Field(True, description="Include linked documentation sections.")] = True,
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
            "parent": _compact_node_to_dict(expanded.parent, include_source=False) if expanded.parent else None,
            "siblings": [_compact_node_to_dict(s, include_source=False) for s in expanded.siblings],
            "callers": [_compact_node_to_dict(c, include_source=False) for c in expanded.callers],
            "callees": [_compact_node_to_dict(c, include_source=False) for c in expanded.callees],
            "docs": [_compact_node_to_dict(d, include_source=False) for d in expanded.docs],
            "package_context": expanded.package_context,
            "query_ms": round(elapsed, 1),
        }
        return await _with_staleness(app, result)


def _register_search_tools(mcp: FastMCP) -> None:
    """Register text_search and vector_search tools."""

    @mcp.tool(
        description=(
            "BM25 keyword search across code entities. Supports quoted phrases, "
            "field-specific queries (name:X, docstring:Y), wildcards (get*User), "
            "and boolean operators (AND, OR). "
            "Returns: {results: [{uid, name, qualified_name, kind, file_path, "
            "line_start, line_end, signature, docstring, score}], count, truncated, query_ms}. "
            "Pass detail='full' to include source code, full docstrings, and caller/callee info."
        ),
    )
    async def text_search(
        query: Annotated[str, Field(description="BM25 query — supports phrases, wildcards, field:value, AND/OR.")],
        label: Annotated[
            str,
            Field("", description="Restrict to one label: Callable, Module, TypeDef, Value, DocSection. Empty = all."),
        ] = "",
        limit: Annotated[int, Field(20, description="Max results to return.", ge=1, le=100)] = 20,
        project: Annotated[
            str, Field("", description="Filter by project name. Empty = auto-detect from workspace.")
        ] = "",
        detail: Annotated[
            str,
            Field(
                "summary",
                description="'summary' (default) or 'full' (add source, full docstrings, call stats).",
            ),
        ] = "summary",
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        app = await _ensure_root(ctx)
        if label and label not in NodeLabel:
            valid = ", ".join(sorted(lbl.value for lbl in NodeLabel))
            return {"error": f"Invalid label: {label!r}. Valid labels: {valid}"}
        clamped = _clamp_limit(limit)
        resolved_projects = [project] if project else await _default_scope_projects(app)
        resolved_scope = ",".join(resolved_projects)

        t0 = time.monotonic()
        try:
            all_results = await app.graph.text_search(query, label=label, limit=clamped + 1, projects=resolved_projects)
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")
        elapsed = (time.monotonic() - t0) * 1000

        total = len(all_results)
        all_results = all_results[:clamped]
        compacted = [_compact_node(r, detail=detail) for r in all_results]
        await _enrich_with_calls(app.graph, compacted, detail=detail)
        return await _with_staleness(
            app, _result(compacted, limit=clamped, query_ms=elapsed, total=total), scope=resolved_scope
        )

    @mcp.tool(
        description=(
            "Semantic similarity search using vector embeddings. "
            "Finds code by meaning, not just name. "
            "Returns: {results: [{uid, name, qualified_name, kind, file_path, "
            "line_start, line_end, signature, docstring, similarity}], count, truncated, query_ms}. "
            "Pass detail='full' to include source code, full docstrings, and caller/callee info."
        ),
    )
    async def vector_search(
        query: Annotated[str, Field(description="Natural language query — describes what the code does.")],
        label: Annotated[
            str,
            Field("", description="Restrict to one label: Callable, Module, TypeDef, Value, DocSection. Empty = all."),
        ] = "",
        limit: Annotated[int, Field(20, description="Max results to return.", ge=1, le=100)] = 20,
        project: Annotated[
            str, Field("", description="Filter by project name. Empty = auto-detect from workspace.")
        ] = "",
        threshold: Annotated[
            float, Field(0.0, description="Minimum cosine similarity to include a result.", ge=0.0, le=1.0)
        ] = 0.0,
        detail: Annotated[
            str,
            Field(
                "summary",
                description="'summary' (default) or 'full' (add source, full docstrings, call stats).",
            ),
        ] = "summary",
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        app = await _ensure_root(ctx)
        if label and label not in NodeLabel:
            valid = ", ".join(sorted(lbl.value for lbl in NodeLabel))
            return {"error": f"Invalid label: {label!r}. Valid labels: {valid}"}
        if not app.vector_enabled:
            if not app.settings.embeddings.enabled:
                return _error(
                    "Vector search unavailable — embeddings are disabled.",
                    code="EMBEDDINGS_DISABLED",
                )
            return _error(
                "Vector search disabled: embedding model mismatch. Run 'atlas index --full' to re-embed.",
                code="MODEL_MISMATCH",
            )
        clamped = _clamp_limit(limit)
        resolved_projects = [project] if project else await _default_scope_projects(app)
        resolved_scope = ",".join(resolved_projects)

        # Embed the query
        assert app.embed is not None  # guaranteed by vector_enabled guard above
        try:
            vector = await app.embed.embed_one(query)
        except EmbeddingError as exc:
            return _error(f"Embedding service unavailable: {exc}", code="EMBED_ERROR")

        t0 = time.monotonic()
        try:
            all_results = await app.graph.vector_search(
                vector, label=label, limit=clamped + 1, projects=resolved_projects, threshold=threshold
            )
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")
        elapsed = (time.monotonic() - t0) * 1000

        total = len(all_results)
        all_results = all_results[:clamped]
        compacted = [_compact_node(r, detail=detail) for r in all_results]
        await _enrich_with_calls(app.graph, compacted, detail=detail)
        return await _with_staleness(
            app, _result(compacted, limit=clamped, query_ms=elapsed, total=total), scope=resolved_scope
        )


def _parse_search_types(search_types: str) -> tuple[list[SearchType] | None, dict[str, Any] | None]:
    """Parse the comma-separated ``search_types`` param. Returns ``(types, error)``."""
    if not search_types:
        return None, None
    try:
        return [SearchType(s.strip()) for s in search_types.split(",") if s.strip()], None
    except ValueError as exc:
        valid = ", ".join(st.value for st in SearchType)
        return None, _error(f"Invalid search_types: {exc}. Valid channels: {valid}", code="INVALID_SEARCH_TYPES")


def _parse_weights(weights: str) -> tuple[dict[str, float] | None, dict[str, Any] | None]:
    """Parse the ``weights`` JSON param. Returns ``(weights, error)``."""
    if not weights:
        return None, None
    try:
        parsed = orjson.loads(weights)
    except orjson.JSONDecodeError:
        return None, _error("Invalid weights JSON", code="INVALID_WEIGHTS")
    if not isinstance(parsed, dict):
        return None, _error('weights must be a JSON object, e.g. {"graph": 2.0}', code="INVALID_WEIGHTS")
    return parsed, None


async def _resolve_hybrid_scope(app: AppContext, scope: str) -> str | None:
    """Resolve hybrid_search's project scope.

    Default (empty scope) expands to include monorepo sub-projects (stored as
    '{root}/{sub}'); explicit glob/comma scope expands against indexed project names.

    Returns ``None`` when a non-empty scope (glob/comma list) matches zero
    projects — callers MUST treat that as an explicit "match nothing"
    restriction (skip searching, return no results), not as "" which
    hybrid_search treats as "no filter at all".
    """
    if not scope:
        return ",".join(await _default_scope_projects(app))
    if "*" not in scope and "," not in scope:
        return scope

    project_rows = await app.graph.get_project_status()
    all_project_names = []
    for row in project_rows:
        node = row.get("n")
        if node:
            props = dict(node.items()) if hasattr(node, "items") else node
            all_project_names.append(props.get("name", ""))
    expanded = expand_scope(scope, all_project_names, app.settings.monorepo.always_include)
    if not expanded:
        return None
    # Pass expanded projects directly — use empty scope and set projects on search calls
    return ",".join(expanded)


def _register_hybrid_tool(mcp: FastMCP) -> None:
    """Register the hybrid_search tool."""

    @mcp.tool(
        description=(
            "Primary search tool — fuses graph name-matching, BM25 keyword, and vector semantic "
            "search via Reciprocal Rank Fusion (RRF). Auto-adjusts weights by query shape. "
            "By default excludes test entities, .pyi stubs, and generated code. "
            "Code entities (Callable, TypeDef, Module, Value) are boosted over documentation. "
            "Set code_only=true to exclude DocSection/DocFile entirely. "
            "Returns: {results: [{uid, name, qualified_name, kind, file_path, line_start, "
            "line_end, signature, docstring, visibility, _labels, rrf_score, sources}], "
            "count, truncated, query_ms}. "
            "Pass detail='full' to include source code, full docstrings, and caller/callee info."
        ),
    )
    async def hybrid_search(
        query: Annotated[str, Field(description="Search query — identifier names, natural language, or mixed.")],
        limit: Annotated[int, Field(20, description="Max results to return.", ge=1, le=100)] = 20,
        search_types: Annotated[
            str, Field("", description="Comma-separated channels to use: graph,vector,bm25. Empty = all.")
        ] = "",
        scope: Annotated[
            str,
            Field(
                "",
                description="Project name filter. Comma-separated names or globs for monorepos. "
                "Empty = auto-detect from workspace.",
            ),
        ] = "",
        weights: Annotated[
            str,
            Field(
                "",
                description='Channel weight overrides as JSON, e.g. {"graph": 2.0, "vector": 0.5}. Empty = auto.',
            ),
        ] = "",
        exclude_tests: Annotated[
            bool | None, Field(None, description="Exclude test files/entities. Default true for non-test queries.")
        ] = None,
        exclude_stubs: Annotated[bool | None, Field(None, description="Exclude .pyi stub files. Default true.")] = None,
        exclude_generated: Annotated[
            bool | None, Field(None, description="Exclude generated code. Default true.")
        ] = None,
        code_only: Annotated[
            bool,
            Field(False, description="Exclude documentation entities (DocSection, DocFile). Return only code."),
        ] = False,
        detail: Annotated[
            str,
            Field(
                "summary",
                description="'summary' (default) or 'full' (add source, full docstrings, call stats).",
            ),
        ] = "summary",
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        app = await _ensure_root(ctx)
        clamped = _clamp_limit(limit)

        types, type_error = _parse_search_types(search_types)
        if type_error:
            return type_error

        weight_dict, weight_error = _parse_weights(weights)
        if weight_error:
            return weight_error

        resolved_scope = await _resolve_hybrid_scope(app, scope)
        if resolved_scope is None:
            # scope explicitly matched zero indexed projects — return no
            # results rather than silently searching every project.
            return await _with_staleness(app, _result([], limit=clamped, query_ms=0.0, total=0), scope=scope)

        t0 = time.monotonic()
        try:
            results = await _hybrid_search(
                graph=app.graph,
                embed=app.embed if app.vector_enabled else None,
                settings=app.settings.search,
                query=query,
                search_types=types,
                limit=clamped + 1,
                scope=resolved_scope,
                weights=weight_dict,
                exclude_tests=exclude_tests,
                exclude_stubs=exclude_stubs,
                exclude_generated=exclude_generated,
                code_only=code_only,
            )
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")
        elapsed = (time.monotonic() - t0) * 1000

        total = len(results)
        results = results[:clamped]

        serialized = []
        for r in results:
            entry: dict[str, Any] = {
                "uid": r.uid,
                "name": r.name,
                "qualified_name": r.qualified_name,
                "kind": r.kind,
                "file_path": r.file_path,
                "line_start": r.line_start,
                "line_end": r.line_end,
                "signature": r.signature,
                "visibility": r.visibility,
                "_labels": r.labels,
                "rrf_score": round(r.rrf_score, 6),
                "sources": r.sources,
            }
            if detail == "full":
                entry["docstring"] = r.docstring or ""
                if r.source:
                    entry["source"] = r.source
            else:
                entry["docstring"] = (
                    r.docstring[:_DOCSTRING_TRUNCATE] + ("..." if len(r.docstring) > _DOCSTRING_TRUNCATE else "")
                    if r.docstring
                    else ""
                )
            serialized.append(entry)

        await _enrich_with_calls(app.graph, serialized, detail=detail)
        return await _with_staleness(
            app, _result(serialized, limit=clamped, query_ms=elapsed, total=total), scope=scope
        )


def _register_info_tools(mcp: FastMCP) -> None:
    """Register index_status and schema_info tools."""

    @mcp.tool(
        description=(
            "Show indexed projects, entity counts, and schema version. "
            "Use this to understand what data is available before querying. "
            "Returns: {projects: [{name, file_count, entity_count, last_indexed_at, git_hash}], "
            "label_counts, vector_indices, text_indices, schema_version, query_ms}."
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
            "List all indexed projects with dependency relationships. "
            "Returns: {results: [{name, file_count, entity_count, last_indexed_at, "
            "git_hash, depends_on, depended_by}], count, truncated, query_ms}."
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
            "properties, and Cypher examples. "
            "Returns: {node_labels, relationship_types, relationship_summary, "
            "kind_discriminators, common_properties, text_searchable_labels, "
            "vector_searchable_labels, cypher_examples, uid_format, schema_version}."
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
            "Check infrastructure health: Memgraph, TEI, Valkey, schema, config, index, pipeline. "
            "Returns: {ok: bool, degraded: bool, checks: [{name, status, message, detail, suggestion}], "
            "elapsed_ms}. degraded is true when any check is WARN/FAIL (e.g. Valkey down = auto-indexing off)."
        ),
    )
    async def health_check(ctx: Context = None) -> dict[str, Any]:  # type: ignore[assignment]
        app = await _ensure_root(ctx)
        report = await run_health_checks(app.settings, graph=app.graph, embed=app.embed, daemon=app.daemon)
        return {
            "ok": report.ok,
            "degraded": report.degraded,
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
            "Catches write ops, invalid labels/rels, missing RETURN/LIMIT, unbalanced syntax. "
            "Returns: {valid: bool, issues: [{level, message}]}."
        ),
    )
    async def validate_cypher(
        query: Annotated[str, Field(description="Cypher query to validate (not executed).")],
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
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
            "How to use Code Atlas tools effectively. Returns: {topic, guide (markdown text), related_topics}."
        ),
    )
    async def get_usage_guide(
        topic: Annotated[
            str,
            Field(
                "",
                description=(
                    "Guide topic: 'searching', 'cypher', 'navigation', 'patterns', 'guidelines'. Empty = quick-start."
                ),
            ),
        ] = "",
    ) -> dict[str, Any]:
        return get_guide(topic)

    @mcp.tool(
        description=(
            "Analyze a question and recommend which search tool + parameters to use. "
            "Returns: {question, recommended_tool, params, reasoning}."
        ),
    )
    async def plan_search_strategy(
        question: Annotated[str, Field(description="The question or task you want to search for.")],
    ) -> dict[str, Any]:
        return plan_strategy(question)


def _register_analysis_tools(mcp: FastMCP) -> None:
    """Register repository analysis and diagram generation tools."""

    @mcp.tool(
        description=(
            "Analyze repository structure, centrality, dependencies, patterns, or quality. "
            "Returns: {analysis, project, ...analysis-specific keys, query_ms}."
        ),
    )
    async def analyze_repo(
        analysis: Annotated[
            Literal["structure", "centrality", "dependencies", "patterns", "quality"],
            Field(
                description=(
                    "Sub-analysis: structure (entity counts, packages, largest modules), "
                    "centrality (hub entities/modules, leaves), "
                    "dependencies (imports, cross-package coupling, circular deps), "
                    "patterns (inheritance, enums, visibility, docstring coverage), "
                    "quality (health score, god modules, circular deps, tangled modules, coupling, instability)."
                ),
            ),
        ],
        project: Annotated[str, Field("", description="Project name. Empty = auto-detect from workspace.")] = "",
        path: Annotated[
            str, Field("", description="Scope analysis to a file or package path prefix. Empty = entire project.")
        ] = "",
        limit: Annotated[int, Field(20, description="Max items per sub-section.", ge=1, le=100)] = 20,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        app = await _ensure_root(ctx)
        project_name = project or derive_project_name(app.settings.project_root)
        clamped = _clamp_limit(limit)
        try:
            return await _analyze_repo(app.graph, analysis, project_name, path=path, limit=clamped)
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")

    @mcp.tool(
        description=("Generate Mermaid diagrams of the codebase. Returns: {type, mermaid, node_count, query_ms}."),
    )
    async def generate_diagram(
        type: Annotated[  # noqa: A002
            Literal["packages", "imports", "inheritance", "module_detail"],
            Field(
                description=(
                    "Diagram type: packages (containment tree), imports (module dependencies), "
                    "inheritance (class hierarchy), module_detail (single module's classes + methods — requires path)."
                ),
            ),
        ],
        project: Annotated[str, Field("", description="Project name. Empty = auto-detect from workspace.")] = "",
        path: Annotated[
            str,
            Field("", description="Scope to a file/package path. Required for module_detail, optional otherwise."),
        ] = "",
        max_nodes: Annotated[int, Field(30, description="Maximum nodes in the diagram.", ge=1, le=100)] = 30,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        app = await _ensure_root(ctx)
        project_name = project or derive_project_name(app.settings.project_root)
        max_nodes = max(1, min(max_nodes, _MAX_LIMIT))
        try:
            return await _generate_diagram(app.graph, type, project_name, path=path, max_nodes=max_nodes)
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")
