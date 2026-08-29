"""MCP server for Code Atlas.

Exposes the Memgraph graph database to AI coding agents via MCP tools.
Auto-starts file watcher + pipeline when Valkey is reachable; falls back
to query-only mode otherwise.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import re
import time
import tomllib
import urllib.parse
import urllib.request
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import orjson
from loguru import logger
from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

from code_atlas.backends import graph_backend_label, use_backends
from code_atlas.backends.sqlite_graph import SqliteGraphClient
from code_atlas.dream import VaultRoot, build_dream_report, report_to_dict
from code_atlas.graph.client import GraphClient, QueryTimeoutError
from code_atlas.indexing.daemon import DaemonManager
from code_atlas.indexing.orchestrator import StalenessChecker
from code_atlas.schema import (
    _CODE_LABELS,
    _DOC_LABELS,
    _EMBEDDABLE_LABELS,
    _EXTERNAL_LABELS,
    _MARKER_LABELS,
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
from code_atlas.search.engine import (
    CompactNode,
    SearchMode,
    SearchType,
    expand_context,
    expand_scope,
    filter_raw_records,
    matches_test_pattern,
)
from code_atlas.search.engine import hybrid_search as _hybrid_search
from code_atlas.search.guidance import (
    _RELATIONSHIP_SUMMARY,
    CYPHER_EXAMPLES,
    ValidationIssue,
    get_guide,
    plan_strategy,
    validate_cypher_explain,
    validate_cypher_static,
)
from code_atlas.server.analysis import (
    _DEFAULT_BLAST_EDGE_TYPES,
    _DEFAULT_TRACE_EDGE_TYPES,
    _padded_limit,
    more_available_notice,
    truncation_notice,
)
from code_atlas.server.analysis import analyze_repo as _analyze_repo
from code_atlas.server.analysis import blast_radius as _blast_radius
from code_atlas.server.analysis import generate_diagram as _generate_diagram
from code_atlas.server.analysis import trace_path as _trace_path
from code_atlas.server.health import run_health_checks
from code_atlas.settings import AtlasSettings, SearchSettings, derive_project_name, find_git_root
from code_atlas.telemetry import get_metrics, get_tracer, init_telemetry, mark_span_error, shutdown_telemetry

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from code_atlas.events import EventBus

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


def _ready_event() -> asyncio.Event:
    """Default ``first_index_ready`` — already set.

    Ordinary restarts against an already-provisioned backend (the common
    case, and every existing caller that builds an ``AppContext`` without
    passing this explicitly) must never block on it.
    """
    event = asyncio.Event()
    event.set()
    return event


@dataclass
class AppContext:
    graph: GraphClient
    #: Owned by the lifespan, shared with the daemon and the health check. Held here
    #: rather than reached for through `daemon.bus`, which is None whenever indexing is
    #: off and left the health check opening a second connection to answer "are you
    #: reachable".
    bus: EventBus
    settings: AtlasSettings
    embed: EmbedClient | None
    staleness: StalenessChecker | None = None
    daemon: DaemonManager = field(default_factory=DaemonManager)
    resolved_root: Path | None = field(default=None, repr=False)
    roots_checked: bool = field(default=False, repr=False)
    vector_enabled: bool = field(default=True, repr=False)
    needs_first_index: bool = field(default=False, repr=False)
    first_index_ready: asyncio.Event = field(default_factory=_ready_event, repr=False)


# ---------------------------------------------------------------------------
# MCP Roots helpers
# ---------------------------------------------------------------------------

_ROOTS_TIMEOUT = 2.0  # seconds — fast fail for broken/missing clients

# Named rather than inlined at the call site so a test can shrink it. As a literal it was
# unpatchable, and the two tests that exercise the timeout branch had no choice but to sit
# through the real five seconds -- a third of that suite's runtime in two tests.
_STALENESS_TIMEOUT_S = 5.0


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
        app.embed = EmbedClient(app.settings.embeddings, app.settings.redis)
        app.vector_enabled = True
        # Per project, not database-wide: comparing against the database default
        # disabled vector search for every project that was not the last one to
        # index this shared store (ATL-135).
        project_model = await app.graph.get_project_embedding_model(derive_project_name(app.settings.project_root))
        if project_model is not None and project_model != app.settings.embeddings.model:
            logger.warning(
                "Embedding model mismatch after root switch (this project indexed under '{}', "
                "current='{}'). Vector search disabled.",
                project_model,
                app.settings.embeddings.model,
            )
            app.vector_enabled = False

    started = await app.daemon.start(app.settings, app.graph, app.bus)
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


_INDEX_READY_TIMEOUT_S = 600.0  # bounded wait for first-index catch-up before failing loudly


class IndexNotReadyError(Exception):
    """Raised by :func:`_ensure_root` when a genuinely fresh backend's first-index
    catch-up doesn't complete within the bounded wait — see ``INDEX_REQUIRED``."""


async def _ensure_root(ctx: Context, *, require_index: bool = True) -> AppContext:
    """Extract AppContext, ensure MCP roots have been checked, and gate on first-index readiness.

    On a genuinely fresh backend (``needs_first_index``), tool calls block until the
    daemon's startup catch-up index completes (``first_index_ready``), bounded by
    ``_INDEX_READY_TIMEOUT_S`` so an unreachable queue fails fast instead of hanging
    forever. *require_index=False* (health_check, index_status) skips the wait so a
    caller can still see why other tools are blocked.
    """
    app: AppContext = ctx.request_context.lifespan_context
    await _maybe_update_root(app, ctx)
    if require_index and app.needs_first_index and not app.first_index_ready.is_set():
        try:
            await asyncio.wait_for(app.first_index_ready.wait(), timeout=_INDEX_READY_TIMEOUT_S)
        except TimeoutError as exc:
            msg = (
                "Backend has never been indexed and the first-index catch-up did not "
                "complete in time. Check health_check/index_status for pipeline state."
            )
            raise IndexNotReadyError(msg) from exc
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
    if node.stale is not None:
        out["stale"] = node.stale
    if node.anchor_hash is not None:
        out["anchor_hash"] = node.anchor_hash
    return out


def _backend_note(graph: Any) -> dict[str, Any]:
    """Announce the embedded backend, and stay silent about the supported one.

    `backend.graph = "auto"` falls back to SQLite whenever Memgraph is unreachable, so on
    a machine without Docker running this is the default outcome rather than an exotic
    one — and ADR-0015 calls SQLite explicitly not a parity replacement. Until now the
    identity appeared only in log lines an MCP stdio client never sees, so an agent had
    no way to know which engine answered it.

    Emitted only in the degraded case: adding a field to every healthy result would cost
    tokens on every call to say nothing.
    """
    from code_atlas.backends.sqlite_graph import SqliteGraphClient  # noqa: PLC0415

    if not isinstance(graph, SqliteGraphClient):
        return {}
    return {
        "backend": "sqlite-embedded",
        "backend_warning": (
            "Answered by the embedded SQLite fallback, not Memgraph. Community detection is "
            "unavailable and some analyses differ. Run health_check for detail."
        ),
    }


def _result(
    records: list[dict[str, Any]],
    *,
    limit: int,
    query_ms: float,
    total: int | None = None,
    has_more: bool = False,
    remedy: str = "raise `limit` (max 100), or narrow the query",
) -> dict[str, Any]:
    """Consistent result envelope.

    Three states, and the third exists because conflating it with the first is how
    the truncation contract came to lie (ATL-111):

    * *total* known, nothing cut — ``truncated: False``.
    * *total* known, rows cut — ``{shown, total, cut, remedy}``.
    * *total* unknown but rows were cut — pass ``has_more=True`` and get
      ``{shown, total: None, cut: None, has_more: True, remedy}``. A caller reading
      ``cut`` sees "unknown" rather than a number that cannot be right.

    Leaving both ``total`` and ``has_more`` unset still yields ``False``, which is
    correct only where the caller genuinely knows nothing was withheld — a user-written
    Cypher LIMIT, or a list built from a bounded set. It is not a default to reach for.
    """
    if total is not None:
        truncated: Any = truncation_notice(min(limit, total), total, remedy)
    elif has_more:
        truncated = more_available_notice(len(records), remedy)
    else:
        truncated = False
    return {
        "results": records,
        "count": len(records),
        "truncated": truncated,
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


def _resolve_test_patterns(search_settings: SearchSettings, exclude_tests: bool | None) -> tuple[str, ...]:
    """Resolve an analyze_repo-family tool's test_patterns tuple.

    Mirrors hybrid_search's exclude_tests resolution: an explicit True/False
    overrides the settings default; None defers to search.test_filter.
    """
    do_tests = search_settings.test_filter if exclude_tests is None else exclude_tests
    return tuple(search_settings.test_patterns) if do_tests else ()


_MAX_TRAVERSAL_DEPTH = 10


def _clamp_depth(depth: int) -> int:
    """Clamp a trace_path/blast_radius traversal depth to [1, 10]."""
    return max(1, min(depth, _MAX_TRAVERSAL_DEPTH))


def _parse_rel_types(rel_types: str, default: tuple[str, ...]) -> tuple[tuple[str, ...], dict[str, Any] | None]:
    """Parse the comma-separated ``edge_types`` param against RelType. Empty = *default*."""
    if not rel_types:
        return default, None
    parsed = tuple(t.strip() for t in rel_types.split(",") if t.strip())
    invalid = [t for t in parsed if t not in RelType]
    if invalid:
        valid = ", ".join(sorted(r.value for r in RelType))
        return default, _error(f"Invalid edge_types: {invalid}. Valid: {valid}", code="INVALID_EDGE_TYPES")
    return parsed, None


def _unverified_staleness(result: dict[str, Any], stale_mode: str) -> dict[str, Any]:
    """What to answer when freshness could not be established within the timeout.

    ``lock`` fails CLOSED. That mode exists precisely to refuse answers from a stale
    index, so serving one because the check timed out defeats the setting the user
    chose — the one place where carrying on is the wrong default (ATL-111).

    ``warn`` still answers, but says the check did not run. Returning the envelope
    untouched was the bug: an absent ``stale`` key is indistinguishable from a verified
    fresh one to anything looking for it.
    """
    if stale_mode == "lock":
        return _error(
            "Could not verify index freshness within 5s, and stale_mode is 'lock'. "
            "Re-run, or set [index] stale_mode = 'warn' to accept unverified results.",
            code="STALE_UNKNOWN",
        )
    return {**result, "stale": None, "stale_check": "timed_out"}


async def _with_staleness(app: AppContext, result: dict[str, Any], *, scope: str = "") -> dict[str, Any]:
    """Annotate a query result envelope with staleness info.

    - ``stale_mode == "ignore"``: return result unchanged.
    - ``stale_mode == "lock"`` and stale: return an error envelope.
    - ``stale_mode == "warn"``: add ``stale``, ``stale_since``, ``changed_files`` keys.

    When staleness is indeterminate (project never indexed or index in progress),
    ``stale`` is set to ``None`` rather than ``True``.

    Also adds ``indexing_pending`` (``{"file-changed": N, "embed-dirty": M}``) when the
    daemon is running and has a non-trivial backlog — lets a caller tell a stale index
    that's actively catching up (e.g. right after MCP startup) apart from one where
    nothing is happening.
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
            timeout=_STALENESS_TIMEOUT_S,
        )
    except TimeoutError, QueryTimeoutError:
        logger.warning("Staleness check timed out")
        return _unverified_staleness(result, stale_mode)

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

    # Surface live catchup/watch backlog when non-trivial — lets a caller tell
    # "stale, but actively catching up" apart from "stale, nothing is happening".
    # Guarded by the same short timeout as the staleness check above so a slow
    # backlog read can never hold up a query response.
    try:
        pending = await asyncio.wait_for(app.daemon.pending_event_counts(), timeout=5.0)
    except Exception:
        pending = None
    if pending and any(v > 0 for v in pending.values()):
        result["indexing_pending"] = pending

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
    """Default project scope: the current project, any monorepo sub-projects, and extra vaults.

    Sub-project entities are stored with ``project_name = '{root}/{sub}'``
    (orchestrator.py, watcher.py). Without this expansion, a search with no
    explicit scope/project resolves to the bare root name and silently
    excludes all sub-project code. ``knowledge.extra_vaults`` projects are
    always appended too, matching ``_resolve_hybrid_scope``'s explicit-scope
    branch.
    """
    root_name = derive_project_name(app.settings.project_root)
    extra_vault_names = [v.project_name for v in app.settings.knowledge.extra_vaults]
    try:
        project_rows = await app.graph.get_project_status()
    except Exception as exc:
        logger.debug("Could not resolve monorepo sub-projects for default scope: {}", exc)
        result = [root_name]
        return result + [n for n in extra_vault_names if n not in result]

    all_names: list[str] = []
    for row in project_rows:
        node = row.get("n")
        if node:
            props = dict(node.items()) if hasattr(node, "items") else node
            name = props.get("name", "")
            if name:
                all_names.append(name)

    siblings = [n for n in all_names if n == root_name or n.startswith(f"{root_name}/")]
    result = siblings or [root_name]
    return result + [n for n in extra_vault_names if n not in result]


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
    auto_index: bool = True,
) -> FastMCP:
    """Create and configure the Code Atlas MCP server.

    *catchup* (default True) runs one blocking delta index pass at startup so
    edits made while the daemon was down are indexed before live consumption.
    Set False to skip it (faster startup at the cost of missing offline edits).

    *auto_index* False is the stronger form: no watcher, no consumers, no catch-up
    — the server only reads. Indexing is per-worktree, not per-session, so when
    several agent sessions share one checkout the extra servers contribute nothing
    but lease contention and duplicate watchers over the same files. Exactly one
    indexer per worktree is still required; this flag is for the others.
    """

    @asynccontextmanager
    async def app_lifespan(server: FastMCP) -> AsyncGenerator[AppContext]:  # noqa: PLR0915
        init_telemetry(settings.observability, role="mcp", project=derive_project_name(settings.project_root))

        # Declared type stays GraphClient (the network backend) — SqliteGraphClient
        # is a structurally-compatible fallback, but full retyping of every
        # downstream signature to the union is an explicitly deferred, mechanical
        # follow-up (see backends/__init__.py factory docstring).
        stack = AsyncExitStack()
        backends = await stack.enter_async_context(use_backends(settings))
        graph: GraphClient = backends.graph  # type: ignore[invalid-assignment]
        bus: EventBus = backends.bus  # type: ignore[invalid-assignment]
        try:
            await graph.ping()
        except Exception as exc:
            logger.error("Cannot reach {} — {}", graph_backend_label(graph, settings), exc)
            await stack.aclose()
            raise

        logger.info("MCP connected to {}", graph_backend_label(graph, settings))

        # "Never bootstrapped" must be read BEFORE ensure_schema() — ensure_schema()
        # sets the version as part of applying a fresh schema, so checking after
        # would always see a version and needs_first_index would never be True.
        needs_first_index = await graph.get_schema_version() is None
        await graph.ensure_schema()
        first_index_ready = asyncio.Event()
        if not needs_first_index:
            first_index_ready.set()

        # find_communities is Memgraph-only — the clustering itself is pure Python now (no
        # MAGE), but the two reads it clusters (module inventory + module-pair CALLS
        # aggregation) are still raw Cypher with no GraphBackend method, so on the embedded
        # backend it's not just non-functional, it's unreachable. Drop it from tools/list
        # entirely rather than leaving it listed with a guaranteed-error response. Adding
        # those two portable backend methods removes this branch and the analysis.py guard.
        # Safe to remove here: FastMCP's lifespan setup fully completes (Server.run enters the
        # lifespan context before creating the ServerSession) before any client request —
        # including tools/list — is processed, so there's no race with a caller listing tools
        # mid-startup.
        if isinstance(graph, SqliteGraphClient):
            server.remove_tool("find_communities")

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

            embed = EmbedClient(settings.embeddings, settings.redis)
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
        # Decided here and fixed for the life of the process, so every tool can share
        # one dict rather than re-deriving it per call.
        global _BACKEND_NOTE  # noqa: PLW0603 — process-wide, set once at startup
        _BACKEND_NOTE = _backend_note(graph)
        app_ctx = AppContext(
            graph=graph,
            bus=bus,
            settings=settings,
            embed=embed,
            staleness=staleness,
            daemon=daemon,
            resolved_root=settings.project_root,
            vector_enabled=vector_enabled,
            needs_first_index=needs_first_index,
            first_index_ready=first_index_ready,
        )

        # Register handler for roots/list_changed notification so we re-probe
        # on next tool call.  _mcp_server.notification_handlers is private API
        # in FastMCP — the only way to register notification handlers today.
        try:
            raw = server._mcp_server  # noqa: SLF001

            async def _on_roots_changed(*_args: object, **_kwargs: object) -> None:
                app_ctx.roots_checked = False
                logger.debug("Received roots/list_changed — will re-probe on next tool call")

            raw.notification_handlers["notifications/roots/list_changed"] = _on_roots_changed  # type: ignore[index]
        except Exception:
            logger.debug("Could not register roots/list_changed handler — root updates via notification disabled")

        # Auto-start watcher + pipeline if Valkey is reachable. catchup runs one
        # blocking delta index pass before consumers start — on a repo that's
        # drifted this can take well over a minute (real embedding calls per
        # changed entity), which would hold the MCP stdio handshake past most
        # clients' connect timeout. Run daemon startup in the background instead
        # so tools are reachable immediately; health_check reports pipeline state
        # and result staleness already surfaces an index that's still catching up.
        daemon_start_task = _spawn_indexing(
            daemon,
            settings,
            graph,
            bus,
            catchup=catchup,
            auto_index=auto_index,
            first_index_ready=first_index_ready,
        )

        try:
            yield app_ctx
        finally:
            if daemon_start_task is not None:
                daemon_start_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await daemon_start_task
            await app_ctx.daemon.stop()
            await stack.aclose()
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
            "Call get_usage_guide('guidelines') for tips on structuring code for better search results. "
            "A result's `truncated` is false when nothing was withheld, otherwise an object carrying "
            "{shown, total, cut, remedy} — read `cut` before concluding a short list is a complete one. "
            "The search tools cannot count matches they did not fetch, so there `total` and `cut` are "
            "null with `has_more: true`: more exist, quantity unknown. Treat null as unknown, not zero."
        ),
        host=host,
        port=port,
        lifespan=app_lifespan,
    )

    _install_tool_tracing(mcp)
    _install_backend_stamp(mcp)
    _register_query_tools(mcp)
    _register_search_tools(mcp)
    _register_hybrid_tool(mcp)
    _register_info_tools(mcp)
    _register_knowledge_tools(mcp)
    _register_subagent_tools(mcp)
    _register_analysis_tools(mcp)
    _register_traversal_tools(mcp)
    return mcp


# ---------------------------------------------------------------------------
# Tool registration — split to stay under statement limits
# ---------------------------------------------------------------------------


def _keep_node_row(row: dict[str, Any], patterns: list[str]) -> bool:
    """Whether a get_node partial-match row survives test filtering."""
    node = row.get("n") or {}
    return not matches_test_pattern(node.get("file_path") or "", node.get("name") or "", patterns)


def _register_node_tools(mcp: FastMCP) -> None:
    """Register the get_node tool (separated for statement-count limits)."""

    @mcp.tool(
        description=(
            "Find code entities by name. "
            "Cascade: exact (uid + name) → partial (suffix > prefix > contains), filling any "
            "remaining slots in the result budget rather than stopping at the first match. "
            "Results ranked by relevance. Use get_context to expand a result. "
            "Returns: {results: [{uid, name, qualified_name, kind, file_path, "
            "line_start, line_end, signature, docstring}], count, truncated, query_ms}. "
            "Pass detail='full' to include source code, full docstrings, and caller/callee info. "
            "Use offset to page beyond the first `limit` results."
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
                    "DocFile, DocSection, Note, ExternalSymbol. Empty = all."
                ),
            ),
        ] = "",
        limit: Annotated[int, Field(20, description="Max results to return.", ge=1, le=100)] = 20,
        offset: Annotated[int, Field(0, description="Skip this many results (for paging beyond limit).", ge=0)] = 0,
        detail: Annotated[
            str,
            Field(
                "summary",
                description="'summary' (default) or 'full' (add source, full docstrings, call stats).",
            ),
        ] = "summary",
        exclude_tests: Annotated[
            bool | None,
            Field(
                None,
                description=(
                    "Exclude test entities from PARTIAL matches. Exact uid/name matches are never "
                    "filtered — you asked for that name. Default true — override to include."
                ),
            ),
        ] = None,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        try:
            app = await _ensure_root(ctx)
        except IndexNotReadyError as exc:
            return _error(str(exc), code="INDEX_REQUIRED")
        clamped = _clamp_limit(limit)
        test_patterns = _resolve_test_patterns(app.settings.search, exclude_tests)
        patterns = list(test_patterns)

        if label and label not in NodeLabel:
            valid = ", ".join(sorted(lbl.value for lbl in NodeLabel))
            return {"error": f"Invalid label: {label!r}. Valid labels: {valid}"}
        t0 = time.monotonic()
        found: list[dict[str, Any]] | None = None
        total: int | None = None
        page_end = offset + clamped
        # One extra row to detect truncation past this page, padded when filtering is on
        # so discarded rows backfill from real candidates instead of under-delivering.
        peek = _padded_limit(page_end, test_patterns) + 1

        try:
            # Stage A: Exact matches (uid + exact name) — 1 RTT
            exact = await app.graph.get_node_exact_matches(name, label, peek)
            seen: dict[str, dict[str, Any]] = {}
            ordered_uids: list[str] = []
            for r in exact:
                uid = r["n"]["uid"]
                if uid not in seen:
                    seen[uid] = r
                    ordered_uids.append(uid)

            # Stage A results are never test-filtered: an exact name match is what the
            # caller asked for by name. Filtering them would make 51.8% of this repo's
            # distinct entity names unresolvable, including production symbols that only
            # look test-shaped (test_filter, test_patterns, from_test). Exempting Stage A
            # also means the gate below can keep counting raw rows — the "a page of test
            # exact matches suppresses Stage B" hazard only exists if Stage A is filtered.
            #
            # Stage B: Partial matches (suffix > prefix > contains) — 1 RTT. Runs
            # whenever Stage A didn't fill the requested budget, not only when it
            # found nothing — otherwise a single exact hit silently hides
            # same-named siblings (e.g. "_catchup" hiding "_catchup_vault").
            if len(seen) < page_end:
                partial = await app.graph.get_node_partial_matches(name, label, peek)
                # Deduplicate by uid (keeping highest _match_score), skipping anything Stage A already has
                partial_best: dict[str, dict[str, Any]] = {}
                for r in partial:
                    uid = r["n"]["uid"]
                    if uid in seen or (patterns and not _keep_node_row(r, patterns)):
                        continue
                    if r.get("_match_score", 0) > partial_best.get(uid, {}).get("_match_score", -1):
                        partial_best[uid] = r
                for uid in sorted(partial_best, key=lambda u: partial_best[u].get("_match_score", 0), reverse=True):
                    seen[uid] = partial_best[uid]
                    ordered_uids.append(uid)

            total = len(seen)
            found = [seen[uid] for uid in ordered_uids[offset:page_end]]
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")

        elapsed = (time.monotonic() - t0) * 1000
        ranked = _rank_results([_compact_node(r, detail=detail) for r in found])
        await _enrich_with_calls(app.graph, ranked, detail=detail)
        return await _with_staleness(app, _result(ranked, limit=page_end, query_ms=elapsed, total=total))


# The degraded-backend note, computed once at startup and merged into every tool's
# answer. Module-level rather than threaded through 23 signatures because the backend
# is chosen once in `app_lifespan` and cannot change while the process runs -- there is
# no per-call state to carry.
_BACKEND_NOTE: dict[str, Any] = {}


def _spawn_indexing(
    daemon: DaemonManager,
    settings: AtlasSettings,
    graph: GraphClient,
    bus: EventBus,
    *,
    catchup: bool,
    auto_index: bool,
    first_index_ready: asyncio.Event,
) -> asyncio.Task | None:
    """Start the watcher + pipeline in the background, or explain why we did not.

    Returns the startup task, or ``None`` when indexing is off for this process.
    """
    if not auto_index:
        # Nothing will ever set this, and every tool call that needs an index would
        # otherwise block on it for the full readiness timeout before answering.
        first_index_ready.set()
        daemon.disabled_reason = "indexing disabled (--no-index)"
        logger.info(
            "Query-only mode (--no-index): no watcher, no pipeline, no catch-up. Another process must index {}",
            settings.project_root,
        )
        return None

    task = asyncio.get_running_loop().create_task(
        daemon.start(settings, graph, bus, catchup=catchup, first_index_ready=first_index_ready)
    )

    def _on_daemon_started(finished: asyncio.Task) -> None:
        if finished.cancelled():
            return
        exc = finished.exception()
        if exc is not None:
            logger.exception("Daemon startup failed", exc_info=exc)
        elif finished.result():
            logger.info("Auto-indexing active (watching {})", settings.project_root)
        else:
            logger.info("Query-only mode (no Valkey)")

    task.add_done_callback(_on_daemon_started)
    return task


def _install_tool_tracing(mcp: FastMCP) -> None:
    """Wrap tool registration so every MCP call opens a span and records a metric.

    `_tracer` has existed in this module since telemetry was added and was never
    used: every trace the system produced started somewhere in the middle of the
    stack (`graph.execute`, `hybrid_search`, `embed.embed_one`) with no parent
    naming the tool an agent actually called. Traces answered "what did the graph
    do" but never "which request asked for it", which is the whole point of having
    them on an agent-facing server.

    Installed through the same `mcp.tool` seam as `_install_backend_stamp` -- one
    interception rather than 23 decorated functions -- and composes with it: this
    runs first, so the span encloses the envelope stamping as well as the tool body.

    Errors are recorded two ways because they arrive two ways. A raised exception
    gets `record_exception` + ERROR status; a returned `{"error": ...}` payload (how
    most tools here report failure) would otherwise look like a clean span, so the
    status is set from the payload too.
    """
    register = mcp.tool

    def tracing_tool(*d_args: Any, **d_kwargs: Any) -> Any:
        decorate = register(*d_args, **d_kwargs)

        def wrap(fn: Any) -> Any:
            tool_name = fn.__name__

            @functools.wraps(fn)
            async def traced(*args: Any, **kwargs: Any) -> Any:
                started = time.perf_counter()
                status = "ok"
                with _tracer.start_as_current_span(f"mcp.tool.{tool_name}", attributes={"mcp.tool": tool_name}) as span:
                    try:
                        result = await fn(*args, **kwargs)
                    except Exception as exc:
                        status = "exception"
                        mark_span_error(span, exc)
                        raise
                    else:
                        if isinstance(result, dict):
                            _annotate_tool_span(span, result)
                            if result.get("error"):
                                status = "error"
                                mark_span_error(span, description=str(result["error"])[:200])
                        return result
                    finally:
                        elapsed = time.perf_counter() - started
                        span.set_attribute("mcp.status", status)
                        attrs = {"tool": tool_name, "status": status}
                        get_metrics().tool_calls.add(1, attrs)
                        get_metrics().tool_latency.record(elapsed, {"tool": tool_name})

            # functools.wraps sets __wrapped__, which inspect.signature follows, so
            # FastMCP still builds the schema from the real parameters.
            return decorate(traced)

        return wrap

    mcp.tool = tracing_tool  # type: ignore[method-assign]


def _annotate_tool_span(span: Any, result: dict[str, Any]) -> None:
    """Copy the few result fields worth filtering traces on.

    Deliberately a fixed allowlist of scalars, not the payload: spans travel over the
    wire on every call, and tool results here carry source code and docstrings.
    """
    for key in ("count", "truncated", "query_ms", "code"):
        value = result.get(key)
        if isinstance(value, (int, float, bool, str)):
            span.set_attribute(f"mcp.{key}", value)


def _install_backend_stamp(mcp: FastMCP) -> None:
    """Wrap tool registration so every dict answer carries the backend note.

    One interception instead of 23 edits. `_backend_note` had a single caller, and only
    8 of the 23 tools route through the `_result` envelope, so there is no shared
    payload seam to extend -- the registration decorator is the only place all of them
    pass through.

    Must be called **before** the `_register_*` functions, since it works by replacing
    the decorator they use.
    """
    register = mcp.tool

    def stamping_tool(*d_args: Any, **d_kwargs: Any) -> Any:
        decorate = register(*d_args, **d_kwargs)

        def wrap(fn: Any) -> Any:
            @functools.wraps(fn)
            async def stamped(*args: Any, **kwargs: Any) -> Any:
                result = await fn(*args, **kwargs)
                # Only dict answers, and never clobber a tool that already said it
                # (index_status names the backend in its own payload shape).
                if not _BACKEND_NOTE or not isinstance(result, dict) or "backend" in result:
                    return result
                return {**result, **_BACKEND_NOTE}

            # functools.wraps sets __wrapped__, which inspect.signature follows, so
            # FastMCP still builds the schema from the real parameters.
            return decorate(stamped)

        return wrap

    mcp.tool = stamping_tool  # type: ignore[method-assign]


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
        try:
            app = await _ensure_root(ctx)
        except IndexNotReadyError as exc:
            return _error(str(exc), code="INDEX_REQUIRED")
        clamped = _clamp_limit(limit)

        if _WRITE_KEYWORDS.search(_strip_cypher_string_literals(query)):
            return _error("Write operations are not allowed via MCP", code="WRITE_REJECTED")

        # Deliberate exception (see graph/protocol.py, ADR-0015): arbitrary agent-authored
        # Cypher has no SQL translation, so this is the one place that still calls
        # graph.execute() directly. Guard explicitly instead of letting SqliteGraphClient's
        # NotImplementedError surface as a generic QUERY_ERROR.
        if isinstance(app.graph, SqliteGraphClient):
            return _error(
                "cypher_query requires the Memgraph backend — the sqlite backend has no Cypher translation layer.",
                code="UNSUPPORTED_BACKEND",
            )

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
            "Pass a uid from get_node or hybrid_search results. docs includes both DocSection "
            "and Note entries; a Note linked via an explicit anchors: frontmatter entry carries "
            "stale (bool — content changed since the anchor was recorded) and anchor_hash. "
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
        try:
            app = await _ensure_root(ctx)
        except IndexNotReadyError as exc:
            return _error(str(exc), code="INDEX_REQUIRED")
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
            "Pass detail='full' to include source code, full docstrings, and caller/callee info. "
            "Use offset to page beyond the first `limit` results."
        ),
    )
    async def text_search(
        query: Annotated[str, Field(description="BM25 query — supports phrases, wildcards, field:value, AND/OR.")],
        label: Annotated[
            str,
            Field("", description="Restrict to one label: Callable, Module, TypeDef, Value, DocSection. Empty = all."),
        ] = "",
        limit: Annotated[int, Field(20, description="Max results to return.", ge=1, le=100)] = 20,
        offset: Annotated[int, Field(0, description="Skip this many results (for paging beyond limit).", ge=0)] = 0,
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
        try:
            app = await _ensure_root(ctx)
        except IndexNotReadyError as exc:
            return _error(str(exc), code="INDEX_REQUIRED")
        if label and label not in NodeLabel:
            valid = ", ".join(sorted(lbl.value for lbl in NodeLabel))
            return {"error": f"Invalid label: {label!r}. Valid labels: {valid}"}
        clamped = _clamp_limit(limit)
        page_end = offset + clamped
        resolved_projects = [project] if project else await _default_scope_projects(app)
        resolved_scope = ",".join(resolved_projects)

        t0 = time.monotonic()
        try:
            all_results = await app.graph.text_search(
                query, label=label, limit=page_end * 3 + 1, projects=resolved_projects
            )
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")
        all_results = filter_raw_records(all_results, app.settings.search)
        elapsed = (time.monotonic() - t0) * 1000

        # Fetched `page_end * 3 + 1` to survive post-filtering, so a full page means
        # "at least one more", never a total (ATL-111).
        has_more = len(all_results) > page_end
        all_results = all_results[offset:page_end]
        compacted = [_compact_node(r, detail=detail) for r in all_results]
        await _enrich_with_calls(app.graph, compacted, detail=detail)
        return await _with_staleness(
            app, _result(compacted, limit=page_end, query_ms=elapsed, has_more=has_more), scope=resolved_scope
        )

    @mcp.tool(
        description=(
            "Semantic similarity search using vector embeddings. "
            "Finds code by meaning, not just name. "
            "Returns: {results: [{uid, name, qualified_name, kind, file_path, "
            "line_start, line_end, signature, docstring, similarity}], count, truncated, query_ms}. "
            "Pass detail='full' to include source code, full docstrings, and caller/callee info. "
            "Use offset to page beyond the first `limit` results."
        ),
    )
    async def vector_search(  # noqa: PLR0911
        query: Annotated[str, Field(description="Natural language query — describes what the code does.")],
        label: Annotated[
            str,
            Field("", description="Restrict to one label: Callable, Module, TypeDef, Value, DocSection. Empty = all."),
        ] = "",
        limit: Annotated[int, Field(20, description="Max results to return.", ge=1, le=100)] = 20,
        offset: Annotated[int, Field(0, description="Skip this many results (for paging beyond limit).", ge=0)] = 0,
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
        try:
            app = await _ensure_root(ctx)
        except IndexNotReadyError as exc:
            return _error(str(exc), code="INDEX_REQUIRED")
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
        page_end = offset + clamped
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
                vector, label=label, limit=page_end * 3 + 1, projects=resolved_projects, threshold=threshold
            )
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")
        all_results = filter_raw_records(all_results, app.settings.search)
        elapsed = (time.monotonic() - t0) * 1000

        # Fetched `page_end * 3 + 1` to survive post-filtering, so a full page means
        # "at least one more", never a total (ATL-111).
        has_more = len(all_results) > page_end
        all_results = all_results[offset:page_end]
        compacted = [_compact_node(r, detail=detail) for r in all_results]
        await _enrich_with_calls(app.graph, compacted, detail=detail)
        return await _with_staleness(
            app, _result(compacted, limit=page_end, query_ms=elapsed, has_more=has_more), scope=resolved_scope
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
    # Extra vaults (global overspanning vault, harness memory dir) stay in scope
    # even when the caller explicitly narrows to one project — a cross-project
    # tooling gotcha is relevant regardless of what code you're scoped to.
    always_include = [
        *app.settings.monorepo.always_include,
        *(v.project_name for v in app.settings.knowledge.extra_vaults),
    ]
    expanded = expand_scope(scope, all_project_names, always_include)
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
            "Code entities (Callable, TypeDef, Module, Value) are boosted over documentation by default "
            "(mode='blended'); pass mode='knowledge' for why/decision/gotcha-shaped questions to invert that. "
            "Set code_only=true (or mode='code') to exclude DocSection/DocFile/Note entirely. "
            "Returns: {results: [{uid, name, qualified_name, kind, file_path, line_start, "
            "line_end, signature, docstring, visibility, _labels, score, rrf_score, sources}], "
            "count, truncated, query_ms}. "
            "Ordered by `score` — rrf_score after the visibility/label boosts. Re-sorting on "
            "`rrf_score` yields a different order than the one returned. "
            "Pass detail='full' to include source code, full docstrings, and caller/callee info. "
            "Use offset to page beyond the first `limit` results."
        ),
    )
    async def hybrid_search(  # noqa: PLR0911
        query: Annotated[str, Field(description="Search query — identifier names, natural language, or mixed.")],
        limit: Annotated[int, Field(20, description="Max results to return.", ge=1, le=100)] = 20,
        offset: Annotated[int, Field(0, description="Skip this many results (for paging beyond limit).", ge=0)] = 0,
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
            Field(False, description="Exclude documentation entities (DocSection, DocFile, Note). Return only code."),
        ] = False,
        mode: Annotated[
            str,
            Field(
                "blended",
                description="Knowledge-participation mode: 'blended' (default — knowledge ranked slightly "
                "below code), 'knowledge' (notes/docs outrank code — for why/decision/gotcha questions), "
                "or 'code' (equivalent to code_only=true).",
            ),
        ] = "blended",
        detail: Annotated[
            str,
            Field(
                "summary",
                description="'summary' (default) or 'full' (add source, full docstrings, call stats).",
            ),
        ] = "summary",
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        try:
            app = await _ensure_root(ctx)
        except IndexNotReadyError as exc:
            return _error(str(exc), code="INDEX_REQUIRED")
        clamped = _clamp_limit(limit)
        page_end = offset + clamped

        types, type_error = _parse_search_types(search_types)
        if type_error:
            return type_error

        weight_dict, weight_error = _parse_weights(weights)
        if weight_error:
            return weight_error

        try:
            search_mode = SearchMode(mode) if mode else SearchMode.BLENDED
        except ValueError:
            valid = ", ".join(m.value for m in SearchMode)
            return _error(f"Invalid mode: {mode!r}. Valid modes: {valid}", code="INVALID_MODE")

        resolved_scope = await _resolve_hybrid_scope(app, scope)
        if resolved_scope is None:
            # scope explicitly matched zero indexed projects — return no
            # results rather than silently searching every project.
            return await _with_staleness(app, _result([], limit=page_end, query_ms=0.0, total=0), scope=scope)

        t0 = time.monotonic()
        try:
            results = await _hybrid_search(
                graph=app.graph,
                embed=app.embed if app.vector_enabled else None,
                settings=app.settings.search,
                query=query,
                search_types=types,
                limit=page_end + 1,
                scope=resolved_scope,
                weights=weight_dict,
                exclude_tests=exclude_tests,
                exclude_stubs=exclude_stubs,
                exclude_generated=exclude_generated,
                code_only=code_only,
                mode=search_mode,
                secondary_projects=frozenset(v.project_name for v in app.settings.knowledge.extra_vaults),
            )
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")
        elapsed = (time.monotonic() - t0) * 1000

        # `limit=page_end + 1` above buys exactly one fact: whether a further row
        # exists. It is not a count, and reporting it as one made `cut` incapable of
        # exceeding 1 on a repo with thousands of matches (ATL-111).
        has_more = len(results) > page_end
        results = results[offset:page_end]

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
                # The value the list is ordered by. `rrf_score` is the raw fusion score
                # before the visibility/label/project multipliers, so sorting on it gives
                # a different order than the one returned.
                "score": round(r.ranked_score, 6),
                "rrf_score": round(r.rrf_score, 6),
                "sources": r.sources,
            }
            # Emitted only when set, so an ordinary hit costs no tokens to say
            # nothing. A demoted result with no stated reason reads as a broken
            # ranker; a disputed one that looks settled is worse.
            if r.superseded_by:
                entry["superseded_by"] = r.superseded_by
                entry["caveat"] = f"Superseded by {r.superseded_by} — this is the note its author replaced."
            if r.contradicts_with:
                entry["contradicts_with"] = list(r.contradicts_with)
                entry["caveat"] = (
                    "Unresolved contradiction with " + ", ".join(r.contradicts_with) + " — neither is settled."
                )
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
            app, _result(serialized, limit=page_end, query_ms=elapsed, has_more=has_more), scope=scope
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
        # Exempted from the first-index gate — must stay reachable so a blocked
        # caller can see the pipeline state that's blocking the other tools.
        app = await _ensure_root(ctx, require_index=False)
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
            label_counts = await app.graph.get_label_counts()

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
            **_backend_note(app.graph),
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
        try:
            app = await _ensure_root(ctx)
        except IndexNotReadyError as exc:
            return _error(str(exc), code="INDEX_REQUIRED")
        t0 = time.monotonic()

        try:
            projects_raw = await app.graph.get_project_status()
            if not projects_raw:
                return _result([], limit=0, query_ms=0)

            # Collect DEPENDS_ON relationships
            depends_records = await app.graph.get_project_dependency_edges()
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
                "marker": sorted(lbl.value for lbl in _MARKER_LABELS),
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
        # Exempted from the first-index gate — must stay reachable so a blocked
        # caller can see the pipeline state that's blocking the other tools.
        app = await _ensure_root(ctx, require_index=False)
        report = await run_health_checks(app.settings, graph=app.graph, bus=app.bus, embed=app.embed, daemon=app.daemon)
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


def _register_knowledge_tools(mcp: FastMCP) -> None:
    """Register knowledge_health (dream-mode deterministic lint report)."""

    @mcp.tool(
        description=(
            "Deterministic dream-mode lint report for the knowledge vault: inbox digest, "
            "orphan notes (no LINKS_TO edges), dangling LINKS_TO/DERIVED_FROM/SUPERSEDES "
            "references, duplicate-id conflicts across vault files, high-similarity note pairs, "
            "and cross-project promotion candidates. Spans this project's vault plus any "
            "configured [knowledge] extra_vaults. Deterministic only — disposition "
            "(KEEP/MERGE/PROMOTE/DROP) is a separate, agent-side judgment call. "
            "Returns: {inbox_count, inbox_paths, orphan_notes, duplicate_ids, dangling_links, "
            "similar_pairs, promotion_candidates, memory_index_issues, broken_anchors, query_ms}."
        ),
    )
    async def knowledge_health(ctx: Context = None) -> dict[str, Any]:  # type: ignore[assignment]
        try:
            app = await _ensure_root(ctx)
        except IndexNotReadyError as exc:
            return _error(str(exc), code="INDEX_REQUIRED")
        t0 = time.monotonic()

        project_name = derive_project_name(app.settings.project_root)
        vault_roots = [
            VaultRoot(path=app.settings.project_root / app.settings.knowledge.vault_path, project_name=project_name)
        ]
        vault_roots.extend(
            VaultRoot(path=Path(v.path).expanduser().resolve(), project_name=v.project_name)
            for v in app.settings.knowledge.extra_vaults
        )

        try:
            report = await build_dream_report(app.graph, vault_roots)
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")

        elapsed = (time.monotonic() - t0) * 1000
        return {**report_to_dict(report), "query_ms": round(elapsed, 1)}


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
            if isinstance(app.graph, SqliteGraphClient):
                # Deliberate exception (see graph/protocol.py, ADR-0015): no SQL translation
                # for arbitrary Cypher — say so explicitly rather than routing through
                # validate_cypher_explain's generic "EXPLAIN failed: ..." exception message.
                issues.append(
                    ValidationIssue(
                        "info",
                        "Live EXPLAIN check skipped — the active backend (sqlite) has no Cypher "
                        "engine; only static checks ran.",
                    )
                )
            else:
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


def _register_analysis_tools(mcp: FastMCP) -> None:  # noqa: PLR0915
    """Register repository analysis and diagram generation tools."""

    @mcp.tool(
        description=(
            "Analyze repository structure, centrality, dependencies, patterns, quality, dead code, "
            "complexity hotspots, communities, git-derived signals, or a whole-module skeleton. "
            "Returns: {analysis, project, ...analysis-specific keys, query_ms}."
        ),
    )
    async def analyze_repo(
        analysis: Annotated[
            Literal[
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
            ],
            Field(
                description=(
                    "Sub-analysis: structure (entity counts, packages, largest modules), "
                    "centrality (hub entities/modules, leaves), "
                    "dependencies (imports, cross-package coupling, circular deps), "
                    "patterns (inheritance, enums, visibility, docstring coverage), "
                    "quality (health score, god modules, circular deps, tangled modules, coupling, instability), "
                    "dead_code (Callables/TypeDefs with zero incoming CALLS edges), "
                    "complexity (top Callables by LOC-span proxy, not true cyclomatic complexity), "
                    "communities (which subsystems the repo has — clusters MODULES over the "
                    "call/import graph aggregated to module level; Memgraph backend only), "
                    "git_signals (commit-count hotspots, bus-factor risks, co-change pairs — requires "
                    "'atlas mine-git-history' to have been run; empty lists otherwise), "
                    "module_summary (dense text skeleton of everything under 'path' — signatures, first "
                    "docstring lines, internal edges, fan-in/fan-out; requires path — see summarize_module)."
                ),
            ),
        ],
        project: Annotated[str, Field("", description="Project name. Empty = auto-detect from workspace.")] = "",
        path: Annotated[
            str, Field("", description="Scope analysis to a file or package path prefix. Empty = entire project.")
        ] = "",
        limit: Annotated[
            int,
            Field(
                20,
                description="Max items per sub-section (module_summary scales it x10 for its entity budget).",
                ge=1,
                le=100,
            ),
        ] = 20,
        exclude_tests: Annotated[
            bool | None,
            Field(
                None,
                description="Exclude test files/entities from ranked/listed results (hub entities, largest "
                "modules, hotspots, community members, etc.). Default true — override to include tests.",
            ),
        ] = None,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        try:
            app = await _ensure_root(ctx)
        except IndexNotReadyError as exc:
            return _error(str(exc), code="INDEX_REQUIRED")
        project_name = project or derive_project_name(app.settings.project_root)
        clamped = _clamp_limit(limit)
        test_patterns = _resolve_test_patterns(app.settings.search, exclude_tests)
        try:
            return await _analyze_repo(
                app.graph, analysis, project_name, path=path, limit=clamped, test_patterns=test_patterns
            )
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")

    @mcp.tool(
        description=(
            "Diagram the codebase. Returns {type, format, node_count, query_ms} plus either "
            "`mermaid` (format='mermaid', small graphs — renders as a picture) or `outline` "
            "(format='outline', larger import graphs — community-grouped adjacency, ~5x cheaper "
            "in tokens than Mermaid and far easier to follow at scale). Check `format` before reading."
        ),
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
        exclude_tests: Annotated[
            bool | None,
            Field(None, description="Exclude test modules from the imports graph. Default true."),
        ] = None,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        try:
            app = await _ensure_root(ctx)
        except IndexNotReadyError as exc:
            return _error(str(exc), code="INDEX_REQUIRED")
        project_name = project or derive_project_name(app.settings.project_root)
        max_nodes = max(1, min(max_nodes, _MAX_LIMIT))
        test_patterns = _resolve_test_patterns(app.settings.search, exclude_tests)
        try:
            return await _generate_diagram(
                app.graph, type, project_name, path=path, max_nodes=max_nodes, test_patterns=test_patterns
            )
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")

    # Shortcut tools (ADR-0013): thin top-level wrappers delegating to
    # analyze_repo with the analysis pre-set — no duplicated query logic.

    @mcp.tool(
        description=(
            "Shortcut for analyze_repo(analysis='dead_code'). Callables/TypeDefs with no incoming "
            "CALLS/USES_TYPE/IMPORTS/INHERITS/IMPLEMENTS/OVERRIDES edge and no call into their members, "
            "excluding dunder methods and test files. "
            "TREAT AS A LEAD, NOT A VERDICT — verify each hit against source before deleting anything. "
            "Known false positives: a caller defined INSIDE another function is not indexed, so its "
            "callees look dead (this hides every nested handler, e.g. decorator-registered tools); "
            "entities referenced only through a dispatch table or by reflection have no static edge; "
            "and dynamic dispatch is invisible generally. "
            "Returns: {analysis, project, dead_code_count, dead_code: [{name, qualified_name, label, "
            "kind, file_path, line_start}], truncated, query_ms}."
        ),
    )
    async def find_dead_code(
        project: Annotated[str, Field("", description="Project name. Empty = auto-detect from workspace.")] = "",
        path: Annotated[
            str, Field("", description="Scope analysis to a file or package path prefix. Empty = entire project.")
        ] = "",
        limit: Annotated[int, Field(20, description="Max items to return.", ge=1, le=100)] = 20,
        exclude_tests: Annotated[
            bool | None, Field(None, description="Exclude test files/entities. Default true — override to include.")
        ] = None,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        try:
            app = await _ensure_root(ctx)
        except IndexNotReadyError as exc:
            return _error(str(exc), code="INDEX_REQUIRED")
        project_name = project or derive_project_name(app.settings.project_root)
        clamped = _clamp_limit(limit)
        test_patterns = _resolve_test_patterns(app.settings.search, exclude_tests)
        try:
            return await _analyze_repo(
                app.graph, "dead_code", project_name, path=path, limit=clamped, test_patterns=test_patterns
            )
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")

    @mcp.tool(
        description=(
            "Shortcut for analyze_repo(analysis='complexity'). Top Callables by LOC-span "
            "(line_end - line_start) — a crude proxy, not true cyclomatic complexity. "
            "Returns: {analysis, project, hotspots: [{name, qualified_name, kind, file_path, "
            "line_start, line_end, loc_span}], query_ms}."
        ),
    )
    async def find_complexity_hotspots(
        project: Annotated[str, Field("", description="Project name. Empty = auto-detect from workspace.")] = "",
        path: Annotated[
            str, Field("", description="Scope analysis to a file or package path prefix. Empty = entire project.")
        ] = "",
        limit: Annotated[int, Field(20, description="Max items to return.", ge=1, le=100)] = 20,
        exclude_tests: Annotated[
            bool | None, Field(None, description="Exclude test files/entities. Default true — override to include.")
        ] = None,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        try:
            app = await _ensure_root(ctx)
        except IndexNotReadyError as exc:
            return _error(str(exc), code="INDEX_REQUIRED")
        project_name = project or derive_project_name(app.settings.project_root)
        clamped = _clamp_limit(limit)
        test_patterns = _resolve_test_patterns(app.settings.search, exclude_tests)
        try:
            return await _analyze_repo(
                app.graph, "complexity", project_name, path=path, limit=clamped, test_patterns=test_patterns
            )
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")

    @mcp.tool(
        description=(
            "Shortcut for analyze_repo(analysis='communities'). Answers 'what subsystems does this "
            "codebase have?' — clusters MODULES, not individual callables: the callable-level CALLS "
            "graph is aggregated up to the modules owning each endpoint (weights summed) and merged "
            "with module-to-module IMPORTS, then partitioned by deterministic greedy modularity "
            "(same input always gives the same answer). Memgraph backend only. Communities of size "
            "< 2 (isolated modules) are dropped as noise; returned largest first. "
            "Returns: {analysis, project, granularity: 'module', module_count, edge_count, "
            "modularity, community_count, communities: [{community_id, size, members: [{uid, name, "
            "qualified_name, label, file_path}]}], noise_threshold, query_ms}."
        ),
    )
    async def find_communities(
        project: Annotated[str, Field("", description="Project name. Empty = auto-detect from workspace.")] = "",
        path: Annotated[
            str, Field("", description="Scope analysis to a file or package path prefix. Empty = entire project.")
        ] = "",
        limit: Annotated[
            int,
            Field(20, description="Max communities to return (also caps members shown per community).", ge=1, le=100),
        ] = 20,
        exclude_tests: Annotated[
            bool | None,
            Field(
                None,
                description="Exclude test modules from the clustered graph. Default true — override to include. "
                "Test modules are dropped before the graph is built, so test connectivity cannot bridge two "
                "production subsystems; including them typically pairs each module with its own test modules.",
            ),
        ] = None,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        try:
            app = await _ensure_root(ctx)
        except IndexNotReadyError as exc:
            return _error(str(exc), code="INDEX_REQUIRED")
        project_name = project or derive_project_name(app.settings.project_root)
        clamped = _clamp_limit(limit)
        test_patterns = _resolve_test_patterns(app.settings.search, exclude_tests)
        try:
            return await _analyze_repo(
                app.graph, "communities", project_name, path=path, limit=clamped, test_patterns=test_patterns
            )
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")

    @mcp.tool(
        description=(
            "Shortcut for analyze_repo(analysis='git_signals'). Commit-count hotspots, bus-factor risks "
            "(files with <=1 distinct author), and top co-change pairs (files frequently committed together) "
            "mined from git history. Requires 'atlas mine-git-history' to have been run first — otherwise "
            "returns empty lists with mined=false, not an error. "
            "Returns: {analysis, project, mined, hotspots: [{name, qualified_name, file_path, commit_count, "
            "author_count, days_since_last_commit}], bus_factor_risks: [{name, qualified_name, file_path, "
            "commit_count, author_count}], co_change_pairs: [{a, a_file_path, b, b_file_path, count}], query_ms}."
        ),
    )
    async def find_hotspots(
        project: Annotated[str, Field("", description="Project name. Empty = auto-detect from workspace.")] = "",
        path: Annotated[
            str, Field("", description="Scope analysis to a file or package path prefix. Empty = entire project.")
        ] = "",
        limit: Annotated[int, Field(20, description="Max items per list to return.", ge=1, le=100)] = 20,
        exclude_tests: Annotated[
            bool | None, Field(None, description="Exclude test files/entities. Default true — override to include.")
        ] = None,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        try:
            app = await _ensure_root(ctx)
        except IndexNotReadyError as exc:
            return _error(str(exc), code="INDEX_REQUIRED")
        project_name = project or derive_project_name(app.settings.project_root)
        clamped = _clamp_limit(limit)
        test_patterns = _resolve_test_patterns(app.settings.search, exclude_tests)
        try:
            return await _analyze_repo(
                app.graph, "git_signals", project_name, path=path, limit=clamped, test_patterns=test_patterns
            )
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")

    @mcp.tool(
        description=(
            "Shortcut for analyze_repo(analysis='module_summary'). Dense text skeleton of an entire "
            "module or package in one call — read this instead of opening the files. For every entity "
            "under 'path': signature, visibility, line span and the FIRST docstring line only (no "
            "bodies, no full docstrings). Plus the intra-scope CALLS/INHERITS/IMPLEMENTS/USES_TYPE "
            "adjacency, the scope boundary (FAN-IN: who outside calls in — the thing get_context cannot "
            "tell you; FAN-OUT: what this scope depends on, external packages marked *), and linked "
            "notes/docs. Edge annotations like [confidence=ambiguous] mark non-default CALLS edge "
            "properties. The outline is self-describing (SCOPE/NAMES/LEGEND header). "
            "Returns: {analysis, project, path, modules, entity_count, internal_edge_count, "
            "fan_in_count, fan_out_count, truncated, outline, query_ms}, or {error, code} with "
            "'PATH_REQUIRED' when path is empty / 'NOT_FOUND' when nothing is indexed under it."
        ),
    )
    async def summarize_module(
        path: Annotated[
            str,
            Field(description="File or package path prefix to summarize, e.g. 'src/code_atlas/graph' — required."),
        ],
        project: Annotated[str, Field("", description="Project name. Empty = auto-detect from workspace.")] = "",
        limit: Annotated[
            int,
            Field(20, description="Budget knob: entities are capped at limit x 10, edges at limit x 30.", ge=1, le=100),
        ] = 20,
        exclude_tests: Annotated[
            bool | None,
            Field(
                None,
                description="Exclude test callers/dependencies from the fan-in/fan-out lists. Default true — "
                "override to include. Entities inside 'path' are never filtered: you asked for that path.",
            ),
        ] = None,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        try:
            app = await _ensure_root(ctx)
        except IndexNotReadyError as exc:
            return _error(str(exc), code="INDEX_REQUIRED")
        project_name = project or derive_project_name(app.settings.project_root)
        clamped = _clamp_limit(limit)
        test_patterns = _resolve_test_patterns(app.settings.search, exclude_tests)
        try:
            return await _analyze_repo(
                app.graph, "module_summary", project_name, path=path, limit=clamped, test_patterns=test_patterns
            )
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")


def _register_traversal_tools(mcp: FastMCP) -> None:
    """Register trace_path and blast_radius — information-retrieval family (ADR-0013).

    Unlike analyze_repo/generate_diagram's project+path+limit signature, these
    are anchored at specific entity uid(s), matching get_node/get_context.
    """

    @mcp.tool(
        description=(
            "Find the shortest path between two entities, bounded by max_depth hops. "
            "Traverses CALLS|IMPORTS|USES_TYPE edges by default (override with edge_types). "
            "Each hop reports its edge type and endpoints; CALLS hops also carry "
            "confidence ('resolved'/'ambiguous') and strategy (see ADR-0014). "
            "Returns: {found, from_uid, to_uid, hop_count, hops: [{from, to, edge_type, "
            "confidence, strategy}], query_ms}, or {found: false, message} if no path "
            "exists within max_depth."
        ),
    )
    async def trace_path(
        from_uid: Annotated[str, Field(description="uid of the starting entity (from get_node/hybrid_search).")],
        to_uid: Annotated[str, Field(description="uid of the target entity.")],
        max_depth: Annotated[int, Field(6, description="Maximum hops to search.", ge=1, le=10)] = 6,
        edge_types: Annotated[
            str,
            Field("", description="Comma-separated relationship types to traverse. Empty = CALLS,IMPORTS,USES_TYPE."),
        ] = "",
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        try:
            app = await _ensure_root(ctx)
        except IndexNotReadyError as exc:
            return _error(str(exc), code="INDEX_REQUIRED")
        types, type_error = _parse_rel_types(edge_types, _DEFAULT_TRACE_EDGE_TYPES)
        if type_error:
            return type_error
        depth = _clamp_depth(max_depth)
        try:
            return await _trace_path(app.graph, from_uid, to_uid, max_depth=depth, edge_types=types)
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")

    @mcp.tool(
        description=(
            "Depth-limited transitive closure of callers/callees/both from an entity — "
            "'what would be affected if I change this'. Traverses the dependency edges by "
            "default — CALLS, USES_TYPE, IMPLEMENTS, OVERRIDES, INHERITS, REFERENCES, "
            "REGISTERED_BY, IMPORTS (override with edge_types). Each hit carries `via`, the "
            "edge types by which it reaches the entity, so a dependent found through "
            "USES_TYPE is not read as a caller. Containment (DEFINES/CONTAINS) is excluded: "
            "it would make changing one method 'affect' everything its class touches. "
            "Flags entities reachable only via an ambiguous edge as ambiguous_only=true — a "
            "heuristic, not a guarantee (see ADR-0014): no path made entirely of "
            "confidence:'resolved' edges reaches it within max_depth. "
            "Returns: {uid, direction, max_depth, affected_count, affected: [{uid, name, "
            "qualified_name, label, file_path, min_depth, direction, via, ambiguous_only}], "
            "truncated, query_ms}."
        ),
    )
    async def blast_radius(
        uid: Annotated[str, Field(description="uid of the entity to analyze (from get_node/hybrid_search).")],
        direction: Annotated[
            Literal["callers", "callees", "both"],
            Field(
                "callers",
                description="callers (who transitively depends on this), callees (what this depends on), or both.",
            ),
        ] = "callers",
        max_depth: Annotated[int, Field(3, description="Maximum hops to traverse.", ge=1, le=10)] = 3,
        edge_types: Annotated[
            str, Field("", description="Comma-separated relationship types to traverse. Empty = CALLS.")
        ] = "",
        limit: Annotated[int, Field(20, description="Max affected entities to return.", ge=1, le=100)] = 20,
        exclude_tests: Annotated[
            bool | None,
            Field(
                None,
                description=(
                    "Exclude test entities from the affected list and affected_count. Default true — "
                    "override to include. Distinct from the per-entity test_only flag, which reports "
                    "whether a test-free call path reaches the entity, not where the entity lives."
                ),
            ),
        ] = None,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        try:
            app = await _ensure_root(ctx)
        except IndexNotReadyError as exc:
            return _error(str(exc), code="INDEX_REQUIRED")
        types, type_error = _parse_rel_types(edge_types, _DEFAULT_BLAST_EDGE_TYPES)
        if type_error:
            return type_error
        depth = _clamp_depth(max_depth)
        clamped_limit = _clamp_limit(limit)
        test_patterns = _resolve_test_patterns(app.settings.search, exclude_tests)
        try:
            return await _blast_radius(
                app.graph,
                uid,
                direction=direction,
                max_depth=depth,
                edge_types=types,
                limit=clamped_limit,
                test_patterns=test_patterns,
            )
        except QueryTimeoutError as exc:
            return _error(str(exc), code="QUERY_TIMEOUT")
