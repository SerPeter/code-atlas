"""Hybrid search — RRF fusion across graph, vector, and BM25 channels.

Consumed by both the MCP server and the CLI ``atlas search`` command.
"""

from __future__ import annotations

import asyncio
import fnmatch
import re
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import tiktoken
from loguru import logger

from code_atlas.schema import RelType

if TYPE_CHECKING:
    from code_atlas.embeddings import EmbedClient
    from code_atlas.graph import GraphClient
    from code_atlas.settings import SearchSettings

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class SearchType(StrEnum):
    GRAPH = "graph"
    VECTOR = "vector"
    BM25 = "bm25"


@dataclass(frozen=True)
class SearchResult:
    """A single fused search result with provenance."""

    uid: str
    name: str
    qualified_name: str
    kind: str
    file_path: str
    line_start: int | None
    line_end: int | None
    signature: str
    docstring: str
    labels: list[str]
    rrf_score: float
    sources: dict[str, int] = field(default_factory=dict)  # channel → rank
    visibility: str = "public"


@dataclass(frozen=True)
class CompactNode:
    """Lightweight node representation for context expansion results."""

    uid: str
    name: str
    qualified_name: str
    kind: str
    file_path: str
    line_start: int | None = None
    line_end: int | None = None
    signature: str = ""
    docstring: str = ""
    labels: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExpandedContext:
    """Full neighborhood context for a single entity."""

    target: CompactNode
    parent: CompactNode | None = None
    siblings: list[CompactNode] = field(default_factory=list)
    callees: list[CompactNode] = field(default_factory=list)
    callers: list[CompactNode] = field(default_factory=list)
    docs: list[CompactNode] = field(default_factory=list)
    package_context: str = ""


@dataclass(frozen=True)
class ContextItem:
    """A single piece of assembled context with its token cost."""

    role: str  # target | parent | callee | caller | doc | sibling | package
    text: str
    tokens: int
    uid: str = ""
    truncated: bool = False


@dataclass(frozen=True)
class AssembledContext:
    """Budget-aware assembled context ready for LLM consumption."""

    items: list[ContextItem]
    total_tokens: int
    budget: int
    excluded_counts: dict[str, int] = field(default_factory=dict)

    def render(self) -> str:
        """Render all items as a single text block."""
        return "\n\n".join(item.text for item in self.items)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_TOKENIZER_ALIASES: dict[str, str] = {
    "claude": "cl100k_base",
}


@lru_cache(maxsize=4)
def _get_encoding(name: str) -> tiktoken.Encoding:
    """Get a cached tiktoken encoding by name."""
    resolved = _TOKENIZER_ALIASES.get(name, name)
    return tiktoken.get_encoding(resolved)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in *text* using a tiktoken encoding."""
    if not text:
        return 0
    return len(_get_encoding(encoding_name).encode(text))


# ---------------------------------------------------------------------------
# Context rendering helpers
# ---------------------------------------------------------------------------

_ROLE_HEADERS: dict[str, str] = {
    "target": "## Target",
    "parent": "## Class Context",
    "callee": "## Direct Callees",
    "caller": "## Direct Callers",
    "doc": "## Documentation",
    "sibling": "## Sibling Methods",
    "package": "## Package Context",
}

_MIN_USEFUL_TOKENS = 20


def _render_node_text(node: CompactNode, *, include_docstring: bool = False) -> str:
    """Render a CompactNode as compact text for context assembly."""
    parts: list[str] = []

    qn = node.qualified_name or node.name
    loc = node.file_path or ""
    if loc and node.line_start is not None:
        loc += f":{node.line_start}"
        if node.line_end is not None:
            loc += f"-{node.line_end}"

    parts.append(f"# {qn}" + (f" ({loc})" if loc else ""))

    if node.signature:
        parts.append(node.signature)

    if include_docstring and node.docstring:
        parts.append(node.docstring)

    return "\n".join(parts)


def _truncate_to_budget(text: str, max_tokens: int, encoding_name: str) -> str:
    """Truncate *text* to fit within *max_tokens*, cutting at line boundaries."""
    if not text or max_tokens <= 0:
        return ""
    enc = _get_encoding(encoding_name)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated = enc.decode(tokens[:max_tokens])
    # Cut at last newline to avoid mid-line truncation
    last_nl = truncated.rfind("\n")
    if last_nl > 0:
        return truncated[:last_nl]
    return truncated


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------


def _make_target_item(expanded: ExpandedContext, budget: int, tokenizer: str) -> ContextItem:
    """Build the always-included target ContextItem, truncating if needed."""
    text = f"{_ROLE_HEADERS['target']}\n{_render_node_text(expanded.target, include_docstring=True)}"
    tokens = count_tokens(text, tokenizer)
    if tokens > budget > 0:
        text = _truncate_to_budget(text, budget, tokenizer)
        tokens = count_tokens(text, tokenizer)
        return ContextItem(role="target", text=text, tokens=tokens, uid=expanded.target.uid, truncated=True)
    return ContextItem(role="target", text=text, tokens=tokens, uid=expanded.target.uid)


def assemble_context(
    expanded: ExpandedContext,
    budget: int = 8000,
    tokenizer: str = "cl100k_base",
) -> AssembledContext:
    """Assemble context within token budget using priority ordering.

    Priority:
    1. Target code (always included)
    2. Class context (parent)
    3. Direct callees
    4. Direct callers
    5. Documentation
    6. Sibling methods
    7. Package context
    """
    items: list[ContextItem] = []
    used = 0
    excluded: dict[str, int] = {}
    seen_roles: set[str] = set()

    def _try_add(role: str, text: str, uid: str = "") -> bool:
        nonlocal used
        full_text = text
        if role not in seen_roles:
            full_text = f"{_ROLE_HEADERS.get(role, f'## {role.title()}')}\n{text}"

        tokens = count_tokens(full_text, tokenizer)
        if used + tokens <= budget:
            items.append(ContextItem(role=role, text=full_text, tokens=tokens, uid=uid))
            seen_roles.add(role)
            used += tokens
            return True

        remaining = budget - used
        if remaining >= _MIN_USEFUL_TOKENS:
            trunc = _truncate_to_budget(full_text, remaining, tokenizer)
            if trunc:
                t_tokens = count_tokens(trunc, tokenizer)
                items.append(ContextItem(role=role, text=trunc, tokens=t_tokens, uid=uid, truncated=True))
                seen_roles.add(role)
                used += t_tokens
                return True
        return False

    def _add_nodes(role: str, nodes: list[CompactNode], *, include_docstring: bool = False) -> None:
        for i, node in enumerate(nodes):
            if budget - used < _MIN_USEFUL_TOKENS:
                excluded[role] = excluded.get(role, 0) + (len(nodes) - i)
                break
            if not _try_add(role, _render_node_text(node, include_docstring=include_docstring), node.uid):
                excluded[role] = excluded.get(role, 0) + 1

    # Priority 1: Target (always included)
    target_item = _make_target_item(expanded, budget, tokenizer)
    items.append(target_item)
    seen_roles.add("target")
    used += target_item.tokens

    # Priority 2: Class context
    if (
        expanded.parent
        and budget - used >= _MIN_USEFUL_TOKENS
        and not _try_add("parent", _render_node_text(expanded.parent), expanded.parent.uid)
    ):
        excluded["parent"] = 1

    # Priority 3-6: callees, callers, docs, siblings
    _add_nodes("callee", expanded.callees)
    _add_nodes("caller", expanded.callers)
    _add_nodes("doc", expanded.docs, include_docstring=True)
    _add_nodes("sibling", expanded.siblings)

    # Priority 7: Package context
    if (
        expanded.package_context
        and budget - used >= _MIN_USEFUL_TOKENS
        and not _try_add("package", expanded.package_context)
    ):
        excluded["package"] = 1

    if excluded:
        logger.debug("Context assembly excluded: {}", excluded)

    return AssembledContext(items=items, total_tokens=used, budget=budget, excluded_counts=excluded)


# ---------------------------------------------------------------------------
# Scope expansion (monorepo support)
# ---------------------------------------------------------------------------


def expand_scope(
    scope: str,
    all_projects: list[str],
    always_include: list[str] | None = None,
) -> list[str] | None:
    """Expand a scope string into a list of project names.

    - Empty string → ``None`` (no filter — all projects).
    - Single name → ``[name] + always_include``.
    - Glob pattern (``services/*``) → matching projects + always_include.
    - Comma-separated → split + always_include.

    Returns ``None`` for no filtering, or a deduplicated list of project names.
    """
    if not scope:
        return None

    always = always_include or []

    # Comma-separated list
    parts = [s.strip() for s in scope.split(",") if s.strip()]

    matched: list[str] = []
    for part in parts:
        if "*" in part or "?" in part or "[" in part:
            # Glob pattern
            matched.extend(p for p in all_projects if fnmatch.fnmatch(p, part))
        else:
            matched.append(part)

    # Add always_include projects
    for inc in always:
        if inc not in matched:
            matched.append(inc)

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for name in matched:
        if name not in seen:
            seen.add(name)
            result.append(name)

    return result or None


# ---------------------------------------------------------------------------
# RRF fusion (pure function)
# ---------------------------------------------------------------------------

_IDENTIFIER_RE = re.compile(r"^[A-Z][a-zA-Z0-9]+$")  # PascalCase
_SNAKE_RE = re.compile(r"^[a-z][a-z0-9_]+$")  # snake_case
_DOTTED_RE = re.compile(r"\.")  # dotted path


def rrf_fuse(
    ranked_lists: dict[str, list[str]],
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Reciprocal Rank Fusion across multiple ranked lists.

    Parameters
    ----------
    ranked_lists:
        ``{channel_name: [uid, uid, ...]}`` — items ordered by rank (0 = best).
    k:
        RRF smoothing constant.
    weights:
        Per-channel multipliers (default 1.0 for missing channels).

    Returns
    -------
    ``{uid: rrf_score}`` dict sorted by score descending.
    """
    weights = weights or {}
    scores: dict[str, float] = {}
    for channel, uids in ranked_lists.items():
        w = weights.get(channel, 1.0)
        for rank, uid in enumerate(uids):
            scores[uid] = scores.get(uid, 0.0) + w * (1.0 / (k + rank + 1))
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


# ---------------------------------------------------------------------------
# Query analysis heuristic
# ---------------------------------------------------------------------------


def analyze_query(query: str) -> dict[str, float]:
    """Return per-channel weight adjustments based on query shape.

    - Identifier-like (PascalCase, snake_case, dotted, short ≤2 words):
      boost graph + BM25, suppress vector.
    - Natural language (3+ words, no structural patterns):
      boost vector, suppress graph.
    - Default: balanced 1.0 weights.
    """
    stripped = query.strip()
    words = stripped.split()

    # Identifier-like patterns
    is_identifier = (
        _IDENTIFIER_RE.match(stripped)
        or _SNAKE_RE.match(stripped)
        or _DOTTED_RE.search(stripped)
        or (len(words) <= 2 and any(_IDENTIFIER_RE.match(w) or ("_" in w and _SNAKE_RE.match(w)) for w in words))
    )
    if is_identifier:
        return {"graph": 2.0, "vector": 0.5, "bm25": 1.5}

    # Natural language (3+ words, no structural indicators)
    if len(words) >= 3:
        return {"graph": 0.5, "vector": 2.0, "bm25": 1.0}

    return {"graph": 1.0, "vector": 1.0, "bm25": 1.0}


# ---------------------------------------------------------------------------
# Node extraction helpers
# ---------------------------------------------------------------------------


def _extract_uid(record: dict[str, Any]) -> str:
    """Get uid from a search result record (node key or n key)."""
    node = record.get("node") or record.get("n")
    if node is None:
        return ""
    if hasattr(node, "get"):
        return node.get("uid", "")
    if isinstance(node, dict):
        return node.get("uid", "")
    return ""


def _extract_props(record: dict[str, Any]) -> dict[str, Any]:
    """Extract node properties from a search result record."""
    node = record.get("node") or record.get("n")
    if node is None:
        return {}
    if hasattr(node, "items") and hasattr(node, "labels"):
        # neo4j Node object
        props = dict(node.items())
        props["_labels"] = sorted(node.labels)
        return props
    if isinstance(node, dict):
        return dict(node)
    return {}


# ---------------------------------------------------------------------------
# Hybrid search orchestrator
# ---------------------------------------------------------------------------


def _compute_weights(
    settings: SearchSettings,
    query: str,
    explicit: dict[str, float] | None,
) -> dict[str, float]:
    """Merge default, auto-analyzed, and explicit per-channel weights."""
    effective = dict(settings.default_weights)
    for ch, w in analyze_query(query).items():
        effective[ch] = effective.get(ch, 1.0) * w
    if explicit:
        effective.update(explicit)
    return effective


def _build_ranked_lists(
    channel_results: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    """Extract ranked uid lists and node properties from channel results."""
    ranked_lists: dict[str, list[str]] = {}
    props_by_uid: dict[str, dict[str, Any]] = {}

    for channel, results in channel_results.items():
        uids: list[str] = []
        for record in results:
            uid = _extract_uid(record)
            if uid and uid not in props_by_uid:
                props_by_uid[uid] = _extract_props(record)
            if uid:
                uids.append(uid)
        ranked_lists[channel] = uids

    return ranked_lists, props_by_uid


def _build_provenance(ranked_lists: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    """Build rank provenance per uid (1-indexed)."""
    uid_ranks: dict[str, dict[str, int]] = {}
    for channel, uids in ranked_lists.items():
        for rank, uid in enumerate(uids):
            uid_ranks.setdefault(uid, {})[channel] = rank + 1
    return uid_ranks


def _normalize_path(path: str) -> str:
    """Normalize a file path to forward slashes and lowercase for cross-platform matching."""
    return path.replace("\\", "/").lower()


def _is_test_result(result: SearchResult, patterns: list[str]) -> bool:
    """Return True if *result* matches test file/entity patterns."""
    fp = _normalize_path(result.file_path)
    basename = fp.rsplit("/", 1)[-1] if "/" in fp else fp

    for pat in patterns:
        pat_lower = pat.lower()
        if pat_lower.endswith("/"):
            # Directory pattern — check if any path segment matches
            if f"/{pat_lower}" in f"/{fp}/" or fp.startswith(pat_lower):
                return True
        elif fnmatch.fnmatch(basename, pat_lower):
            return True

    # Also check entity name for test_* / *_test patterns
    name = result.name.lower()
    return name.startswith("test_") or name.endswith("_test")


def _is_stub_result(result: SearchResult) -> bool:
    """Return True if *result* comes from a .pyi type-stub file."""
    return _normalize_path(result.file_path).endswith(".pyi")


def _is_generated_result(result: SearchResult, patterns: list[str]) -> bool:
    """Return True if *result* matches generated-code patterns."""
    fp = _normalize_path(result.file_path)
    basename = fp.rsplit("/", 1)[-1] if "/" in fp else fp
    return any(fnmatch.fnmatch(basename, pat.lower()) for pat in patterns)


def _apply_filters(
    results: list[SearchResult],
    settings: SearchSettings,
    *,
    exclude_tests: bool | None = None,
    exclude_stubs: bool | None = None,
    exclude_generated: bool | None = None,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> list[SearchResult]:
    """Apply post-fusion filters to search results.

    ``None`` values fall back to settings defaults.  Exclude filters run first,
    then include-pattern whitelisting narrows further.
    """
    do_tests = settings.test_filter if exclude_tests is None else exclude_tests
    do_stubs = settings.stub_filter if exclude_stubs is None else exclude_stubs
    do_generated = settings.generated_filter if exclude_generated is None else exclude_generated

    filtered: list[SearchResult] = []
    excluded = 0
    for result in results:
        fp = _normalize_path(result.file_path)
        basename = fp.rsplit("/", 1)[-1] if "/" in fp else fp

        # Exclude filters
        if do_tests and _is_test_result(result, settings.test_patterns):
            excluded += 1
            continue
        if do_stubs and _is_stub_result(result):
            excluded += 1
            continue
        if do_generated and _is_generated_result(result, settings.generated_patterns):
            excluded += 1
            continue
        if exclude_patterns and any(fnmatch.fnmatch(basename, p.lower()) for p in exclude_patterns):
            excluded += 1
            continue

        # Include-pattern whitelist (if specified, only matching results pass)
        if include_patterns and not any(fnmatch.fnmatch(basename, p.lower()) for p in include_patterns):
            excluded += 1
            continue

        filtered.append(result)

    if excluded:
        logger.debug("Result filtering excluded {} of {} results", excluded, len(results))

    return filtered


_VIS_BOOST: dict[str, float] = {"public": 1.0, "protected": 0.97, "internal": 0.94, "private": 0.88}


def _boost_results(results: list[SearchResult]) -> list[SearchResult]:
    """Re-rank by RRF score * visibility boost. Preserves relative order for equal boosts."""
    return sorted(
        results,
        key=lambda r: r.rrf_score * _VIS_BOOST.get(r.visibility, 1.0),
        reverse=True,
    )


async def hybrid_search(
    graph: GraphClient,
    embed: EmbedClient | None,
    settings: SearchSettings,
    query: str,
    *,
    search_types: list[SearchType] | None = None,
    limit: int = 20,
    scope: str = "",
    weights: dict[str, float] | None = None,
    exclude_tests: bool | None = None,
    exclude_stubs: bool | None = None,
    exclude_generated: bool | None = None,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> list[SearchResult]:
    """Run hybrid search across selected channels and fuse with RRF.

    Parameters
    ----------
    graph:
        Connected GraphClient instance.
    embed:
        EmbedClient for vector search (None to skip vector channel).
    settings:
        SearchSettings with rrf_k and default_weights.
    query:
        Search query string.
    search_types:
        Channels to search (default: all three).
    limit:
        Max results to return.
    scope:
        Optional project name filter.
    weights:
        Explicit per-channel weight overrides (merged with auto-weights).
    exclude_tests:
        Exclude test entities (None = use settings.test_filter).
    exclude_stubs:
        Exclude .pyi stubs (None = use settings.stub_filter).
    exclude_generated:
        Exclude generated code (None = use settings.generated_filter).
    include_patterns:
        Only include results whose basename matches one of these globs.
    exclude_patterns:
        Exclude results whose basename matches any of these globs.
    """
    if search_types is None:
        search_types = list(SearchType)

    effective_weights = _compute_weights(settings, query, weights)

    # Pre-compute embedding if vector channel is requested
    vector: list[float] | None = None
    if SearchType.VECTOR in search_types and embed is not None:
        try:
            vector = await embed.embed_one(query)
        except Exception as exc:
            logger.warning("Embedding failed, skipping vector channel: {}", exc)
            search_types = [st for st in search_types if st != SearchType.VECTOR]

    # Resolve scope to projects list (monorepo-aware)
    # Supports comma-separated project names (e.g., "auth,shared")
    if scope:
        parts = [s.strip() for s in scope.split(",") if s.strip()]
        scope_projects: list[str] | None = parts
    else:
        scope_projects = None

    # Fire search channels in parallel
    tasks: dict[str, asyncio.Task[list[dict[str, Any]]]] = {}
    fetch_limit = limit * 3  # over-fetch for fusion quality

    if SearchType.GRAPH in search_types:
        tasks["graph"] = asyncio.create_task(graph.graph_search(query, limit=fetch_limit, projects=scope_projects))
    if SearchType.VECTOR in search_types and vector is not None:
        tasks["vector"] = asyncio.create_task(graph.vector_search(vector, limit=fetch_limit, projects=scope_projects))
    if SearchType.BM25 in search_types:
        tasks["bm25"] = asyncio.create_task(graph.text_search(query, limit=fetch_limit, projects=scope_projects))

    # Collect results
    channel_results: dict[str, list[dict[str, Any]]] = {}
    for channel, task in tasks.items():
        try:
            channel_results[channel] = await task
        except Exception as exc:
            logger.warning("Search channel {} failed: {}", channel, exc)
            channel_results[channel] = []

    ranked_lists, props_by_uid = _build_ranked_lists(channel_results)
    fused_scores = rrf_fuse(ranked_lists, k=settings.rrf_k, weights=effective_weights)
    uid_ranks = _build_provenance(ranked_lists)

    # Build all SearchResult objects, apply filters, then slice to limit
    all_results = [
        SearchResult(
            uid=uid,
            name=props_by_uid.get(uid, {}).get("name", ""),
            qualified_name=props_by_uid.get(uid, {}).get("qualified_name", ""),
            kind=props_by_uid.get(uid, {}).get("kind", ""),
            file_path=props_by_uid.get(uid, {}).get("file_path", ""),
            line_start=props_by_uid.get(uid, {}).get("line_start"),
            line_end=props_by_uid.get(uid, {}).get("line_end"),
            signature=props_by_uid.get(uid, {}).get("signature", ""),
            docstring=props_by_uid.get(uid, {}).get("docstring", ""),
            labels=props_by_uid.get(uid, {}).get("_labels", []),
            rrf_score=rrf_score,
            sources=uid_ranks.get(uid, {}),
            visibility=props_by_uid.get(uid, {}).get("visibility", "public"),
        )
        for uid, rrf_score in fused_scores.items()
    ]

    filtered = _apply_filters(
        all_results,
        settings,
        exclude_tests=exclude_tests,
        exclude_stubs=exclude_stubs,
        exclude_generated=exclude_generated,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )
    return _boost_results(filtered)[:limit]


# ---------------------------------------------------------------------------
# Context expansion helpers
# ---------------------------------------------------------------------------


def _node_to_compact(node: Any) -> CompactNode:
    """Convert a neo4j Node object to a CompactNode dataclass."""
    if hasattr(node, "items") and hasattr(node, "labels"):
        props = dict(node.items())
        labels = sorted(node.labels)
    elif isinstance(node, dict):
        props = node
        labels = node.get("_labels", [])
    else:
        return CompactNode(uid="", name="", qualified_name="", kind="", file_path="")

    return CompactNode(
        uid=props.get("uid", ""),
        name=props.get("name", ""),
        qualified_name=props.get("qualified_name", ""),
        kind=props.get("kind", ""),
        file_path=props.get("file_path", ""),
        line_start=props.get("line_start"),
        line_end=props.get("line_end"),
        signature=props.get("signature", ""),
        docstring=props.get("docstring", ""),
        labels=labels,
    )


def _records_to_compact(records: list[dict[str, Any]], key: str = "n") -> list[CompactNode]:
    """Convert a list of query records to CompactNode list."""
    result: list[CompactNode] = []
    for record in records:
        node = record.get(key)
        if node is not None:
            result.append(_node_to_compact(node))
    return result


def _prioritize_callers(callers: list[CompactNode], target_qn: str) -> list[CompactNode]:
    """Rank callers: same-package first, non-test first, shorter qualified_name.

    Parameters
    ----------
    callers:
        Unranked list of caller CompactNodes.
    target_qn:
        Qualified name of the target entity (used for same-package detection).
    """
    target_pkg = target_qn.rsplit(".", 1)[0] if "." in target_qn else ""

    def _sort_key(caller: CompactNode) -> tuple[int, int, int]:
        # Same package = 0 (preferred), different = 1
        caller_pkg = caller.qualified_name.rsplit(".", 1)[0] if "." in caller.qualified_name else ""
        same_pkg = 0 if (target_pkg and caller_pkg == target_pkg) else 1

        # Non-test = 0 (preferred), test = 1
        is_test = 1 if "test" in (caller.file_path or "").lower() else 0

        # Shorter qualified_name preferred
        qn_len = len(caller.qualified_name)

        return (same_pkg, is_test, qn_len)

    return sorted(callers, key=_sort_key)


# ---------------------------------------------------------------------------
# Context expansion
# ---------------------------------------------------------------------------


async def expand_context(
    graph: GraphClient,
    uid: str,
    *,
    include_hierarchy: bool = True,
    include_calls: bool = True,
    call_depth: int = 1,
    include_docs: bool = True,
    max_siblings: int = 5,
    max_callers: int = 10,
) -> ExpandedContext | None:
    """Expand a node into its full neighborhood context.

    Fires sub-queries in parallel via ``asyncio.gather`` for speed.

    Parameters
    ----------
    graph:
        Connected GraphClient instance.
    uid:
        The unique identifier of the target node.
    include_hierarchy:
        Include parent and sibling nodes.
    include_calls:
        Include callers and callees.
    call_depth:
        Max relationship hops for CALLS traversal (1-3).
    include_docs:
        Include documentation nodes linked via DOCUMENTS.
    max_siblings:
        Max sibling entities to return.
    max_callers:
        Max callers to return (over-fetched then prioritized).
    """
    call_depth = max(1, min(call_depth, 3))

    # Always fetch the target node
    target_records = await graph.execute("MATCH (n {uid: $uid}) RETURN n", {"uid": uid})
    if not target_records:
        return None

    target = _node_to_compact(target_records[0].get("n"))

    # Build parallel sub-queries
    coros: dict[str, Any] = {}

    if include_hierarchy:
        coros["parent"] = graph.execute(
            f"MATCH (p)-[:{RelType.DEFINES}]->(n {{uid: $uid}}) RETURN p AS n LIMIT 1",
            {"uid": uid},
        )
        coros["siblings"] = graph.execute(
            f"MATCH (p)-[:{RelType.DEFINES}]->(n {{uid: $uid}}), (p)-[:{RelType.DEFINES}]->(s) "
            f"WHERE s.uid <> $uid RETURN s AS n LIMIT {max_siblings}",
            {"uid": uid},
        )
        coros["package_ctx"] = graph.execute(
            "MATCH (pkg)-[:CONTAINS*1..3]->(target {uid: $uid}) "
            "WHERE pkg:Package OR pkg:Module RETURN pkg.docstring AS docstring LIMIT 1",
            {"uid": uid},
        )

    if include_calls:
        coros["callers"] = graph.execute(
            f"MATCH (caller)-[:{RelType.CALLS}*1..{call_depth}]->(n {{uid: $uid}}) "
            f"RETURN DISTINCT caller AS n LIMIT {max_callers * 2}",
            {"uid": uid},
        )
        coros["callees"] = graph.execute(
            f"MATCH (n {{uid: $uid}})-[:{RelType.CALLS}*1..{call_depth}]->(callee) "
            "RETURN DISTINCT callee AS n LIMIT 20",
            {"uid": uid},
        )

    if include_docs:
        coros["docs"] = graph.execute(
            f"MATCH (doc)-[:{RelType.DOCUMENTS}]->(n {{uid: $uid}}) RETURN doc AS n LIMIT 10",
            {"uid": uid},
        )

    # Fire all sub-queries in parallel
    keys = list(coros.keys())
    results_list = await asyncio.gather(*coros.values(), return_exceptions=True)
    results_map: dict[str, list[dict[str, Any]]] = {}
    for key, result in zip(keys, results_list, strict=True):
        if isinstance(result, BaseException):
            logger.warning("Context sub-query '{}' failed: {}", key, result)
            results_map[key] = []
        else:
            results_map[key] = result

    # Extract results
    parent_nodes = _records_to_compact(results_map.get("parent", []))
    parent = parent_nodes[0] if parent_nodes else None

    siblings = _records_to_compact(results_map.get("siblings", []))

    raw_callers = _records_to_compact(results_map.get("callers", []))
    callers = _prioritize_callers(raw_callers, target.qualified_name)[:max_callers]

    callees = _records_to_compact(results_map.get("callees", []))
    docs = _records_to_compact(results_map.get("docs", []))

    # Package context docstring
    pkg_records = results_map.get("package_ctx", [])
    package_context = ""
    if pkg_records:
        package_context = pkg_records[0].get("docstring", "") or ""

    return ExpandedContext(
        target=target,
        parent=parent,
        siblings=siblings,
        callees=callees,
        callers=callers,
        docs=docs,
        package_context=package_context,
    )
