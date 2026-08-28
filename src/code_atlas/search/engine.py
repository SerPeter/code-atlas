"""Hybrid search — RRF fusion across graph, vector, and BM25 channels.

Consumed by both the MCP server and the CLI ``atlas search`` command.
"""

from __future__ import annotations

import asyncio
import fnmatch
import re
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import tiktoken
from loguru import logger

from code_atlas.schema import NodeLabel
from code_atlas.telemetry import get_meter, get_metrics, get_tracer

if TYPE_CHECKING:
    from typing import Protocol

    from code_atlas.settings import SearchSettings

    class GraphExecutor(Protocol):
        """Structural subset of GraphClient needed by expand_context (neighborhood navigation)."""

        async def get_entity_by_uid(self, uid: str, label: str = "") -> dict[str, Any] | None: ...

        async def get_defining_parent(self, uid: str) -> dict[str, Any] | None: ...

        async def get_sibling_entities(self, uid: str, limit: int) -> list[dict[str, Any]]: ...

        async def get_package_docstring(self, uid: str) -> str | None: ...

        async def get_callers(self, uid: str, label: str, call_depth: int, limit: int) -> list[dict[str, Any]]: ...

        async def get_callees(self, uid: str, label: str, call_depth: int, limit: int) -> list[dict[str, Any]]: ...

        async def get_linked_docs(self, uid: str, label: str, limit: int) -> list[dict[str, Any]]: ...

    class SearchGraph(Protocol):
        """Structural subset of GraphClient needed by hybrid_search's three channels.

        ``limit``/``projects`` are keyword-only here since engine.py always calls
        them by keyword — this keeps the protocol satisfied regardless of where
        an implementation's own optional params (e.g. GraphClient's ``label``) sit.
        """

        async def graph_search(
            self, query: str, *, limit: int, projects: list[str] | None = None
        ) -> list[dict[str, Any]]: ...

        async def text_search(
            self, query: str, *, limit: int, projects: list[str] | None = None
        ) -> list[dict[str, Any]]: ...

        async def vector_search(
            self, vector: list[float], *, limit: int, projects: list[str] | None = None
        ) -> list[dict[str, Any]]: ...

    class EmbedOne(Protocol):
        """Structural subset of EmbedClient needed by hybrid_search's vector channel."""

        async def embed_one(self, text: str) -> list[float]: ...


_tracer = get_tracer(__name__)
_meter = get_meter(__name__)

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
    source: str = ""
    # The value results are actually ordered by: rrf_score after the visibility, label
    # and secondary-project multipliers. Reported separately because showing the raw
    # rrf_score next to a boosted ordering produces a list that is not sorted by the
    # number beside it — `atlas search fetch` printed 0.0078 at rank 5 and 0.0076 at
    # rank 4, which reads as a broken ranker rather than a deliberate demotion.
    ranked_score: float = 0.0
    # Stamped at index time by the supersession sweep, never traversed per query.
    # A note whose author explicitly replaced it must not read as current guidance,
    # and the successor's uid travels with the hit so the reader can follow it.
    superseded_by: str = ""
    # Symmetric and NOT demoted: in an unresolved contradiction neither side is known
    # wrong, so demoting one would be the system picking a winner nobody picked.
    contradicts_with: tuple[str, ...] = ()


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
    source: str = ""
    labels: list[str] = field(default_factory=list)
    # Anchor staleness (§3.6) — set only for docs linked via an explicit
    # anchors: DOCUMENTS edge (link_type='anchor'); None for heuristic docs.
    stale: bool | None = None
    anchor_hash: str | None = None


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

    Returns ``None`` only when *scope* itself is empty (no filtering requested).
    A non-empty *scope* that matches zero projects (e.g. a glob with no matches
    and no ``always_include``) returns an empty list — callers MUST treat that
    as an explicit "match nothing" restriction, not as "no filter".
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

    return result


# ---------------------------------------------------------------------------
# RRF fusion (pure function)
# ---------------------------------------------------------------------------

_IDENTIFIER_RE = re.compile(r"^[A-Z][a-zA-Z0-9]+$")  # PascalCase
_SNAKE_RE = re.compile(r"^[a-z][a-z0-9_]+$")  # snake_case
_DOTTED_RE = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$")  # dotted path (whole token, e.g. pkg.mod.Class)


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
        or _DOTTED_RE.match(stripped)
        or (
            len(words) <= 2
            and any(_IDENTIFIER_RE.match(w) or _DOTTED_RE.match(w) or ("_" in w and _SNAKE_RE.match(w)) for w in words)
        )
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


def matches_test_pattern(file_path: str, name: str, patterns: list[str]) -> bool:
    """Return True if *file_path*/*name* matches configured test file/entity patterns.

    Primitive-typed so callers that don't have a full ``SearchResult`` (e.g. raw
    graph.text_search()/vector_search() records) can reuse the same rule.
    """
    fp = _normalize_path(file_path)
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
    name_lower = name.lower()
    return name_lower.startswith("test_") or name_lower.endswith("_test")


def _is_test_result(result: SearchResult, patterns: list[str]) -> bool:
    """Return True if *result* matches test file/entity patterns."""
    return matches_test_pattern(result.file_path, result.name, patterns)


def _is_stub_result(result: SearchResult) -> bool:
    """Return True if *result* comes from a type-stub file (.pyi or .d.ts)."""
    path = _normalize_path(result.file_path)
    return path.endswith((".pyi", ".d.ts"))


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
    code_only: bool = False,
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
        if code_only and _is_doc_result(result):
            excluded += 1
            continue
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


def _record_file_path_and_name(record: dict[str, Any]) -> tuple[str, str]:
    """Extract (file_path, name) from a raw graph.text_search()/vector_search() record."""
    node = record.get("node") or record.get("n")
    props = dict(node.items()) if hasattr(node, "items") else (node if isinstance(node, dict) else {})
    return props.get("file_path", "") or "", props.get("name", "") or ""


def filter_raw_records(
    records: list[dict[str, Any]],
    settings: SearchSettings,
    *,
    exclude_tests: bool | None = None,
    exclude_stubs: bool | None = None,
    exclude_generated: bool | None = None,
) -> list[dict[str, Any]]:
    """Apply the same test/stub/generated exclusion ``_apply_filters`` uses, directly on raw
    ``{"node": ..., ...}`` records — for callers (text_search/vector_search MCP tools) that
    query the graph directly rather than going through the full hybrid_search/SearchResult
    pipeline. ``None`` values fall back to settings defaults, same as ``_apply_filters``.
    """
    do_tests = settings.test_filter if exclude_tests is None else exclude_tests
    do_stubs = settings.stub_filter if exclude_stubs is None else exclude_stubs
    do_generated = settings.generated_filter if exclude_generated is None else exclude_generated
    if not (do_tests or do_stubs or do_generated):
        return records

    filtered: list[dict[str, Any]] = []
    for record in records:
        file_path, name = _record_file_path_and_name(record)
        if do_tests and matches_test_pattern(file_path, name, settings.test_patterns):
            continue
        if do_stubs and _normalize_path(file_path).endswith((".pyi", ".d.ts")):
            continue
        if do_generated:
            fp = _normalize_path(file_path)
            basename = fp.rsplit("/", 1)[-1] if "/" in fp else fp
            if any(fnmatch.fnmatch(basename, pat.lower()) for pat in settings.generated_patterns):
                continue
        filtered.append(record)
    return filtered


_VIS_BOOST: dict[str, float] = {"public": 1.0, "protected": 0.97, "internal": 0.94, "private": 0.88}

# "blended" (default): knowledge participates in every query, ranked slightly
# below code unless the caller asks for knowledge mode explicitly (Q7).
_LABEL_BOOST_BLENDED: dict[str, float] = {
    "Callable": 1.15,
    "TypeDef": 1.15,
    "Module": 1.10,
    "Value": 1.10,
    "Package": 1.05,
    "DocFile": 0.70,
    "DocSection": 0.70,
    "Note": 0.70,
}

# "knowledge": invert the boost so notes/docs outrank code — for "why/decision/
# gotcha"-shaped queries where the answer lives in prose, not in the AST.
_LABEL_BOOST_KNOWLEDGE: dict[str, float] = {
    "Callable": 0.85,
    "TypeDef": 0.85,
    "Module": 0.90,
    "Value": 0.90,
    "Package": 0.95,
    "DocFile": 1.10,
    "DocSection": 1.15,
    "Note": 1.15,
}

# Results from a secondary (extra-vault: global/memory-dir) project are never
# excluded, just deprioritized when the current project has an equally
# relevant answer — a coarse first cut at "current > global > other" (R6).
_SECONDARY_PROJECT_BOOST = 0.92

_DOC_LABELS = frozenset({"DocFile", "DocSection", "Note"})


class SearchMode(StrEnum):
    """Knowledge-participation mode for hybrid_search's label boosting."""

    CODE = "code"
    KNOWLEDGE = "knowledge"
    BLENDED = "blended"


def _is_doc_result(result: SearchResult) -> bool:
    """Return True if *result* is a documentation entity (DocFile, DocSection, or Note)."""
    return bool(_DOC_LABELS & set(result.labels))


# Demotion, not exclusion. A superseded note stays findable -- it is the provenance
# for the note that replaced it, and the dream-mode archive-stub decision keeps it
# reachable for the same reason. 0.5 is stronger than any label boost and weaker than
# a filter: it loses to its own successor on an equal raw score, without vanishing.
_SUPERSEDED_PENALTY = 0.5


def _boost_results(
    results: list[SearchResult],
    *,
    label_boost: dict[str, float] | None = None,
    secondary_projects: frozenset[str] | None = None,
) -> list[SearchResult]:
    """Re-rank by RRF score * visibility * label * project-scope * supersession."""
    boost_table = label_boost if label_boost is not None else _LABEL_BOOST_BLENDED

    def _project_boost(result: SearchResult) -> float:
        if not secondary_projects:
            return 1.0
        project_name = result.uid.split(":", 1)[0] if ":" in result.uid else ""
        return _SECONDARY_PROJECT_BOOST if project_name in secondary_projects else 1.0

    def _effective(result: SearchResult) -> float:
        return (
            result.rrf_score
            * _VIS_BOOST.get(result.visibility, 1.0)
            * max((boost_table.get(lbl, 1.0) for lbl in result.labels), default=1.0)
            * _project_boost(result)
            * (_SUPERSEDED_PENALTY if result.superseded_by else 1.0)
        )

    # Recorded on each result rather than discarded with the sort key, so a consumer can
    # show the number that produced the order it is looking at.
    scored = [replace(r, ranked_score=_effective(r)) for r in results]
    return sorted(scored, key=lambda r: r.ranked_score, reverse=True)


async def hybrid_search(  # noqa: PLR0912, PLR0915
    graph: SearchGraph,
    embed: EmbedOne | None,
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
    code_only: bool = False,
    mode: SearchMode | str = SearchMode.BLENDED,
    secondary_projects: frozenset[str] | None = None,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    channel_status: dict[str, str] | None = None,
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
    code_only:
        Exclude documentation entities (DocSection, DocFile, Note).
    mode:
        Knowledge-participation mode: "blended" (default, knowledge ranked
        slightly below code — R6/Q7), "knowledge" (invert — notes/docs
        outrank code, for why/decision/gotcha-shaped questions), or "code"
        (equivalent to ``code_only=True`` — knowledge excluded entirely).
    secondary_projects:
        Project names (extra vaults: global/memory-dir) to deprioritize
        without excluding — current project's own results are unaffected.
    include_patterns:
        Only include results whose basename matches one of these globs.
    exclude_patterns:
        Exclude results whose basename matches any of these globs.
    channel_status:
        Optional dict the caller supplies to receive per-channel outcomes
        (``"ok"``, ``"unavailable: ..."``, or ``"error: ..."``) keyed by
        channel name (``graph``/``vector``/``bm25``). Left untouched if
        ``None``. Lets a caller detect silent channel degradation instead
        of an empty/partial result set looking like a complete search.
    """
    with _tracer.start_as_current_span(
        "hybrid_search", attributes={"query": query, "limit": limit, "scope": scope}
    ) as span:
        if search_types is None:
            search_types = list(SearchType)
        requested_types = list(search_types)

        span.set_attribute("search_types", ",".join(st.value for st in search_types))
        effective_weights = _compute_weights(settings, query, weights)

        # Pre-compute embedding if vector channel is requested
        vector: list[float] | None = None
        if SearchType.VECTOR in search_types:
            if embed is None:
                logger.warning("Vector channel requested but no embedding client is configured; skipping.")
                if channel_status is not None:
                    channel_status["vector"] = "unavailable: no embedding client configured"
                search_types = [st for st in search_types if st != SearchType.VECTOR]
            else:
                with _tracer.start_as_current_span("embed_query"):
                    try:
                        vector = await embed.embed_one(query)
                    except Exception as exc:
                        logger.warning("Embedding failed, skipping vector channel: {}", exc)
                        if channel_status is not None:
                            channel_status["vector"] = f"error: {exc}"
                        search_types = [st for st in search_types if st != SearchType.VECTOR]

        # Resolve scope to projects list (monorepo-aware)
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
            tasks["vector"] = asyncio.create_task(
                graph.vector_search(vector, limit=fetch_limit, projects=scope_projects)
            )
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
                if channel_status is not None:
                    channel_status[channel] = f"error: {exc}"

        if channel_status is not None:
            for st in requested_types:
                channel_status.setdefault(st.value, "ok")

        with _tracer.start_as_current_span("rrf_fuse"):
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
                file_path=props_by_uid.get(uid, {}).get("file_path", "") or "",
                line_start=props_by_uid.get(uid, {}).get("line_start"),
                line_end=props_by_uid.get(uid, {}).get("line_end"),
                signature=props_by_uid.get(uid, {}).get("signature", ""),
                docstring=props_by_uid.get(uid, {}).get("docstring", ""),
                labels=props_by_uid.get(uid, {}).get("_labels", []),
                rrf_score=rrf_score,
                sources=uid_ranks.get(uid, {}),
                visibility=props_by_uid.get(uid, {}).get("visibility", "public"),
                source=props_by_uid.get(uid, {}).get("source", "") or "",
                superseded_by=props_by_uid.get(uid, {}).get("superseded_by", "") or "",
                contradicts_with=tuple(props_by_uid.get(uid, {}).get("contradicts_with") or ()),
            )
            for uid, rrf_score in fused_scores.items()
        ]

        mode_value = mode.value if isinstance(mode, SearchMode) else str(mode)
        effective_code_only = code_only or mode_value == SearchMode.CODE
        label_boost = _LABEL_BOOST_KNOWLEDGE if mode_value == SearchMode.KNOWLEDGE else _LABEL_BOOST_BLENDED

        with _tracer.start_as_current_span("filter_and_boost"):
            filtered = _apply_filters(
                all_results,
                settings,
                exclude_tests=exclude_tests,
                exclude_stubs=exclude_stubs,
                exclude_generated=exclude_generated,
                code_only=effective_code_only,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
            )
            results = _boost_results(filtered, label_boost=label_boost, secondary_projects=secondary_projects)[:limit]

        span.set_attribute("results_count", len(results))
        m = get_metrics()
        m.query_count.add(1, {"type": "hybrid"})
        m.search_results_count.record(len(results))
        return results


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
        source=props.get("source", "") or "",
        labels=labels,
    )


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
    graph: GraphExecutor,
    uid: str,
    *,
    label: str | None = None,
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
    with _tracer.start_as_current_span("expand_context", attributes={"uid": uid}) as span:
        return await _expand_context_inner(
            graph,
            uid,
            label=label,
            span=span,
            include_hierarchy=include_hierarchy,
            include_calls=include_calls,
            call_depth=call_depth,
            include_docs=include_docs,
            max_siblings=max_siblings,
            max_callers=max_callers,
        )


async def _expand_context_inner(
    graph: GraphExecutor,
    uid: str,
    *,
    label: str | None = None,
    span: Any,
    include_hierarchy: bool,
    include_calls: bool,
    call_depth: int,
    include_docs: bool,
    max_siblings: int,
    max_callers: int,
) -> ExpandedContext | None:
    """Inner implementation of expand_context (separated to keep span wrapper clean)."""
    call_depth = max(1, min(call_depth, 3))
    if label and label not in NodeLabel:
        msg = f"Invalid node label: {label!r}"
        raise ValueError(msg)
    label_value = label or ""

    # Always fetch the target node
    target_node = await graph.get_entity_by_uid(uid, label_value)
    if target_node is None:
        return None

    target = _node_to_compact(target_node)

    # Build parallel sub-queries, each paired with the fallback value used if
    # it raises — preserves the previous per-key exception isolation (one
    # sub-query failing doesn't blank out the others).
    coros: dict[str, tuple[Any, Any]] = {}

    if include_hierarchy:
        coros["parent"] = (graph.get_defining_parent(uid), None)
        coros["siblings"] = (graph.get_sibling_entities(uid, max_siblings), [])
        coros["package_ctx"] = (graph.get_package_docstring(uid), None)

    if include_calls:
        coros["callers"] = (graph.get_callers(uid, label_value, call_depth, max_callers * 2), [])
        coros["callees"] = (graph.get_callees(uid, label_value, call_depth, 20), [])

    if include_docs:
        coros["docs"] = (graph.get_linked_docs(uid, label_value, 10), [])

    # Fire all sub-queries in parallel
    keys = list(coros.keys())
    results_list = await asyncio.gather(*(coro for coro, _default in coros.values()), return_exceptions=True)
    results_map: dict[str, Any] = {}
    for key, result in zip(keys, results_list, strict=True):
        if isinstance(result, BaseException):
            logger.warning("Context sub-query '{}' failed: {}", key, result)
            results_map[key] = coros[key][1]
        else:
            results_map[key] = result

    # Extract results
    parent_node = results_map.get("parent")
    parent = _node_to_compact(parent_node) if parent_node is not None else None

    siblings = [_node_to_compact(n) for n in results_map.get("siblings", [])]

    raw_callers = [_node_to_compact(n) for n in results_map.get("callers", [])]
    callers = _prioritize_callers(raw_callers, target.qualified_name)[:max_callers]

    callees = [_node_to_compact(n) for n in results_map.get("callees", [])]
    docs = [
        replace(_node_to_compact(rec["node"]), stale=rec.get("stale"), anchor_hash=rec.get("anchor_hash"))
        for rec in results_map.get("docs", [])
        if rec.get("node") is not None
    ]

    # Package context docstring
    package_context = results_map.get("package_ctx") or ""

    span.set_attribute("callers_count", len(callers))
    span.set_attribute("callees_count", len(callees))

    return ExpandedContext(
        target=target,
        parent=parent,
        siblings=siblings,
        callees=callees,
        callers=callers,
        docs=docs,
        package_context=package_context,
    )
