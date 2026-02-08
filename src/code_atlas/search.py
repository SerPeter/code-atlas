"""Hybrid search — RRF fusion across graph, vector, and BM25 channels.

Consumed by both the MCP server and the CLI ``atlas search`` command.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from loguru import logger

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
    if _IDENTIFIER_RE.match(stripped) or _SNAKE_RE.match(stripped) or _DOTTED_RE.search(stripped) or len(words) <= 2:
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

    # Fire search channels in parallel
    tasks: dict[str, asyncio.Task[list[dict[str, Any]]]] = {}
    fetch_limit = limit * 3  # over-fetch for fusion quality

    if SearchType.GRAPH in search_types:
        tasks["graph"] = asyncio.create_task(graph.graph_search(query, limit=fetch_limit, project=scope))
    if SearchType.VECTOR in search_types and vector is not None:
        tasks["vector"] = asyncio.create_task(graph.vector_search(vector, limit=fetch_limit, project=scope))
    if SearchType.BM25 in search_types:
        tasks["bm25"] = asyncio.create_task(graph.text_search(query, limit=fetch_limit, project=scope))

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

    # Build SearchResult objects
    return [
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
        )
        for uid, rrf_score in list(fused_scores.items())[:limit]
    ]
