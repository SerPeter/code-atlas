"""Search package — hybrid search, embeddings, and query guidance."""

from __future__ import annotations

from code_atlas.search.embeddings import EmbedCache, EmbedClient, EmbeddingError, build_embed_text
from code_atlas.search.engine import (
    AssembledContext,
    CompactNode,
    ContextItem,
    ExpandedContext,
    SearchResult,
    SearchType,
    analyze_query,
    assemble_context,
    expand_context,
    expand_scope,
    hybrid_search,
)
from code_atlas.search.guidance import get_guide, plan_strategy, validate_cypher_explain, validate_cypher_static

__all__ = [
    "AssembledContext",
    "CompactNode",
    "ContextItem",
    "EmbedCache",
    "EmbedClient",
    "EmbeddingError",
    "ExpandedContext",
    "SearchResult",
    "SearchType",
    "analyze_query",
    "assemble_context",
    "build_embed_text",
    "expand_context",
    "expand_scope",
    "get_guide",
    "hybrid_search",
    "plan_strategy",
    "validate_cypher_explain",
    "validate_cypher_static",
]
