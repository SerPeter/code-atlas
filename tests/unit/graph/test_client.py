"""Unit tests for pure-function helpers in GraphClient.

No infrastructure required — these test pure functions and data structures.
"""

from __future__ import annotations

from code_atlas.graph.client import (
    _NAME_ROUTED_REL_TYPES,
    _OUT_OF_BAND_REL_TYPES,
    _POST_BATCH_REL_TYPES,
    _UID_ROUTED_REL_TYPES,
    _fuse_bm25_results,
    _sanitize_bm25_query,
    _validate_relationship_routing,
)
from code_atlas.schema import RelType


class TestRelationshipRouting:
    """Every RelType must be routed by exactly one of GraphClient's routing
    mechanisms — the guard against the silent-drop failure class (a new
    RelType added to schema.py but never wired up anywhere)."""

    def test_every_rel_type_is_routed_exactly_once(self):
        groups = [_UID_ROUTED_REL_TYPES, _NAME_ROUTED_REL_TYPES, _POST_BATCH_REL_TYPES, _OUT_OF_BAND_REL_TYPES]
        seen: set[RelType] = set()
        for group in groups:
            overlap = seen & group
            assert not overlap, f"RelTypes routed by more than one mechanism: {overlap}"
            seen |= group
        assert seen == set(RelType), f"RelTypes missing from all routing groups: {set(RelType) - seen}"

    def test_validate_relationship_routing_passes_on_current_schema(self):
        _validate_relationship_routing()  # must not raise

    def test_note_rel_types_are_uid_routed(self):
        assert {RelType.LINKS_TO, RelType.DERIVED_FROM, RelType.SUPERSEDES} <= _UID_ROUTED_REL_TYPES


class TestSanitizeBm25Query:
    """_sanitize_bm25_query neutralizes Tantivy syntax characters that crash
    text_search.search_all (client.py:1690)."""

    def test_leaves_plain_words_untouched(self):
        assert _sanitize_bm25_query("user authentication flow") == "user authentication flow"

    def test_neutralizes_parens_and_brackets(self):
        assert "(" not in _sanitize_bm25_query("embed_batch(texts)")
        assert "[" not in _sanitize_bm25_query("dict[str, Any]")

    def test_neutralizes_colon_and_quote(self):
        sanitized = _sanitize_bm25_query('std::vector "quoted"')
        assert ":" not in sanitized
        assert '"' not in sanitized

    def test_does_not_touch_unaffected_operators(self):
        # These characters did not reproduce the crash empirically and carry
        # meaning in free-text queries (hyphenated/compound words, wildcards).
        assert _sanitize_bm25_query("multi-word") == "multi-word"
        assert _sanitize_bm25_query("embed*") == "embed*"


class TestFuseBm25Results:
    """_fuse_bm25_results replaces cross-index raw-score comparison with
    reciprocal rank fusion (client.py:1704) — BM25 scores are not comparable
    across indices with different corpus statistics."""

    def test_single_index_preserves_rank_order(self):
        index_a = [
            {"node": {"uid": "p:first"}, "score": 9.0},
            {"node": {"uid": "p:second"}, "score": 1.0},
        ]
        fused = _fuse_bm25_results([index_a])
        assert [r["node"]["uid"] for r in fused] == ["p:first", "p:second"]

    def test_raw_score_merge_would_misrank_across_indices(self):
        """The core defect: a weak match in a small/short-doc index (inflated
        BM25 score) must not outrank the TRUE best match in another index's
        own top rank, just because that index's score scale is smaller.
        """
        # Index A (e.g. text_typedef): small corpus, inflated raw scores.
        index_a = [
            {"node": {"uid": "p:TypeA"}, "score": 50.0},
            {"node": {"uid": "p:TypeB"}, "score": 40.0},
        ]
        # Index B (e.g. text_callable): larger corpus, modest raw scores —
        # func_best is genuinely the #1 result in ITS OWN index.
        index_b = [
            {"node": {"uid": "p:func_best"}, "score": 5.0},
            {"node": {"uid": "p:func_other"}, "score": 4.0},
        ]

        fused = _fuse_bm25_results([index_a, index_b])
        uids = [r["node"]["uid"] for r in fused]

        # Raw-score merge would rank TypeA/TypeB (50/40) above func_best (5) —
        # rank fusion instead credits each list's #1 position equally, so
        # func_best (rank 0 in its own index) outranks TypeB (rank 1 in its).
        assert uids.index("p:func_best") < uids.index("p:TypeB")

    def test_dedupes_and_sums_score_for_uid_seen_in_multiple_indices(self):
        index_a = [{"node": {"uid": "p:x"}, "score": 1.0}]
        index_b = [{"node": {"uid": "p:x"}, "score": 1.0}]
        fused = _fuse_bm25_results([index_a, index_b])
        assert len(fused) == 1
        assert fused[0]["score"] == 2 * (1.0 / 61)

    def test_records_without_a_node_are_skipped(self):
        index_a = [{"node": None, "score": 5.0}, {"node": {"uid": "p:ok"}, "score": 1.0}]
        fused = _fuse_bm25_results([index_a])
        assert [r["node"]["uid"] for r in fused] == ["p:ok"]
