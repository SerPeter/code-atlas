"""Unit tests for pure-function helpers in GraphClient.

No infrastructure required — these test pure functions and data structures.
"""

from __future__ import annotations

from code_atlas.graph.client import (
    _NAME_ROUTED_REL_TYPES,
    _OUT_OF_BAND_REL_TYPES,
    _POST_BATCH_REL_TYPES,
    _UID_ROUTED_REL_TYPES,
    _CallLookup,
    _fuse_bm25_results,
    _resolve_one_call,
    _sanitize_bm25_query,
    _validate_relationship_routing,
)
from code_atlas.parsing.ast import ParsedRelationship
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


class TestResolveOneCall:
    """_resolve_one_call (client.py) — CALLS resolution strategies.

    Strategies 4 (project-wide match) and 5 (constructor call) now return every
    candidate instead of only firing when exactly one exists (ADR-0014): the
    caller (resolve_calls) derives confidence from the returned list's length —
    a single candidate is "resolved", more than one is "ambiguous" — rather than
    the resolver silently discarding ambiguous matches.
    """

    PROJECT = "proj"

    def _rel(self, from_uid: str, to_name: str) -> ParsedRelationship:
        return ParsedRelationship(from_qualified_name=from_uid, rel_type=RelType.CALLS, to_name=to_name)

    def test_project_wide_unique_match_resolves(self):
        """Strategy 4, exactly 1 candidate → resolved, strategy=project_unique."""
        lookup = _CallLookup(
            name_to_callables={"helper": [(f"{self.PROJECT}:mod.helper", "mod.py", "public")]},
            import_map={},
            caller_to_parent={},
            parent_children={},
            uid_to_info={f"{self.PROJECT}:mod.helper": ("helper", "mod.py")},
        )
        rel = self._rel(f"{self.PROJECT}:mod.caller", "helper")
        result = _resolve_one_call(self.PROJECT, rel, lookup)
        assert result == ([f"{self.PROJECT}:mod.helper"], "project_unique")

    def test_project_wide_ambiguous_match_returns_all_candidates(self):
        """Strategy 4, >1 candidate → every candidate returned, strategy=project_wide."""
        lookup = _CallLookup(
            name_to_callables={
                "run": [
                    (f"{self.PROJECT}:mod_a.run", "mod_a.py", "public"),
                    (f"{self.PROJECT}:mod_b.run", "mod_b.py", "public"),
                ]
            },
            import_map={},
            caller_to_parent={},
            parent_children={},
            uid_to_info={
                f"{self.PROJECT}:mod_a.run": ("run", "mod_a.py"),
                f"{self.PROJECT}:mod_b.run": ("run", "mod_b.py"),
            },
        )
        rel = self._rel(f"{self.PROJECT}:mod_c.caller", "run")
        result = _resolve_one_call(self.PROJECT, rel, lookup)
        assert result is not None
        candidates, strategy = result
        assert strategy == "project_wide"
        assert set(candidates) == {f"{self.PROJECT}:mod_a.run", f"{self.PROJECT}:mod_b.run"}

    def test_constructor_unique_match_resolves(self):
        """Strategy 5, exactly 1 same-named TypeDef → resolved, strategy=constructor."""
        lookup = _CallLookup(
            name_to_callables={},
            import_map={},
            caller_to_parent={},
            parent_children={f"{self.PROJECT}:mod.Widget": [f"{self.PROJECT}:mod.Widget.__init__"]},
            uid_to_info={f"{self.PROJECT}:mod.Widget.__init__": ("__init__", "mod.py")},
        )
        name_to_typedefs = {"Widget": [(f"{self.PROJECT}:mod.Widget", "mod.py")]}
        rel = self._rel(f"{self.PROJECT}:mod.build", "Widget")
        result = _resolve_one_call(self.PROJECT, rel, lookup, name_to_typedefs)
        assert result == ([f"{self.PROJECT}:mod.Widget.__init__"], "constructor")

    def test_constructor_ambiguous_match_returns_all_init_candidates(self):
        """Strategy 5, >1 same-named TypeDef → every __init__ candidate returned, still tagged constructor."""
        lookup = _CallLookup(
            name_to_callables={},
            import_map={},
            caller_to_parent={},
            parent_children={
                f"{self.PROJECT}:mod_a.Widget": [f"{self.PROJECT}:mod_a.Widget.__init__"],
                f"{self.PROJECT}:mod_b.Widget": [f"{self.PROJECT}:mod_b.Widget.__init__"],
            },
            uid_to_info={
                f"{self.PROJECT}:mod_a.Widget.__init__": ("__init__", "mod_a.py"),
                f"{self.PROJECT}:mod_b.Widget.__init__": ("__init__", "mod_b.py"),
            },
        )
        name_to_typedefs = {
            "Widget": [
                (f"{self.PROJECT}:mod_a.Widget", "mod_a.py"),
                (f"{self.PROJECT}:mod_b.Widget", "mod_b.py"),
            ]
        }
        rel = self._rel(f"{self.PROJECT}:mod_c.build", "Widget")
        result = _resolve_one_call(self.PROJECT, rel, lookup, name_to_typedefs)
        assert result is not None
        candidates, strategy = result
        assert strategy == "constructor"
        assert set(candidates) == {
            f"{self.PROJECT}:mod_a.Widget.__init__",
            f"{self.PROJECT}:mod_b.Widget.__init__",
        }

    def test_constructor_candidates_without_init_are_excluded(self):
        """A same-named TypeDef with no __init__ child contributes no candidate."""
        lookup = _CallLookup(
            name_to_callables={},
            import_map={},
            caller_to_parent={},
            parent_children={f"{self.PROJECT}:mod.Widget": []},
            uid_to_info={},
        )
        name_to_typedefs = {"Widget": [(f"{self.PROJECT}:mod.Widget", "mod.py")]}
        rel = self._rel(f"{self.PROJECT}:mod.build", "Widget")
        result = _resolve_one_call(self.PROJECT, rel, lookup, name_to_typedefs)
        assert result is None

    def test_no_match_returns_none(self):
        lookup = _CallLookup(
            name_to_callables={},
            import_map={},
            caller_to_parent={},
            parent_children={},
            uid_to_info={},
        )
        rel = self._rel(f"{self.PROJECT}:mod.func", "print")
        result = _resolve_one_call(self.PROJECT, rel, lookup, {})
        assert result is None
