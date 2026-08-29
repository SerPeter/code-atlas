"""Unit tests for pure-function helpers in GraphClient.

No infrastructure required — these test pure functions and data structures.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import sys
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tenacity import wait_none

from code_atlas.graph.client import (
    _CODE_ENTITY_KINDS,
    _INFERRED_IMPLEMENTS_WEIGHT,
    _NAME_ROUTED_REL_TYPES,
    _OUT_OF_BAND_REL_TYPES,
    _POST_BATCH_REL_TYPES,
    _TYPE_REF_FACTS,
    _TYPE_REF_RANK,
    _UID_ROUTED_REL_TYPES,
    GraphClient,
    QueryTimeoutError,
    _active_tx_var,
    _call_edge_weight,
    _CallEdgeFacts,
    _CallLookup,
    _combine_call_edge_facts,
    _direct_call_lines,
    _format_path_hops,
    _fuse_bm25_results,
    _resolve_one_call,
    _sanitize_bm25_query,
    _test_callable_uids,
    _validate_relationship_routing,
)
from code_atlas.parsing.ast import ParsedRelationship
from code_atlas.parsing.languages.config import _MODULE_KINDS as _CONFIG_MODULE_KINDS
from code_atlas.parsing.languages.containerfile import _STAGE_KIND
from code_atlas.parsing.languages.hcl import _BLOCK_SPECS
from code_atlas.parsing.languages.hcl import _KINDS as _HCL_FILE_KINDS
from code_atlas.parsing.languages.sql import _KIND_COLUMN, _KIND_INDEX, _KIND_TABLE, _KIND_VIEW
from code_atlas.schema import CallableKind, RelType, TypeDefKind
from code_atlas.settings import AtlasSettings


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


class TestCodeEntityKinds:
    """``_CODE_ENTITY_KINDS`` separates invocable code from config/infra
    declarations for the dead-code analysis. It has to stay *derived* from the
    schema enums: a hand-maintained list would need editing per new language,
    which is the drift trap `_DEFAULT_INCLUDE` already demonstrates."""

    def test_is_exactly_the_union_of_the_schema_kind_enums(self):
        assert frozenset(CallableKind) | frozenset(TypeDefKind) == _CODE_ENTITY_KINDS

    def test_excludes_every_terraform_block_kind(self):
        """Sourced from the parser's own table, so a new Terraform block type is
        covered without touching this test."""
        terraform_kinds = {spec[0] for spec in _BLOCK_SPECS.values()} | set(_HCL_FILE_KINDS.values())
        assert not (terraform_kinds & _CODE_ENTITY_KINDS)

    def test_excludes_config_and_infra_declaration_kinds(self):
        infra_kinds = {
            _STAGE_KIND,
            _KIND_TABLE,
            _KIND_VIEW,
            _KIND_COLUMN,
            _KIND_INDEX,
            "sql_function",
            "terraform_local",
            "k8s_resource",
            "compose_service",
            "ci_job",
            "ansible_play",
            "ansible_task",
            "ansible_handler",
            "xml_element",
            "xml_setting",
        } | set(_CONFIG_MODULE_KINDS.values())
        assert not (infra_kinds & _CODE_ENTITY_KINDS)

    def test_includes_the_shell_parsers_real_functions(self):
        """Shell is the one new-parser language that emits genuinely invocable
        entities (and same-file CALLS edges) — it must not be filtered out."""
        assert CallableKind.FUNCTION in _CODE_ENTITY_KINDS


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


class TestCallSiteProvenance:
    """A CALLS edge records where the CALL happens, not where the caller is defined
    (ATL-105). In any caller longer than a few statements those are different lines, and
    the caller's `def` is the less useful of the two.
    """

    def test_the_first_call_site_wins_and_the_count_accumulates(self):
        """`line` deliberately does not follow the best-evidenced observation — "which
        evidence is strongest" and "where does this first appear" are different questions.
        """
        first = _CallEdgeFacts("ambiguous", "project_wide", 3, False, line=90, site_count=1)
        second = _CallEdgeFacts("resolved", "same_file", 1, False, line=12, site_count=1)

        combined = _combine_call_edge_facts(first, second)

        # Strategy/confidence follow the BEST evidence (candidate_count 1)...
        assert (combined.confidence, combined.strategy, combined.candidate_count) == ("resolved", "same_file", 1)
        # ...while the line is the EARLIEST site, regardless of which one won above.
        assert combined.line == 12
        assert combined.site_count == 2

    def test_a_missing_line_never_masks_a_known_one(self):
        """Languages other than Python do not record a line yet, so None must be absorbed
        rather than compared — min() over a None would raise, and defaulting it to 0 would
        beat every real line."""
        known = _CallEdgeFacts("resolved", "import", 1, False, line=7, site_count=1)
        unknown = _CallEdgeFacts("resolved", "import", 1, False, line=None, site_count=1)

        assert _combine_call_edge_facts(known, unknown).line == 7
        assert _combine_call_edge_facts(unknown, known).line == 7
        assert _combine_call_edge_facts(unknown, unknown).line is None

    def test_call_lines_are_reported_only_for_direct_dependents(self):
        """At depth 1 the incident edge starts at the affected entity, so its lines are
        that entity's own. Deeper, the same edge belongs to an intermediate hop and its
        lines name a DIFFERENT file — misleading rather than merely imprecise."""
        assert _direct_call_lines({"min_depth": 1, "via_lines": [42, 17, 42]}) == {"at_lines": [17, 42]}
        assert _direct_call_lines({"min_depth": 2, "via_lines": [42]}) == {}
        # Absent, not null: an absent key reads as "not applicable", null as "we looked".
        assert _direct_call_lines({"min_depth": 1, "via_lines": [None]}) == {}
        assert _direct_call_lines({"min_depth": 1}) == {}


class TestTestCandidateHygiene:
    """Production code cannot depend on test code, so a non-test call site must not
    resolve onto a test definition (ATL-103).

    The filter is deliberately ASYMMETRIC. "Production does not depend on tests" is an
    architectural invariant; "tests do not call production" is the opposite of true, so a
    test caller filters nothing. Graphify (round-3 competitor read) applies a symmetric
    preference; that half is not grounded in an invariant and is not copied.

    The damage this prevents is not only a wrong edge. `candidate_count` is the surviving
    list's length and `weight` is 1/candidate_count, so one same-named fixture also halves
    the weight of the RIGHT edge — which is what reaches Leiden and blast_radius ranking.
    """

    PROJECT = "proj"
    PROD_UID = "proj:mod.helper"
    TEST_UID = "proj:tests.test_mod.helper"

    def _rel(self, from_uid: str, to_name: str) -> ParsedRelationship:
        return ParsedRelationship(from_qualified_name=from_uid, rel_type=RelType.CALLS, to_name=to_name)

    def _lookup(self) -> _CallLookup:
        return _CallLookup(
            name_to_callables={
                "helper": [
                    (self.PROD_UID, "src/mod.py", "public"),
                    (self.TEST_UID, "tests/test_mod.py", "public"),
                ]
            },
            import_map={},
            caller_to_parent={},
            parent_children={},
            uid_to_info={
                self.PROD_UID: ("helper", "src/mod.py"),
                self.TEST_UID: ("helper", "tests/test_mod.py"),
            },
        )

    def test_a_production_call_site_ignores_a_same_named_test_definition(self):
        """The real edge resolves alone — so candidate_count is 1 and weight stays 1.0."""
        lookup = self._lookup()
        test_callables = _test_callable_uids(lookup, ["tests/", "test_*.py"])
        assert test_callables == frozenset({self.TEST_UID})

        rel = self._rel("proj:src.other.caller", "helper")
        result = _resolve_one_call(self.PROJECT, rel, lookup, None, test_callables)
        assert result == ([self.PROD_UID], "project_unique")
        # Without the filter this is the regression: two candidates, so the production
        # edge is tagged ambiguous and its weight halved.
        assert _resolve_one_call(self.PROJECT, rel, lookup, None, frozenset()) == (
            [self.PROD_UID, self.TEST_UID],
            "project_wide",
        )

    def test_a_test_call_site_is_not_filtered_at_all(self):
        """Calling production code is what a test is FOR — the invariant only runs one way.

        The caller is a distinct uid from either candidate AND lives in a different test
        file, so neither ``non_self`` nor the same-file rung can account for the result:
        both definitions survive precisely because no filter ran.

        (Co-locate the caller with the test definition instead and Strategy 3 resolves it
        to that definition on its own — which is why the symmetric "prefer test candidates
        for test callers" half of Graphify's rule buys nothing here.)
        """
        caller_uid = "proj:tests.test_other.test_something"
        lookup = self._lookup()
        lookup.uid_to_info[caller_uid] = ("test_something", "tests/test_other.py")
        test_callables = _test_callable_uids(lookup, ["tests/", "test_*.py"])
        assert caller_uid in test_callables

        rel = self._rel(caller_uid, "helper")
        result = _resolve_one_call(self.PROJECT, rel, lookup, None, test_callables)
        assert result == ([self.PROD_UID, self.TEST_UID], "project_wide")

    def test_an_all_test_candidate_set_still_produces_an_edge(self):
        """Falling back beats dropping: a diluted edge outranks a silent absence."""
        lookup = _CallLookup(
            name_to_callables={"fixture_only": [(self.TEST_UID, "tests/test_mod.py", "public")]},
            import_map={},
            caller_to_parent={},
            parent_children={},
            uid_to_info={self.TEST_UID: ("fixture_only", "tests/test_mod.py")},
        )
        test_callables = _test_callable_uids(lookup, ["tests/", "test_*.py"])
        assert test_callables == frozenset({self.TEST_UID})

        rel = self._rel("proj:src.other.caller", "fixture_only")
        result = _resolve_one_call(self.PROJECT, rel, lookup, None, test_callables)
        assert result == ([self.TEST_UID], "project_unique")

    def test_the_filter_follows_the_configured_patterns_not_the_defaults(self):
        """A project that configures its own test_patterns filters by the same rule that
        decides from_test — which is why the uid set is derived from the effective list."""
        lookup = _CallLookup(
            name_to_callables={"helper": [("proj:spec.mod.helper", "spec/mod.py", "public")]},
            import_map={},
            caller_to_parent={},
            parent_children={},
            uid_to_info={"proj:spec.mod.helper": ("helper", "spec/mod.py")},
        )
        assert _test_callable_uids(lookup, ["tests/"]) == frozenset()
        assert _test_callable_uids(lookup, ["spec/"]) == frozenset({"proj:spec.mod.helper"})


class TestNonCallEdgeQuality:
    """Weight was scoped to CALLS because only that resolver had a candidate set.

    That reasoning never covered USES_TYPE strategy 3 (project-wide *uniqueness* —
    the shape ADR-0022 demoted for calls) or an inferred IMPLEMENTS (derived from
    method-set containment, declared nowhere). Both were written indistinguishable
    from a structural fact, so anything scoring a path read them as certainties.
    """

    def test_a_guessed_type_use_is_not_worth_an_exact_import_match(self):
        import_conf, import_w = _TYPE_REF_FACTS["import"]
        guess_conf, guess_w = _TYPE_REF_FACTS["project_unique"]

        assert import_conf == "resolved"
        assert guess_conf == "ambiguous", "project-wide uniqueness is a guess, not a resolution"
        assert guess_w < import_w, "a guessed type-use outranking an import match is ADR-0022's failure"

    def test_a_same_file_type_use_is_as_good_as_an_import(self):
        """Both are lexically grounded — the name was looked up in a namespace that
        actually contains it, which is exactly ADR-0022's test."""
        assert _TYPE_REF_FACTS["same_file"] == _TYPE_REF_FACTS["import"]

    def test_every_strategy_is_ranked_and_scored(self):
        """A rung with no entry would KeyError at write time, on a path only some
        codebases reach — so pin the two tables against each other instead."""
        assert set(_TYPE_REF_RANK) == set(_TYPE_REF_FACTS)

    def test_rank_runs_strongest_first(self):
        weights = [_TYPE_REF_FACTS[st][1] for st in _TYPE_REF_RANK]
        assert weights == sorted(weights, reverse=True)

    def test_an_inferred_conformance_edge_is_damped(self):
        """ADR-0025 derives IMPLEMENTS from method-set containment. It is the best
        evidence available and may not claim to be a fact."""
        assert _call_edge_weight(1, from_test=False) > _INFERRED_IMPLEMENTS_WEIGHT


class TestCallEdgeWeight:
    """_call_edge_weight (client.py) — the numeric weight amending ADR-0014.

    ADR-0014 rejected a float confidence as premature; the amendment stores the
    raw facts (candidate_count, from_test) and derives the scalar here, so the
    derivation is the only thing that needs retuning.
    """

    def test_single_candidate_production_call_is_the_base_weight(self):
        assert _call_edge_weight(1, from_test=False) == 1.0

    def test_evidence_is_split_across_ambiguous_candidates(self):
        assert _call_edge_weight(4, from_test=False) == 0.25
        assert _call_edge_weight(2, from_test=False) > _call_edge_weight(3, from_test=False)

    def test_test_caller_ranks_below_the_same_call_from_production(self):
        assert _call_edge_weight(1, from_test=True) < _call_edge_weight(1, from_test=False)
        assert _call_edge_weight(3, from_test=True) < _call_edge_weight(3, from_test=False)

    def test_weight_is_always_strictly_positive(self):
        """MAGE's Leiden divides gamma by the sum of edge weights, so a zero total
        produces NaN and silently meaningless communities — the floor is load-bearing."""
        for count in (0, 1, 2, 100, 1_000_000, 10_000_000):
            for from_test in (False, True):
                assert _call_edge_weight(count, from_test=from_test) > 0.0


class TestCombineCallEdgeFacts:
    """_combine_call_edge_facts (client.py) — N call sites collapse to one edge.

    Replaces the previous last-write-wins assignment, whose stored confidence
    depended on parse order.
    """

    RESOLVED = _CallEdgeFacts("resolved", "same_file", 1, False)
    AMBIGUOUS = _CallEdgeFacts("ambiguous", "project_wide", 3, False)

    def test_best_evidenced_observation_wins_in_either_order(self):
        forward = _combine_call_edge_facts(self.RESOLVED, self.AMBIGUOUS)
        backward = _combine_call_edge_facts(self.AMBIGUOUS, self.RESOLVED)

        assert forward == backward
        assert forward.confidence == "resolved"
        assert forward.strategy == "same_file"
        assert forward.candidate_count == 1

    def test_equal_evidence_keeps_the_first_observation(self):
        first = _CallEdgeFacts("resolved", "import", 1, False)
        second = _CallEdgeFacts("resolved", "sibling", 1, False)

        assert _combine_call_edge_facts(first, second).strategy == "import"

    def test_edge_is_from_test_only_when_every_call_site_was(self):
        test_site = _CallEdgeFacts("resolved", "import", 1, True)
        prod_site = _CallEdgeFacts("resolved", "import", 1, False)

        assert _combine_call_edge_facts(test_site, test_site).from_test is True
        assert _combine_call_edge_facts(test_site, prod_site).from_test is False
        assert _combine_call_edge_facts(prod_site, test_site).from_test is False

    def test_from_test_is_combined_independently_of_the_evidence_comparison(self):
        """The losing observation still contributes its from_test — one production
        caller makes the edge production-relevant regardless of which site had
        the better-evidenced resolution."""
        best = _CallEdgeFacts("resolved", "import", 1, True)
        worse = _CallEdgeFacts("ambiguous", "project_wide", 5, True)

        combined = _combine_call_edge_facts(worse, best)

        assert combined.candidate_count == 1
        assert combined.from_test is True


class TestConstructorInjection:
    """GraphClient accepts a pre-built driver, bypassing AsyncGraphDatabase.driver()."""

    def test_injected_driver_is_used_directly(self, tmp_path: Path):
        settings = AtlasSettings(project_root=tmp_path)
        fake_driver = MagicMock()

        with patch("code_atlas.graph.client.AsyncGraphDatabase.driver") as mock_driver_factory:
            client = GraphClient(settings, driver=fake_driver)

        assert client._driver is fake_driver
        mock_driver_factory.assert_not_called()

    def test_no_injected_driver_falls_back_to_settings_based_construction(self, tmp_path: Path):
        settings = AtlasSettings(project_root=tmp_path)
        sentinel_driver = MagicMock()

        with patch(
            "code_atlas.graph.client.AsyncGraphDatabase.driver", return_value=sentinel_driver
        ) as mock_driver_factory:
            client = GraphClient(settings)

        mock_driver_factory.assert_called_once()
        assert client._driver is sentinel_driver


class _FakeRel(dict):
    """Minimal stand-in for a neo4j Relationship object: dict props + .type."""

    def __init__(self, type_: str, **props: object) -> None:
        super().__init__(**props)
        self.type = type_


class TestFormatPathHops:
    """_format_path_hops (client.py) — renders a Cypher path into per-hop dicts for trace_path_between."""

    def test_includes_confidence_and_strategy(self):
        nodes = [{"uid": "p:a", "name": "a"}, {"uid": "p:b", "name": "b"}]
        rels = [_FakeRel("CALLS", confidence="resolved", strategy="import")]

        hops = _format_path_hops(nodes, rels)

        assert hops == [
            {
                "from": {"uid": "p:a", "name": "a"},
                "to": {"uid": "p:b", "name": "b"},
                "edge_type": "CALLS",
                "confidence": "resolved",
                "strategy": "import",
            }
        ]

    def test_includes_weight_and_from_test(self):
        """The weighting amendment to ADR-0014 — weight explains an equal-hop tie-break,
        from_test shows the hop runs through a test caller."""
        nodes = [{"uid": "p:a", "name": "a"}, {"uid": "p:b", "name": "b"}]
        rels = [_FakeRel("CALLS", confidence="resolved", strategy="import", weight=0.25, from_test=True)]

        hops = _format_path_hops(nodes, rels)

        assert hops[0]["weight"] == 0.25
        assert hops[0]["from_test"] is True

    def test_omits_confidence_when_absent(self):
        """A non-CALLS edge (e.g. IMPORTS) has no confidence/strategy property to surface."""
        nodes = [{"uid": "p:a", "name": "a"}, {"uid": "p:b", "name": "b"}]
        rels = [_FakeRel("IMPORTS")]

        hops = _format_path_hops(nodes, rels)

        assert "confidence" not in hops[0]
        assert "strategy" not in hops[0]
        assert "weight" not in hops[0]
        assert "from_test" not in hops[0]
        assert hops[0]["edge_type"] == "IMPORTS"

    def test_multi_hop(self):
        nodes = [{"uid": "p:a", "name": "a"}, {"uid": "p:b", "name": "b"}, {"uid": "p:c", "name": "c"}]
        rels = [_FakeRel("CALLS"), _FakeRel("CALLS")]

        hops = _format_path_hops(nodes, rels)

        assert len(hops) == 2
        assert hops[0]["to"]["uid"] == "p:b"
        assert hops[1]["from"]["uid"] == "p:b"


def _client_with_fake_execute(tmp_path: Path) -> Any:
    """A GraphClient with a fake driver (never touched) and ``.execute`` overridden
    as an AsyncMock — for asserting the Cypher text/params a query-construction
    method produces, without a real Memgraph connection.

    Returns ``Any`` (not ``GraphClient``) deliberately: ``.execute`` is
    monkey-patched to an ``AsyncMock``, which the real ``GraphClient.execute``
    signature doesn't structurally support — callers use ``.execute.side_effect``/
    ``.call_args_list`` the same way tests elsewhere mock a ``MagicMock()`` stand-in.
    """
    settings = AtlasSettings(project_root=tmp_path)
    client = GraphClient(settings, driver=MagicMock())
    client.execute = AsyncMock()  # type: ignore[invalid-assignment]  # query-text capture, not a real DB call
    return client


class TestAnalysisQueryConstruction:
    """Query-text regression coverage for the analysis/diagram methods (graph/protocol.py's
    GraphBackend) — moved here from tests/unit/server/test_analysis.py now that query
    construction lives in GraphClient rather than server/analysis.py.
    """

    async def test_dependency_external_counts_query_includes_path_scope(self, tmp_path: Path):
        """external_imports must be scoped like internal_imports, not report whole-project counts."""
        client = _client_with_fake_execute(tmp_path)
        client.execute.side_effect = [[], []]

        await client.get_dependency_external_counts("code-atlas", "src/foo")

        ext_pkg_query = client.execute.call_args_list[0][0][0]
        ext_sym_query = client.execute.call_args_list[1][0][0]
        assert "$path" in ext_pkg_query
        assert "$path" in ext_sym_query

    async def test_structure_overview_external_deps_query_includes_path_scope(self, tmp_path: Path):
        """_analyze_structure has the same inconsistency as external_imports: fix both."""
        client = _client_with_fake_execute(tmp_path)
        client.execute.side_effect = [[], [], [], []]

        await client.get_structure_overview("code-atlas", "src/foo", 20)

        ext_query = client.execute.call_args_list[3][0][0]
        assert "$path" in ext_query

    async def test_dead_code_query_excludes_calls_and_dunders(self, tmp_path: Path):
        client = _client_with_fake_execute(tmp_path)
        client.execute.return_value = []

        await client.get_dead_code_candidates("code-atlas", "")

        query = client.execute.call_args[0][0]
        # "Unused" is not "no CALLS edge": a class is used by being annotated, subclassed
        # or imported, and constructing it calls its __init__ rather than the class. The
        # CALLS-only test reported 29 live entities dead out of 30 in one package.
        assert "USES_TYPE" in query
        assert "INHERITS" in query
        assert "CALLS" in query
        assert "DEFINES" in query  # a call into a member keeps its owner alive
        assert "NOT n.name STARTS WITH '__'" in query

    async def test_dead_code_query_gates_on_invocable_kinds(self, tmp_path: Path):
        """Config/infra declarations share the Callable/TypeDef labels with real code
        but can never receive a CALLS edge — without the kind gate every Terraform
        resource and k8s object is reported as dead."""
        client = _client_with_fake_execute(tmp_path)
        client.execute.return_value = []

        await client.get_dead_code_candidates("code-atlas", "")

        query, params = client.execute.call_args[0]
        assert "n.kind IN $code_kinds" in query
        assert set(params["code_kinds"]) == _CODE_ENTITY_KINDS

    async def test_complexity_hotspots_query_computes_and_sorts_loc_span(self, tmp_path: Path):
        client = _client_with_fake_execute(tmp_path)
        client.execute.return_value = []

        await client.get_complexity_hotspots("code-atlas", "", 20)

        query = client.execute.call_args[0][0]
        assert "line_end - n.line_start" in query
        assert "ORDER BY loc_span DESC" in query

    async def test_git_signals_queries_target_the_right_properties(self, tmp_path: Path):
        client = _client_with_fake_execute(tmp_path)
        client.execute.side_effect = [[], [], []]

        await client.get_git_signals_data("code-atlas", "src/foo", 20, 1)

        hotspot_query, hotspot_params = client.execute.call_args_list[0][0]
        assert "git_commit_count" in hotspot_query
        assert "$path" in hotspot_query
        assert hotspot_params["path"] == "src/foo"
        bus_factor_query = client.execute.call_args_list[1][0][0]
        assert "git_author_count" in bus_factor_query
        co_change_query = client.execute.call_args_list[2][0][0]
        assert "CO_CHANGES_WITH" in co_change_query
        assert "$path" in co_change_query

    async def test_module_detail_methods_and_inherits_queries_are_bounded(self, tmp_path: Path):
        """Both the methods and inheritance queries must be bounded by max_nodes,
        not just the top-level entities query, so a module with large classes
        can't blow past the requested output size.
        """
        client = _client_with_fake_execute(tmp_path)
        client.execute.side_effect = [
            [{"name": "mod", "qn": "pkg.mod", "uid": "proj:pkg.mod"}],
            [],
            [],
            [],
        ]

        await client.get_diagram_module_detail("code-atlas", "pkg/mod", 5)

        methods_query = client.execute.call_args_list[2][0][0]
        inherits_query = client.execute.call_args_list[3][0][0]
        assert "LIMIT" in methods_query.upper()
        assert "LIMIT" in inherits_query.upper()


class TestResolveCallsEdgeProperties:
    """resolve_calls persists the ADR-0014 amendment's raw facts plus the derived weight.

    Asserts the Cypher SET clause and the parameter payload rather than a live
    graph — a missing property here is invisible downstream (Memgraph is
    schemaless for edge properties and MAGE reads an absent weight as 1.0
    without erroring), so the write payload is the thing worth pinning.
    """

    PROJECT = "proj"
    CALLER = "proj:mod.caller"

    def _lookup(self, caller_fp: str, targets: list[tuple[str, str, str]]) -> _CallLookup:
        info = {self.CALLER: ("caller", caller_fp)}
        for uid, fp, _vis in targets:
            info[uid] = ("helper", fp)
        return _CallLookup(
            name_to_callables={"helper": targets},
            import_map={},
            caller_to_parent={},
            parent_children={},
            uid_to_info=info,
        )

    async def _write_call(
        self,
        tmp_path: Path,
        caller_fp: str,
        targets: list[tuple[str, str, str]],
        *,
        test_patterns: tuple[str, ...] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        client = _client_with_fake_execute(tmp_path)
        client.execute_write = AsyncMock()
        rel = ParsedRelationship(from_qualified_name=self.CALLER, rel_type=RelType.CALLS, to_name="helper")

        await client.resolve_calls(
            self.PROJECT,
            [rel],
            lookup=self._lookup(caller_fp, targets),
            name_to_typedefs={},
            test_patterns=test_patterns,
        )

        query, params = client.execute_write.call_args[0]
        return query, params["rels"]

    async def test_resolved_production_call_writes_full_weight_and_raw_facts(self, tmp_path: Path):
        query, rels = await self._write_call(tmp_path, "src/mod.py", [("proj:other.helper", "other.py", "public")])

        assert "e.candidate_count = r.candidate_count" in query
        assert "e.from_test = r.from_test" in query
        assert "e.weight = r.weight" in query
        assert "e.line = r.line" in query
        assert "e.site_count = r.site_count" in query
        assert rels == [
            {
                "f": "proj:mod.caller",
                "t": "proj:other.helper",
                "confidence": "resolved",
                "strategy": "project_unique",
                "candidate_count": 1,
                "from_test": False,
                "weight": 1.0,
                # This fixture builds relationships directly rather than parsing source,
                # so there is no call site to record — None is the honest value, and the
                # parser-fed path is covered by the integration corpus.
                "line": None,
                "site_count": 1,
            }
        ]

    async def test_ambiguous_call_records_the_count_and_splits_the_weight(self, tmp_path: Path):
        targets = [("proj:a.helper", "a.py", "public"), ("proj:b.helper", "b.py", "public")]

        _query, rels = await self._write_call(tmp_path, "src/mod.py", targets)

        assert {r["t"] for r in rels} == {"proj:a.helper", "proj:b.helper"}
        assert all(r["confidence"] == "ambiguous" for r in rels)
        assert all(r["candidate_count"] == 2 for r in rels)
        assert all(r["weight"] == 0.5 for r in rels)

    async def test_caller_in_a_test_directory_is_flagged_and_damped(self, tmp_path: Path):
        _query, rels = await self._write_call(
            tmp_path, "tests/unit/check_mod.py", [("proj:other.helper", "other.py", "public")]
        )

        assert rels[0]["from_test"] is True
        assert rels[0]["weight"] < 1.0

    async def test_test_patterns_override_the_search_settings_default(self, tmp_path: Path):
        """The graph layer must not hardcode test policy — a project configuring
        its own patterns gets its own from_test verdict."""
        targets = [("proj:other.helper", "other.py", "public")]

        _q1, default_rels = await self._write_call(tmp_path, "spec/mod_spec.py", targets)
        _q2, custom_rels = await self._write_call(tmp_path, "spec/mod_spec.py", targets, test_patterns=("*_spec.py",))

        assert default_rels[0]["from_test"] is False
        assert custom_rels[0]["from_test"] is True

    async def test_unknown_caller_defaults_to_non_test(self, tmp_path: Path):
        """uid_to_info is Callable-scoped; a caller absent from it has no path to
        match, and guessing "test" there would damp real production edges."""
        client = _client_with_fake_execute(tmp_path)
        client.execute_write = AsyncMock()
        lookup = _CallLookup(
            name_to_callables={"helper": [("proj:other.helper", "other.py", "public")]},
            import_map={},
            caller_to_parent={},
            parent_children={},
            uid_to_info={"proj:other.helper": ("helper", "other.py")},
        )
        rel = ParsedRelationship(from_qualified_name=self.CALLER, rel_type=RelType.CALLS, to_name="helper")

        await client.resolve_calls(self.PROJECT, [rel], lookup=lookup, name_to_typedefs={})

        assert client.execute_write.call_args[0][1]["rels"][0]["from_test"] is False


class TestWeightAwareTraversalQueries:
    """trace_path_between / compute_blast_radius Cypher — the weight-aware parts."""

    async def test_trace_path_breaks_equal_hop_ties_by_path_weight(self, tmp_path: Path):
        client = _client_with_fake_execute(tmp_path)
        client.execute.side_effect = [[{"from_exists": True, "to_exists": True}], []]

        await client.trace_path_between("p:a", "p:b", 4, ("CALLS", "IMPORTS"))

        query = client.execute.call_args_list[1][0][0]
        assert "coalesce(r.weight, 1.0)" in query
        assert "AS path_weight" in query
        assert "ORDER BY hops, path_weight DESC" in query

    async def test_trace_path_reports_no_path_weight_when_no_path_exists(self, tmp_path: Path):
        client = _client_with_fake_execute(tmp_path)
        client.execute.side_effect = [[{"from_exists": True, "to_exists": True}], []]

        result = await client.trace_path_between("p:a", "p:b", 4, ("CALLS",))

        assert result["found"] is False
        assert result["path_weight"] is None

    async def test_blast_radius_scores_best_path_and_flags_test_only_reachability(self, tmp_path: Path):
        client = _client_with_fake_execute(tmp_path)
        client.execute.side_effect = [
            [
                {
                    "uid": "p:x",
                    "name": "x",
                    "qn": "m.x",
                    "label": "Callable",
                    "file_path": "m.py",
                    "min_depth": 1,
                    "confidence_score": 0.25,
                }
            ],
            [],
            [],
        ]

        results = await client.compute_blast_radius("p:a", "in", ("CALLS",), 3)

        all_query, resolved_query, production_query = (c[0][0] for c in client.execute.call_args_list)
        assert "max(reduce(w = 1.0" in all_query
        assert "coalesce(r.weight, 1.0)" in all_query
        assert "AS confidence_score" in all_query
        # coalesce, not a bare equality: an absent confidence means STRUCTURAL (ADR-0028),
        # and since blast_radius widened past CALLS most hops legitimately carry none. A
        # bare `r.confidence = 'resolved'` marked every one of them ambiguous.
        assert "coalesce(r.confidence, 'resolved') = 'resolved'" in resolved_query
        assert "NOT coalesce(r.from_test, false)" in production_query
        assert results[0]["confidence_score"] == 0.25
        assert results[0]["ambiguous_only"] is True
        assert results[0]["test_only"] is True

    async def test_blast_radius_clears_the_flags_when_a_clean_path_exists(self, tmp_path: Path):
        client = _client_with_fake_execute(tmp_path)
        client.execute.side_effect = [
            [
                {
                    "uid": "p:x",
                    "name": "x",
                    "qn": "m.x",
                    "label": "Callable",
                    "file_path": "m.py",
                    "min_depth": 1,
                    "confidence_score": 1.0,
                }
            ],
            [{"uid": "p:x"}],
            [{"uid": "p:x"}],
        ]

        results = await client.compute_blast_radius("p:a", "in", ("CALLS",), 3)

        assert results[0]["ambiguous_only"] is False
        assert results[0]["test_only"] is False
        assert results[0]["confidence_score"] == 1.0


class TestCrossLanguageCandidateHygiene:
    """A call resolves only within its own call-namespace group (ATL-113, ADR-0030 axis 2).

    The pool was keyed by bare name alone, so a TypeScript ``render`` and a Python
    ``render`` competed for the same call. The quiet damage is not the extra candidate:
    where only one language's definition exists, ``project_unique`` fires and reports the
    cross-language match as confidence "resolved" — a confidently wrong edge rather than
    an ambiguous one.

    Unlike the test filter, this one is STRICT. Falling back to the unfiltered pool is
    right when every definition lives in test code (a production→test call is unusual,
    not impossible); it is wrong across languages, because a Python function cannot be
    reached from a TypeScript call site under any reading.
    """

    PROJECT = "proj"
    PY_UID = "proj:app.views.render"
    TS_UID = "proj:web.ui.render"
    TS_CALLER = "proj:web.page.mount"
    PY_CALLER = "proj:app.main.boot"

    def _rel(self, from_uid: str, to_name: str) -> ParsedRelationship:
        return ParsedRelationship(from_qualified_name=from_uid, rel_type=RelType.CALLS, to_name=to_name)

    def _lookup(self, *, with_ts: bool = True) -> _CallLookup:
        callables = [(self.PY_UID, "app/views.py", "public")]
        if with_ts:
            callables.append((self.TS_UID, "web/ui.ts", "public"))
        info = {
            self.PY_UID: ("render", "app/views.py"),
            self.TS_CALLER: ("mount", "web/page.ts"),
            self.PY_CALLER: ("boot", "app/main.py"),
        }
        if with_ts:
            info[self.TS_UID] = ("render", "web/ui.ts")
        return _CallLookup(
            name_to_callables={"render": callables},
            import_map={},
            caller_to_parent={},
            parent_children={},
            uid_to_info=info,
        )

    def test_a_typescript_call_picks_the_typescript_definition(self):
        """Both definitions exist; only the same-group one survives, so weight stays 1.0."""
        lookup = self._lookup()
        result = _resolve_one_call(self.PROJECT, self._rel(self.TS_CALLER, "render"), lookup)
        assert result == ([self.TS_UID], "project_unique")

    def test_a_python_call_picks_the_python_definition(self):
        lookup = self._lookup()
        result = _resolve_one_call(self.PROJECT, self._rel(self.PY_CALLER, "render"), lookup)
        assert result == ([self.PY_UID], "project_unique")

    def test_a_cross_language_only_match_resolves_to_nothing(self):
        """The load-bearing case, and it needs a NEGATIVE assertion.

        Only a Python ``render`` exists and a TypeScript file calls ``render``. Before
        this filter the call resolved to the Python definition as ``project_unique`` —
        strategy name "unique", confidence "resolved". A wrong edge wearing full
        confidence is worse than no edge, so the correct answer is None.
        """
        lookup = self._lookup(with_ts=False)
        assert _resolve_one_call(self.PROJECT, self._rel(self.TS_CALLER, "render"), lookup) is None

    def test_an_unmapped_extension_disables_the_filter_rather_than_emptying_it(self):
        """A language the map does not know must lose precision, never edges."""
        lookup = _CallLookup(
            name_to_callables={"render": [(self.PY_UID, "app/views.py", "public")]},
            import_map={},
            caller_to_parent={},
            parent_children={},
            uid_to_info={
                self.PY_UID: ("render", "app/views.py"),
                "proj:weird.caller": ("caller", "src/thing.zzz"),
            },
        )
        result = _resolve_one_call(self.PROJECT, self._rel("proj:weird.caller", "render"), lookup)
        assert result == ([self.PY_UID], "project_unique")

    def test_typescript_and_javascript_share_one_namespace(self):
        """Grouping is by namespace, not grammar — a .ts file really does call into .js."""
        lookup = _CallLookup(
            name_to_callables={"helper": [("proj:lib.helper", "lib/helper.js", "public")]},
            import_map={},
            caller_to_parent={},
            parent_children={},
            uid_to_info={
                "proj:lib.helper": ("helper", "lib/helper.js"),
                self.TS_CALLER: ("mount", "web/page.ts"),
            },
        )
        result = _resolve_one_call(self.PROJECT, self._rel(self.TS_CALLER, "helper"), lookup)
        assert result == (["proj:lib.helper"], "project_unique")

    def test_c_and_cpp_share_one_namespace(self):
        """`.h` is routed to either grammar by a content sniff, so the split is unstable
        per file — and C/C++ interoperate by design. One group is the correct answer."""
        lookup = _CallLookup(
            name_to_callables={"buf_init": [("proj:buf.buf_init", "src/buf.c", "public")]},
            import_map={},
            caller_to_parent={},
            parent_children={},
            uid_to_info={
                "proj:buf.buf_init": ("buf_init", "src/buf.c"),
                "proj:app.run": ("run", "src/app.cpp"),
            },
        )
        result = _resolve_one_call(self.PROJECT, self._rel("proj:app.run", "buf_init"), lookup)
        assert result == (["proj:buf.buf_init"], "project_unique")

    def test_java_and_csharp_do_not_share_a_namespace(self):
        """They share jvm.py. Sharing a walker is an implementation detail; sharing a
        call namespace is a language fact, and they do not."""
        lookup = _CallLookup(
            name_to_callables={"Parse": [("proj:J.Parse", "src/J.java", "public")]},
            import_map={},
            caller_to_parent={},
            parent_children={},
            uid_to_info={
                "proj:J.Parse": ("Parse", "src/J.java"),
                "proj:C.Run": ("Run", "src/C.cs"),
            },
        )
        assert _resolve_one_call(self.PROJECT, self._rel("proj:C.Run", "Parse"), lookup) is None

    def test_a_single_language_project_is_unaffected(self):
        """The churn bound: with one language present nothing is filtered."""
        lookup = _CallLookup(
            name_to_callables={
                "helper": [
                    ("proj:a.helper", "src/a.py", "public"),
                    ("proj:b.helper", "src/b.py", "public"),
                ]
            },
            import_map={},
            caller_to_parent={},
            parent_children={},
            uid_to_info={
                "proj:a.helper": ("helper", "src/a.py"),
                "proj:b.helper": ("helper", "src/b.py"),
                "proj:c.caller": ("caller", "src/c.py"),
            },
        )
        result = _resolve_one_call(self.PROJECT, self._rel("proj:c.caller", "helper"), lookup)
        assert result == (["proj:a.helper", "proj:b.helper"], "project_wide")


class TestMarkerLabelStamping:
    """Every node this module creates must carry the :Entity marker.

    Three hot-path queries match a node by uid alone and now do it as
    ``MATCH (a:Entity {uid: ...})``, so an entity node written without the marker is
    invisible to relationship linking, package containment and cross-project import
    resolution. Nothing raises when that happens -- the MATCH simply finds nothing
    and the edge is never written, which is how an unmarked fixture turned
    ``analyze_repo(analysis="structure")`` into an empty package list. A silent
    failure mode earns a tripwire rather than a convention.
    """

    # MERGE/CREATE of a node pattern that pins a label: `MERGE (var:<chain>`.
    # Relationship writes bind an already-matched variable (`MERGE (a)-[...]->(b)`)
    # and have no colon after the variable, so they do not match.
    _NODE_WRITE = re.compile(r"\b(?:MERGE|CREATE) \((?P<var>\w+):(?P<chain>[^\s)]+)")

    # SchemaVersion is meta, not an entity: a singleton with no uid, never an edge
    # endpoint, and deliberately outside _ENTITY_LABELS.
    _EXEMPT = ("NodeLabel.SCHEMA_VERSION",)

    def _node_writes(self):
        src = inspect.getsource(sys.modules[GraphClient.__module__])
        return [
            (src[: m.start()].count("\n") + 1, m.group("chain"))
            for m in self._NODE_WRITE.finditer(src)
            if not any(x in m.group("chain") for x in self._EXEMPT)
        ]

    def test_every_created_entity_node_is_marked(self):
        unmarked = [(line, chain) for line, chain in self._node_writes() if "NodeLabel.ENTITY" not in chain]
        assert not unmarked, f"entity nodes created without the :Entity marker: {unmarked}"

    def test_scan_actually_finds_the_node_writes(self):
        """Guards the guard: a regex that stops matching would pass the test above
        vacuously, which is the exact failure mode it exists to catch."""
        assert len(self._node_writes()) >= 8


class TestPatientWritePath:
    """The schema/DDL write path must actually be patient when a write blocks.

    Its whole reason to exist is that a concurrent writer can hold storage access for
    minutes. But it delegated to ``execute_write``, which bounds every call with
    ``asyncio.wait_for(write_timeout_s)`` and raises ``QueryTimeoutError`` -- and both
    retry predicates listed only ``TransientError``. A write that *blocks*, which is
    precisely what a busy writer produces, therefore failed on the first attempt and
    the 20-attempt schedule never ran.
    """

    @staticmethod
    def _fast_retry(monkeypatch):
        """Keep the retry *policy* under test but drop its backoff to zero.

        The real schedule is 20 attempts with up to 15s waits — ~4 minutes to exhaust.
        Patching the wait, rather than the stop condition, means these tests still
        exercise the real attempt count and the real retry predicate.
        """
        # `.retry` is attached by tenacity's decorator at runtime; ty cannot see it.
        retrying = GraphClient._execute_write_patient.retry  # ty: ignore[unresolved-attribute]
        monkeypatch.setattr(retrying, "wait", wait_none())

    @staticmethod
    def _client(tmp_path: Path, *, run_side_effect):
        settings = AtlasSettings(project_root=tmp_path)
        session = AsyncMock()
        session.run = AsyncMock(side_effect=run_side_effect)
        session_cm = MagicMock()
        session_cm.__aenter__ = AsyncMock(return_value=session)
        session_cm.__aexit__ = AsyncMock(return_value=False)
        driver = MagicMock()
        driver.session = MagicMock(return_value=session_cm)
        client = GraphClient(settings, driver=driver)
        client._write_timeout_s = 0.05  # keep the test fast; the mechanism is what matters
        return client, session

    async def test_retries_a_blocking_write_instead_of_failing_once(self, tmp_path: Path, monkeypatch):
        """Two blocked attempts then success must succeed, not raise."""
        calls = {"n": 0}

        async def run(*_a, **_k):
            calls["n"] += 1
            if calls["n"] <= 2:
                await asyncio.sleep(10)  # blocks past write_timeout_s
            result = AsyncMock()
            result.consume = AsyncMock(return_value=None)
            return result

        self._fast_retry(monkeypatch)
        client, _ = self._client(tmp_path, run_side_effect=run)
        await client._execute_write_patient("CREATE INDEX ON :Thing(uid);")
        assert calls["n"] == 3, "a blocked write was not retried"

    async def test_gives_up_eventually_rather_than_hanging(self, tmp_path: Path, monkeypatch):
        """Patience is bounded — a permanently wedged server must surface, not hang."""

        async def run(*_a, **_k):
            await asyncio.sleep(10)

        self._fast_retry(monkeypatch)
        client, _ = self._client(tmp_path, run_side_effect=run)
        with pytest.raises(QueryTimeoutError):
            await client._execute_write_patient("CREATE INDEX ON :Thing(uid);")

    async def test_defers_to_execute_write_inside_a_managed_transaction(self, tmp_path: Path):
        """A managed transaction owns its own retries; the patient path must not
        open a second session inside one."""
        client, _ = self._client(tmp_path, run_side_effect=AsyncMock())
        tx = AsyncMock()
        tx_result = AsyncMock()
        tx_result.consume = AsyncMock(return_value=None)
        tx.run = AsyncMock(return_value=tx_result)
        token = _active_tx_var.set(tx)
        try:
            await client._execute_write_patient("MATCH (n) SET n.x = 1")
        finally:
            _active_tx_var.reset(token)
        tx.run.assert_awaited_once()
        client._driver.session.assert_not_called()


class TestFixtureMarkerStamping:
    """Test fixtures that build nodes with raw Cypher must stamp :Entity too.

    Product code is covered by TestMarkerLabelStamping; fixtures were the gap, and the
    gap is not cosmetic. Fourteen queries now match on :Entity, so an unmarked fixture
    node is invisible to them -- silently, with no error. That is exactly how
    ``test_upsert_with_documents_rels`` began asserting on an empty edge list: it
    pre-created its target with a raw CREATE carrying only the Callable label, and the
    doc-link query stopped finding it. (Phrased without the literal pattern on purpose --
    the scan below covers this file too, and a worked example in prose would match it.)

    A sweep at the time found 43 such creations across three files, after an earlier
    automated audit had reported "exactly two" -- which is why this is a test and not a
    one-off cleanup.
    """

    _NODE_WRITE = re.compile(
        r"(?:CREATE|MERGE)\s*\(\w*:"
        r"((?:\{NodeLabel\.\w+\}|[A-Za-z_]\w*)(?::(?:\{NodeLabel\.\w+\}|[A-Za-z_]\w*))*)"
    )

    def _unmarked(self) -> list[tuple[str, int, str]]:
        from code_atlas.schema import _ENTITY_LABELS

        values = {lbl.value for lbl in _ENTITY_LABELS}
        enum_names = {lbl.name for lbl in _ENTITY_LABELS}
        tests_root = Path(__file__).resolve().parents[2]
        out: list[tuple[str, int, str]] = []
        for path in sorted(tests_root.rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            for m in self._NODE_WRITE.finditer(text):
                chain = m.group(1)
                names = {
                    (mm.group(1) if (mm := re.fullmatch(r"\{NodeLabel\.(\w+)\}", part)) else part)
                    for part in chain.split(":")
                }
                if not (names & values or names & enum_names):
                    continue  # not an entity label (SchemaVersion, a plain alias, ...)
                if "Entity" in names or "ENTITY" in names:
                    continue
                rel = path.relative_to(tests_root).as_posix()
                out.append((rel, text[: m.start()].count("\n") + 1, chain))
        return out

    def test_no_fixture_creates_an_unmarked_entity_node(self):
        unmarked = self._unmarked()
        assert not unmarked, f"fixtures creating entity nodes without :Entity: {unmarked}"

    def test_the_scan_still_finds_node_writes(self):
        """Guards the guard: a regex that matched nothing would pass the test above
        vacuously, which is the failure mode it exists to catch."""
        text = "\n".join(
            p.read_text(encoding="utf-8", errors="replace") for p in (Path(__file__).resolve().parents[2]).rglob("*.py")
        )
        assert len(self._NODE_WRITE.findall(text)) >= 40


class _StubResult:
    def __init__(self, rows):
        self._rows = rows

    def __aiter__(self):
        async def gen():
            for row in self._rows:
                yield row

        return gen()

    async def consume(self):
        return None


class _StubSession:
    def __init__(self, rows):
        self._rows = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False

    async def run(self, _query, _params=None):
        return _StubResult(self._rows)


class _StubDriver:
    def __init__(self, rows):
        self._rows = rows

    def session(self):
        return _StubSession(self._rows)


def _client_with(tmp_path, rows) -> GraphClient:
    """A client over a stub driver. The cast is the same deferred-retyping convention
    used at the other construction sites: _StubDriver implements the two methods this
    path touches, not the whole neo4j AsyncDriver surface."""
    return GraphClient(AtlasSettings(project_root=tmp_path), driver=cast("Any", _StubDriver(rows)))


class TestGraphQueryTiming:
    """Round-trips are attributed to the method that made them.

    The query text cannot be the label: `cypher_query` passes agent-authored Cypher
    straight through, so the label set would be unbounded and would carry user content
    into the metrics store. Method names are a closed set at exactly the granularity
    "which read is costing me" needs.

    This test also pins the stack depth `caller_name` walks. That depth is the fragile
    part -- a decorator or an extra wrapper on `execute` silently shifts it, and the
    symptom would be a dashboard full of `_execute_inner` rather than an error.
    """

    @staticmethod
    def _capture(monkeypatch):
        import code_atlas.telemetry as tel

        recorded: list[tuple] = []
        monkeypatch.setattr(
            tel._metrics,
            "graph_query_seconds",
            type("H", (), {"record": lambda _s, v, a=None: recorded.append((v, a))})(),
        )
        monkeypatch.setattr(tel, "_enabled", True)
        return recorded

    async def test_a_read_is_labelled_with_the_calling_method(self, monkeypatch, tmp_path: Path):
        recorded = self._capture(monkeypatch)
        client = _client_with(tmp_path, [{"n": 1}])

        await client.ping()  # ping() -> execute() -> the timed block

        assert len(recorded) == 1
        elapsed, attrs = recorded[0]
        assert attrs == {"op": "ping", "kind": "read"}, "wrong stack depth — see caller_name"
        assert elapsed >= 0

    async def test_a_write_is_labelled_and_marked_as_a_write(self, monkeypatch, tmp_path: Path):
        recorded = self._capture(monkeypatch)
        client = _client_with(tmp_path, [])

        async def set_schema_version_probe():
            await client.execute_write("RETURN 1")

        await set_schema_version_probe()

        assert [a for _v, a in recorded] == [{"op": "set_schema_version_probe", "kind": "write"}]

    async def test_nothing_is_recorded_while_telemetry_is_off(self, monkeypatch, tmp_path: Path):
        """The frame walk is the only part of this with a real cost, and it must not
        happen for the overwhelmingly common case of telemetry being disabled."""
        import code_atlas.telemetry as tel

        recorded = self._capture(monkeypatch)
        monkeypatch.setattr(tel, "_enabled", False)
        client = _client_with(tmp_path, [{"n": 1}])

        await client.ping()

        assert recorded == []
