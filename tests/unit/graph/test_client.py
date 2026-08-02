"""Unit tests for pure-function helpers in GraphClient.

No infrastructure required — these test pure functions and data structures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

from code_atlas.graph.client import (
    _CODE_ENTITY_KINDS,
    _NAME_ROUTED_REL_TYPES,
    _OUT_OF_BAND_REL_TYPES,
    _POST_BATCH_REL_TYPES,
    _UID_ROUTED_REL_TYPES,
    GraphClient,
    _call_edge_weight,
    _CallEdgeFacts,
    _CallLookup,
    _combine_call_edge_facts,
    _format_path_hops,
    _fuse_bm25_results,
    _resolve_one_call,
    _sanitize_bm25_query,
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

if TYPE_CHECKING:
    from pathlib import Path


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
        assert rels == [
            {
                "f": "proj:mod.caller",
                "t": "proj:other.helper",
                "confidence": "resolved",
                "strategy": "project_unique",
                "candidate_count": 1,
                "from_test": False,
                "weight": 1.0,
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
        assert "r.confidence = 'resolved'" in resolved_query
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
