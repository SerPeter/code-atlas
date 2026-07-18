"""Unit tests for pure-function helpers in GraphClient.

No infrastructure required — these test pure functions and data structures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

from code_atlas.graph.client import (
    _NAME_ROUTED_REL_TYPES,
    _OUT_OF_BAND_REL_TYPES,
    _POST_BATCH_REL_TYPES,
    _UID_ROUTED_REL_TYPES,
    GraphClient,
    _CallLookup,
    _format_path_hops,
    _fuse_bm25_results,
    _resolve_one_call,
    _sanitize_bm25_query,
    _validate_relationship_routing,
)
from code_atlas.parsing.ast import ParsedRelationship
from code_atlas.schema import RelType
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

    def test_omits_confidence_when_absent(self):
        """A non-CALLS edge (e.g. IMPORTS) has no confidence/strategy property to surface."""
        nodes = [{"uid": "p:a", "name": "a"}, {"uid": "p:b", "name": "b"}]
        rels = [_FakeRel("IMPORTS")]

        hops = _format_path_hops(nodes, rels)

        assert "confidence" not in hops[0]
        assert "strategy" not in hops[0]
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
        assert "NOT ()-[:CALLS]->(n)" in query
        assert "NOT n.name STARTS WITH '__'" in query

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
