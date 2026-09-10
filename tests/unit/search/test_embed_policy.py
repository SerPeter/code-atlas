"""Indexing a file and embedding its entities are two decisions (ATL-166).

The policy that separates them is recomputed from settings at every site that needs it
rather than stamped onto nodes -- a node property would mean a ``SCHEMA_VERSION`` bump,
and a bump drops the vector indices while recreating them only conditionally (ADR-0024).

What these tests pin is the pair of claims the feature stands on:

* **Excluded is not indexed-less.** A node the policy excludes keeps its name, its edges
  and its FTS document. Only the vector goes.
* **Excluded is not invisible.** BM25 and graph still return it, and the file-level node
  (``config_file`` / ``xml_document``) is never excluded -- it keeps its vector and carries
  the file body. There is deliberately no third compensation in fusion: ATL-166 added one,
  and ADR-0052 records why RRF rank space cannot express it.

The first is easy to half-implement in a way that passes a naive test: gate one of the
three sites and the queue fills anyway.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
from unittest.mock import AsyncMock

import pytest

from code_atlas.indexing.orchestrator import _reclaim_excluded_embeddings, _reconcile_missing_embeddings
from code_atlas.search.embeddings import DEFAULT_EXCLUDE_KINDS, EmbedPolicy
from code_atlas.search.engine import _build_ranked_lists, rrf_fuse
from code_atlas.settings import EmbeddingSettings

if TYPE_CHECKING:
    from collections.abc import Collection


def _policy(**kwargs: Any) -> EmbedPolicy:
    return EmbedPolicy.from_settings(EmbeddingSettings(**kwargs))


class TestPolicyResolution:
    """Two axes. The kind axis carries the shipped default and replaces it when set; the
    path axis is the user's instrument and starts empty. `include` beats both."""

    def test_the_default_excludes_the_generic_fallback_kinds(self):
        policy = _policy()
        assert policy.exclude_kinds == set(DEFAULT_EXCLUDE_KINDS)
        assert not policy.allows("config_setting", "conf/app.json")
        assert not policy.allows("config_section", "conf/app.json")

    def test_the_default_keeps_the_file_level_node_and_every_recognised_dialect(self):
        """The default is aimed at data no dialect could read. A node a dialect *did*
        produce is a semantic target and keeps its vector, with no re-admit line."""
        policy = _policy()
        assert policy.allows("config_file", "conf/app.json")
        assert policy.allows("ci_job", ".github/workflows/ci.yml")
        assert policy.allows("k8s_resource", "deploy/svc.yaml")
        assert policy.allows("function", "src/mod.py")

    def test_exclude_kinds_replaces_rather_than_extends(self):
        policy = _policy(exclude_kinds=["ci_job"])
        assert not policy.allows("ci_job", "ci.yml")
        assert policy.allows("config_setting", "conf/app.json"), "the default should have been replaced, not merged"

    def test_an_empty_kind_list_is_a_real_answer_not_a_missing_one(self):
        """``exclude_kinds = []`` means "embed everything", and must not read as unset."""
        assert _policy(exclude_kinds=[]).allows("config_setting", "conf/app.json")

    def test_the_path_axis_takes_gitignore_patterns(self):
        """One field, not the `exclude`/`extend_exclude` pair `[scope]` has: the path
        default is empty, so a second field would do exactly what the first one does."""
        policy = _policy(exclude=["vendor/**", "generated/**"])
        assert policy.allows("function", "src/mod.py")
        assert not policy.allows("function", "vendor/lib.py")
        assert not policy.allows("function", "generated/pb2.py")

    def test_include_beats_both_axes(self):
        policy = _policy(exclude=["ops/**"], include=["ops/critical/"])
        assert not policy.allows("function", "ops/tool.py")
        assert policy.allows("function", "ops/critical/tool.py")
        assert policy.allows("config_setting", "ops/critical/app.json"), "include must beat the kind axis too"

    def test_the_axes_are_independent(self):
        """A kind rule must not imply a path rule or the reverse -- they answer different
        questions and a user reaches for them for different reasons."""
        kinds_only = _policy(exclude_kinds=["config_setting"])
        assert kinds_only.allows("function", "anywhere/at/all.py")
        paths_only = _policy(exclude_kinds=[], exclude=["conf/**"])
        assert paths_only.allows("config_setting", "src/app.json")
        assert not paths_only.allows("function", "conf/gen.py")

    def test_a_permissive_policy_says_so(self):
        """The callers skip the whole check on this, so it has to be exact."""
        assert _policy(exclude_kinds=[]).is_permissive
        assert not _policy().is_permissive
        assert not _policy(exclude_kinds=[], exclude=["vendor/**"]).is_permissive

    def test_an_entity_with_no_file_path_is_judged_on_its_kind_alone(self):
        """Reference nodes carry no path. A path rule cannot match one, and must not be
        read as matching everything or as matching nothing in a way that flips the kind."""
        policy = _policy(exclude=["**"])
        assert policy.allows("external_package", "")
        assert not policy.allows("config_setting", "")


# ---------------------------------------------------------------------------
# Gate 2 — the embed stage
# ---------------------------------------------------------------------------


class _Consumer:
    """Just the policy filter off EmbedConsumer, without building one."""

    def __init__(self, policy: EmbedPolicy) -> None:
        self._embed_policy = policy


class TestTheEmbedStageGate:
    """Defence in depth. The AST stage stops work being queued; this catches what is
    already on the stream -- a poison-parked event, an abandoned PEL entry, or work
    published before the policy changed."""

    @staticmethod
    def _props(kind: str, file_path: str) -> dict[str, Any]:
        return {"uid": f"p:{kind}", "kind": kind, "file_path": file_path, "name": "x", "_label": "Value"}

    def _filter(self, policy: EmbedPolicy, props: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from code_atlas.indexing.consumers import EmbedConsumer

        return EmbedConsumer._allowed_by_policy(_Consumer(policy), props)  # ty: ignore[invalid-argument-type]

    def test_an_excluded_entity_is_dropped_from_the_batch(self):
        props = [self._props("config_setting", "conf/app.json"), self._props("function", "src/mod.py")]
        kept = self._filter(_policy(), props)
        assert [p["kind"] for p in kept] == ["function"]

    def test_a_permissive_policy_returns_the_batch_untouched(self):
        """Not merely equal -- the same object, because this runs per batch."""
        props = [self._props("config_setting", "conf/app.json")]
        assert self._filter(_policy(exclude_kinds=[]), props) is props

    def test_a_missing_kind_or_path_does_not_crash_the_batch(self):
        """``read_entity_texts`` returns whatever the graph holds, and a node written by
        an older version may hold neither."""
        assert self._filter(_policy(), [{"uid": "p:x"}]) == [{"uid": "p:x"}]


# ---------------------------------------------------------------------------
# Gate 3 — the re-queue sweep
# ---------------------------------------------------------------------------


class _ReconcileGraph:
    """Reports unembedded entities, and honours the query-side kind filter like both
    real backends do -- so a test can tell "filtered in SQL" from "filtered in Python"."""

    def __init__(self, rows: list[tuple[str, str, str, str]]) -> None:
        self.rows = rows
        self.kind_filters: list[list[str]] = []

    async def find_unembedded_entities(
        self, project_name: str, *, limit: int = 5000, exclude_kinds: Collection[str] = ()
    ) -> list[tuple[str, str, str, str]]:
        skip = sorted(exclude_kinds)
        self.kind_filters.append(skip)
        return [r for r in self.rows if r[2] not in set(skip)][:limit]


class TestTheReQueueGate:
    """The site that bites. Ungated, every excluded entity is re-queued on every run, the
    embed stage drops them all, the next pass finds exactly the same set, and the
    reconcile loop gives up with a warning about lost work -- permanently, on a graph
    where nothing is wrong."""

    ROWS: ClassVar = [
        ("p:blob", "Value", "config_setting", "conf/app.json"),
        ("p:fn", "Callable", "function", "src/mod.py"),
    ]

    async def test_an_excluded_entity_is_never_re_queued(self):
        graph, bus = _ReconcileGraph(list(self.ROWS)), AsyncMock()
        queued = await _reconcile_missing_embeddings(graph, bus, ["p"], _policy())  # ty: ignore[invalid-argument-type]
        assert queued == {"p:fn"}

    async def test_a_real_hole_is_still_healed(self):
        """Non-vacuity: a gate that re-queued nothing at all would pass the test above."""
        graph, bus = _ReconcileGraph(list(self.ROWS)), AsyncMock()
        await _reconcile_missing_embeddings(graph, bus, ["p"], _policy())  # ty: ignore[invalid-argument-type]
        published = bus.publish_many.call_args[0][1]
        assert [e.entity.qualified_name for e in published] == ["p:fn"]

    async def test_the_kind_filter_is_pushed_into_the_query(self):
        """So the per-project cap is spent on real holes. Without this, two thousand
        excluded config nodes fill a 5,000-row page and the actual holes never surface."""
        graph, bus = _ReconcileGraph(list(self.ROWS)), AsyncMock()
        await _reconcile_missing_embeddings(graph, bus, ["p"], _policy())  # ty: ignore[invalid-argument-type]
        assert graph.kind_filters == [sorted(DEFAULT_EXCLUDE_KINDS)]

    async def test_an_include_list_moves_the_kind_decision_out_of_the_query(self):
        """``include`` beats ``exclude_kinds``, and that decision needs the path and the
        kind together -- which the query cannot do. So it filters nothing and the policy
        is applied in full afterwards, or an included path would never be re-queued."""
        rows = [("p:kept", "Value", "config_setting", "ops/critical/app.json"), *self.ROWS]
        graph, bus = _ReconcileGraph(rows), AsyncMock()
        policy = _policy(include=["ops/critical/"])
        queued = await _reconcile_missing_embeddings(graph, bus, ["p"], policy)  # ty: ignore[invalid-argument-type]
        assert graph.kind_filters == [[]], "the query must not pre-filter when include can re-admit"
        assert queued == {"p:kept", "p:fn"}


# ---------------------------------------------------------------------------
# The reclaim sweep
# ---------------------------------------------------------------------------


class _ReclaimGraph:
    def __init__(self, rows: list[tuple[str, str, str]]) -> None:
        self.rows = rows
        self.kind_filters: list[list[str]] = []
        self.cleared: list[str] = []

    async def find_embedded_entities(
        self, project_name: str, *, kinds: Collection[str] = ()
    ) -> list[tuple[str, str, str]]:
        wanted = sorted(kinds)
        self.kind_filters.append(wanted)
        return [r for r in self.rows if not wanted or r[1] in set(wanted)]

    async def clear_embeddings_for_uids(self, uids: list[str]) -> int:
        self.cleared.extend(uids)
        return len(uids)


class TestTheReclaimSweep:
    """A policy applied only at write time leaves every vector bought under the old one
    in place. This is what makes a policy change take effect on the next ``atlas index``
    instead of on a ``--reset-embeddings``, which would re-bill the whole database."""

    ROWS: ClassVar = [
        ("p:blob", "config_setting", "conf/app.json"),
        ("p:fn", "function", "src/mod.py"),
    ]

    async def test_a_now_excluded_vector_is_reclaimed(self):
        graph = _ReclaimGraph(list(self.ROWS))
        assert await _reclaim_excluded_embeddings(graph, ["p"], _policy()) == 1  # ty: ignore[invalid-argument-type]
        assert graph.cleared == ["p:blob"]

    async def test_a_permissive_policy_reads_nothing_at_all(self):
        """No policy, no sweep -- not even the read. Every index runs this."""
        graph = _ReclaimGraph(list(self.ROWS))
        assert await _reclaim_excluded_embeddings(graph, ["p"], _policy(exclude_kinds=[])) == 0  # ty: ignore[invalid-argument-type]
        assert graph.kind_filters == []

    async def test_a_path_only_policy_still_sweeps(self):
        """`is_permissive` is false here, and the kind axis is empty, so the sweep has to
        read everything and decide in Python rather than short-circuiting on no kinds."""
        graph = _ReclaimGraph([("p:fn", "function", "vendor/lib.py"), ("p:g", "function", "src/mod.py")])
        policy = _policy(exclude_kinds=[], exclude=["vendor/**"])
        assert await _reclaim_excluded_embeddings(graph, ["p"], policy) == 1  # ty: ignore[invalid-argument-type]
        assert graph.cleared == ["p:fn"]


# ---------------------------------------------------------------------------
# What fusion pays for
# ---------------------------------------------------------------------------


def _lists(**channels: list[str]) -> dict[str, list[str]]:
    return {name: list(uids) for name, uids in channels.items()}


# Natural-language weights: the regime the retired floor was written for, and the one it
# damaged most. ``analyze_query`` returns these for any query of three words or more.
_NL_WEIGHTS = {"graph": 0.5, "vector": 2.0, "bm25": 1.0}


class TestFusionPaysOnlyForRanksAChannelReturned:
    """ATL-166 appended policy-excluded uids to the tail of the vector list so a
    natural-language query could not gate them off the page. ATL-184 removed it: RRF rank
    space has no epsilon to append them at (ADR-0052).

    At ``k=60`` the curve is nearly flat across a fetched window -- the 63rd of 63 rows
    still pays ``2/124``, 98.4% of an entire rank-1 BM25 hit -- while the adjacent-rank
    differential that decides *order* is 2.6e-4. The floor was ~62x coarser than the
    ordering it was forbidden to disturb, so it inverted the very comparison it was meant
    to make fair."""

    def test_a_saturated_vector_channel_does_not_promote_the_excluded_cohort(self):
        """A saturated vector channel (60 unrelated hits, neither rival among them) and a
        BM25 list whose rank-1 entry is real code followed by excluded blobs.

        This test was a strict xfail for as long as the floor existed. The margin is
        exactly ``1/61 - 1/62 = 1/3782`` -- BM25's adjacent-rank differential, and the
        entire budget any floor had to stay under. The floor paid 2/121, 62x that.
        """
        filler = [f"p:v{i}" for i in range(60)]
        excluded = [f"p:x{i}" for i in range(20)]
        scores = rrf_fuse(_lists(vector=filler, bm25=["p:code", *excluded]), k=60, weights=_NL_WEIGHTS)

        assert scores["p:code"] > scores["p:x0"], (
            "a rank-1 BM25 hit must outrank an entity that no channel ranked first"
        )
        assert scores["p:code"] - scores["p:x0"] == pytest.approx(1 / 3782)
        ranking = list(scores)
        assert not [uid for uid in ranking[:20] if uid in set(excluded)], (
            "the excluded cohort must not occupy the first page; the floor put 11 there"
        )

    def test_an_entity_with_no_vector_is_still_buried_by_a_saturated_channel(self):
        """The grievance behind ADR-0047 section 4, which this change deliberately does
        not fix -- recorded as a test so the next reader finds a fact, not a memory.

        The corpus's rank-1 BM25 hit still fuses *below* all 60 vector rows, because the
        worst of them pays ``2/120 = 0.01667`` against its ``1/61 = 0.01639``. That is the
        2.0 vector weight over a shortlist, and it applies identically to an entity that
        merely missed the shortlist and to one the policy excluded. Pricing those two
        states differently is what ATL-184 undid; neither is rescued here.
        """
        filler = [f"p:v{i}" for i in range(60)]
        scores = rrf_fuse(_lists(vector=filler, bm25=["p:code", "p:absent"]), k=60, weights=_NL_WEIGHTS)
        ranking = list(scores)

        assert ranking.index("p:code") == 60, "the rank-1 BM25 hit lands behind all 60 vector rows"
        assert scores["p:code"] < min(scores[uid] for uid in filler)
        # Same state, same price: neither uid owns a vector row, and nothing distinguishes
        # "excluded by policy" from "absent from the shortlist" at query time any more.
        assert scores["p:code"] > scores["p:absent"], "ordering still comes from the ranks BM25 returned"

    def test_provenance_reports_only_the_ranks_the_channels_returned(self):
        """``sources`` is public MCP payload. Nothing may claim a vector hit it did not
        get -- the guard that stops a synthetic rank being re-invented."""
        from code_atlas.search.engine import _build_provenance

        channel_results = {
            "vector": [{"node": {"uid": "p:fn", "kind": "function", "file_path": "src/mod.py"}}],
            "bm25": [{"node": {"uid": "p:blob", "kind": "config_setting", "file_path": "conf/app.json"}}],
        }
        ranked, _props = _build_ranked_lists(channel_results)
        provenance = _build_provenance(ranked)

        assert provenance["p:blob"] == {"bm25": 1}
        assert provenance["p:fn"] == {"vector": 1}


@pytest.mark.parametrize("kind", DEFAULT_EXCLUDE_KINDS)
def test_every_default_excluded_kind_is_one_the_generic_handler_emits(kind: str):
    """The default is only defensible because these kinds come from exactly one code path
    -- ``config.py``'s structural fallback. If one of them were also emitted by a dialect
    handler, excluding it would take vectors off entities somebody understood.

    The fallback has two branches -- the key-tree one for YAML/JSON/TOML and the element
    one for XML -- and both mean equally "no handler recognised this"."""
    from code_atlas.parsing.languages import config

    fallback = {
        config._GENERIC_MODULE_KIND,
        config._GENERIC_SECTION_KIND,
        config._GENERIC_SETTING_KIND,
        config._XML_DOCUMENT_KIND,
        config._XML_ELEMENT_KIND,
        config._XML_SETTING_KIND,
    }
    assert kind in fallback
