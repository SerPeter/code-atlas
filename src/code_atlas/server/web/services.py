"""Application layer for the web UI.

The middle of three layers:

* ``controllers`` own HTTP — routing, status codes, template selection.
* ``services`` (here) own the *use case* — what a view needs and how to assemble it.
* the graph backend and :mod:`code_atlas.server.analysis` own data access.

The rule that makes the split worth having: **a service never writes Cypher and never
imports Litestar.** Reaching past ``GraphBackend`` into raw queries would bypass the
SQLite backend entirely, and a number shown in the UI that disagrees with the same
number from an MCP tool is worse than no UI at all — so both read the same backend
methods.

A service may call :mod:`code_atlas.server.analysis` where it needs genuine analysis,
but not merely to reach data: routing a label tally through the MCP-facing dispatcher
buys a shared number at the cost of depending on that analysis's entire output shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from code_atlas.server.architecture import analyse
from code_atlas.server.web.schemas import (
    ArchitectureHealth,
    CoverageCaveat,
    CycleDetail,
    EdgeEvidence,
    EntityDetail,
    ProjectOverview,
    ProjectRef,
    RelatedEntity,
    SearchHit,
    SearchPage,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from code_atlas.graph.protocol import GraphBackend
    from code_atlas.search.engine import CompactNode, EmbedOne, SearchResult
    from code_atlas.server.architecture import Cycle
    from code_atlas.settings import SearchSettings

# Label tallies are whole-project aggregates; the limit only bounds the ranked lists
# this view does not read.
_STRUCTURE_LIMIT = 20
_ENTITY_LIMIT = 200
_EDGE_LIMIT = 500
_DSM_LIMIT = 60


class ProjectNotIndexedError(LookupError):
    """Raised when the requested project has no graph data.

    Distinct from "the project is empty": one means run `atlas index`, the other means
    the index ran and found nothing. Serving an empty graph for the first would be the
    silent-success failure ATL-110 removed from the CLI.
    """

    def __init__(self, project: str) -> None:
        super().__init__(project)
        self.project = project


class ProjectViewService:
    """Assembles the project-scoped views.

    Scope is deliberate and is what keeps every query bounded: a view covers **one
    project**, with other projects reachable but not loaded. The alternative — render
    everything — is an unbounded query against a graph that is already ~30k nodes for
    eight projects, and it is offered as an explicit opt-in rather than a default.
    """

    def __init__(self, graph: GraphBackend, project: str) -> None:
        self._graph = graph
        self._project = project

    @property
    def project(self) -> str:
        return self._project

    async def overview(self) -> ProjectOverview:
        """The landing view for the current project."""
        statuses = await self._graph.get_project_status()
        current = next((s for s in statuses if s.get("project") == self._project), None)
        if current is None:
            raise ProjectNotIndexedError(self._project)

        # Straight to the data layer rather than through `analyze_repo`. Routing the
        # overview through the MCP-facing dispatcher coupled this service to that
        # analysis's whole contract — largest_modules, packages, external_deps — for a
        # label tally that is four lines. Both read `get_structure_overview`, so the
        # numbers still share one source; only the needless coupling is gone.
        structure = await self._graph.get_structure_overview(self._project, "", _STRUCTURE_LIMIT)
        label_counts: dict[str, int] = {}
        for row in structure.get("counts", []):
            label = str(row.get("label", ""))
            if label:
                label_counts[label] = label_counts.get(label, 0) + int(row.get("cnt") or 0)

        others = tuple(
            ProjectRef(
                name=str(s.get("project", "")),
                entities=int(s.get("entities") or 0),
                is_current=s.get("project") == self._project,
            )
            for s in statuses
            if s.get("project")
        )

        return ProjectOverview(
            project=self._project,
            entity_count=int(current.get("entities") or 0),
            module_count=int(label_counts.get("Module", 0)),
            indexed_at=_as_str(current.get("indexed_at")),
            git_hash=_as_str(current.get("git_hash")),
            label_counts=label_counts,
            other_projects=others,
            caveat=_coverage_caveat(label_counts),
        )


def _as_str(value: object) -> str | None:
    """Normalise a graph value to a display string, preserving "absent"."""
    if value is None or value == "":
        return None
    return str(value)


def _coverage_caveat(label_counts: dict[str, int]) -> CoverageCaveat:
    """State what these numbers do not cover.

    Placeholder shape for now: the real signal is which languages in this project had no
    grammar installed, which ATL-110 already records per index run but does not yet
    persist to the graph. Wiring that through is ATL-119's dependency, not this story's —
    what matters here is that every view carries the field from the start, so adding the
    data later cannot be forgotten.
    """
    return CoverageCaveat(note="" if label_counts else "No entities indexed for this project.")


class EntityNotFoundError(LookupError):
    """The requested uid is not in the graph."""

    def __init__(self, uid: str) -> None:
        super().__init__(uid)
        self.uid = uid


class SearchViewService:
    """Search, and the entity detail a result leads to.

    Separate from :class:`ProjectViewService` because it is a different use case with
    different collaborators, not because the file was getting long: this one needs the
    search engine and an embedding client, the overview needs neither.
    """

    def __init__(
        self,
        graph: GraphBackend,
        project: str,
        *,
        search_settings: SearchSettings,
        embed: EmbedOne | None = None,
    ) -> None:
        self._graph = graph
        self._project = project
        self._settings = search_settings
        self._embed = embed

    async def search(self, query: str, *, limit: int = 20) -> SearchPage:
        """Ranked results for *query*, scoped to this project.

        Ranking is whatever :func:`hybrid_search` produces — the same call the MCP tool
        makes. A UI that re-ranked would give a human a different answer from the agent
        looking at the same graph, which is the divergence the service layer exists to
        prevent.
        """
        from code_atlas.search.engine import hybrid_search  # noqa: PLC0415

        if not query.strip():
            return SearchPage(query=query, hits=(), more_available=False)

        # One row beyond the page: enough to know whether more exist, and deliberately
        # not treated as a count (ATL-111).
        results = await hybrid_search(
            self._graph,
            self._embed,
            self._settings,
            query,
            limit=limit + 1,
            scope=self._project,
        )
        more = len(results) > limit
        return SearchPage(
            query=query,
            hits=tuple(_as_hit(r) for r in results[:limit]),
            more_available=more,
        )

    async def detail(self, uid: str) -> EntityDetail:
        """One entity, its neighbourhood, and the evidence behind each edge."""
        from code_atlas.search.engine import expand_context  # noqa: PLC0415

        context = await expand_context(self._graph, uid)
        if context is None:
            raise EntityNotFoundError(uid)

        evidence = await self._edge_evidence(context.target.file_path)
        target = context.target

        return EntityDetail(
            uid=target.uid,
            name=target.name,
            qualified_name=target.qualified_name,
            kind=target.kind,
            label=target.labels[0] if target.labels else "",
            file_path=target.file_path,
            line_start=target.line_start,
            line_end=target.line_end,
            signature=target.signature,
            docstring=target.docstring,
            parent=_as_related(context.parent, None) if context.parent else None,
            callers=tuple(
                _as_related(n, evidence.get((n.qualified_name, target.qualified_name))) for n in context.callers
            ),
            callees=tuple(
                _as_related(n, evidence.get((target.qualified_name, n.qualified_name))) for n in context.callees
            ),
            docs=tuple(_as_related(n, None) for n in context.docs),
            caveat=CoverageCaveat(),
        )

    async def _edge_evidence(self, file_path: str) -> dict[tuple[str, str], EdgeEvidence]:
        """``(from_qn, to_qn) -> evidence`` for edges around *file_path*'s module.

        Sourced from ``get_module_summary``, which already returns a decoded ``props``
        dict per edge — "all relationship properties, whatever they are ... so new CALLS
        edge properties surface without a backend change", per its own contract. Using
        it costs a module's worth of edges to annotate one entity's, and buys not adding
        an entity-scoped read to both backends for data that is already reachable.

        ``expand_context`` cannot supply this: ``CompactNode`` is a node, and evidence
        belongs to the edge that reached it — the same node reached two ways has two
        different claims behind it.
        """
        if not file_path:
            return {}
        try:
            summary = await self._graph.get_module_summary(self._project, file_path, _ENTITY_LIMIT, _EDGE_LIMIT)
        except KeyError, ValueError:
            # A path the summary cannot resolve costs evidence, never the view.
            return {}

        found: dict[tuple[str, str], EdgeEvidence] = {}
        for group in ("internal_edges", "fan_in", "fan_out"):
            for row in summary.get(group, []):
                key = (str(row.get("from_qn", "")), str(row.get("to_qn", "")))
                if not all(key):
                    continue
                props = row.get("props") or {}
                found[key] = EdgeEvidence(
                    rel_type=str(row.get("rel_type", "")),
                    strategy=str(props.get("strategy") or ""),
                    confidence=str(props.get("confidence") or ""),
                    weight=_as_float(props.get("weight")),
                    line=_as_int(props.get("line")),
                    site_count=_as_int(props.get("site_count")),
                )
        return found


def _as_hit(result: SearchResult) -> SearchHit:
    """Project a ``SearchResult`` onto the view model."""
    return SearchHit(
        uid=result.uid,
        name=result.name,
        qualified_name=result.qualified_name,
        kind=result.kind,
        label=result.labels[0] if result.labels else "",
        file_path=result.file_path,
        line_start=result.line_start,
        signature=result.signature,
        score=round(result.rrf_score, 4),
        channels=tuple(sorted(result.sources)),
    )


def _as_related(node: CompactNode, evidence: EdgeEvidence | None) -> RelatedEntity:
    return RelatedEntity(
        uid=node.uid,
        name=node.name,
        qualified_name=node.qualified_name,
        kind=node.kind,
        file_path=node.file_path,
        line_start=node.line_start,
        evidence=evidence,
    )


def _as_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _as_int(value: object) -> int | None:
    return int(value) if isinstance(value, int) else None


class ArchitectureViewService:
    """The "is this becoming a big ball of mud" view.

    Deliberately not a node-link graph: a force-directed blob looks like a hairball at
    every level of health, so it cannot answer whether things are getting worse. A design
    structure matrix can, because a clean architecture and a rotten one produce
    categorically different pictures.

    All arithmetic lives in :mod:`code_atlas.server.architecture` as pure functions over
    an edge list, so the numbers are checkable against hand-worked graphs rather than
    only against a live database.
    """

    def __init__(self, graph: GraphBackend, project: str) -> None:
        self._graph = graph
        self._project = project

    async def health(self, *, dsm_limit: int = _DSM_LIMIT) -> ArchitectureHealth:
        raw = await self._graph.get_module_import_edges(self._project, "")
        edges = [
            (str(r.get("from_mod", "")), str(r.get("to_mod", "")))
            for r in raw.get("direct", [])
            if r.get("from_mod") and r.get("to_mod")
        ]
        nodes = sorted({n for edge in edges for n in edge})

        metrics = analyse(nodes, edges)

        shown = metrics.order[:dsm_limit]
        position = {name: i for i, name in enumerate(shown)}
        # Deduplicated: a repeated import between the same module pair is one mark, and
        # counting it twice would make the matrix look denser than the graph is.
        marks = tuple(
            sorted({(position[src], position[dst]) for src, dst in edges if src in position and dst in position})
        )

        return ArchitectureHealth(
            project=self._project,
            module_count=metrics.module_count,
            edge_count=metrics.edge_count,
            propagation_cost=metrics.propagation_cost,
            core_size=metrics.core_size,
            largest_cycle=metrics.largest_cycle,
            fan_in_gini=metrics.fan_in_gini,
            cycles=_cycle_details(metrics.cycles[:10], edges),
            dsm_order=shown,
            dsm_marks=marks,
            dsm_truncated=len(metrics.order) > len(shown),
            caveat=_architecture_caveat(metrics.module_count),
        )


def _cycle_details(cycles: Sequence[Cycle], edges: Sequence[tuple[str, str]]) -> tuple[CycleDetail, ...]:
    """Attach to each cycle the edges that close it.

    A cycle's edges are exactly those with both endpoints inside it — every one is part
    of some loop, so every one is a candidate for the cut that breaks it. Listing the
    members alone would say a subsystem is tangled without saying which import to remove,
    which is the only thing a reader can act on.
    """
    details: list[CycleDetail] = []
    for cycle in cycles:
        members = set(cycle.members)
        internal = sorted({(src, dst) for src, dst in edges if src in members and dst in members})
        details.append(CycleDetail(members=cycle.members, edges=tuple(internal)))
    return tuple(details)


def _architecture_caveat(module_count: int) -> CoverageCaveat:
    """What these numbers do not cover.

    Propagation cost over a graph whose C++ named-function capture sits at 0.690
    (ATL-096) is a LOWER BOUND, and "8% - you are fine" over partial extraction is
    exactly the confident wrong answer this project keeps removing. The per-language
    coverage data is recorded per index run but not yet persisted to the graph, so this
    states the honest general case until it is.
    """
    if module_count == 0:
        return CoverageCaveat(note="No module dependencies indexed - nothing to measure.")
    return CoverageCaveat(
        note=(
            f"Computed over {module_count} modules. Any language whose extraction is "
            "incomplete makes this a lower bound, not a ceiling."
        )
    )
