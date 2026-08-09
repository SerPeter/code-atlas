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

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from code_atlas.server.analysis import _DEFAULT_BLAST_EDGE_TYPES
from code_atlas.server.architecture import analyse
from code_atlas.server.web.layout import force_layout, node_size
from code_atlas.server.web.naming import breadcrumb
from code_atlas.server.web.schemas import (
    AffectedEntity,
    ArchitectureHealth,
    ArchitectureTrend,
    BlastRadiusView,
    CommunityRef,
    CoverageCaveat,
    CycleDetail,
    DepthGroup,
    EdgeEvidence,
    EntityDetail,
    MapEdge,
    MapNode,
    ModuleMap,
    PathHop,
    ProjectOverview,
    ProjectRef,
    RelatedEntity,
    SearchHit,
    SearchPage,
    TracePathView,
    TrendPoint,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from code_atlas.graph.protocol import GraphBackend
    from code_atlas.search.engine import CompactNode, EmbedOne, SearchResult
    from code_atlas.server.analysis import ModuleGraph
    from code_atlas.server.architecture import Cycle
    from code_atlas.settings import SearchSettings

# Label tallies are whole-project aggregates; the limit only bounds the ranked lists
# this view does not read.
_STRUCTURE_LIMIT = 20
_ENTITY_LIMIT = 200
_EDGE_LIMIT = 500
_DSM_LIMIT = 60
# Sigma renders far more than this comfortably; the bound is readability, not WebGL.
_MAP_NODE_LIMIT = 1500
_COMMUNITY_MEMBER_LIMIT = 200
_EXTERNAL_RING = 165.0
# The page a reader sees, and the ceiling the resolved-only filter runs over.
_IMPACT_PAGE = 50
_IMPACT_CEILING = 500


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
        statuses = [_project_props(row) for row in await self._graph.get_project_status()]
        current = next((s for s in statuses if s.get("name") == self._project), None)
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
                name=str(s.get("name", "")),
                entities=int(s.get("entity_count") or 0),
                is_current=s.get("name") == self._project,
            )
            for s in statuses
            if s.get("name")
        )

        return ProjectOverview(
            project=self._project,
            entity_count=int(current.get("entity_count") or 0),
            module_count=int(label_counts.get("Module", 0)),
            indexed_at=_as_timestamp(current.get("last_indexed_at")),
            git_hash=_as_str(current.get("git_hash")),
            label_counts=label_counts,
            other_projects=others,
            caveat=_coverage_caveat(label_counts),
        )


def _project_props(row: dict[str, Any]) -> dict[str, Any]:
    """Unwrap a ``get_project_status`` row into the Project node's own properties.

    Both backends return ``[{"n": <node>}]`` — Memgraph from ``RETURN n``, SQLite from
    ``_row_to_node`` — and the properties are ``name``/``entity_count``/
    ``last_indexed_at``, not ``project``/``entities``/``indexed_at``. Reading the wrapper
    dict directly finds none of them, so every project compared unequal and the landing
    page 404'd as "not indexed" against a fully indexed graph.

    The unit tests missed it because the fake returned a flattened shape that no backend
    produces. `mcp.py`'s ``index_status`` had the unwrapping right all along.
    """
    node = row.get("n", row)
    return dict(node.items()) if hasattr(node, "items") else dict(node)


def _as_str(value: object) -> str | None:
    """Normalise a graph value to a display string, preserving "absent"."""
    if value is None or value == "":
        return None
    return str(value)


def _as_timestamp(value: object) -> str | None:
    """Render an index time for a human.

    `update_project_metadata` writes `last_indexed_at` as `time.time()`, so the stored
    value is a float. Passing it through `str()` put `1786176798.014237` on the landing
    page — technically the data, and useless as an answer to "when was this indexed".
    Anything that is not a number is passed through unchanged, since older rows and the
    SQLite path may hold an ISO string already.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return datetime.fromtimestamp(float(value), tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
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
        # ranked_score, not rrf_score: the raw fusion score is the value *before* the
        # visibility/label boosts, so showing it against the boosted ordering renders a
        # list that is not sorted by the number beside it. The CLI and the MCP tools were
        # fixed for this; this third consumer was missed.
        score=round(result.ranked_score, 4),
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
            trend=await self._trend(),
        )

    async def _trend(self) -> ArchitectureTrend | None:
        """How these numbers have moved across recorded index runs.

        Read-only here. Snapshots are written on the index path (ATL-121), because a
        history written when someone opens a page would record who looked at it rather
        than how the code changed.
        """
        from code_atlas.server.architecture_history import MAX_SNAPSHOTS, load, trend  # noqa: PLC0415

        try:
            snapshots = await load(self._graph, self._project)
        except Exception:  # a missing history costs the trend, never the view
            return None

        movement = trend(snapshots)
        if movement is None:
            return None

        return ArchitectureTrend(
            points=tuple(
                TrendPoint(
                    at=s.at,
                    commit=s.commit[:12],
                    modules=s.modules,
                    propagation_cost=s.propagation_cost,
                    core_size=s.core_size,
                    largest_cycle=s.largest_cycle,
                )
                for s in snapshots[-movement.count :]
            ),
            direction=movement.direction,
            propagation_delta=movement.propagation_delta,
            core_delta=movement.core_delta,
            coverage_changed=movement.coverage_changed,
            note=(f"Comparing the last {movement.count} index runs; at most {MAX_SNAPSHOTS} are kept."),
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


class MapViewService:
    """The "how is this codebase organised" view.

    Clusters modules into subsystems and draws them. Module granularity is not a
    simplification for the sake of the picture — at callable granularity the
    CALLS+IMPORTS subgraph puts ~95% of production code in one community at every usable
    resolution (see ``_analyze_communities``), so an entity-level map would be a hairball
    that also happened to be wrong.

    The clustering comes from :func:`build_module_graph`, the same code path
    ``find_communities`` uses, so the map and the MCP tool cannot disagree about the same
    project.
    """

    def __init__(self, graph: GraphBackend, project: str, *, test_patterns: tuple[str, ...] = ()) -> None:
        self._graph = graph
        self._project = project
        self._test_patterns = test_patterns

    async def map(
        self,
        *,
        node_limit: int = _MAP_NODE_LIMIT,
        include_external: bool = True,
        show_tests: bool = False,
        show_noncode: bool = False,
    ) -> ModuleMap:
        """Modules, their dependencies, and the subsystems they fall into.

        Tests and non-code files are excluded by default. On the real graph 69 of 126
        modules are tests and another twelve are CI YAML, a Dockerfile and TOML — 64% of
        the picture, mirroring the production structure or participating in no import
        graph at all. Both are counted and reported rather than silently dropped.
        """
        from code_atlas.server.analysis import (  # noqa: PLC0415
            build_module_graph,
            fetch_first_hop_external,
        )

        unavailable = self._unsupported_reason()
        if unavailable:
            return _empty_map(self._project, unavailable)

        module_graph = await build_module_graph(self._graph, self._project, "", test_patterns=self._test_patterns)
        if not module_graph.modules:
            return _empty_map(self._project, "", caveat=CoverageCaveat(note="No modules indexed for this project."))

        hidden_tests, hidden_noncode = 0, 0
        excluded: set[str] = set()
        for qn, info in module_graph.modules.items():
            path = str(info.get("file_path") or "")
            if not show_tests and _is_test_module(path, str(info.get("name") or ""), self._test_patterns):
                excluded.add(qn)
                hidden_tests += 1
            elif not show_noncode and _is_noncode_module(path):
                excluded.add(qn)
                hidden_noncode += 1

        community_of = {qn: idx for idx, group in enumerate(module_graph.partition) for qn in group}
        visible_partition = [[qn for qn in group if qn not in excluded] for group in module_graph.partition]
        kept = _largest_first(visible_partition, node_limit)
        truncated = len(kept) < len(module_graph.modules) - len(excluded)

        external_rows = await fetch_first_hop_external(self._graph, self._project) if include_external else []
        external = _external_nodes(external_rows, kept)

        # Position is a function of the edges now (ATL-123). The clustered ring it
        # replaces put every node in a band and encoded nothing.
        positions = force_layout(sorted(kept), {e: w for e, w in module_graph.edges.items() if set(e) <= kept})
        positions.update(_external_positions(external))

        # Undirected for size: a module coupled to ten others is equally central
        # whichever way the arrows point, and counting in+out would double a mutual pair.
        degree = _degree_by_module(module_graph.edges, kept)
        nodes = tuple(
            MapNode(
                id=qn,
                label=breadcrumb(
                    qualified_name=qn,
                    file_path=str(module_graph.modules[qn].get("file_path") or ""),
                    label="Module",
                ).short,
                community=community_of.get(qn, -1),
                size=node_size(degree.get(qn, 0)),
                x=positions.get(qn, (0.0, 0.0))[0],
                y=positions.get(qn, (0.0, 0.0))[1],
                project=self._project,
            )
            for qn in sorted(kept)
        ) + tuple(
            MapNode(
                id=qn,
                label=qn.rsplit(".", 1)[-1],
                community=-1,
                size=node_size(1),
                x=positions.get(qn, (0.0, 0.0))[0],
                y=positions.get(qn, (0.0, 0.0))[1],
                project=owner,
                is_external=True,
            )
            for qn, owner in sorted(external.items())
        )

        edges = _map_edges(module_graph.directed, kept, community_of) + _external_edges(external_rows, kept, external)

        communities = tuple(
            CommunityRef(
                id=idx,
                size=len(group),
                label=_community_label(group),
                members=tuple(sorted(group)[:_COMMUNITY_MEMBER_LIMIT]),
            )
            for idx, group in enumerate(module_graph.partition)
            if len(group) >= 2
        )

        return ModuleMap(
            project=self._project,
            nodes=nodes,
            edges=edges,
            communities=communities,
            modularity=_modularity_of(module_graph),
            truncated=truncated,
            hidden_tests=hidden_tests,
            hidden_noncode=hidden_noncode,
            caveat=_map_caveat(len(module_graph.modules), truncated, node_limit),
        )

    def _unsupported_reason(self) -> str:
        """Why this backend cannot produce the map, or empty if it can.

        Checked before any query rather than caught after one: the failure on SQLite is a
        missing capability, not a runtime error, and a half-drawn map is worse than none
        because a map with modules silently missing still looks complete.
        """
        from code_atlas.backends.sqlite_graph import SqliteGraphClient  # noqa: PLC0415

        if isinstance(self._graph, SqliteGraphClient):
            return (
                "Community detection is not available on the SQLite backend — the module inventory "
                "and module-pair CALLS aggregation it clusters are still raw Cypher reads. "
                "Run against Memgraph to see the map."
            )
        return ""


def _modularity_of(module_graph: ModuleGraph) -> float:
    """Partition quality, from the same function ``find_communities`` reports."""
    from code_atlas.server.analysis import _modularity  # noqa: PLC0415

    return round(_modularity(module_graph.partition, module_graph.edges), 4)


# Extensions that carry no import graph: they are indexed on purpose and belong in
# search, but a dependency map draws them as isolated dots. On the real project the CI
# workflows even formed their own "community", which is noise in a picture about coupling.
_NONCODE_SUFFIXES = (
    ".yml",
    ".yaml",
    ".toml",
    ".json",
    ".ini",
    ".cfg",
    ".lock",
    ".md",
    ".txt",
    ".dockerfile",
)
_NONCODE_NAMES = ("dockerfile", "containerfile", "makefile")


def _is_noncode_module(file_path: str) -> bool:
    path = file_path.replace("\\", "/").lower()
    leaf = path.rsplit("/", 1)[-1]
    return leaf in _NONCODE_NAMES or path.endswith(_NONCODE_SUFFIXES)


def _is_test_module(file_path: str, name: str, patterns: tuple[str, ...]) -> bool:
    """Whether a module is test code.

    Delegates to the same matcher `exclude_tests` uses in the MCP tools. A second notion
    of "is a test" would drift from the first, and then the map and the tools would
    disagree about the same file.
    """
    from code_atlas.search.engine import matches_test_pattern  # noqa: PLC0415
    from code_atlas.settings import SearchSettings  # noqa: PLC0415

    effective = list(patterns) if patterns else list(SearchSettings().test_patterns)
    return matches_test_pattern(file_path, name, effective)


def _empty_map(project: str, unavailable: str, *, caveat: CoverageCaveat | None = None) -> ModuleMap:
    return ModuleMap(
        project=project,
        nodes=(),
        edges=(),
        communities=(),
        modularity=0.0,
        truncated=False,
        caveat=caveat or CoverageCaveat(note=unavailable),
        unavailable=unavailable,
    )


def _largest_first(partition: list[list[str]], node_limit: int) -> set[str]:
    """The modules that fit, taking whole communities largest-first.

    Truncating by community rather than by module keeps every drawn subsystem complete.
    Slicing a flat list would cut communities in half and show a subsystem missing the
    modules that explain it — a picture that is not merely partial but actively
    misleading, because the gap is invisible.
    """
    kept: set[str] = set()
    for group in partition:
        if len(kept) + len(group) > node_limit:
            continue
        kept.update(group)
    return kept


def _degree_by_module(edges: dict[tuple[str, str], float], kept: set[str]) -> dict[str, int]:
    degree: dict[str, int] = {}
    for a, b in edges:
        if a in kept and b in kept:
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
    return degree


def _map_edges(
    directed: dict[tuple[str, str], float], kept: set[str], community_of: dict[str, int]
) -> tuple[MapEdge, ...]:
    """``source`` depends on ``target`` — a real orientation, not a sort order.

    This reads the directed view. It previously read the undirected one, whose key is
    ``(a, b) if a < b else (b, a)``, so every rendered edge pointed alphabetically: on the
    real graph all 615 had ``source < target``. Any arrowhead drawn from that was
    dictionary order wearing the costume of a dependency.
    """
    return tuple(
        MapEdge(
            source=depender,
            target=dependency,
            weight=round(weight, 4),
            crosses_community=community_of.get(depender, -1) != community_of.get(dependency, -2),
        )
        for (depender, dependency), weight in sorted(directed.items())
        if depender in kept and dependency in kept
    )


def _external_nodes(rows: list[dict[str, Any]], kept: set[str]) -> dict[str, str]:
    """``module_qn -> owning project`` for first-hop modules outside this project."""
    found: dict[str, str] = {}
    for row in rows:
        source = str(row.get("from_mod") or "")
        target = str(row.get("to_mod") or "")
        owner = str(row.get("to_project") or "")
        if source in kept and target and owner:
            found[target] = owner
    return found


def _external_edges(rows: list[dict[str, Any]], kept: set[str], external: dict[str, str]) -> tuple[MapEdge, ...]:
    """Edges reaching out of the project, deduplicated per pair.

    Weight is fixed at 1.0 rather than summed: these come from IMPORTS, which carries no
    ADR-0017 weight, and inventing one would make a cross-project edge look better
    evidenced than a call edge that actually was measured.
    """
    pairs = {
        (str(row.get("from_mod") or ""), str(row.get("to_mod") or ""))
        for row in rows
        if str(row.get("from_mod") or "") in kept and str(row.get("to_mod") or "") in external
    }
    return tuple(MapEdge(source=a, target=b, weight=1.0, crosses_community=True) for a, b in sorted(pairs))


def _external_positions(external: dict[str, str]) -> dict[str, tuple[float, float]]:
    """Park external modules on an outer ring, clear of the project's own clusters."""
    if not external:
        return {}
    step = 2 * math.pi / len(external)
    return {
        qn: (_EXTERNAL_RING * math.cos(i * step), _EXTERNAL_RING * math.sin(i * step))
        for i, qn in enumerate(sorted(external))
    }


def _community_label(group: list[str]) -> str:
    """Name a subsystem by the longest package prefix its modules share.

    A generated name beats an integer id — "community 3" tells a reader nothing they can
    act on. Falls back to the first member when the modules share no prefix at all.
    """
    if not group:
        return "empty"
    parts = [qn.split(".") for qn in sorted(group)]
    shared: list[str] = []
    for segments in zip(*parts, strict=False):
        if len(set(segments)) != 1:
            break
        shared.append(segments[0])
    return ".".join(shared) if shared else sorted(group)[0]


def _map_caveat(module_count: int, truncated: bool, node_limit: int) -> CoverageCaveat:
    if truncated:
        return CoverageCaveat(
            note=(
                f"{module_count} modules indexed; showing the largest communities that fit within "
                f"{node_limit} nodes. Communities are kept whole, so the cut falls between subsystems."
            )
        )
    return CoverageCaveat(note=f"Clustered over {module_count} modules.")


class ImpactViewService:
    """ "What breaks if I change this", and "how do these two connect".

    Both views delegate to :mod:`code_atlas.server.analysis` — the same functions the
    ``blast_radius`` and ``trace_path`` MCP tools call. Re-implementing either traversal
    in Cypher here would let the UI and the tool drift apart on the same question, and the
    UI would be the one nobody notices was wrong.
    """

    def __init__(self, graph: GraphBackend, project: str, *, test_patterns: tuple[str, ...] = ()) -> None:
        self._graph = graph
        self._project = project
        self._test_patterns = test_patterns

    async def blast(
        self,
        uid: str,
        *,
        direction: str = "callers",
        max_depth: int = 3,
        limit: int = _IMPACT_PAGE,
        resolved_only: bool = False,
    ) -> BlastRadiusView:
        """The dependency closure around *uid*, grouped by distance.

        Per ADR-0029 this traverses dependency edges only — DEFINES and CONTAINS are
        excluded, because counting containment makes "what does changing this method
        affect" mean nothing.
        """
        from code_atlas.server.analysis import blast_radius  # noqa: PLC0415

        # Fetched at the view's own ceiling rather than at `limit`, so the resolved-only
        # filter runs over the whole considered set instead of over one page. Filtering a
        # page and paging a filtered set give different answers, and the second is the one
        # a reader assumes they are looking at.
        result = await blast_radius(
            self._graph,
            uid,
            direction=direction,
            max_depth=max_depth,
            edge_types=_DEFAULT_BLAST_EDGE_TYPES,
            limit=_IMPACT_CEILING,
            test_patterns=self._test_patterns,
        )
        if result.get("error"):
            return _blast_error(uid, direction, max_depth, str(result["error"]))

        considered = [_as_affected(row) for row in result.get("affected", [])]
        kept = [e for e in considered if not e.ambiguous_only] if resolved_only else considered
        page = kept[:limit]

        return BlastRadiusView(
            uid=uid,
            target_name=_target_name(uid),
            direction=direction,
            max_depth=max_depth,
            groups=_group_by_depth(page),
            affected_count=int(result.get("affected_count") or 0),
            shown=len(page),
            considered=len(considered),
            resolved_only=resolved_only,
            truncated=len(page) < len(kept),
            remedy="Raise the limit, lower the depth, or narrow the direction.",
            caveat=_impact_caveat(result, considered, resolved_only),
        )

    async def trace(self, from_uid: str, to_uid: str, *, max_depth: int = 6) -> TracePathView:
        """The shortest path between two entities, hop by hop."""
        from code_atlas.server.analysis import trace_path  # noqa: PLC0415

        result = await trace_path(self._graph, from_uid, to_uid, max_depth=max_depth)
        if result.get("error"):
            return TracePathView(from_uid=from_uid, to_uid=to_uid, found=False, error=str(result["error"]))
        if not result.get("found"):
            return TracePathView(
                from_uid=from_uid,
                to_uid=to_uid,
                found=False,
                message=str(result.get("message") or f"No path within {max_depth} hops."),
            )

        return TracePathView(
            from_uid=from_uid,
            to_uid=to_uid,
            found=True,
            hops=tuple(_as_hop(hop) for hop in result.get("hops", [])),
            hop_count=result.get("hop_count"),
            path_weight=result.get("path_weight"),
        )


def _as_affected(row: dict[str, Any]) -> AffectedEntity:
    return AffectedEntity(
        uid=str(row.get("uid") or ""),
        name=str(row.get("name") or ""),
        qualified_name=str(row.get("qualified_name") or ""),
        label=str(row.get("label") or ""),
        file_path=str(row.get("file_path") or ""),
        depth=int(row.get("min_depth") or 0),
        via=tuple(str(v) for v in (row.get("via") or []) if v),
        via_lines=tuple(int(v) for v in (row.get("via_lines") or []) if isinstance(v, int)),
        ambiguous_only=bool(row.get("ambiguous_only", False)),
        test_only=bool(row.get("test_only", False)),
        confidence_score=float(row.get("confidence_score", 1.0) or 0.0),
    )


def _as_hop(hop: dict[str, Any]) -> PathHop:
    source = hop.get("from") or {}
    target = hop.get("to") or {}
    return PathHop(
        from_uid=str(source.get("uid") or ""),
        from_name=str(source.get("name") or ""),
        to_uid=str(target.get("uid") or ""),
        to_name=str(target.get("name") or ""),
        edge_type=str(hop.get("edge_type") or ""),
        confidence=str(hop.get("confidence") or ""),
        strategy=str(hop.get("strategy") or ""),
        weight=_as_float(hop.get("weight")),
        at_line=_as_int(hop.get("at_line")),
        from_test=bool(hop.get("from_test", False)),
    )


def _group_by_depth(entities: list[AffectedEntity]) -> tuple[DepthGroup, ...]:
    """Nearest first. Distance is the strongest signal of how likely a break is."""
    by_depth: dict[int, list[AffectedEntity]] = {}
    for entity in entities:
        by_depth.setdefault(entity.depth, []).append(entity)
    return tuple(DepthGroup(depth=d, entities=tuple(by_depth[d])) for d in sorted(by_depth))


def _target_name(uid: str) -> str:
    """A readable name for the analysed entity.

    The traversal never returns the target itself, so there is nothing to read a real
    name from — the uid's own tail is the honest fallback rather than a second lookup.
    """
    return uid.rsplit(":", 1)[-1] or uid


def _blast_error(uid: str, direction: str, max_depth: int, error: str) -> BlastRadiusView:
    return BlastRadiusView(
        uid=uid,
        target_name=uid.rsplit(":", 1)[-1] or uid,
        direction=direction,
        max_depth=max_depth,
        groups=(),
        affected_count=0,
        shown=0,
        considered=0,
        resolved_only=False,
        truncated=False,
        remedy="",
        caveat=CoverageCaveat(note=error),
        error=error,
    )


def _impact_caveat(result: dict[str, Any], considered: list[AffectedEntity], resolved_only: bool) -> CoverageCaveat:
    """Say what the closure did and did not cover.

    ``ambiguous_only`` is a heuristic, not a guarantee — an edge type carrying no
    confidence property always counts as not-resolved — so a filtered list must not read
    as a verified one.
    """
    total = int(result.get("affected_count") or 0)
    if not total:
        return CoverageCaveat(note="Nothing depends on this entity within the traversed depth.")

    parts = [f"{total} affected within depth {result.get('max_depth')}."]
    if len(considered) < total:
        parts.append(f"The nearest {len(considered)} were examined.")
    if resolved_only:
        dropped = sum(1 for e in considered if e.ambiguous_only)
        parts.append(
            f"{dropped} reached only by a guessed path are hidden — that flag is a heuristic, "
            "so treat what remains as better-evidenced, not verified."
        )
    return CoverageCaveat(note=" ".join(parts))
