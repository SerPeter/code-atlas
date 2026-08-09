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

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import msgspec
from loguru import logger

from code_atlas.server.analysis import _DEFAULT_BLAST_EDGE_TYPES
from code_atlas.server.web.kinds import KINDS, classify
from code_atlas.server.web.layout import force_layout
from code_atlas.server.web.naming import SEPARATOR, breadcrumb
from code_atlas.server.web.schemas import (
    AffectedEntity,
    ArchitectureHealth,
    ArchitectureTrend,
    ArchitectureView,
    BlastRadiusView,
    ChannelFilter,
    CommunityRef,
    CoverageCaveat,
    CycleDetail,
    CycleRow,
    DepthGroup,
    DetailEvidenceMix,
    DetailRelated,
    DetailView,
    DsmCell,
    DsmRow,
    EdgeEvidence,
    EntityDetail,
    ImpactHopGroup,
    ImpactRoot,
    ImpactRow,
    ImpactView,
    KindDef,
    KindFilter,
    KindTally,
    MapEdge,
    MapNode,
    MapPayload,
    MetricCard,
    PageChrome,
    PathHop,
    ProjectChoice,
    ProjectOverview,
    ProjectPicker,
    ProjectRef,
    ReferenceRow,
    RelatedEntity,
    ScopeOption,
    SearchHit,
    SearchPage,
    SearchRow,
    SearchView,
    TracePathView,
    TrendPoint,
    TrendRow,
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
# The design's 1,500-node cap: past it the map truncates and says so.
_MAP_NODE_LIMIT = 1500
_ENTITY_SCOPE_LIMIT = 1500
# The page a reader sees, and the ceiling the resolved-only filter runs over.
_IMPACT_PAGE = 50
_IMPACT_CEILING = 500

# Containment anchors weigh far less than any real call in the layout's attraction
# term — the call graph, not the containment tree, should decide where things sit.
_DEFINES_WEIGHT = 0.25

# Full-scope entity defaults: value-and-documentation kinds start hidden so the first
# paint is a picture of the call structure rather than a wall. Hidden is COUNTED — the
# kind rows in the rail carry the numbers and the toggles that bring them back.
_FULL_SCOPE_HIDDEN = ("constant", "env_var", "doc_file", "doc_section", "knowledge_note")
# The skeleton kinds a filter can never remove: containers anchor everything else, and
# classes are the fold target — hiding them would vanish their folded methods silently.
_UNHIDEABLE_KINDS = frozenset({"module", "package", "class", "method"})
_ENTITY_FULL_LIMIT = 4000

# Full-scope layouts are the one expensive computation in this module (minutes of numpy
# at thousands of nodes); they are deterministic per graph, so the last few are kept.
_LAYOUT_CACHE: dict[tuple, dict[str, tuple[float, float]]] = {}
_LAYOUT_CACHE_MAX = 8

# ADR-0028's four states, strongest first. Shared by every view that draws a chip.
_EV_RANK = {"structural": 3, "resolved": 2, "guessed": 1, "unknown": 0}

_CHANNELS = ("graph", "keyword", "semantic")


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
    project**, with other projects reachable but not loaded.
    """

    def __init__(self, graph: GraphBackend, project: str) -> None:
        self._graph = graph
        self._project = project

    @property
    def project(self) -> str:
        return self._project

    async def overview(self) -> ProjectOverview:
        """The project's headline numbers — the chrome and the export read these."""
        statuses = [_project_props(row) for row in await self._graph.get_project_status()]
        current = next((s for s in statuses if s.get("name") == self._project), None)
        if current is None:
            raise ProjectNotIndexedError(self._project)

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


class ChromeService:
    """Header chip, indexed note and footer — the values every page shows.

    Never raises: an unindexed project still gets a shell, because the shell is where
    the "run atlas index" explanation lives.
    """

    def __init__(self, graph: GraphBackend, projects: tuple[str, ...]) -> None:
        self._graph = graph
        self._projects = projects or ("",)

    async def chrome(self) -> PageChrome:
        try:
            rows = await self._graph.get_project_status()
            statuses = {str(s.get("name") or ""): s for s in map(_project_props, rows)}
        except Exception:  # the shell must render even when the backend is down
            statuses = {}
        chosen = [statuses[name] for name in self._projects if name in statuses]

        if len(chosen) == 1:
            current = chosen[0]
            name = str(current.get("name") or self._projects[0])
            entities = int(current.get("entity_count") or 0)
            # Relative when the stored stamp is numeric; the absolute string otherwise —
            # older rows and the SQLite path hold ISO strings, and "Not indexed" over a
            # real index would be the confident wrong answer.
            ago = _ago(current.get("last_indexed_at")) or _as_timestamp(current.get("last_indexed_at"))
            indexed_note = f"indexed {ago}" if ago else "not indexed yet"
            commit = str(current.get("git_hash") or "")[:7]
            footer = f"Indexed {ago}" if ago else "Not indexed"
            if commit:
                footer += f" · commit {commit}"
            meta = f"{entities:,} entities"
        elif chosen:
            name = f"{len(chosen)} projects"
            entities = sum(int(c.get("entity_count") or 0) for c in chosen)
            indexed_note = "mixed index times"
            footer = f"Indexed at mixed times · {len(chosen)} projects"
            meta = f"{entities:,} entities combined"
        else:
            name = self._projects[0] or "no project"
            indexed_note = "not indexed yet"
            footer = "Not indexed"
            meta = "no entities"

        module_count = await self._module_count()
        coverage = (
            f"Computed over {module_count} modules. Languages with incomplete extraction make this a lower bound."
            if module_count
            else "Nothing indexed yet — run `atlas index` to build the graph."
        )

        return PageChrome(
            project_name=name,
            project_meta=meta,
            indexed_note=indexed_note,
            footer_indexed=footer,
            coverage_note=coverage,
            backend_note=_backend_note(self._graph),
        )

    async def _module_count(self) -> int:
        try:
            structure = await self._graph.get_structure_overview(self._projects[0], "", _STRUCTURE_LIMIT)
        except Exception:
            return 0
        return sum(int(row.get("cnt") or 0) for row in structure.get("counts", []) if str(row.get("label")) == "Module")


def _backend_note(graph: GraphBackend) -> str:
    from code_atlas.backends.sqlite_graph import SqliteGraphClient  # noqa: PLC0415

    if isinstance(graph, SqliteGraphClient):
        return "Backend sqlite · map and impact need Memgraph"
    return "Backend memgraph · all views available"


def _project_props(row: dict[str, Any]) -> dict[str, Any]:
    """Unwrap a ``get_project_status`` row into the Project node's own properties.

    Both backends return ``[{"n": <node>}]`` — Memgraph from ``RETURN n``, SQLite from
    ``_row_to_node`` — and the properties are ``name``/``entity_count``/
    ``last_indexed_at``, not ``project``/``entities``/``indexed_at``. Reading the wrapper
    dict directly finds none of them, so every project compared unequal and the landing
    page 404'd as "not indexed" against a fully indexed graph.
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
    value is a float. Anything that is not a number is passed through unchanged, since
    older rows and the SQLite path may hold an ISO string already.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return datetime.fromtimestamp(float(value), tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    return str(value)


def _ago(value: object) -> str:
    """ "2 hours ago" — the design's header and footer speak in relative time."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return ""
    seconds = max(0.0, (datetime.now(tz=UTC) - datetime.fromtimestamp(float(value), tz=UTC)).total_seconds())
    if seconds < 90:
        return "just now"
    minutes = seconds / 60
    if minutes < 90:
        return f"{round(minutes)} minutes ago"
    hours = minutes / 60
    if hours < 36:
        return f"{round(hours)} hour{'s' if round(hours) != 1 else ''} ago"
    days = hours / 24
    if days < 14:
        return f"{round(days)} day{'s' if round(days) != 1 else ''} ago"
    weeks = days / 7
    return f"{round(weeks)} weeks ago"


def _coverage_caveat(label_counts: dict[str, int]) -> CoverageCaveat:
    """State what these numbers do not cover."""
    return CoverageCaveat(note="" if label_counts else "No entities indexed for this project.")


class EntityNotFoundError(LookupError):
    """The requested uid is not in the graph."""

    def __init__(self, uid: str) -> None:
        super().__init__(uid)
        self.uid = uid


def _evidence_state(evidence: EdgeEvidence | None) -> str:
    """An edge's ADR-0028 state, for the chip that renders it.

    ``None`` — the edge was not found among the module's annotated edges — is
    **unknown**: evidence was never looked up, which must never render like "there is
    nothing". A structural relationship (IMPORTS, DEFINES, INHERITS…) is a fact. A
    CALLS edge with no recorded strategy or confidence was likewise never looked up.
    """
    if evidence is None:
        return "unknown"
    if evidence.rel_type and evidence.rel_type != "CALLS":
        return "structural"
    if evidence.confidence == "ambiguous":
        return "guessed"
    if evidence.strategy or evidence.confidence:
        return "resolved"
    return "unknown"


class SearchViewService:
    """Search, and the entity detail a result leads to."""

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
        """Ranked results for *query*, scoped to this project — the JSON contract.

        Ranking is whatever :func:`hybrid_search` produces — the same call the MCP tool
        makes. A UI that re-ranked would give a human a different answer from the agent
        looking at the same graph.
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

    async def search_view(
        self,
        query: str,
        *,
        channels: tuple[str, ...] = _CHANNELS,
        kinds: tuple[str, ...] = (),
        limit: int = 20,
        entities: int = 0,
    ) -> SearchView:
        """The search page, in the design's shape.

        Filters narrow what was **fetched**; they cannot reveal matches the search did
        not retrieve, so every count here names the fetched set and never a total.
        """
        page = await self.search(query, limit=limit)
        hits = list(page.hits)
        active_channels = tuple(c for c in _CHANNELS if c in channels) or _CHANNELS
        shown = [h for h in hits if any(c in active_channels for c in h.channels) and (not kinds or h.kind in kinds)]

        rows = tuple(
            SearchRow(
                uid=h.uid,
                label=breadcrumb(qualified_name=h.qualified_name, file_path=h.file_path, kind=h.kind).full or h.name,
                kind=h.kind or h.label.lower(),
                loc=_loc(h.file_path, h.line_start),
                sig=h.signature or h.qualified_name,
                channels=h.channels,
                strength={3: "found three ways", 2: "found two ways"}.get(len(h.channels), "found one way"),
                score=f"{h.score:.2f}",
            )
            for h in shown
        )

        if not query.strip():
            note = "Type a query to search this project."
        elif len(shown) == len(hits):
            note = (
                f"{len(hits)} results fetched · more exist, quantity unknown"
                if page.more_available
                else f"{len(hits)} results fetched"
            )
        else:
            note = f"{len(shown)} of {len(hits)} fetched results shown · more exist, quantity unknown"

        def _channel_url(channel: str) -> str:
            if channel in active_channels:
                active = [c for c in active_channels if c != channel]
            else:
                active = [*active_channels, channel]
            return _search_url(query, tuple(c for c in _CHANNELS if c in active), kinds)

        seen_kinds = sorted({h.kind for h in hits if h.kind})

        def _kind_url(kind: str) -> str:
            active = tuple(k for k in kinds if k != kind) if kind in kinds else (*kinds, kind)
            return _search_url(query, active_channels, active)

        return SearchView(
            query=query,
            rows=rows,
            channels=tuple(
                ChannelFilter(
                    id=c,
                    label=c.capitalize(),
                    count=f"{sum(1 for h in hits if c in h.channels)} hits",
                    on=c in active_channels,
                    url=_channel_url(c),
                )
                for c in _CHANNELS
            ),
            kind_filters=tuple(
                KindFilter(
                    id=k,
                    label_count=f"{k} · {sum(1 for h in hits if h.kind == k)}",
                    on=k in kinds,
                    url=_kind_url(k),
                )
                for k in seen_kinds
            ),
            result_note=note,
            searching_note=(
                f"Searching {self._project} · {entities:,} entities" if entities else f"Searching {self._project}"
            ),
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

    async def detail_view(self, uid: str) -> DetailView:
        """The entity page, in the design's shape."""
        detail = await self.detail(uid)
        crumb = breadcrumb(
            qualified_name=detail.qualified_name, file_path=detail.file_path, kind=detail.kind, label=detail.label
        )
        lines = (
            f"{detail.line_start}–{detail.line_end}"  # noqa: RUF001  # en dash: a range, not a hyphen
            if detail.line_start and detail.line_end
            else str(detail.line_start or "")
        )

        def related(row: RelatedEntity) -> DetailRelated:
            return DetailRelated(
                uid=row.uid,
                label=breadcrumb(qualified_name=row.qualified_name, file_path=row.file_path, kind=row.kind).full
                or row.name,
                rel=(row.evidence.rel_type.lower() if row.evidence and row.evidence.rel_type else "calls"),
                ev=_evidence_state(row.evidence),
            )

        callers = tuple(related(r) for r in detail.callers)
        callees = tuple(related(r) for r in detail.callees)
        unknown_callers = sum(1 for c in callers if c.ev == "unknown")
        parent_crumb = (
            breadcrumb(
                qualified_name=detail.parent.qualified_name,
                file_path=detail.parent.file_path,
                kind=detail.parent.kind,
            ).full
            if detail.parent
            else ""
        )

        return DetailView(
            uid=detail.uid,
            name=crumb.full or detail.qualified_name,
            short_name=detail.name or crumb.symbol,
            kind=detail.kind or detail.label.lower(),
            file=detail.file_path,
            lines=f"lines {lines}" if lines else "",
            file_lines=f"{detail.file_path} · lines {lines}" if lines else detail.file_path,
            parent_line=f"Defined in {parent_crumb}" if parent_crumb else "",
            signature=detail.signature,
            paragraphs=tuple(p.strip() for p in detail.docstring.split("\n\n") if p.strip()),
            callers=callers,
            callees=callees,
            docs=tuple(
                DetailRelated(uid=d.uid, label=d.qualified_name or d.name, rel="documents", ev="structural")
                for d in detail.docs
            ),
            evidence_mix=tuple(
                DetailEvidenceMix(ev=state, n=sum(1 for c in callers if c.ev == state))
                for state in ("structural", "resolved", "guessed", "unknown")
            ),
            caller_note=(
                f"{unknown_callers} of {len(callers)} callers carry no evidence — "
                "they were not looked up, not disproved."
                if callers
                else "No callers indexed. Callers reached only through dynamic dispatch are not counted."
            ),
            callers_count_note=f"{len(callers)} indexed",
            callees_count_note=f"{len(callees)} indexed",
        )

    async def _edge_evidence(self, file_path: str) -> dict[tuple[str, str], EdgeEvidence]:
        """``(from_qn, to_qn) -> evidence`` for edges around *file_path*'s module."""
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


def _loc(file_path: str, line: int | None) -> str:
    leaf = file_path.replace("\\", "/").rsplit("/", 1)[-1]
    return f"{leaf}:{line}" if line else leaf


def _search_url(query: str, channels: tuple[str, ...], kinds: tuple[str, ...]) -> str:
    from urllib.parse import urlencode  # noqa: PLC0415

    params: list[tuple[str, str]] = [("q", query)]
    if tuple(channels) != _CHANNELS:
        params.extend(("channel", c) for c in channels)
    params.extend(("kind", k) for k in kinds)
    return "/search?" + urlencode(params)


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
        # list that is not sorted by the number beside it.
        score=round(result.ranked_score, 4),
        channels=tuple(sorted(_CHANNEL_NAMES.get(s, s) for s in result.sources)),
    )


# The engine's channel ids, translated to the names a reader sees. "bm25" describes
# the algorithm; "keyword" describes what it does — and the rail's toggles, the row
# dots and the hit counts must all speak the same three words.
_CHANNEL_NAMES = {"bm25": "keyword", "vector": "semantic"}


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
    """

    # Published propagation-cost measurements, for scale only. They were computed on
    # other languages and other tools; the rail says to treat them as order-of-magnitude.
    _REFERENCES = (
        ReferenceRow(label="A large browser codebase, before refactoring", value="~17%"),
        ReferenceRow(label="The same codebase, after", value="~2%"),
        ReferenceRow(label="Linux kernel", value="~0.3%"),
    )

    def __init__(self, graph: GraphBackend, project: str) -> None:
        self._graph = graph
        self._project = project

    async def health(self, *, dsm_limit: int = _DSM_LIMIT) -> ArchitectureHealth:
        """The JSON contract — unchanged, still computed over the import graph."""
        from code_atlas.server.architecture import analyse  # noqa: PLC0415

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

    async def view(self, *, dsm_limit: int = _DSM_LIMIT) -> ArchitectureView:
        """The architecture page, in the design's shape.

        Built on the same module graph as the map (CALLS + IMPORTS, per-pair evidence)
        so the matrix and the picture cannot disagree. On SQLite that graph is not
        available and the page says so rather than drawing a partial one.
        """
        from code_atlas.server.architecture import analyse  # noqa: PLC0415

        source = await self._edge_source()
        if isinstance(source, str):
            return ArchitectureView(
                cards=(),
                dsm_rows=(),
                dsm_caption="",
                cycles=(),
                trend_rows=(),
                references=self._REFERENCES,
                unavailable=source,
            )
        pair_ev, module_paths = source

        edges = sorted(pair_ev)
        nodes = sorted({n for e in edges for n in e})
        if not nodes:
            return ArchitectureView(
                cards=(),
                dsm_rows=(),
                dsm_caption="",
                cycles=(),
                trend_rows=(),
                references=self._REFERENCES,
                unavailable="No module dependencies indexed — nothing to measure.",
            )
        metrics = analyse(nodes, edges)

        # The matrix shows the most connected modules; ordering within the shown set
        # still comes from the dependency sort, so cycles stay above the diagonal.
        degree: dict[str, int] = dict.fromkeys(nodes, 0)
        for a, b in edges:
            degree[a] += 1
            degree[b] += 1
        top = set(sorted(nodes, key=lambda n: -degree[n])[:dsm_limit])
        shown = [n for n in metrics.order if n in top]
        position = {name: i for i, name in enumerate(shown)}

        label_of = {n: _module_map_label(n, module_paths.get(n, "")) for n in nodes}
        mark_ev: dict[tuple[int, int], str] = {}
        for a, b in edges:
            if a in position and b in position:
                mark_ev[position[a], position[b]] = pair_ev[a, b]

        dsm_rows = tuple(
            DsmRow(
                label=label_of[row],
                cells=tuple(
                    _dsm_cell(r, c, mark_ev.get((r, c)), label_of[row], label_of[col]) for c, col in enumerate(shown)
                ),
            )
            for r, row in enumerate(shown)
        )

        back_edges = [
            (a, b, pair_ev[a, b]) for a, b in edges if a in position and b in position and position[b] > position[a]
        ]
        cycles = tuple(
            CycleRow(
                members=f"{label_of[a]}  ⇄  {label_of[b]}",
                closing=f"{label_of[a]} → {label_of[b]}  ·  {ev}",
                closing2=f"{label_of[b]} → {label_of[a]}  ·  {pair_ev.get((b, a), 'unknown')}",
                note="Both directions are indexed; cutting either breaks the cycle.",
                ev=ev,
            )
            for a, b, ev in back_edges[:10]
        )

        fan_in: dict[str, int] = dict.fromkeys(nodes, 0)
        for _a, b in edges:
            fan_in[b] += 1
        total_fan = sum(fan_in.values())
        top5 = sum(sorted(fan_in.values(), reverse=True)[:5])
        core_modules = round(metrics.core_size * metrics.module_count)

        cards = (
            MetricCard(
                label="Propagation cost",
                value=f"{metrics.propagation_cost * 100:.1f}%",
                note=(
                    f"Share of module pairs where one can reach the other. Computed over "
                    f"{metrics.module_count} modules; any language whose extraction is incomplete "
                    "makes this a lower bound."
                ),
            ),
            MetricCard(
                label="Cycles",
                value=f"{len(back_edges)} cycle{'s' if len(back_edges) != 1 else ''}",
                note=(
                    (
                        f"Ten of the {len(back_edges)} are listed below with both of their edges."
                        if len(back_edges) > 10
                        else "Each is listed below with both of its edges."
                    )
                    if back_edges
                    else "No dependency cycles among the modules shown."
                ),
            ),
            MetricCard(
                label="Core size",
                value=f"{core_modules} module{'s' if core_modules != 1 else ''}",
                note="Modules that both depend on and are depended on by the bulk of the system.",
            ),
            MetricCard(
                label="Fan-in concentration",
                value=f"{round(top5 / max(1, total_fan) * 100)}%",
                note="Share of all incoming dependencies landing on the five most depended-upon modules.",
            ),
        )

        return ArchitectureView(
            cards=cards,
            dsm_rows=dsm_rows,
            dsm_caption=(
                f"Showing {len(shown)} of {metrics.module_count} modules, the most connected. "
                "Rows depend on columns; a mark above the diagonal closes a cycle."
            ),
            cycles=cycles,
            cycles_caption=(
                f"read off the matrix — {len(back_edges)} above the diagonal, ten listed, both edges named"
                if len(back_edges) > 10
                else "read off the matrix — both edges named, cut either"
            ),
            trend_rows=await self._trend_rows(),
            references=self._REFERENCES,
        )

    async def _edge_source(self) -> tuple[dict[tuple[str, str], str], dict[str, str]] | str:
        """Directed module pairs with evidence plus module file paths, or a reason
        string when unavailable.

        Memgraph gets the full CALLS+IMPORTS graph the map draws. SQLite cannot serve
        the CALLS aggregation, but its IMPORT edges are real structural facts — so the
        page still works there, computed over imports alone.
        """
        from code_atlas.backends.sqlite_graph import SqliteGraphClient  # noqa: PLC0415
        from code_atlas.server.analysis import build_module_graph  # noqa: PLC0415

        if isinstance(self._graph, SqliteGraphClient):
            raw = await self._graph.get_module_import_edges(self._project, "")
            pairs = {
                (str(r.get("from_mod")), str(r.get("to_mod"))): "structural"
                for r in raw.get("direct", [])
                if r.get("from_mod") and r.get("to_mod") and r.get("from_mod") != r.get("to_mod")
            }
            return pairs, {}
        try:
            module_graph = await build_module_graph(self._graph, self._project, "")
        except Exception as exc:
            logger.debug("Architecture view unavailable for {}: {}", self._project, exc)
            return "The module graph could not be built. Run health_check for the backend's own account of why."
        paths = {qn: str(info.get("file_path") or "") for qn, info in module_graph.modules.items()}
        return {pair: module_graph.evidence.get(pair, "unknown") for pair in module_graph.directed}, paths

    async def _trend_rows(self) -> tuple[TrendRow, ...]:
        """Across index runs, with the coverage rule the design states verbatim:
        direction is unclear whenever the index itself changed size by more than a
        tenth — a metric that moved because extraction improved is not decay."""
        from code_atlas.server.architecture_history import load  # noqa: PLC0415

        try:
            snapshots = list(await load(self._graph, self._project))
        except Exception:  # a missing history costs the trend, never the view
            return ()
        snapshots.reverse()  # newest first, as the design's table reads

        rows = []
        for i, snap in enumerate(snapshots):
            prev = snapshots[i + 1] if i + 1 < len(snapshots) else None
            direction, accented = "—", False
            if prev is not None and prev.modules:
                grew = abs(snap.modules - prev.modules) / prev.modules
                delta = (snap.propagation_cost - prev.propagation_cost) * 100
                if grew > 0.1:
                    direction, accented = f"unclear — index grew {round(grew * 100)}%", True
                elif delta > 0.05:
                    direction, accented = f"worse +{delta:.1f} pts", True
                elif delta < -0.05:
                    direction = f"better −{abs(delta):.1f} pts"  # noqa: RUF001  # a real minus sign
                else:
                    direction = "unchanged"
            rows.append(
                TrendRow(
                    date=snap.at[:10],
                    commit=snap.commit[:7],
                    modules=f"{snap.modules} modules",
                    propagation=f"{snap.propagation_cost * 100:.1f}%",
                    largest_cycle=str(snap.largest_cycle),
                    direction=direction,
                    accented=accented,
                )
            )
        return tuple(rows)

    async def _trend(self) -> ArchitectureTrend | None:
        """How these numbers have moved across recorded index runs (JSON contract)."""
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


def _dsm_cell(r: int, c: int, ev: str | None, row_label: str, col_label: str) -> DsmCell:
    if ev is None:
        return DsmCell(mark="diag" if r == c else "")
    above = c > r
    return DsmCell(
        mark="cycle" if above else "dep",
        title=f"{row_label} → {col_label} ({ev}{', closes a cycle' if above else ''})",
    )


def _module_label(qn: str) -> str:
    return breadcrumb(qualified_name=qn, label="Module").short or qn


def _external_root(qn: str, known: dict[str, str]) -> str | None:
    """The drawn node an external link lands on: the symbol's root package.

    "ext/opentelemetry.sdk.trace.export.SpanExporter" belongs to
    "ext/opentelemetry"; a slash-shaped ref ("ext/actions/checkout") is its own
    package. Links to anything the inventory does not name are dropped rather than
    drawn against an invented node.
    """
    if qn in known:
        return known[qn]
    rest = qn.removeprefix("ext/")
    root = "ext/" + rest.split(".", 1)[0]
    return known.get(root)


def _doc_label(file_path: str) -> str:
    """A documentation file's breadcrumb: its directory and its name.

    Doc paths are not dotted qualified names, so the module breadcrumb logic would
    say things like "CHANGELOG, CHANGELOG.md" — one thing twice. A root-level file
    is just its own name, which for a README identifies plenty.
    """
    parts = file_path.replace("\\", "/").strip("/").split("/")
    if len(parts) == 1:
        return parts[0]
    return f"{parts[-2]}{SEPARATOR}{parts[-1]}"


def _module_map_label(qn: str, file_path: str) -> str:
    """``package``, separator, ``file.py`` — the design's module-level breadcrumb.

    Never a bare filename: ``conftest`` four times on one map identifies nothing. The
    package path keeps its root only when the module sits directly under it;
    deeper modules drop it ("parsing.ast, builder.py"), exactly as the reference
    labels its own nodes.
    """
    leaf = file_path.replace("\\", "/").rsplit("/", 1)[-1] or qn.rsplit(".", 1)[-1]
    package = qn.split(".")[:-1]
    if len(package) > 1:
        package = package[1:]
    if not package:
        return leaf
    return f"{'.'.join(package)}{SEPARATOR}{leaf}"


def _cycle_details(cycles: Sequence[Cycle], edges: Sequence[tuple[str, str]]) -> tuple[CycleDetail, ...]:
    """Attach to each cycle the edges that close it."""
    details: list[CycleDetail] = []
    for cycle in cycles:
        members = set(cycle.members)
        internal = sorted({(src, dst) for src, dst in edges if src in members and dst in members})
        details.append(CycleDetail(members=cycle.members, edges=tuple(internal)))
    return tuple(details)


def _architecture_caveat(module_count: int) -> CoverageCaveat:
    if module_count == 0:
        return CoverageCaveat(note="No module dependencies indexed - nothing to measure.")
    return CoverageCaveat(
        note=(
            f"Computed over {module_count} modules. Any language whose extraction is "
            "incomplete makes this a lower bound, not a ceiling."
        )
    )


class MapViewService:
    """The map — the design's two levels over the real graph.

    Module level aggregates the entity graph one node per module (the same clustering
    ``find_communities`` reports, so the map and the MCP tool cannot disagree). Entity
    level draws the graph database's own nodes, scoped to one module.
    """

    def __init__(self, graph: GraphBackend, project: str, *, test_patterns: tuple[str, ...] = ()) -> None:
        self._graph = graph
        self._project = project
        self._test_patterns = test_patterns

    async def map(  # noqa: PLR0912, PLR0915  # one pass over one table, so every count closes
        self,
        *,
        node_limit: int = _MAP_NODE_LIMIT,
        show_tests: bool = False,
        show_noncode: bool = False,
        show_external: bool = False,
        projects: tuple[str, ...] = (),
    ) -> MapPayload:
        """The module level: one node per module, edges rolled up, evidence carried.

        Tests and non-code files are hidden by default and **counted** — on the real
        graph they are 64% of the picture, mirroring the production structure or
        participating in no import graph at all.
        """
        from code_atlas.server.analysis import (  # noqa: PLC0415
            build_module_graph,
            fetch_doc_modules,
            fetch_external_imports,
            fetch_first_hop_external,
        )

        unavailable = self._unsupported_reason()
        if unavailable:
            return _empty_map(self._project, unavailable)

        selected = tuple(projects) or (self._project,)
        multi = len(selected) > 1

        nodes_all: dict[str, dict[str, Any]] = {}
        directed: dict[tuple[str, str], float] = {}
        evidence: dict[tuple[str, str], str] = {}
        undirected: dict[tuple[str, str], float] = {}
        communities: list[CommunityRef] = []
        community_of: dict[str, int] = {}
        partition_groups: list[list[str]] = []
        doc_group_names: dict[int, str] = {}

        for project in selected:
            try:
                module_graph = await build_module_graph(self._graph, project, "", test_patterns=self._test_patterns)
            except Exception as exc:  # the map is the landing page; it may not take it down
                logger.debug("Map unavailable for {}: {}", project, exc)
                return _empty_map(
                    self._project,
                    "The module map could not be built for this project. "
                    "Run health_check for the backend's own account of why.",
                )
            prefix = f"{project}:" if multi else ""
            base = len(partition_groups)
            for qn, info in module_graph.modules.items():
                nodes_all[prefix + qn] = {**info, "project": project}
            for (a, b), w in module_graph.directed.items():
                directed[prefix + a, prefix + b] = w
            for pair, state in module_graph.evidence.items():
                evidence[prefix + pair[0], prefix + pair[1]] = state
            for (a, b), w in module_graph.edges.items():
                undirected[prefix + a, prefix + b] = w
            for idx, group in enumerate(module_graph.partition):
                members = [prefix + qn for qn in group]
                partition_groups.append(members)
                for member in members:
                    community_of[member] = base + idx

            # Documentation files are DocFile/Note nodes, not Modules, so the module
            # inventory never sees them — merged here as non-code files behind the
            # same toggle as the CI YAML, in their own files community. A DOCUMENTS
            # link is a structural fact; it aggregates onto the module that owns the
            # documented entity.
            try:
                doc_rows, doc_links = await fetch_doc_modules(self._graph, project)
            except Exception as exc:  # docs are additive; their failure is not the map's
                logger.debug("Doc modules unavailable for {}: {}", project, exc)
                doc_rows, doc_links = [], []
            path_key = {str(info.get("file_path") or ""): prefix + qn for qn, info in module_graph.modules.items()}
            doc_members: list[str] = []
            for row in doc_rows:
                path = str(row.get("file_path") or "")
                qn = str(row.get("qn") or "") or path.replace("\\", "/").replace("/", ".")
                key = prefix + qn
                if not path or key in nodes_all:
                    continue
                nodes_all[key] = {
                    "uid": str(row.get("uid") or ""),
                    "name": str(row.get("name") or "") or path.rsplit("/", 1)[-1],
                    "qn": qn,
                    "file_path": path,
                    "project": project,
                    "is_doc": True,
                }
                path_key.setdefault(path, key)
                doc_members.append(key)
            for row in doc_links:
                a = path_key.get(str(row.get("from_path") or ""))
                b = path_key.get(str(row.get("to_path") or ""))
                if not a or not b or a == b:
                    continue
                weight = float(row.get("links") or 1.0)
                directed[a, b] = directed.get((a, b), 0.0) + weight
                if _EV_RANK["structural"] > _EV_RANK.get(evidence.get((a, b), ""), -1):
                    evidence[a, b] = "structural"
                key2 = (min(a, b), max(a, b))
                undirected[key2] = undirected.get(key2, 0.0) + weight
            if doc_members:
                doc_idx = len(partition_groups)
                partition_groups.append(doc_members)
                for member in doc_members:
                    community_of[member] = doc_idx
                tops = {
                    str(nodes_all[m]["file_path"]).replace("\\", "/").strip("/").split("/", 1)[0] for m in doc_members
                }
                doc_group_names[doc_idx] = (prefix + tops.pop()) if len(tops) == 1 else (prefix + "documentation")

            # External packages are the third-party boundary — 690 import edges on
            # this repo that the module level never showed. A symbol import
            # aggregates onto its root package, so the map draws one dashed node
            # per library rather than a scatter of loose names.
            try:
                ext_rows, ext_links = await fetch_external_imports(self._graph, project)
            except Exception as exc:  # externals are additive; their failure is not the map's
                logger.debug("External imports unavailable for {}: {}", project, exc)
                ext_rows, ext_links = [], []
            ext_members: list[str] = []
            ext_known: dict[str, str] = {}
            for row in ext_rows:
                ext_qn = str(row.get("qn") or "")
                if not ext_qn:
                    continue
                key = prefix + ext_qn
                ext_known[ext_qn] = key
                if key in nodes_all:
                    continue
                nodes_all[key] = {
                    "uid": str(row.get("uid") or ""),
                    "name": str(row.get("name") or "") or ext_qn.removeprefix("ext/"),
                    "qn": ext_qn,
                    "file_path": "",
                    "project": project,
                    "is_external": True,
                }
                ext_members.append(key)
            for row in ext_links:
                target = _external_root(str(row.get("to_qn") or ""), ext_known)
                source = path_key.get(str(row.get("from_path") or ""))
                if not source or not target or source == target:
                    continue
                weight = float(row.get("links") or 1.0)
                directed[source, target] = directed.get((source, target), 0.0) + weight
                if _EV_RANK["structural"] > _EV_RANK.get(evidence.get((source, target), ""), -1):
                    evidence[source, target] = "structural"
                key2 = (min(source, target), max(source, target))
                undirected[key2] = undirected.get(key2, 0.0) + weight
            if ext_members:
                ext_idx = len(partition_groups)
                partition_groups.append(ext_members)
                for member in ext_members:
                    community_of[member] = ext_idx
                doc_group_names[ext_idx] = prefix + "external packages"

        # Cross-project imports appear when more than one project is loaded — the
        # modal's own promise. They are IMPORTS, so their evidence is structural.
        if multi:
            selected_set = set(selected)
            for project in selected:
                for row in await fetch_first_hop_external(self._graph, project):
                    to_project = str(row.get("to_project") or "")
                    if to_project not in selected_set:
                        continue
                    a = f"{project}:{row.get('from_mod')}"
                    b = f"{to_project}:{row.get('to_mod')}"
                    if a in nodes_all and b in nodes_all and (a, b) not in directed:
                        directed[a, b] = 1.0
                        evidence[a, b] = "structural"
                        undirected[min(a, b), max(a, b)] = undirected.get((min(a, b), max(a, b)), 0.0) + 1.0

        if not nodes_all:
            return _empty_map(self._project, "", caveat=CoverageCaveat(note="No modules indexed for this project."))

        # Classify every module once; the counts below all derive from this one table.
        kind_of: dict[str, str] = {}
        test_count = noncode_count = 0
        external_count = 0
        for qn, info in nodes_all.items():
            path = str(info.get("file_path") or "")
            if info.get("is_external"):
                kind_of[qn] = "external"
                external_count += 1
            elif info.get("is_doc"):
                kind_of[qn] = "noncode"
                noncode_count += 1
            elif _is_test_module(path, str(info.get("name") or ""), self._test_patterns):
                kind_of[qn] = "test"
                test_count += 1
            elif _is_noncode_module(path):
                kind_of[qn] = "noncode"
                noncode_count += 1
            else:
                kind_of[qn] = "code"

        excluded = {
            qn
            for qn, kind in kind_of.items()
            if (kind == "test" and not show_tests)
            or (kind == "noncode" and not show_noncode)
            or (kind == "external" and not show_external)
        }
        visible_partition = [[qn for qn in group if qn not in excluded] for group in partition_groups]
        kept = _largest_first(visible_partition, node_limit)
        truncated = len(kept) < len(nodes_all) - len(excluded)

        # Every community rides along, singletons included: the sidebar's totals are
        # summed from this list, so a group left out would make the arithmetic on
        # screen fail to close (acceptance check 1).
        for idx, group in enumerate(partition_groups):
            if not group:
                continue
            members_kind = {kind_of[m] for m in group}
            communities.append(
                CommunityRef(
                    id=idx,
                    name=doc_group_names.get(idx) or _community_label([m.split(":", 1)[-1] for m in group]),
                    count=len(group),
                    color=f"var(--atlas-c{min(8, idx)})",
                    files=members_kind == {"noncode"},
                )
            )

        max_w = max(directed.values(), default=1.0) or 1.0
        scaled = {pair: _scale_weight(w, max_w) for pair, w in directed.items()}
        kept_edges = {pair: w for pair, w in scaled.items() if pair[0] in kept and pair[1] in kept}
        positions = force_layout(sorted(kept), kept_edges)

        degree = _degree_of(undirected, kept)
        nodes = tuple(
            MapNode(
                id=qn,
                label=(
                    _doc_label(str(nodes_all[qn].get("file_path") or ""))
                    if nodes_all[qn].get("is_doc")
                    else _external_label(str(nodes_all[qn].get("qn") or qn))
                    if nodes_all[qn].get("is_external")
                    else _module_map_label(qn.split(":", 1)[-1], str(nodes_all[qn].get("file_path") or ""))
                ),
                community=community_of.get(qn, -1),
                deg=degree.get(qn, 0),
                kind=kind_of[qn],
                x=round(positions.get(qn, (500.0, 500.0))[0], 1),
                y=round(positions.get(qn, (500.0, 500.0))[1], 1),
                uid=str(nodes_all[qn].get("uid") or ""),
                path=str(nodes_all[qn].get("file_path") or ""),
            )
            for qn in sorted(kept)
        )
        edges = tuple(
            MapEdge(s=a, t=b, w=round(w, 2), ev=evidence.get((a, b), "unknown"))
            for (a, b), w in sorted(kept_edges.items())
        )

        scope_options = await self._scope_options()
        return MapPayload(
            project=self._project,
            level="module",
            nodes=nodes,
            edges=edges,
            communities=tuple(communities),
            kinds=_kind_defs(),
            caveat=_map_caveat(len(nodes_all), truncated, node_limit),
            module_total=len(nodes_all),
            edge_total=len(directed),
            test_count=test_count,
            noncode_count=noncode_count,
            external_count=external_count,
            entity_total=await self._entity_total(selected),
            truncated=truncated,
            scope_options=scope_options,
            default_scope=_default_scope(scope_options),
        )

    async def entity_map(  # noqa: PLR0912, PLR0915  # inventory, fold, filter, anchor and tally share one table
        self,
        scope: str,
        *,
        expand_methods: bool = False,
        hidden: tuple[str, ...] | None = None,
        node_limit: int | None = None,
        show_tests: bool = False,
        show_noncode: bool = False,
    ) -> MapPayload:
        """The entity level: the graph's own nodes — one module, or the whole project.

        An empty *scope* draws every entity the project indexes. Methods fold into the
        class that holds them by default, with their calls rewired to the class; kind
        filters hide what they name and count what they hid; and the tallies always
        cover the **full** inventory, so nothing removed from the drawing reads as
        absent from the module.

        *hidden* is the exact set of kinds to hide; ``None`` means the level's own
        default — value and documentation kinds at full scope, nothing when scoped.
        """
        full = not scope
        requested = tuple(hidden) if hidden is not None else (_FULL_SCOPE_HIDDEN if full else ())
        hidden_kinds = tuple(k for k in requested if k not in _UNHIDEABLE_KINDS)
        limit = node_limit if node_limit is not None else (_ENTITY_FULL_LIMIT if full else _ENTITY_SCOPE_LIMIT)
        scope_options = await self._scope_options()

        # One row shape for both sources: (from_qn, to_qn, rel_type, weight, confidence, strategy).
        if full:
            unavailable = self._unsupported_reason()
            if unavailable:
                return _empty_map(self._project, unavailable, level="entity", scope_options=scope_options)
            from code_atlas.server.analysis import fetch_entity_graph  # noqa: PLC0415

            try:
                rows, raw_edges = await fetch_entity_graph(self._graph, self._project)
            except Exception as exc:  # the map may not take the page down
                logger.debug("Entity graph unavailable for {}: {}", self._project, exc)
                return _empty_map(
                    self._project,
                    "The entity graph could not be fetched for this project. "
                    "Run health_check for the backend's own account of why.",
                    level="entity",
                    scope_options=scope_options,
                )
            edge_rows = [
                (
                    str(r.get("from_qn") or ""),
                    str(r.get("to_qn") or ""),
                    str(r.get("rel_type") or ""),
                    _as_float(r.get("weight")) or 1.0,
                    str(r.get("confidence") or ""),
                    str(r.get("strategy") or ""),
                )
                for r in raw_edges
            ]
        else:
            summary = await self._graph.get_module_summary(self._project, scope, _ENTITY_SCOPE_LIMIT, _EDGE_LIMIT)
            rows = list(summary.get("entities") or [])
            edge_rows = []
            for row in summary.get("internal_edges") or []:
                props = row.get("props") or {}
                edge_rows.append(
                    (
                        str(row.get("from_qn") or ""),
                        str(row.get("to_qn") or ""),
                        str(row.get("rel_type") or ""),
                        _as_float(props.get("weight")) or 1.0,
                        str(props.get("confidence") or ""),
                        str(props.get("strategy") or ""),
                    )
                )

        if not rows:
            return _empty_map(
                self._project,
                "",
                caveat=CoverageCaveat(note=f"No entities indexed under {scope or 'this project'}."),
                level="entity",
                scope=scope,
                scope_options=scope_options,
            )

        entities: dict[str, dict[str, Any]] = {}
        for r in rows:
            qn = str(r.get("qn") or r.get("uid") or "")
            if not qn:
                continue
            entities[qn] = {
                "uid": str(r.get("uid") or ""),
                "name": str(r.get("name") or ""),
                "kind": classify(str(r.get("label") or ""), str(r.get("kind") or "")),
                "file_path": str(r.get("file_path") or ""),
                "lines": _lines_of(r),
            }

        # Full scope honours the same test and non-code filters the module level
        # does — without them half the picture is test scaffolding mirroring the
        # production structure. Hidden is counted, never silently dropped. A view
        # deliberately scoped INTO a test module keeps showing it.
        test_count = noncode_count = 0
        if full:
            production: dict[str, dict[str, Any]] = {}
            for qn, info in entities.items():
                path = info["file_path"]
                if not show_tests and _is_test_module(path, info["name"], self._test_patterns):
                    test_count += 1
                elif not show_noncode and _is_noncode_module(path):
                    noncode_count += 1
                else:
                    production[qn] = info
            entities = production
            if not entities:
                return _empty_map(
                    self._project,
                    "",
                    caveat=CoverageCaveat(note="Every indexed entity is test or non-code, and both are filtered."),
                    level="entity",
                    scope=scope,
                    scope_options=scope_options,
                )

        # A scoped view synthesises its host module when the summary omits it — the
        # design draws the module as the graph's anchor. At full scope the modules are
        # entities in their own right, fetched like everything else.
        module_qn = "" if full else _host_module_qn(entities)
        if module_qn and module_qn not in entities:
            entities[module_qn] = {
                "uid": "",
                "name": module_qn.rsplit(".", 1)[-1],
                "kind": "module",
                "file_path": scope,
                "lines": "",
            }

        # Counted before folding or filtering: the tally answers "what is indexed
        # here", not "what survived the display settings".
        inventory: dict[str, int] = {}
        for info in entities.values():
            inventory[info["kind"]] = inventory.get(info["kind"], 0) + 1

        owner = {} if expand_methods else _method_owners(entities)
        hidden_set = set(hidden_kinds)
        drawn = {qn: info for qn, info in entities.items() if qn not in owner and info["kind"] not in hidden_set}

        # Aggregate dependency edges onto the drawn set, carrying the strongest
        # evidence. CONTAINS is containment, not dependency — it becomes the same
        # "defines" scaffolding the qualified-name walk synthesises, which is what
        # holds a stub library together as one shape.
        edges: dict[tuple[str, str], float] = {}
        edge_ev: dict[tuple[str, str], str] = {}
        edge_rel: dict[tuple[str, str], str] = {}
        for from_qn, to_qn, rel_type, weight, confidence, strategy in edge_rows:
            a = owner.get(from_qn, from_qn)
            b = owner.get(to_qn, to_qn)
            if a == b or a not in drawn or b not in drawn:
                continue
            if rel_type == "CONTAINS":
                if (a, b) not in edges:
                    edges[a, b] = _DEFINES_WEIGHT
                    edge_ev[a, b] = "structural"
                    edge_rel[a, b] = "defines"
                continue
            state = _evidence_state(EdgeEvidence(rel_type=rel_type, strategy=strategy, confidence=confidence))
            edges[a, b] = edges.get((a, b), 0.0) + weight
            edge_rel[a, b] = "calls"
            if _EV_RANK[state] > _EV_RANK.get(edge_ev.get((a, b), ""), -1):
                edge_ev[a, b] = state

        # Containment is structural fact: the module DEFINES its members, a class its
        # own. Drawn as edges they anchor every entity — without them, anything with
        # no resolved call floats detached at the map's edge, which reads as
        # "isolated" when the truth is "contained". They are scaffolding, not signal,
        # so they get a "defines" rel (the canvas draws them as faint hairlines) and a
        # low weight. At full scope an entity with no drawn ancestor simply has no
        # anchor — there is no single host to fall back to, and modules connect
        # through their own import edges.
        for qn in drawn:
            if qn == module_qn:
                continue
            definer = qn.rsplit(".", 1)[0] if "." in qn else ""
            while definer and definer not in drawn:
                definer = definer.rsplit(".", 1)[0] if "." in definer else ""
            anchor = definer or module_qn
            if anchor and anchor != qn and (anchor, qn) not in edges:
                edges[anchor, qn] = _DEFINES_WEIGHT
                edge_ev[anchor, qn] = "structural"
                edge_rel[anchor, qn] = "defines"

        # Degree over everything drawable, so a truncation keeps the most connected
        # rather than the alphabetically first.
        degree: dict[str, int] = {}
        held: dict[str, int] = {}
        for class_qn in owner.values():
            held[class_qn] = held.get(class_qn, 0) + 1
        for a, b in edges:
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1

        if len(drawn) > limit:
            ranked = sorted(drawn, key=lambda qn: (-degree.get(qn, 0), qn))
            kept = set(ranked[:limit])
            truncated = True
        else:
            kept = set(drawn)
            truncated = False
        max_w = max(edges.values(), default=1.0) or 1.0
        scaled = {pair: _scale_weight(w, max_w) for pair, w in edges.items() if pair[0] in kept and pair[1] in kept}

        # A full-scope layout is the one expensive computation here — deterministic
        # per graph, so it is cached against the index stamp.
        if full and len(kept) > 800:
            key = (
                self._project,
                expand_methods,
                hidden_kinds,
                show_tests,
                show_noncode,
                await self._index_stamp(),
                len(kept),
                len(scaled),
            )
            positions = _LAYOUT_CACHE.get(key)
            if positions is None:
                positions = force_layout(sorted(kept), scaled)
                while len(_LAYOUT_CACHE) >= _LAYOUT_CACHE_MAX:
                    _LAYOUT_CACHE.pop(next(iter(_LAYOUT_CACHE)))
                _LAYOUT_CACHE[key] = positions
        else:
            positions = force_layout(sorted(kept), scaled)

        drawn_count: dict[str, int] = {}
        for qn in kept:
            drawn_count[drawn[qn]["kind"]] = drawn_count.get(drawn[qn]["kind"], 0) + 1

        entity_comm, entity_communities = await self._entity_communities(entities)

        nodes = tuple(
            MapNode(
                id=qn,
                label=_entity_label(qn, drawn[qn]),
                community=entity_comm.get(qn, -1),
                deg=degree.get(qn, 0) + held.get(qn, 0),
                kind=drawn[qn]["kind"],
                x=round(positions.get(qn, (500.0, 500.0))[0], 1),
                y=round(positions.get(qn, (500.0, 500.0))[1], 1),
                uid=drawn[qn]["uid"],
                path=drawn[qn]["file_path"],
                lines=drawn[qn]["lines"],
            )
            for qn in sorted(kept)
        )

        return MapPayload(
            project=self._project,
            level="entity",
            nodes=nodes,
            edges=tuple(
                MapEdge(s=a, t=b, w=round(w, 2), ev=edge_ev.get((a, b), "unknown"), rel=edge_rel.get((a, b), "calls"))
                for (a, b), w in sorted(scaled.items())
            ),
            communities=entity_communities,
            kinds=_kind_defs(),
            caveat=CoverageCaveat(note=f"{len(entities)} entities indexed under {scope or 'this project'}."),
            module_total=await self._module_total(),
            test_count=test_count,
            noncode_count=noncode_count,
            entity_total=await self._entity_total((self._project,)),
            truncated=truncated,
            scope_options=scope_options,
            default_scope=_default_scope(scope_options),
            scope=scope,
            scope_name="Whole project" if full else _scope_name(scope, scope_options),
            in_module=len(entities),
            collapsed=not expand_methods,
            hidden_kinds=hidden_kinds,
            tally=tuple(
                KindTally(id=kind.id, in_module=inventory[kind.id], drawn=drawn_count.get(kind.id, 0))
                for kind in KINDS
                if inventory.get(kind.id)
            ),
        )

    async def _index_stamp(self) -> str:
        """When this project was last indexed — the layout cache's freshness key."""
        try:
            statuses = [_project_props(row) for row in await self._graph.get_project_status()]
        except Exception:
            return ""
        current = next((st for st in statuses if st.get("name") == self._project), {})
        return str(current.get("last_indexed_at") or "")

    async def _entity_communities(
        self, entities: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, int], tuple[CommunityRef, ...]]:
        """Each entity's community — its module's, from the same partition the module
        map draws — plus the community table whose counts sum to the inventory.

        Externals form their own community: the first-party/third-party boundary is
        exactly what colour should show. Files outside the partition (docs, notes)
        get a named bucket rather than silently colouring as something else.
        """
        from code_atlas.server.analysis import build_module_graph  # noqa: PLC0415

        community_of_path: dict[str, int] = {}
        names: dict[int, str] = {}
        try:
            module_graph = await build_module_graph(self._graph, self._project, "", test_patterns=self._test_patterns)
            for idx, group in enumerate(module_graph.partition):
                names[idx] = _community_label(group)
                for qn in group:
                    path = str(module_graph.modules.get(qn, {}).get("file_path") or "")
                    if path:
                        community_of_path[path] = idx
        except Exception as exc:  # colour degrades to neutral, the map survives
            logger.debug("Entity communities unavailable for {}: {}", self._project, exc)

        ext_id = len(names)
        misc_id = ext_id + 1
        entity_comm: dict[str, int] = {}
        counts: dict[int, int] = {}
        for qn, info in entities.items():
            if info["kind"] in {"external_package", "external_symbol"}:
                cid = ext_id
            else:
                cid = community_of_path.get(info["file_path"], misc_id)
            entity_comm[qn] = cid
            counts[cid] = counts.get(cid, 0) + 1

        names[ext_id] = "external packages"
        names[misc_id] = "docs & other files"
        communities = tuple(
            CommunityRef(
                id=cid,
                name=names.get(cid, f"community {cid}"),
                count=counts[cid],
                color=f"var(--atlas-c{min(8, cid)})",
                files=cid == misc_id,
            )
            for cid in sorted(counts)
        )
        return entity_comm, communities

    async def _scope_options(self) -> tuple[ScopeOption, ...]:
        """Every module the entity level can be pointed at — the rail arranges them
        into a file tree, so the list must be complete, not merely the largest."""
        try:
            overview = await self._graph.get_structure_overview(self._project, "", 500)
        except Exception:  # the picker is a convenience; its failure is not the view's
            return ()
        return tuple(
            ScopeOption(
                id=str(row.get("file_path") or row.get("name") or ""),
                label=str(row.get("name") or row.get("file_path") or ""),
                entities=int(row.get("cnt") or row.get("entities") or 0),
            )
            for row in (overview.get("largest_modules") or [])
            if row.get("file_path") or row.get("name")
        )

    async def _entity_total(self, selected: tuple[str, ...]) -> int:
        try:
            statuses = [_project_props(row) for row in await self._graph.get_project_status()]
        except Exception:
            return 0
        return sum(int(s.get("entity_count") or 0) for s in statuses if s.get("name") in selected)

    async def _module_total(self) -> int:
        try:
            overview = await self._graph.get_structure_overview(self._project, "", _STRUCTURE_LIMIT)
        except Exception:
            return 0
        return sum(int(r.get("cnt") or 0) for r in overview.get("counts", []) if str(r.get("label")) == "Module")

    def _unsupported_reason(self) -> str:
        """Why this backend cannot produce the map, or empty if it can."""
        from code_atlas.backends.sqlite_graph import SqliteGraphClient  # noqa: PLC0415

        if isinstance(self._graph, SqliteGraphClient):
            return (
                "Community detection is not available on the SQLite backend — the module inventory "
                "and module-pair CALLS aggregation it clusters are still raw Cypher reads. "
                "Run against Memgraph to see the map."
            )
        return ""


def _scale_weight(weight: float, max_weight: float) -> float:
    """ADR-0017 aggregates into the design's 1-3 band.

    The canvas thickness formula (``EW[ev] * (0.7 + w * 0.1)``) and the layout's
    attraction term both expect the mock's discrete 1..3; real aggregates span four
    orders of magnitude, so the square root spreads the low end instead of letting one
    heavy pair flatten everything else to 1.
    """
    return 1.0 + 2.0 * (max(weight, 0.0) / max_weight) ** 0.5


def _kind_defs() -> tuple[KindDef, ...]:
    return tuple(KindDef(id=k.id, label=k.label, shape=k.shape, note=k.note) for k in KINDS)


def _lines_of(row: dict[str, Any]) -> str:
    start, end = row.get("line_start"), row.get("line_end")
    if isinstance(start, int) and isinstance(end, int):
        return f"{start}–{end}"  # noqa: RUF001  # en dash: a range, not a hyphen
    return ""


def _external_label(qn: str) -> str:
    """The library path with its symbol — a bare "batched" answers nothing.

    Slash-shaped refs (GitHub Actions) stay whole: splitting "actions/checkout"
    would invent structure it does not have.
    """
    rest = qn.removeprefix("ext/")
    if "." in rest:
        package, symbol = rest.rsplit(".", 1)
        return f"{package}{SEPARATOR}{symbol}"
    return rest


def _entity_label(qn: str, info: dict[str, Any]) -> str:
    """``file.py › Class › symbol`` — the design's entity-level breadcrumb."""  # noqa: RUF002
    if info["kind"] in {"external_package", "external_symbol"}:
        return _external_label(qn)
    crumb = breadcrumb(qualified_name=qn, file_path=info["file_path"], kind=info["kind"])
    leaf = crumb.path.rsplit("/", 1)[-1]
    parts = [p for p in (leaf, crumb.owner, crumb.symbol) if p]
    # A module names itself; "client.py › client" would say one thing twice.  # noqa: RUF003
    if info["kind"] in {"module", "package"} and len(parts) == 2 and parts[0].rsplit(".", 1)[0] == parts[1]:
        parts = [parts[0]]
    return SEPARATOR.join(parts) or info["name"] or qn


def _scope_name(scope: str, options: tuple[ScopeOption, ...]) -> str:
    for option in options:
        if option.id == scope:
            return option.label
    return scope


def _default_scope(options: tuple[ScopeOption, ...]) -> str:
    """The first module to show when the entity level is opened without one.

    Largest *production* module: the largest overall is a test file, which is a poor
    first thing to show someone opening the entity level.
    """
    production = [o for o in options if not _looks_like_test(o.id)]
    chosen = production or list(options)
    return chosen[0].id if chosen else ""


def _looks_like_test(path: str) -> bool:
    lowered = path.replace("\\", "/").lower()
    return "/tests/" in f"/{lowered}" or lowered.rsplit("/", 1)[-1].startswith("test_")


# Extensions that carry no import graph: they are indexed on purpose and belong in
# search, but a dependency map draws them as isolated dots.
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


def _host_module_qn(entities: dict[str, dict[str, Any]]) -> str:
    """The scope's own module, read off the entity names.

    The module's qualified name is the shortest parent shared by its members — a
    top-level function is one segment deeper, a method two. An explicit Module row
    wins when the summary carries one.
    """
    for qn, info in entities.items():
        if info["kind"] in {"module", "package"}:
            return qn
    parents = [qn.rsplit(".", 1)[0] for qn in entities if "." in qn]
    if not parents:
        return ""
    # Sorted in place rather than `min(..., key=len)` — the key-callable widens the
    # element type to Sized under ty, losing str.
    parents.sort(key=lambda parent: (len(parent), parent))
    return parents[0]


def _method_owners(entities: dict[str, dict[str, Any]]) -> dict[str, str]:
    """``method_qn -> owning class_qn`` for every method whose class is present.

    A method whose class is not in this scope stays drawn — folding it into nothing
    would remove it from the picture.
    """
    classes = {qn for qn, info in entities.items() if info["kind"] == "class"}
    owners: dict[str, str] = {}
    for qn, info in entities.items():
        if info["kind"] != "method" or "." not in qn:
            continue
        parent = qn.rsplit(".", 1)[0]
        if parent in classes:
            owners[qn] = parent
    return owners


def _empty_map(
    project: str,
    unavailable: str,
    *,
    caveat: CoverageCaveat | None = None,
    level: str = "module",
    scope: str = "",
    scope_options: tuple[ScopeOption, ...] = (),
) -> MapPayload:  # an empty map still names its level and scope
    return MapPayload(
        project=project,
        level=level,
        nodes=(),
        edges=(),
        communities=(),
        kinds=_kind_defs(),
        caveat=caveat or CoverageCaveat(note=unavailable),
        unavailable=unavailable,
        scope=scope,
        scope_options=scope_options,
        default_scope=_default_scope(scope_options),
    )


def _largest_first(partition: list[list[str]], node_limit: int) -> set[str]:
    """The modules that fit, taking whole communities largest-first.

    Truncating by community rather than by module keeps every drawn subsystem complete.
    Slicing a flat list would cut communities in half and show a subsystem missing the
    modules that explain it.
    """
    kept: set[str] = set()
    # Copied then sorted in place: `sorted(..., key=len)` resolves the element type
    # from `len` and widens it to Sized, losing list[str].
    ordered = list(partition)
    ordered.sort(key=len, reverse=True)
    for group in ordered:
        if len(kept) + len(group) > node_limit:
            continue
        kept.update(group)
    return kept


def _degree_of(edges: dict[tuple[str, str], float], kept: set[str]) -> dict[str, int]:
    degree: dict[str, int] = {}
    for a, b in edges:
        if a in kept and b in kept:
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
    return degree


def _community_label(group: list[str]) -> str:
    """Name a subsystem by the longest package prefix its modules share."""
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

    The page works at module level over the same graph the map draws; the JSON routes
    keep the entity-level traversals the ``blast_radius`` and ``trace_path`` MCP tools
    use, so neither view can drift from the tools.
    """

    def __init__(self, graph: GraphBackend, project: str, *, test_patterns: tuple[str, ...] = ()) -> None:
        self._graph = graph
        self._project = project
        self._test_patterns = test_patterns

    async def module_impact(  # noqa: PLR0912  # the BFS and its grouping belong together
        self, subject: str = "", *, confident_only: bool = False, max_hops: int = 3
    ) -> ImpactView:
        """Everything that transitively depends on one module, grouped by distance.

        A row's evidence is the **weakest** claim anywhere along the path that reached
        it — a dependent three structural hops away plus one guess is a guess.
        """
        from code_atlas.backends.sqlite_graph import SqliteGraphClient  # noqa: PLC0415
        from code_atlas.server.analysis import build_module_graph  # noqa: PLC0415

        if isinstance(self._graph, SqliteGraphClient):
            return _impact_unavailable(
                "Module impact runs over the CALLS+IMPORTS module graph, which the SQLite "
                "backend cannot aggregate. Run against Memgraph."
            )
        try:
            module_graph = await build_module_graph(self._graph, self._project, "", test_patterns=self._test_patterns)
        except Exception as exc:
            logger.debug("Impact unavailable for {}: {}", self._project, exc)
            return _impact_unavailable("The module graph could not be built for this project.")
        if not module_graph.modules:
            return _impact_unavailable("No modules indexed for this project.")

        degree: dict[str, int] = dict.fromkeys(module_graph.modules, 0)
        for a, b in module_graph.edges:
            if a in degree:
                degree[a] += 1
            if b in degree:
                degree[b] += 1
        ranked = sorted(module_graph.modules, key=lambda qn: (-degree[qn], qn))
        root = subject if subject in module_graph.modules else (ranked[0] if ranked else "")
        if not root:
            return _impact_unavailable("No modules indexed for this project.")

        def label_of(qn: str) -> str:
            return _module_map_label(qn, str(module_graph.modules[qn].get("file_path") or ""))

        # BFS against the arrows: who depends on the root, then who depends on them.
        seen: dict[str, tuple[int, str, str]] = {root: (0, "structural", "")}
        frontier = [root]
        for hop in range(1, max_hops + 1):
            nxt: list[str] = []
            for depender, dependency in module_graph.directed:
                if dependency not in frontier or depender in seen:
                    continue
                ev = module_graph.evidence.get((depender, dependency), "unknown")
                carried = seen[dependency][1]
                worst = ev if _EV_RANK[ev] < _EV_RANK[carried] else carried
                rel = "imports" if ev == "structural" else "calls"
                seen[depender] = (hop, worst, rel)
                nxt.append(depender)
            frontier = nxt
            if not frontier:
                break

        kept_count = dropped = 0
        groups: list[ImpactHopGroup] = []
        for hop in range(1, max_hops + 1):
            rows: list[ImpactRow] = []
            for qn, (at, worst, rel) in sorted(seen.items(), key=lambda kv: kv[0]):
                if at != hop:
                    continue
                ok = worst in {"structural", "resolved"}
                if confident_only and not ok:
                    dropped += 1
                    continue
                kept_count += 1
                rows.append(
                    ImpactRow(
                        id=qn,
                        label=label_of(qn),
                        rel=rel,
                        path_note=(
                            "path fully resolved"
                            if ok
                            else "path passes a guess"
                            if worst == "guessed"
                            else "path passes an unlooked-up edge"
                        ),
                        ev=worst,
                        url=_impact_url(qn, confident_only),
                    )
                )
            if rows:
                groups.append(
                    ImpactHopGroup(
                        hop_label=f"{hop} hop{'s' if hop != 1 else ''} away",
                        count_label=f"{len(rows)} module{'s' if len(rows) != 1 else ''}",
                        rows=tuple(rows),
                    )
                )

        total_note = (
            f"{kept_count} modules on fully resolved paths · {dropped} hidden because their path "
            "passes a guess or an unlooked-up edge"
            if confident_only
            else f"{kept_count} modules depend on this, directly or transitively, within {max_hops} hops"
        )

        return ImpactView(
            subject_id=root,
            subject_label=label_of(root),
            total_note=total_note,
            groups=tuple(groups),
            roots=tuple(
                ImpactRoot(
                    id=qn,
                    label=label_of(qn),
                    on=qn == root,
                    url=_impact_url(qn, confident_only),
                )
                for qn in ranked[:10]
            ),
            confident_only=confident_only,
            confident_url=_impact_url(root, not confident_only),
        )

    async def blast(
        self,
        uid: str,
        *,
        direction: str = "callers",
        max_depth: int = 3,
        limit: int = _IMPACT_PAGE,
        resolved_only: bool = False,
    ) -> BlastRadiusView:
        """The dependency closure around *uid*, grouped by distance (JSON contract).

        Per ADR-0029 this traverses dependency edges only — DEFINES and CONTAINS are
        excluded, because counting containment makes "what does changing this method
        affect" mean nothing.
        """
        from code_atlas.server.analysis import blast_radius  # noqa: PLC0415

        # Fetched at the view's own ceiling rather than at `limit`, so the resolved-only
        # filter runs over the whole considered set instead of over one page.
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
        """The shortest path between two entities, hop by hop (JSON contract)."""
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


def _impact_url(subject: str, confident: bool) -> str:
    from urllib.parse import urlencode  # noqa: PLC0415

    params: list[tuple[str, str]] = [("subject", subject)]
    if confident:
        params.append(("confident", "1"))
    return "/impact?" + urlencode(params)


def _impact_unavailable(reason: str) -> ImpactView:
    return ImpactView(
        subject_id="",
        subject_label="",
        total_note="",
        groups=(),
        roots=(),
        confident_only=False,
        confident_url="",
        unavailable=reason,
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
    """A readable name for the analysed entity."""
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
    """Say what the closure did and did not cover."""
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


class ProjectPickerService:
    """Every indexed project, arranged the way a monorepo actually is."""

    def __init__(self, graph: GraphBackend, project: str) -> None:
        self._graph = graph
        self._project = project

    async def picker(self, selected: tuple[str, ...] = ()) -> ProjectPicker:
        rows = [_project_props(row) for row in await self._graph.get_project_status()]
        chosen = tuple(selected) or (self._project,)

        flat: dict[str, ProjectChoice] = {}
        for row in rows:
            name = str(row.get("name") or "")
            if not name:
                continue
            days = _days_since(row.get("last_indexed_at"))
            entities = int(row.get("entity_count") or 0)
            unindexed = entities == 0 and not row.get("last_indexed_at")
            flat[name] = ProjectChoice(
                name=name,
                label=name.rsplit("/", 1)[-1],
                entities=entities,
                modules=0,
                indexed_at=_as_timestamp(row.get("last_indexed_at")),
                git_hash=_as_str(row.get("git_hash")),
                is_current=name in chosen,
                days_since_indexed=days,
                state="unindexed" if unindexed else "stale" if (days is not None and days > 14) else "fresh",
                indexed_ago=_ago(row.get("last_indexed_at")),
            )

        # `a/b` is a sub-project of `a` when `a` is itself indexed. Without that check a
        # path-shaped name would invent a parent that does not exist in the graph.
        roots: list[ProjectChoice] = []
        for name in sorted(flat):
            parent = name.rsplit("/", 1)[0] if "/" in name else ""
            if parent and parent in flat:
                continue
            kids = tuple(flat[child] for child in sorted(flat) if "/" in child and child.rsplit("/", 1)[0] == name)
            roots.append(msgspec.structs.replace(flat[name], children=kids))

        selected_entities = sum(c.entities for c in flat.values() if c.name in chosen)
        return ProjectPicker(
            projects=tuple(roots),
            selected=chosen,
            selected_modules=selected_entities,
            cost_note=_picker_cost(len(chosen), selected_entities),
        )


def _days_since(value: object) -> int | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    delta = datetime.now(tz=UTC) - datetime.fromtimestamp(float(value), tz=UTC)
    return max(delta.days, 0)


def _picker_cost(count: int, entities: int) -> str:
    """What the current selection implies, before it is applied."""
    if count <= 1:
        return ""
    if entities > 20_000:
        return f"{count} projects, ~{entities:,} entities — a large combined graph, slower to lay out."
    return f"{count} projects, ~{entities:,} entities combined."
