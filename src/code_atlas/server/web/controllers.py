"""HTTP layer for the web UI.

Owns routing, status codes and template selection — and nothing else. Every handler
here is a thin translation between a request and a service call; if a handler starts
making decisions about *what* to show rather than *how* to return it, that logic
belongs in :mod:`code_atlas.server.web.services`.

Handlers return view models (``msgspec.Struct``) for JSON routes and
``Template`` for HTML ones. The JSON routes exist because the graph canvas is a
client-side island fed by fetch — that is the one part of this UI HTMX cannot express.
"""

from __future__ import annotations

from litestar import Controller, get
from litestar.di import NamedDependency  # noqa: TC002 — see the runtime-import note below
from litestar.exceptions import NotFoundException
from litestar.params import FromPath, FromQuery  # noqa: TC002 — see the runtime-import note below
from litestar.response import Template

# Imported at RUNTIME, not under TYPE_CHECKING, and that is load-bearing rather than an
# oversight. Litestar resolves dependency injection and response types by reading a
# handler's type hints at registration time, so with `from __future__ import
# annotations` every hint is a string it must evaluate — and a name that only exists
# under TYPE_CHECKING raises `NameError: name 'ProjectViewService' is not defined`
# before the app finishes constructing. Any type appearing in a handler signature has to
# be importable for real.
from code_atlas.server.web.schemas import (  # noqa: TC001
    ArchitectureHealth,
    BlastRadiusView,
    EntityDetail,
    ModuleMap,
    ProjectOverview,
    ProjectPicker,
    SearchPage,
    TracePathView,
)
from code_atlas.server.web.services import (
    ArchitectureViewService,
    EntityNotFoundError,
    ImpactViewService,
    MapViewService,
    ProjectNotIndexedError,
    ProjectPickerService,
    ProjectViewService,
    SearchViewService,
)


class ProjectController(Controller):
    """The current project's views."""

    path = "/"

    @get("/", name="index")
    async def index(  # every argument is one control in the rail
        self,
        view_service: NamedDependency[ProjectViewService],
        map_service: NamedDependency[MapViewService],
        show_tests: FromQuery[bool] = False,
        show_noncode: FromQuery[bool] = False,
        level: FromQuery[str] = "module",
        # NOT `scope`: Litestar reserves that name and injects its ASGI ScopeState,
        # which then travelled into a Cypher parameter and failed inside the driver.
        module: FromQuery[str] = "",
        expand: FromQuery[bool] = False,
        direction: FromQuery[str] = "arrows",
        hops: FromQuery[int] = 1,
        labels: FromQuery[str] = "some",
        focus: FromQuery[int] = -1,
    ) -> Template:
        """Landing page — the map, already open.

        Not a dashboard: someone opening this wants to see their codebase, and a page of
        counts is a step in the way of that. The overview still exists at /overview.
        """
        try:
            # Only to prove the project is indexed. A map of nothing and a project that
            # was never indexed look identical, and the fix for the second is different.
            await view_service.overview()
        except ProjectNotIndexedError as exc:
            return Template("not_indexed.html", context={"project": exc.project}, status_code=404)

        return await _render_map(
            map_service,
            level=level,
            scope=module,
            expand=expand,
            show_tests=show_tests,
            show_noncode=show_noncode,
            direction=direction,
            hops=hops,
            labels=labels,
            focus=focus,
        )

    @get("/overview", name="overview")
    async def overview(self, view_service: NamedDependency[ProjectViewService]) -> Template:
        """The counts, for when that is the question."""
        try:
            overview = await view_service.overview()
        except ProjectNotIndexedError as exc:
            # A project with no graph data is not an empty project. Saying "run atlas
            # index" is the whole difference between the two (ATL-110).
            return Template("not_indexed.html", context={"project": exc.project}, status_code=404)
        return Template(
            "index.html",
            context={"overview": overview, "project": overview.project, "active": "overview"},
        )

    @get("/projects", name="projects")
    async def projects(
        self,
        picker_service: NamedDependency[ProjectPickerService],
        project: FromQuery[list[str]] | None = None,
    ) -> Template:
        """The project picker — multi-select, with monorepo children nested."""
        picker = await picker_service.picker(tuple(project or ()))
        return Template("projects.html", context={"picker": picker, "project": picker.selected[0]})

    @get("/api/projects", name="api_projects")
    async def api_projects(
        self,
        picker_service: NamedDependency[ProjectPickerService],
        project: FromQuery[list[str]] | None = None,
    ) -> ProjectPicker:
        return await picker_service.picker(tuple(project or ()))

    @get("/api/overview", name="api_overview")
    async def api_overview(self, view_service: NamedDependency[ProjectViewService]) -> ProjectOverview:
        """The same view model as JSON, for the client-side canvas."""
        try:
            return await view_service.overview()
        except ProjectNotIndexedError as exc:
            raise NotFoundException(detail=f"Project {exc.project!r} has no index. Run 'atlas index'.") from exc


async def _render_map(  # each argument is one control in the rail
    map_service: MapViewService,
    *,
    level: str,
    scope: str,
    expand: bool,
    show_tests: bool,
    show_noncode: bool,
    direction: str,
    hops: int,
    labels: str,
    focus: int,
) -> Template:
    """Render whichever level was asked for, with the display settings echoed back.

    Display settings ride in the query string rather than in a session: the page is
    server-rendered, so a setting is a different URL — which also makes any view
    shareable by pasting the address.
    """
    if level == "entity":
        target = scope or await _default_scope(map_service)
        module_map = await map_service.entity_map(target, expand_methods=expand)
    else:
        module_map = await map_service.map(show_tests=show_tests, show_noncode=show_noncode)

    return Template(
        "map.html",
        context={
            "map": module_map,
            "project": module_map.project,
            "active": "map",
            "level": module_map.level,
            "show_tests": show_tests,
            "show_noncode": show_noncode,
            "direction": direction if direction in {"arrows", "plain", "curved"} else "arrows",
            "hops": hops if hops in {1, 2, 3} else 1,
            "labels": labels if labels in {"few", "some", "all"} else "some",
            "focus": focus,
        },
    )


async def _default_scope(map_service: MapViewService) -> str:
    """The first module to show when the entity level is opened without one."""
    options = await map_service._scope_options()  # noqa: SLF001  # same package, one caller
    if not options:
        return ""
    # Largest *production* module. The largest overall is a test file, which is a poor
    # first thing to show someone opening the entity level.
    production = [o for o in options if not _looks_like_test(o.id)]
    return (production or options)[0].id


def _looks_like_test(path: str) -> bool:
    lowered = path.replace("\\", "/").lower()
    return "/tests/" in f"/{lowered}" or lowered.rsplit("/", 1)[-1].startswith("test_")


class HealthController(Controller):
    """Liveness, for scripts and for the browser to tell "server down" from "no data"."""

    path = "/healthz"

    @get("/", name="healthz")
    async def healthz(self, view_service: NamedDependency[ProjectViewService]) -> dict[str, str]:
        return {"status": "ok", "project": view_service.project}


class SearchController(Controller):
    """Search, and the entity a result leads to.

    The way in: without it every other view needs you to already know an entity name.
    """

    path = "/"

    @get("/search", name="search")
    async def search(
        self,
        search_service: NamedDependency[SearchViewService],
        q: FromQuery[str] = "",
        limit: FromQuery[int] = 20,
    ) -> Template:
        page = await search_service.search(q, limit=min(max(limit, 1), 100))
        return Template("search.html", context={"page": page})

    @get("/entity/{uid:path}", name="entity")
    async def entity(self, search_service: NamedDependency[SearchViewService], uid: FromPath[str]) -> Template:
        try:
            detail = await search_service.detail(uid.lstrip("/"))
        except EntityNotFoundError as exc:
            return Template("not_found.html", context={"uid": exc.uid}, status_code=404)
        return Template("entity.html", context={"detail": detail})

    @get("/api/search", name="api_search")
    async def api_search(
        self,
        search_service: NamedDependency[SearchViewService],
        q: FromQuery[str] = "",
        limit: FromQuery[int] = 20,
    ) -> SearchPage:
        return await search_service.search(q, limit=min(max(limit, 1), 100))

    @get("/api/entity/{uid:path}", name="api_entity")
    async def api_entity(self, search_service: NamedDependency[SearchViewService], uid: FromPath[str]) -> EntityDetail:
        try:
            return await search_service.detail(uid.lstrip("/"))
        except EntityNotFoundError as exc:
            raise NotFoundException(detail=f"No entity with uid {exc.uid!r}") from exc


class ArchitectureController(Controller):
    """The mud view — a design structure matrix, not another node-link graph."""

    path = "/architecture"

    @get("/", name="architecture")
    async def architecture(self, architecture_service: NamedDependency[ArchitectureViewService]) -> Template:
        return Template("architecture.html", context={"health": await architecture_service.health()})

    @get("/api", name="api_architecture")
    async def api_architecture(
        self, architecture_service: NamedDependency[ArchitectureViewService]
    ) -> ArchitectureHealth:
        return await architecture_service.health()


class MapController(Controller):
    """The community map — modules as nodes, subsystems as clusters."""

    path = "/map"

    @get("/", name="map")
    async def map_page(
        self,
        map_service: NamedDependency[MapViewService],
        show_tests: FromQuery[bool] = False,
        show_noncode: FromQuery[bool] = False,
    ) -> Template:
        """Kept so /map still resolves; the canonical address is now /."""
        module_map = await map_service.map(show_tests=show_tests, show_noncode=show_noncode)
        return Template(
            "map.html",
            context={
                "map": module_map,
                "project": module_map.project,
                "active": "map",
                "show_tests": show_tests,
                "show_noncode": show_noncode,
            },
        )

    @get("/api", name="api_map")
    async def api_map(
        self, map_service: NamedDependency[MapViewService], external: FromQuery[bool] = True
    ) -> ModuleMap:
        """The map as JSON — this is what the canvas fetches and renders."""
        return await map_service.map(include_external=external)


class ImpactController(Controller):
    """ "What breaks if I change this", and "how do these two connect"."""

    path = "/impact"

    @get("/", name="impact")
    async def impact(
        self,
        impact_service: NamedDependency[ImpactViewService],
        uid: FromQuery[str] = "",
        direction: FromQuery[str] = "callers",
        depth: FromQuery[int] = 3,
        resolved_only: FromQuery[bool] = False,
        to: FromQuery[str] = "",
    ) -> Template:
        """One page for both questions — a path is the natural follow-up to an impact list."""
        if not uid:
            return Template("impact.html", context={"blast": None, "trace": None})

        safe_direction, safe_depth = _impact_params(direction, depth)
        blast = await impact_service.blast(
            uid, direction=safe_direction, max_depth=safe_depth, resolved_only=resolved_only
        )
        trace = await impact_service.trace(uid, to) if to else None
        return Template(
            "impact.html",
            context={"blast": blast, "trace": trace, "resolved_only": resolved_only, "to": to},
        )

    @get("/api/blast", name="api_blast")
    async def api_blast(
        self,
        impact_service: NamedDependency[ImpactViewService],
        uid: FromQuery[str],
        direction: FromQuery[str] = "callers",
        depth: FromQuery[int] = 3,
        resolved_only: FromQuery[bool] = False,
    ) -> BlastRadiusView:
        safe_direction, safe_depth = _impact_params(direction, depth)
        return await impact_service.blast(
            uid, direction=safe_direction, max_depth=safe_depth, resolved_only=resolved_only
        )

    @get("/api/trace", name="api_trace")
    async def api_trace(
        self,
        impact_service: NamedDependency[ImpactViewService],
        uid: FromQuery[str],
        to: FromQuery[str],
        depth: FromQuery[int] = 6,
    ) -> TracePathView:
        return await impact_service.trace(uid, to, max_depth=max(1, min(depth, 10)))


def _impact_params(direction: str, depth: int) -> tuple[str, int]:
    """Clamp the query string to what the traversal will accept.

    Bounded at the edge rather than deeper in: an unbounded depth on a dense graph is a
    denial of service against the reader's own machine, and the service should not have
    to defend itself from its own HTTP layer.
    """
    safe = direction if direction in {"callers", "callees", "both"} else "callers"
    return safe, max(1, min(depth, 10))
