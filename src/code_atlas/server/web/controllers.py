"""HTTP layer for the web UI.

Owns routing, status codes and template selection — and nothing else. Every handler
here is a thin translation between a request and a service call; if a handler starts
making decisions about *what* to show rather than *how* to return it, that logic
belongs in :mod:`code_atlas.server.web.services`.

Handlers return view models (``msgspec.Struct``) for JSON routes and ``Template`` for
HTML ones. The JSON routes exist because the map view is a client-side island fed by
fetch — the design implements that screen as one component whose rail, canvas and
context panel all react to the same state, and the port keeps that shape.
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
# under TYPE_CHECKING raises `NameError` before the app finishes constructing.
from code_atlas.server.web.schemas import (  # noqa: TC001
    ArchitectureHealth,
    BlastRadiusView,
    EntityDetail,
    MapPayload,
    ProjectOverview,
    ProjectPicker,
    SearchPage,
    TracePathView,
)
from code_atlas.server.web.services import (
    ArchitectureViewService,
    ChromeService,
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
    async def index(
        self,
        view_service: NamedDependency[ProjectViewService],
        chrome_service: NamedDependency[ChromeService],
    ) -> Template:
        """Landing page — the map, already open.

        Someone opening this wants to see their codebase, and a page of counts is a
        step in the way of that. The map's own state (level, filters, display
        settings) lives in the query string, read client-side by the island.
        """
        chrome = await chrome_service.chrome()
        try:
            # Only to prove the project is indexed. A map of nothing and a project that
            # was never indexed look identical, and the fix for the second is different.
            await view_service.overview()
        except ProjectNotIndexedError as exc:
            return Template(
                "not_indexed.html",
                context={"project": exc.project, "chrome": chrome, "active": "map"},
                status_code=404,
            )
        return Template("map.html", context={"chrome": chrome, "active": "map"})

    @get("/settings", name="settings")
    async def settings(
        self,
        view_service: NamedDependency[ProjectViewService],
        chrome_service: NamedDependency[ChromeService],
    ) -> Template:
        """Appearance, map defaults, and what the index last did."""
        chrome = await chrome_service.chrome()
        try:
            overview = await view_service.overview()
        except ProjectNotIndexedError:
            overview = None
        return Template(
            "settings.html",
            context={"chrome": chrome, "active": "settings", "overview": overview},
        )

    @get("/api/projects", name="api_projects")
    async def api_projects(
        self,
        picker_service: NamedDependency[ProjectPickerService],
        selected_projects: NamedDependency[tuple[str, ...]],
    ) -> ProjectPicker:
        """What the projects dialog renders."""
        return await picker_service.picker(selected_projects)

    @get("/api/overview", name="api_overview")
    async def api_overview(self, view_service: NamedDependency[ProjectViewService]) -> ProjectOverview:
        try:
            return await view_service.overview()
        except ProjectNotIndexedError as exc:
            raise NotFoundException(detail=f"Project {exc.project!r} has no index. Run 'atlas index'.") from exc


class HealthController(Controller):
    """Liveness, for scripts and for the browser to tell "server down" from "no data"."""

    path = "/healthz"

    @get("/", name="healthz")
    async def healthz(self, view_service: NamedDependency[ProjectViewService]) -> dict[str, str]:
        return {"status": "ok", "project": view_service.project}


class SearchController(Controller):
    """Search, and the entity a result leads to."""

    path = "/"

    @get("/search", name="search")
    async def search(
        self,
        search_service: NamedDependency[SearchViewService],
        view_service: NamedDependency[ProjectViewService],
        chrome_service: NamedDependency[ChromeService],
        q: FromQuery[str] = "",
        channel: FromQuery[list[str]] | None = None,
        kind: FromQuery[list[str]] | None = None,
        limit: FromQuery[int] = 20,
    ) -> Template:
        chrome = await chrome_service.chrome()
        try:
            entities = (await view_service.overview()).entity_count
        except ProjectNotIndexedError:
            entities = 0
        page = await search_service.search_view(
            q,
            channels=tuple(channel or ("graph", "keyword", "semantic")),
            kinds=tuple(kind or ()),
            limit=min(max(limit, 1), 100),
            entities=entities,
        )
        return Template("search.html", context={"page": page, "chrome": chrome, "active": "search"})

    @get("/entity/{uid:path}", name="entity")
    async def entity(
        self,
        search_service: NamedDependency[SearchViewService],
        chrome_service: NamedDependency[ChromeService],
        uid: FromPath[str],
    ) -> Template:
        chrome = await chrome_service.chrome()
        try:
            detail = await search_service.detail_view(uid.lstrip("/"))
        except EntityNotFoundError as exc:
            return Template(
                "not_found.html",
                context={"uid": exc.uid, "chrome": chrome, "active": "search"},
                status_code=404,
            )
        return Template("entity.html", context={"detail": detail, "chrome": chrome, "active": "search"})

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
    async def architecture(
        self,
        architecture_service: NamedDependency[ArchitectureViewService],
        chrome_service: NamedDependency[ChromeService],
    ) -> Template:
        return Template(
            "architecture.html",
            context={
                "arch": await architecture_service.view(),
                "chrome": await chrome_service.chrome(),
                "active": "architecture",
            },
        )

    @get("/api", name="api_architecture")
    async def api_architecture(
        self, architecture_service: NamedDependency[ArchitectureViewService]
    ) -> ArchitectureHealth:
        return await architecture_service.health()


class MapController(Controller):
    """The map data — what the client-side island fetches."""

    path = "/map"

    @get("/", name="map")
    async def map_page(self, chrome_service: NamedDependency[ChromeService]) -> Template:
        """Kept so /map still resolves; the canonical address is /."""
        return Template("map.html", context={"chrome": await chrome_service.chrome(), "active": "map"})

    @get("/api", name="api_map")
    async def api_map(
        self,
        map_service: NamedDependency[MapViewService],
        selected_projects: NamedDependency[tuple[str, ...]],
        level: FromQuery[str] = "module",
        module: FromQuery[str] = "",
        expand: FromQuery[bool] = False,
        show_tests: FromQuery[bool] = False,
        show_noncode: FromQuery[bool] = False,
    ) -> MapPayload:
        """Whichever level was asked for, in the shape map.js renders."""
        if level == "entity":
            payload = await map_service.entity_map(module or "", expand_methods=expand)
            if module or not payload.default_scope:
                return payload
            return await map_service.entity_map(payload.default_scope, expand_methods=expand)
        return await map_service.map(show_tests=show_tests, show_noncode=show_noncode, projects=selected_projects)


class ImpactController(Controller):
    """ "What breaks if I change this", and "how do these two connect"."""

    path = "/impact"

    @get("/", name="impact")
    async def impact(
        self,
        impact_service: NamedDependency[ImpactViewService],
        chrome_service: NamedDependency[ChromeService],
        subject: FromQuery[str] = "",
        confident: FromQuery[bool] = False,
    ) -> Template:
        view = await impact_service.module_impact(subject, confident_only=confident)
        return Template(
            "impact.html",
            context={"impact": view, "chrome": await chrome_service.chrome(), "active": "impact"},
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
