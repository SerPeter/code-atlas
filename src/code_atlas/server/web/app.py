"""Composition root for the web UI.

Wires the three layers together and owns every decision about *where things come from*:
the graph backend, the project name, the template directory, the static assets. Nothing
below this module reaches out to construct its own dependencies.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

from litestar import Litestar, Request
from litestar.di import NamedDependency, Provide
from litestar.enums import ScopeType
from litestar.middleware import ASGIMiddleware
from litestar.static_files import create_static_files_router
from litestar.template.config import TemplateConfig

from code_atlas.server.web import STATIC_DIR, TEMPLATES_DIR
from code_atlas.server.web.controllers import (
    ArchitectureController,
    HealthController,
    ImpactController,
    MapController,
    ProjectController,
    SearchController,
)
from code_atlas.server.web.services import (
    ArchitectureViewService,
    ChromeService,
    ImpactViewService,
    MapViewService,
    ProjectPickerService,
    ProjectViewService,
    SearchViewService,
)
from code_atlas.telemetry import get_metrics, get_tracer, mark_span_error

if TYPE_CHECKING:
    from code_atlas.graph.protocol import GraphBackend
    from code_atlas.search.engine import EmbedOne
    from code_atlas.settings import SearchSettings

_tracer = get_tracer(__name__)


class TelemetryMiddleware(ASGIMiddleware):
    """A span and a latency sample per request.

    Identified by Litestar's route template rather than the request path. `/entity/abc123`
    and `/entity/def456` are the same route, and using the raw path would make every
    entity view its own metric series -- a metric with unbounded cardinality is how a
    time-series database gets taken down by its own instrumentation. Traces keep the
    concrete path, which is what traces are for.

    Litestar has already resolved `scope["route_handler"]` by the time middleware runs,
    so the template is available before the request is served, not only after.

    A consequence of that ordering, verified rather than assumed: an unrouted path
    never reaches this middleware at all, so 404s are not counted. That is the right
    trade here -- it means a 404 sweep cannot mint one metric series per probed URL --
    but the request counter is "requests that matched a route", not "requests received".
    """

    #: HTTP only. A websocket has no status code or route latency in the sense measured
    #: here, and the UI has no websocket routes anyway.
    scopes = (ScopeType.HTTP,)

    async def handle(self, scope: Any, receive: Any, send: Any, next_app: Any) -> None:
        method = scope.get("method", "")
        route = _route_template(scope)
        started = time.perf_counter()
        status = 0

        async def send_wrapper(message: Any) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        with _tracer.start_as_current_span(
            f"web {method} {route}",
            attributes={"http.request.method": method, "http.route": route, "url.path": scope.get("path", "")},
        ) as span:
            try:
                await next_app(scope, receive, send_wrapper)
            except Exception as exc:
                mark_span_error(span, exc)
                status = 500
                raise
            finally:
                elapsed = time.perf_counter() - started
                span.set_attribute("http.response.status_code", status)
                attrs = {"method": method, "route": route, "status": str(status)}
                get_metrics().web_requests.add(1, attrs)
                get_metrics().web_latency.record(elapsed, {"method": method, "route": route})


def _route_template(scope: Any) -> str:
    """The registered path pattern for this request, or a stable placeholder.

    ``paths`` is a set; sorted() so a handler registered under two paths does not
    alternate between them from request to request and split its own series. The
    placeholder covers a handler that exposes no ``paths`` -- not an unmatched request,
    which never gets this far.
    """
    handler = scope.get("route_handler")
    paths = sorted(getattr(handler, "paths", ()) or ())
    return paths[0] if paths else "unmatched"


def _static_version() -> str:
    """A short fingerprint of the static assets, for cache-busting links."""
    import hashlib  # noqa: PLC0415

    digest = hashlib.sha256()
    for name in sorted(("design.css", "app.js", "map.js")):
        path = STATIC_DIR / name
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:10]


def create_app(
    graph: GraphBackend,
    project: str,
    *,
    search_settings: SearchSettings | None = None,
    embed: EmbedOne | None = None,
    debug: bool = False,
) -> Litestar:
    """Build the ``atlas ui`` application.

    *graph* is injected rather than constructed here so the caller owns its lifecycle —
    the CLI opens it, serves, and closes it. That also makes the app trivially testable
    against a fake backend, which is why the service layer depends on the
    ``GraphBackend`` protocol and not on ``GraphClient``.
    """
    from litestar.plugins.jinja import JinjaTemplateEngine  # noqa: PLC0415

    from code_atlas.settings import SearchSettings as _SearchSettings  # noqa: PLC0415

    # Defaulted here rather than at the signature: a mutable default is a trap, and
    # the UI must still work when the caller has no settings to hand (tests do not).
    search_settings = search_settings or _SearchSettings()

    async def provide_selected_projects(request: Request) -> tuple[str, ...]:
        """The projects the dialog picked, from its cookie; the CLI's project otherwise.

        A cookie rather than a query parameter, so every link on every page stays clean
        and the selection survives navigation. The first name is the primary project —
        the one search, impact and the entity level are scoped to.
        """
        raw = unquote(request.cookies.get("atlas_projects", ""))
        names = tuple(name for name in raw.split(",") if name.strip())
        return names or (project,)

    async def provide_primary_project(selected_projects: NamedDependency[tuple[str, ...]]) -> str:
        return selected_projects[0]

    async def provide_view_service(primary_project: NamedDependency[str]) -> ProjectViewService:
        return ProjectViewService(graph, primary_project)

    async def provide_chrome_service(selected_projects: NamedDependency[tuple[str, ...]]) -> ChromeService:
        return ChromeService(graph, selected_projects)

    async def provide_search_service(primary_project: NamedDependency[str]) -> SearchViewService:
        return SearchViewService(graph, primary_project, search_settings=search_settings, embed=embed)

    async def provide_architecture_service(primary_project: NamedDependency[str]) -> ArchitectureViewService:
        return ArchitectureViewService(graph, primary_project)

    async def provide_map_service(primary_project: NamedDependency[str]) -> MapViewService:
        return MapViewService(graph, primary_project)

    async def provide_impact_service(primary_project: NamedDependency[str]) -> ImpactViewService:
        return ImpactViewService(graph, primary_project)

    async def provide_picker_service(primary_project: NamedDependency[str]) -> ProjectPickerService:
        return ProjectPickerService(graph, primary_project)

    template_config: TemplateConfig[Any] = TemplateConfig(directory=TEMPLATES_DIR, engine=JinjaTemplateEngine)
    # A version stamp on every static link, derived from the assets' own bytes. A
    # browser holding yesterday's map.js against today's payload renders half-truths
    # that look like data bugs; a changed URL cannot be served from cache.
    template_config.engine_instance.engine.globals["static_v"] = _static_version()

    return Litestar(
        route_handlers=[
            ProjectController,
            SearchController,
            ArchitectureController,
            MapController,
            ImpactController,
            HealthController,
            # Vendored, never CDN — `atlas ui` must work offline, and the static export
            # (ATL-120) must not reach the network at all. See ADR-0033.
            create_static_files_router(path="/static", directories=[STATIC_DIR], name="static"),
        ],
        dependencies={
            "selected_projects": Provide(provide_selected_projects),
            "primary_project": Provide(provide_primary_project),
            "view_service": Provide(provide_view_service),
            "chrome_service": Provide(provide_chrome_service),
            "search_service": Provide(provide_search_service),
            "architecture_service": Provide(provide_architecture_service),
            "map_service": Provide(provide_map_service),
            "impact_service": Provide(provide_impact_service),
            "picker_service": Provide(provide_picker_service),
        },
        # Annotated rather than inlined: `TemplateConfig` is generic in its engine and
        # Litestar's own parameter is `TemplateConfig[EngineType] | None`, so the
        # inferred `TemplateConfig[JinjaTemplateEngine]` reads as a variance error at
        # the call site. Naming the wider type is the fix; the runtime value is the same.
        middleware=[TelemetryMiddleware()],
        template_config=template_config,
        debug=debug,
    )
