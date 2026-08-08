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
from litestar.exceptions import NotFoundException
from litestar.response import Template

# Imported at RUNTIME, not under TYPE_CHECKING, and that is load-bearing rather than an
# oversight. Litestar resolves dependency injection and response types by reading a
# handler's type hints at registration time, so with `from __future__ import
# annotations` every hint is a string it must evaluate — and a name that only exists
# under TYPE_CHECKING raises `NameError: name 'ProjectViewService' is not defined`
# before the app finishes constructing. Any type appearing in a handler signature has to
# be importable for real.
from code_atlas.server.web.schemas import ProjectOverview  # noqa: TC001
from code_atlas.server.web.services import ProjectNotIndexedError, ProjectViewService


class ProjectController(Controller):
    """The current project's views."""

    path = "/"

    @get("/", name="index")
    async def index(self, view_service: ProjectViewService) -> Template:
        """Landing page — the project this server was started from."""
        try:
            overview = await view_service.overview()
        except ProjectNotIndexedError as exc:
            # A project with no graph data is not an empty project. Saying "run atlas
            # index" is the whole difference between the two (ATL-110).
            return Template(
                "not_indexed.html",
                context={"project": exc.project},
                status_code=404,
            )
        return Template("index.html", context={"overview": overview})

    @get("/api/overview", name="api_overview")
    async def api_overview(self, view_service: ProjectViewService) -> ProjectOverview:
        """The same view model as JSON, for the client-side canvas."""
        try:
            return await view_service.overview()
        except ProjectNotIndexedError as exc:
            raise NotFoundException(detail=f"Project {exc.project!r} has no index. Run 'atlas index'.") from exc


class HealthController(Controller):
    """Liveness, for scripts and for the browser to tell "server down" from "no data"."""

    path = "/healthz"

    @get("/", name="healthz")
    async def healthz(self, view_service: ProjectViewService) -> dict[str, str]:
        return {"status": "ok", "project": view_service.project}
