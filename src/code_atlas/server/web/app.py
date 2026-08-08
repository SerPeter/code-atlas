"""Composition root for the web UI.

Wires the three layers together and owns every decision about *where things come from*:
the graph backend, the project name, the template directory, the static assets. Nothing
below this module reaches out to construct its own dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from litestar import Litestar
from litestar.di import Provide
from litestar.static_files import create_static_files_router
from litestar.template.config import TemplateConfig

from code_atlas.server.web.controllers import HealthController, ProjectController
from code_atlas.server.web.services import ProjectViewService

if TYPE_CHECKING:
    from code_atlas.graph.protocol import GraphBackend

_WEB_ROOT = Path(__file__).parent
TEMPLATES_DIR = _WEB_ROOT / "templates"
STATIC_DIR = _WEB_ROOT / "static"


def create_app(graph: GraphBackend, project: str, *, debug: bool = False) -> Litestar:
    """Build the ``atlas ui`` application.

    *graph* is injected rather than constructed here so the caller owns its lifecycle —
    the CLI opens it, serves, and closes it. That also makes the app trivially testable
    against a fake backend, which is why the service layer depends on the
    ``GraphBackend`` protocol and not on ``GraphClient``.
    """
    from litestar.contrib.jinja import JinjaTemplateEngine  # noqa: PLC0415

    async def provide_view_service() -> ProjectViewService:
        return ProjectViewService(graph, project)

    template_config: TemplateConfig[Any] = TemplateConfig(directory=TEMPLATES_DIR, engine=JinjaTemplateEngine)

    return Litestar(
        route_handlers=[
            ProjectController,
            HealthController,
            # Vendored, never CDN — `atlas ui` must work offline, and the static export
            # (ATL-120) must not reach the network at all. See ADR-0033.
            create_static_files_router(path="/static", directories=[STATIC_DIR], name="static"),
        ],
        dependencies={"view_service": Provide(provide_view_service)},
        # Annotated rather than inlined: `TemplateConfig` is generic in its engine and
        # Litestar's own parameter is `TemplateConfig[EngineType] | None`, so the
        # inferred `TemplateConfig[JinjaTemplateEngine]` reads as a variance error at
        # the call site. Naming the wider type is the fix; the runtime value is the same.
        template_config=template_config,
        debug=debug,
    )
