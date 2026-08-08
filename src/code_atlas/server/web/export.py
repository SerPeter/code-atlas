"""Static HTML export — the second renderer over the same components.

`atlas ui --export out.html` writes one self-contained file: no server, no Memgraph, no
network. It is deliberately *not* a parallel implementation. It calls the same view
services and renders the same template partials the live server does, so the two cannot
disagree about the same project. A second implementation would drift, and the export is
the one nobody would notice had gone stale.

What differs is asset delivery and data delivery, and only that: the server links its
JS and serves the map payload from an endpoint; the export inlines both. The markup in
between is byte-identical because it is literally the same partials.

**Nothing here may reach the network.** A `file://` document has an *empty* hostname,
which is why the telemetry in `@cosmograph/cosmograph` would have fired from an export
despite skipping localhost (ADR-0033). The vendored bundles are MIT and phone home
nowhere, and the map data is embedded rather than fetched.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import msgspec
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from markupsafe import Markup

from code_atlas.server.web import STATIC_DIR, TEMPLATES_DIR
from code_atlas.server.web.services import (
    ArchitectureViewService,
    MapViewService,
    ProjectNotIndexedError,
    ProjectViewService,
)

if TYPE_CHECKING:
    from pathlib import Path

    from code_atlas.graph.protocol import GraphBackend
    from code_atlas.server.web.schemas import ModuleMap

# Load order matters: sigma takes graphology as a peer, not a bundled dependency.
_VENDOR_BUNDLES = ("vendor/graphology-0.26.0.umd.min.js", "vendor/sigma-3.0.3.min.js")


@dataclass(frozen=True)
class ExportResult:
    """What was written, and what it covers."""

    path: Path
    bytes_written: int
    project: str
    node_count: int
    map_available: bool

    @property
    def size_mb(self) -> float:
        return self.bytes_written / 1_048_576


class StaticExporter:
    """Renders the project's views into one portable document."""

    def __init__(self, graph: GraphBackend, project: str) -> None:
        self._graph = graph
        self._project = project

    async def render(self, *, generated_at: datetime | None = None) -> str:
        """Build the document. Raises :class:`ProjectNotIndexedError` if there is nothing to show."""
        return (await self._build(generated_at))[0]

    async def _build(self, generated_at: datetime | None) -> tuple[str, ModuleMap]:
        """Render once and hand back the map alongside it.

        Rendering and then re-querying for the summary would run the traversal twice and
        could report a different graph from the one embedded in the file.
        """
        overview = await ProjectViewService(self._graph, self._project).overview()
        module_map = await MapViewService(self._graph, self._project).map()
        health = await ArchitectureViewService(self._graph, self._project).health()

        stamp = generated_at or datetime.now(UTC)
        document = (
            self._environment()
            .get_template("export.html")
            .render(
                overview=overview,
                map=module_map,
                health=health,
                # Markup, because these are the payload rather than text to display. The
                # environment autoescapes (matching Litestar's engine, so the shared partials
                # render identically under both drivers) and would otherwise turn every
                # quote in the bundle into an entity.
                map_data_json=Markup(_embed_json(module_map)),
                vendor_js=Markup(_read_assets(_VENDOR_BUNDLES)),
                map_js=Markup(_read_assets(("map.js",))),
                generated_at=stamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                indexed_at=overview.indexed_at,
            )
        )
        return document, module_map

    async def write(self, path: Path, *, generated_at: datetime | None = None) -> ExportResult:
        """Render and write to *path*."""
        document, module_map = await self._build(generated_at)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document, encoding="utf-8")

        return ExportResult(
            path=path,
            bytes_written=len(document.encode("utf-8")),
            project=self._project,
            node_count=len(module_map.nodes),
            map_available=module_map.is_available,
        )

    def _environment(self) -> Environment:
        """A Jinja environment matching the live server's.

        ``autoescape`` is True because Litestar's engine sets it, and the partials are
        shared — an environment that escaped differently would render the same markup two
        ways, which is exactly the divergence this module exists to prevent.

        ``StrictUndefined`` is the one deliberate difference. The server can survive a
        missing variable as a blank cell on a page someone will reload; an export is
        written once and mailed to somebody, so a silently empty section would ship.
        """
        return Environment(
            loader=FileSystemLoader(str(TEMPLATES_DIR)),
            autoescape=True,
            undefined=StrictUndefined,
        )


def _read_assets(relative: tuple[str, ...]) -> str:
    """Concatenate vendored assets, each isolated in its own IIFE-safe boundary."""
    return "\n;\n".join((STATIC_DIR / name).read_text(encoding="utf-8") for name in relative)


def _embed_json(payload: Any) -> str:
    """Serialise *payload* for embedding inside a ``<script>`` element.

    ``<`` is escaped even though it is valid JSON: a ``</script>`` sequence appearing
    anywhere in the data — a docstring, a module name — would terminate the element early
    and break the document. The HTML parser sees the raw text before any JSON parser
    does, so this cannot be handled downstream.
    """
    return msgspec.json.encode(payload).decode("utf-8").replace("<", "\\u003c")


async def export_project(graph: GraphBackend, project: str, path: Path) -> ExportResult:
    """Write *project*'s snapshot to *path*.

    Raises :class:`ProjectNotIndexedError` when the project has no graph data — writing a
    file full of zeroes would be worse, since the reader has no way to tell it apart from
    a project that really is empty.
    """
    return await StaticExporter(graph, project).write(path)


__all__ = ["ExportResult", "ProjectNotIndexedError", "StaticExporter", "export_project"]
