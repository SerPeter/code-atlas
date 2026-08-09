"""Static HTML export — the second renderer over the same components.

`atlas ui --export out.html` writes one self-contained file: no server, no Memgraph, no
network. It is deliberately *not* a parallel implementation. It embeds the same map
payload the live `/map/api` route serves and the same `map.js` that renders it, so the
two cannot disagree about the same project.

**Nothing here may reach the network.** A `file://` document has an *empty* hostname,
which is why third-party telemetry would have fired from an export despite skipping
localhost (ADR-0033). The fonts are folded into the stylesheet as data URIs and the map
data is embedded rather than fetched.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import msgspec
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from markupsafe import Markup

from code_atlas.server.web import STATIC_DIR, TEMPLATES_DIR
from code_atlas.server.web.services import (
    ChromeService,
    MapViewService,
    ProjectNotIndexedError,
    ProjectViewService,
)

if TYPE_CHECKING:
    from pathlib import Path

    from code_atlas.graph.protocol import GraphBackend
    from code_atlas.server.web.schemas import MapPayload


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
    """Renders the project's map into one portable document."""

    def __init__(self, graph: GraphBackend, project: str) -> None:
        self._graph = graph
        self._project = project

    async def render(self, *, generated_at: datetime | None = None) -> str:
        """Build the document. Raises :class:`ProjectNotIndexedError` if there is nothing to show."""
        return (await self._build(generated_at))[0]

    async def _build(self, generated_at: datetime | None) -> tuple[str, MapPayload]:
        """Render once and hand back the payload alongside it.

        Rendering and then re-querying for the summary would run the traversal twice
        and could report a different graph from the one embedded in the file.
        """
        # Only to prove the project is indexed — a file full of zeroes would be worse
        # than no file, since the reader cannot tell it from a genuinely empty project.
        await ProjectViewService(self._graph, self._project).overview()
        payload = await MapViewService(self._graph, self._project).map()
        chrome = await ChromeService(self._graph, (self._project,)).chrome()

        stamp = generated_at or datetime.now(UTC)
        document = (
            self._environment()
            .get_template("export.html")
            .render(
                chrome=chrome,
                # Markup, because these are the payload rather than text to display. The
                # environment autoescapes (matching Litestar's engine) and would
                # otherwise turn every quote in the bundle into an entity.
                map_data_json=Markup(_embed_json(payload)),
                map_js=Markup(_read_asset("map.js")),
                design_css=Markup(_inlined_css()),
                generated_at=stamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
            )
        )
        return document, payload

    async def write(self, path: Path, *, generated_at: datetime | None = None) -> ExportResult:
        """Render and write to *path*."""
        document, payload = await self._build(generated_at)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document, encoding="utf-8")

        return ExportResult(
            path=path,
            bytes_written=len(document.encode("utf-8")),
            project=self._project,
            node_count=len(payload.nodes),
            map_available=payload.is_available,
        )

    def _environment(self) -> Environment:
        """A Jinja environment matching the live server's.

        ``StrictUndefined`` is the one deliberate difference. The server can survive a
        missing variable as a blank cell on a page someone will reload; an export is
        written once and mailed to somebody, so a silently empty section would ship.
        """
        return Environment(
            loader=FileSystemLoader(str(TEMPLATES_DIR)),
            autoescape=True,
            undefined=StrictUndefined,
        )


def _read_asset(relative: str) -> str:
    return (STATIC_DIR / relative).read_text(encoding="utf-8")


def _inlined_css() -> str:
    """The stylesheet with its font files folded in as data URIs.

    `design.css` points at `/static/vendor/archivo-*.woff2`, which resolves through the
    server and nowhere else. An export opened from a filesystem would silently fall back
    to the system font — the page would still render, so nothing would look broken, and
    the design would simply be gone. Embedding costs ~33% over the raw bytes and removes
    the failure entirely.
    """
    css = (STATIC_DIR / "design.css").read_text(encoding="utf-8")
    for font in sorted((STATIC_DIR / "vendor").glob("archivo-*.woff2")):
        encoded = base64.b64encode(font.read_bytes()).decode("ascii")
        css = css.replace(
            f'url("/static/vendor/{font.name}")',
            f'url("data:font/woff2;base64,{encoded}")',
        )
    return css


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
