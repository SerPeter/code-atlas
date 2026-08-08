"""The `atlas ui` web interface.

Three layers, outermost first:

* :mod:`controllers` — HTTP: routing, status codes, template selection.
* :mod:`services` — the use case: what a view needs and how to assemble it.
* the graph backend and :mod:`code_atlas.server.analysis` — data access.

:mod:`app` is the composition root and the only module that knows where anything
comes from. :mod:`schemas` holds the view models that cross the service/HTTP boundary.

Every dependency here lives behind the optional ``ui`` extra, so nothing in this
package may be imported from the CLI or MCP server without guarding for it.
"""

from pathlib import Path

# Asset locations live here rather than in `app`, so the static exporter can find them
# without importing Litestar — export needs jinja2 alone.
_WEB_ROOT = Path(__file__).parent
TEMPLATES_DIR = _WEB_ROOT / "templates"
STATIC_DIR = _WEB_ROOT / "static"

__all__ = ["STATIC_DIR", "TEMPLATES_DIR"]
