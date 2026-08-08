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
