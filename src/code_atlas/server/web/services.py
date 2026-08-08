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

from typing import TYPE_CHECKING

from code_atlas.server.web.schemas import CoverageCaveat, ProjectOverview, ProjectRef

if TYPE_CHECKING:
    from code_atlas.graph.protocol import GraphBackend

# Label tallies are whole-project aggregates; the limit only bounds the ranked lists
# this view does not read.
_STRUCTURE_LIMIT = 20


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
    project**, with other projects reachable but not loaded. The alternative — render
    everything — is an unbounded query against a graph that is already ~30k nodes for
    eight projects, and it is offered as an explicit opt-in rather than a default.
    """

    def __init__(self, graph: GraphBackend, project: str) -> None:
        self._graph = graph
        self._project = project

    @property
    def project(self) -> str:
        return self._project

    async def overview(self) -> ProjectOverview:
        """The landing view for the current project."""
        statuses = await self._graph.get_project_status()
        current = next((s for s in statuses if s.get("project") == self._project), None)
        if current is None:
            raise ProjectNotIndexedError(self._project)

        # Straight to the data layer rather than through `analyze_repo`. Routing the
        # overview through the MCP-facing dispatcher coupled this service to that
        # analysis's whole contract — largest_modules, packages, external_deps — for a
        # label tally that is four lines. Both read `get_structure_overview`, so the
        # numbers still share one source; only the needless coupling is gone.
        structure = await self._graph.get_structure_overview(self._project, "", _STRUCTURE_LIMIT)
        label_counts: dict[str, int] = {}
        for row in structure.get("counts", []):
            label = str(row.get("label", ""))
            if label:
                label_counts[label] = label_counts.get(label, 0) + int(row.get("cnt") or 0)

        others = tuple(
            ProjectRef(
                name=str(s.get("project", "")),
                entities=int(s.get("entities") or 0),
                is_current=s.get("project") == self._project,
            )
            for s in statuses
            if s.get("project")
        )

        return ProjectOverview(
            project=self._project,
            entity_count=int(current.get("entities") or 0),
            module_count=int(label_counts.get("Module", 0)),
            indexed_at=_as_str(current.get("indexed_at")),
            git_hash=_as_str(current.get("git_hash")),
            label_counts=label_counts,
            other_projects=others,
            caveat=_coverage_caveat(label_counts),
        )


def _as_str(value: object) -> str | None:
    """Normalise a graph value to a display string, preserving "absent"."""
    if value is None or value == "":
        return None
    return str(value)


def _coverage_caveat(label_counts: dict[str, int]) -> CoverageCaveat:
    """State what these numbers do not cover.

    Placeholder shape for now: the real signal is which languages in this project had no
    grammar installed, which ATL-110 already records per index run but does not yet
    persist to the graph. Wiring that through is ATL-119's dependency, not this story's —
    what matters here is that every view carries the field from the start, so adding the
    data later cannot be forgotten.
    """
    return CoverageCaveat(note="" if label_counts else "No entities indexed for this project.")
