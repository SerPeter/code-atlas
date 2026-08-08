"""View models for the web UI.

``msgspec.Struct`` rather than pydantic: Litestar serialises these directly, and the
graph payloads this will eventually carry (thousands of nodes and edges per view) are
the one place in this project where serialisation cost is on the hot path.

These are the boundary between the service layer and HTTP. A controller returns one of
these; a service builds one. Neither the graph's row dicts nor Litestar's request types
cross that line in either direction.
"""

from __future__ import annotations

import msgspec


class ProjectRef(msgspec.Struct, frozen=True):
    """A project the UI can switch to."""

    name: str
    entities: int
    is_current: bool


class CoverageCaveat(msgspec.Struct, frozen=True):
    """What the numbers on a view do not cover.

    Every view that reports an aggregate carries one of these. A picture reads as
    complete in a way a tool result does not, so a view built on partial extraction has
    to say so — C++ named-function capture sits at 0.690 (ATL-096), and a confident
    architecture score over that would be exactly the failure this project spends its
    effort eliminating.
    """

    languages_missing_grammar: tuple[str, ...] = ()
    note: str = ""

    @property
    def is_complete(self) -> bool:
        return not self.languages_missing_grammar and not self.note


class ProjectOverview(msgspec.Struct, frozen=True):
    """The landing view: what this project is, and how much of it we can see."""

    project: str
    entity_count: int
    module_count: int
    indexed_at: str | None
    git_hash: str | None
    label_counts: dict[str, int]
    other_projects: tuple[ProjectRef, ...]
    caveat: CoverageCaveat


class SearchHit(msgspec.Struct, frozen=True):
    """One fused search result."""

    uid: str
    name: str
    qualified_name: str
    kind: str
    label: str
    file_path: str
    line_start: int | None
    signature: str
    score: float
    channels: tuple[str, ...]


class SearchPage(msgspec.Struct, frozen=True):
    """A page of results, and an honest statement of what it left out.

    ``more_available`` rather than a count: the search fetches one row beyond the page
    to learn *whether* more exist, which is not the same as knowing how many. Reporting
    the fetch size as a total is exactly the lie ATL-111 removed from the MCP tools, and
    a UI repeating it would be worse — a list on screen reads as the whole answer.
    """

    query: str
    hits: tuple[SearchHit, ...]
    more_available: bool


class EdgeEvidence(msgspec.Struct, frozen=True):
    """Why the graph believes one entity reaches another (ADR-0028).

    A caller found by matching an import is a very different claim from one found by
    matching a bare name across the whole project, and until this view existed a human
    had no way to see which they were looking at.

    ``strategy`` empty and ``confidence`` empty means a **structural** edge — DEFINES,
    IMPORTS, INHERITS. Those are facts rather than guesses, which is why an absent
    confidence coalesces to "resolved" everywhere else in the codebase (ADR-0029).
    """

    rel_type: str
    strategy: str = ""
    confidence: str = ""
    weight: float | None = None
    line: int | None = None
    site_count: int | None = None

    @property
    def is_structural(self) -> bool:
        return not self.confidence and not self.strategy

    @property
    def is_guess(self) -> bool:
        """True when the resolver could not pin this down to one target."""
        return self.confidence == "ambiguous"


class RelatedEntity(msgspec.Struct, frozen=True):
    """A neighbour of the entity being viewed, with the edge that reached it."""

    uid: str
    name: str
    qualified_name: str
    kind: str
    file_path: str
    line_start: int | None
    evidence: EdgeEvidence | None = None


class EntityDetail(msgspec.Struct, frozen=True):
    """Everything the graph knows about one entity."""

    uid: str
    name: str
    qualified_name: str
    kind: str
    label: str
    file_path: str
    line_start: int | None
    line_end: int | None
    signature: str
    docstring: str
    parent: RelatedEntity | None
    callers: tuple[RelatedEntity, ...]
    callees: tuple[RelatedEntity, ...]
    docs: tuple[RelatedEntity, ...]
    caveat: CoverageCaveat
