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
