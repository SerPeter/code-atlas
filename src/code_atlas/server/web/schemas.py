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


class TrendPoint(msgspec.Struct, frozen=True):
    """One recorded index run, for the trend table."""

    at: str
    commit: str
    modules: int
    propagation_cost: float
    core_size: float
    largest_cycle: int

    @property
    def propagation_pct(self) -> str:
        return f"{self.propagation_cost * 100:.1f}%"

    @property
    def core_pct(self) -> str:
        return f"{self.core_size * 100:.1f}%"


class ArchitectureTrend(msgspec.Struct, frozen=True):
    """How the architecture numbers have moved across recorded index runs.

    ``direction`` is ``"unclear"`` whenever coverage moved enough to explain the change on
    its own. A propagation cost that rose because a language's extraction improved is not
    a codebase that decayed, and calling that "worse" would be a confident wrong answer.

    ``note`` carries the retention bound, because a window silently capped at fifty runs
    reads as the whole history.
    """

    points: tuple[TrendPoint, ...]
    direction: str
    propagation_delta: float
    core_delta: float
    coverage_changed: bool
    note: str

    @property
    def has_trend(self) -> bool:
        """One point is not a trend; a line drawn through it invents a direction."""
        return len(self.points) >= 2

    @property
    def propagation_delta_pct(self) -> str:
        return f"{self.propagation_delta * 100:+.1f}%"


class CycleDetail(msgspec.Struct, frozen=True):
    """One dependency cycle, with the edges that actually close it.

    Members alone are not actionable — "these six modules are tangled" does not say
    which import to cut. The edges do, so they travel with the cycle rather than being
    a second lookup the view might skip.
    """

    members: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]

    @property
    def size(self) -> int:
        return len(self.members)


class ArchitectureHealth(msgspec.Struct, frozen=True):
    """The mud report, ready to render.

    ``dsm_order`` and ``dsm_marks`` are the matrix: modules on both axes in dependency
    order, a mark where the row depends on the column. Marks below the diagonal are
    layering; marks **above** it are cycles, and they are what the view is for.

    Capped at ``dsm_limit`` modules because an N x N grid is quadratic in the page.
    ``dsm_truncated`` says so rather than silently showing a corner of the matrix as if
    it were the whole thing.
    """

    project: str
    module_count: int
    edge_count: int
    propagation_cost: float
    core_size: float
    largest_cycle: int
    fan_in_gini: float
    cycles: tuple[CycleDetail, ...]
    dsm_order: tuple[str, ...]
    dsm_marks: tuple[tuple[int, int], ...]
    dsm_truncated: bool
    caveat: CoverageCaveat
    trend: ArchitectureTrend | None = None

    @property
    def propagation_pct(self) -> str:
        return f"{self.propagation_cost * 100:.1f}%"

    @property
    def core_pct(self) -> str:
        return f"{self.core_size * 100:.1f}%"


class MapNode(msgspec.Struct, frozen=True):
    """One module on the map.

    ``x``/``y`` are computed server-side. The client runs no force simulation on first
    paint — a layout that settles in the browser makes the same graph look different on
    every reload, which destroys the one thing a map is for: recognising it again.
    """

    id: str
    label: str
    community: int
    size: float
    x: float
    y: float
    project: str
    is_external: bool = False


class MapEdge(msgspec.Struct, frozen=True):
    """A weighted module-to-module dependency.

    ``weight`` is the stored edge weight (ADR-0017: ``1 / candidate_count``, halved when
    unverified, quartered when test-origin), summed over the pair. Thickness therefore
    tracks how well-evidenced a dependency is, not merely how often it appears.
    """

    source: str
    target: str
    weight: float
    crosses_community: bool = False


class CommunityRef(msgspec.Struct, frozen=True):
    """A detected subsystem — id, size, and the modules in it."""

    id: int
    size: int
    label: str
    members: tuple[str, ...]


class ModuleMap(msgspec.Struct, frozen=True):
    """The map view model.

    ``unavailable`` is not an error string: community detection needs MAGE, which the
    SQLite backend does not have. Rendering a partial map there would be worse than
    rendering none, because a map with modules silently missing still looks complete.
    """

    project: str
    nodes: tuple[MapNode, ...]
    edges: tuple[MapEdge, ...]
    communities: tuple[CommunityRef, ...]
    modularity: float
    truncated: bool
    caveat: CoverageCaveat
    unavailable: str = ""

    @property
    def is_available(self) -> bool:
        return not self.unavailable

    @property
    def external_count(self) -> int:
        return sum(1 for n in self.nodes if n.is_external)


class AffectedEntity(msgspec.Struct, frozen=True):
    """One entity a change to the target could reach.

    ``via`` is load-bearing. It names the edge types that land on the *target*, so a
    dependent found through REFERENCES or USES_TYPE is never read as a caller — ADR-0029
    calls moving that from an omission to real output the only place it was useful, and a
    rendered list is the easiest place to lose it again.
    """

    uid: str
    name: str
    qualified_name: str
    label: str
    file_path: str
    depth: int
    via: tuple[str, ...]
    via_lines: tuple[int, ...] = ()
    ambiguous_only: bool = False
    test_only: bool = False
    confidence_score: float = 1.0

    @property
    def is_call(self) -> bool:
        """Whether this really is a caller/callee, rather than some other dependency."""
        return "CALLS" in self.via


class DepthGroup(msgspec.Struct, frozen=True):
    """Everything reachable in exactly *depth* hops."""

    depth: int
    entities: tuple[AffectedEntity, ...]


class BlastRadiusView(msgspec.Struct, frozen=True):
    """What a change to one entity could reach.

    ``affected_count`` is the true total from the traversal, not the page size — the
    analysis computes the whole closure and then slices, so unlike a search this one
    genuinely knows (ATL-111).

    ``considered`` is what the resolved-only filter was applied over. Filtering a page is
    not the same as paging a filtered set, and saying which happened is the difference
    between an honest count and a plausible one.
    """

    uid: str
    target_name: str
    direction: str
    max_depth: int
    groups: tuple[DepthGroup, ...]
    affected_count: int
    shown: int
    considered: int
    resolved_only: bool
    truncated: bool
    remedy: str
    caveat: CoverageCaveat
    error: str = ""

    @property
    def is_found(self) -> bool:
        return not self.error


class PathHop(msgspec.Struct, frozen=True):
    """One edge on a traced path, with the claim behind it."""

    from_uid: str
    from_name: str
    to_uid: str
    to_name: str
    edge_type: str
    confidence: str = ""
    strategy: str = ""
    weight: float | None = None
    at_line: int | None = None
    from_test: bool = False

    @property
    def is_guess(self) -> bool:
        return self.confidence == "ambiguous"

    @property
    def is_structural(self) -> bool:
        return not self.confidence and not self.strategy


class TracePathView(msgspec.Struct, frozen=True):
    """How two entities connect, hop by hop."""

    from_uid: str
    to_uid: str
    found: bool
    hops: tuple[PathHop, ...] = ()
    hop_count: int | None = None
    path_weight: float | None = None
    message: str = ""
    error: str = ""

    @property
    def has_guessed_hop(self) -> bool:
        """A path is only as trustworthy as its weakest hop."""
        return any(hop.is_guess for hop in self.hops)
