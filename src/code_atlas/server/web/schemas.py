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
    """One node on the map, in the shape map.js reads.

    ``x``/``y`` are in the canvas's 0-1000 space, computed server-side with the design's
    own layout — a layout that settles in the browser makes the same graph look
    different on every reload, which destroys the one thing a map is for: recognising
    it again. ``deg`` is the undirected dependency degree; the client turns it into a
    radius relative to the space each node actually gets.
    """

    id: str
    label: str
    community: int
    deg: int
    kind: str
    x: float
    y: float
    # Empty when the graph holds no such thing — the client hides the affordance
    # rather than linking to a page that would 404.
    uid: str = ""
    # The module's file path — what the entity level's scope selector keys on.
    path: str = ""
    lines: str = ""


class MapEdge(msgspec.Struct, frozen=True):
    """A directed dependency: ``s`` depends on ``t``.

    ``w`` is the ADR-0017 aggregate weight scaled into the design's 1-3 band (the
    canvas's thickness formula expects that range). ``ev`` is the strongest ADR-0028
    claim among the entity edges the pair aggregates — and "unknown" means the edge was
    never looked up, which the canvas draws as the thinnest dotted line, not as absence.

    ``rel`` distinguishes containment from dependency: a "defines" edge is the module or
    class holding its member — structural scaffolding the canvas draws as a faint
    hairline, so a dense scope's call graph stays readable instead of drowning under a
    starburst of anchors.
    """

    s: str
    t: str
    w: float
    ev: str
    rel: str = ""


class CommunityRef(msgspec.Struct, frozen=True):
    """A detected community, as the rail lists it.

    ``count`` is the community's size **in the index** — the drawn share is computed
    client-side from the nodes actually on the canvas, so the two numbers can never
    silently disagree.
    """

    id: int
    name: str
    count: int
    color: str
    files: bool = False


class KindDef(msgspec.Struct, frozen=True):
    """One of the twelve drawable kinds — id, display label, silhouette."""

    id: str
    label: str
    shape: str
    note: str = ""


class KindTally(msgspec.Struct, frozen=True):
    """A kind's count over the module's whole inventory, and how many are drawn.

    ``in_module`` is counted before folding or truncation — a kind the module holds
    must never read as absent because the drawing dropped it.
    """

    id: str
    in_module: int
    drawn: int


class ScopeOption(msgspec.Struct, frozen=True):
    """A module the entity level can be scoped to."""

    id: str
    label: str
    entities: int


class MapPayload(msgspec.Struct, frozen=True):
    """Everything the map island renders, for one level.

    ``unavailable`` is not an error string: community detection needs raw Cypher the
    SQLite backend cannot serve. Rendering a partial map there would be worse than
    rendering none, because a map with modules silently missing still looks complete.
    """

    project: str
    level: str
    nodes: tuple[MapNode, ...]
    edges: tuple[MapEdge, ...]
    communities: tuple[CommunityRef, ...]
    kinds: tuple[KindDef, ...]
    caveat: CoverageCaveat
    unavailable: str = ""
    # The totals every sidebar sentence derives from. All counted over the index,
    # not the drawn subset — the client states the drawn share separately.
    module_total: int = 0
    edge_total: int = 0
    test_count: int = 0
    noncode_count: int = 0
    entity_total: int = 0
    truncated: bool = False
    scope_options: tuple[ScopeOption, ...] = ()
    default_scope: str = ""
    # Entity level only.
    scope: str = ""
    scope_name: str = ""
    in_module: int = 0
    collapsed: bool = False
    tally: tuple[KindTally, ...] = ()

    @property
    def is_available(self) -> bool:
        return not self.unavailable


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


class ProjectChoice(msgspec.Struct, frozen=True):
    """One selectable project in the picker.

    ``children`` carries the monorepo shape: `trading-bot` really does have four
    sub-projects, and they are separate graph projects rather than folders — selecting
    the parent does not pull them in.
    """

    name: str
    label: str
    entities: int
    modules: int
    indexed_at: str | None
    git_hash: str | None
    is_current: bool
    days_since_indexed: int | None = None
    children: tuple[ProjectChoice, ...] = ()
    # The design's three states, decided server-side so the dialog never re-derives
    # them: "unindexed" (no graph data — not selectable, its fix is `atlas index`),
    # "stale" (older than a fortnight), "fresh". The two failure states must not look
    # alike because their remedies differ.
    state: str = "fresh"
    # "2 hours ago" — relative, because the dialog compares recency at a glance.
    indexed_ago: str = ""

    @property
    def is_stale(self) -> bool:
        """Old enough that the code has probably moved on.

        A fortnight is arbitrary but has to be *something*: a view of two-week-old data
        that looks live is the failure the coverage rules exist to prevent.
        """
        return self.days_since_indexed is not None and self.days_since_indexed > 14

    @property
    def short_hash(self) -> str:
        return (self.git_hash or "")[:7]


class ProjectPicker(msgspec.Struct, frozen=True):
    """Every indexed project, and what selecting them would cost.

    ``cost_note`` exists because combining projects is the one action here with a real
    performance cliff — the picker should say so before the map has to.
    """

    projects: tuple[ProjectChoice, ...]
    selected: tuple[str, ...]
    selected_modules: int
    cost_note: str

    @property
    def total_projects(self) -> int:
        return sum(1 + len(p.children) for p in self.projects)


class PageChrome(msgspec.Struct, frozen=True):
    """What the shell needs on every page: header chip, indexed note, footer."""

    project_name: str
    project_meta: str
    indexed_note: str
    footer_indexed: str
    coverage_note: str
    backend_note: str


class SearchRow(msgspec.Struct, frozen=True):
    """One search result, ready to render as the design draws it."""

    uid: str
    label: str
    kind: str
    loc: str
    sig: str
    # The channels that found this hit, out of graph/keyword/semantic.
    channels: tuple[str, ...]
    strength: str
    score: str


class ChannelFilter(msgspec.Struct, frozen=True):
    """A retrieval channel in the rail — hit count over the fetched set only."""

    id: str
    label: str
    count: str
    on: bool
    url: str


class KindFilter(msgspec.Struct, frozen=True):
    """A kind chip in the rail — label carries its count over the fetched set."""

    id: str
    label_count: str
    on: bool
    url: str


class SearchView(msgspec.Struct, frozen=True):
    """The search page, filters applied.

    ``result_note`` never states a total: the search cannot count matches it did not
    fetch, so the note says what was fetched and that more exist, quantity unknown.
    """

    query: str
    rows: tuple[SearchRow, ...]
    channels: tuple[ChannelFilter, ...]
    kind_filters: tuple[KindFilter, ...]
    result_note: str
    searching_note: str


class ImpactRoot(msgspec.Struct, frozen=True):
    """A subject the rail offers — the highest-degree modules."""

    id: str
    label: str
    on: bool
    url: str


class ImpactRow(msgspec.Struct, frozen=True):
    """One dependent module. ``ev`` is the WEAKEST evidence anywhere along the path
    that reached it, not the evidence of its last hop."""

    id: str
    label: str
    rel: str
    path_note: str
    ev: str
    url: str


class ImpactHopGroup(msgspec.Struct, frozen=True):
    """Everything reached at exactly one distance."""

    hop_label: str
    count_label: str
    rows: tuple[ImpactRow, ...]


class ImpactView(msgspec.Struct, frozen=True):
    """The impact page: what transitively depends on one module."""

    subject_id: str
    subject_label: str
    total_note: str
    groups: tuple[ImpactHopGroup, ...]
    roots: tuple[ImpactRoot, ...]
    confident_only: bool
    confident_url: str
    unavailable: str = ""


class MetricCard(msgspec.Struct, frozen=True):
    """One architecture metric, with the caveat naming its population."""

    label: str
    value: str
    note: str


class DsmCell(msgspec.Struct, frozen=True):
    """One matrix cell. ``mark`` is '' (empty), 'diag', 'dep' or 'cycle'."""

    mark: str
    title: str = ""


class DsmRow(msgspec.Struct, frozen=True):
    """One matrix row — the module and its dependencies on every column."""

    label: str
    cells: tuple[DsmCell, ...]


class CycleRow(msgspec.Struct, frozen=True):
    """One cycle read off the matrix, both closing edges named so either can be cut."""

    members: str
    closing: str
    closing2: str
    note: str
    ev: str


class TrendRow(msgspec.Struct, frozen=True):
    """One recorded index run in the trend table.

    ``direction`` reads "unclear — index grew N%" whenever coverage moved by more than
    a tenth between runs: a metric that moved because extraction improved is not a
    codebase that decayed.
    """

    date: str
    commit: str
    modules: str
    propagation: str
    largest_cycle: str
    direction: str
    accented: bool


class ReferenceRow(msgspec.Struct, frozen=True):
    """A published propagation-cost measurement, for scale only."""

    label: str
    value: str


class ArchitectureView(msgspec.Struct, frozen=True):
    """The architecture page — deliberately not a node-link graph."""

    cards: tuple[MetricCard, ...]
    dsm_rows: tuple[DsmRow, ...]
    dsm_caption: str
    cycles: tuple[CycleRow, ...]
    trend_rows: tuple[TrendRow, ...]
    references: tuple[ReferenceRow, ...]
    unavailable: str = ""
    # Restates the cut whenever the inventory lists fewer cycles than the matrix
    # marks — a capped list that looks complete is the failure this page bans.
    cycles_caption: str = "read off the matrix — both edges named, cut either"


class DetailRelated(msgspec.Struct, frozen=True):
    """A neighbour row on the entity page — breadcrumb, relationship, evidence."""

    uid: str
    label: str
    rel: str
    ev: str


class DetailEvidenceMix(msgspec.Struct, frozen=True):
    """How many of the entity's callers carry each evidence state."""

    ev: str
    n: int


class DetailView(msgspec.Struct, frozen=True):
    """The entity page, in the design's shape."""

    uid: str
    name: str
    short_name: str
    kind: str
    file: str
    lines: str
    file_lines: str
    parent_line: str
    signature: str
    paragraphs: tuple[str, ...]
    callers: tuple[DetailRelated, ...]
    callees: tuple[DetailRelated, ...]
    docs: tuple[DetailRelated, ...]
    evidence_mix: tuple[DetailEvidenceMix, ...]
    caller_note: str
    callers_count_note: str
    callees_count_note: str
