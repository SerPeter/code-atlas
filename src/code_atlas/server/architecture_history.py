"""Architecture-health snapshots over time (ATL-121).

A propagation cost of 8.4% is close to meaningless on its own: against the published
anchors it sits somewhere between refactored Mozilla (~2%) and pre-refactor Mozilla
(~17%), which is most of the useful range. The same number *rising from 6% over ten
index runs* is unambiguous. Trajectory is the question the mud view exists to answer, so
the numbers have to accumulate.

Separate from :mod:`code_atlas.server.architecture`, which is deliberately pure — no
graph client, no I/O — so its metrics stay checkable against hand-worked examples. This
module is the half that touches the database, and keeping the split visible is what
stops the arithmetic from quietly acquiring a dependency on a live backend.

Snapshots live as a bounded list of JSON strings on the ``Project`` node. That is a
plain property write through ``update_project_metadata``, which both backends already
implement — no new label, no constraint, and no schema migration to verify.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from code_atlas.graph.protocol import GraphBackend
    from code_atlas.server.architecture import ArchitectureMetrics

# Bounded on purpose: a daemon-indexed repo would otherwise grow this without limit.
# Fifty runs is enough to see a trend and costs roughly ten kilobytes.
MAX_SNAPSHOTS = 50

_PROPERTY = "architecture_snapshots"


@dataclass(frozen=True)
class Snapshot:
    """One index run's architecture numbers.

    ``modules`` and ``skipped_languages`` are the coverage this was computed over, and
    they are not decoration. A propagation cost that rose because C++ extraction improved
    is not a codebase that decayed, and without the coverage the two are
    indistinguishable — which would make the trend actively misleading rather than merely
    incomplete.
    """

    at: str
    commit: str
    modules: int
    edges: int
    propagation_cost: float
    core_size: float
    largest_cycle: int
    fan_in_gini: float
    skipped_languages: tuple[str, ...] = ()

    @property
    def coverage_changed_from(self) -> str:
        """Human phrasing used when comparing two snapshots of unequal coverage."""
        return f"{self.modules} modules"


def snapshot_from_metrics(
    metrics: ArchitectureMetrics,
    *,
    commit: str = "",
    skipped_languages: tuple[str, ...] = (),
    at: datetime | None = None,
) -> Snapshot:
    """Freeze *metrics* into a recordable snapshot."""
    stamp = at or datetime.now(UTC)
    return Snapshot(
        at=stamp.isoformat(timespec="seconds"),
        commit=commit or "",
        modules=metrics.module_count,
        edges=metrics.edge_count,
        propagation_cost=round(metrics.propagation_cost, 6),
        core_size=round(metrics.core_size, 6),
        largest_cycle=metrics.largest_cycle,
        fan_in_gini=round(metrics.fan_in_gini, 6),
        skipped_languages=tuple(skipped_languages),
    )


def decode(raw: Any) -> list[Snapshot]:
    """Read snapshots off a Project node property, discarding anything unreadable.

    Tolerant by design: this is a history, and one malformed entry written by an older
    version must not cost the reader the other forty-nine.
    """
    if not isinstance(raw, (list, tuple)):
        return []
    found: list[Snapshot] = []
    for item in raw:
        try:
            data = json.loads(item) if isinstance(item, str) else item
            found.append(
                Snapshot(
                    at=str(data["at"]),
                    commit=str(data.get("commit") or ""),
                    modules=int(data["modules"]),
                    edges=int(data.get("edges") or 0),
                    propagation_cost=float(data["propagation_cost"]),
                    core_size=float(data["core_size"]),
                    largest_cycle=int(data.get("largest_cycle") or 1),
                    fan_in_gini=float(data.get("fan_in_gini") or 0.0),
                    skipped_languages=tuple(str(x) for x in (data.get("skipped_languages") or [])),
                )
            )
        except TypeError, ValueError, KeyError, json.JSONDecodeError:
            continue
    return found


def encode(snapshots: list[Snapshot]) -> list[str]:
    """Serialise for storage, newest last, trimmed to :data:`MAX_SNAPSHOTS`."""
    return [
        json.dumps(asdict(s) | {"skipped_languages": list(s.skipped_languages)}) for s in snapshots[-MAX_SNAPSHOTS:]
    ]


async def load(graph: GraphBackend, project: str) -> list[Snapshot]:
    """Every recorded snapshot for *project*, oldest first."""
    rows = await graph.get_project_status(project)
    for row in rows:
        node = row.get("n", row)
        value = node.get(_PROPERTY) if hasattr(node, "get") else None
        if value:
            return decode(value)
    return []


async def record(graph: GraphBackend, project: str, snapshot: Snapshot) -> bool:
    """Append *snapshot* to *project*'s history. Returns whether it was written.

    **Never raises.** This is telemetry about an index run, taken on the index path, and
    a failure to record how healthy the architecture is must not be able to fail the
    indexing itself. A caller that had to wrap this in its own try/except would
    eventually forget.
    """
    try:
        history = await load(graph, project)
        history.append(snapshot)
        await graph.update_project_metadata(project, **{_PROPERTY: encode(history)})
    except Exception as exc:  # broad by design: see the docstring — telemetry may not break indexing
        logger.debug("Could not record architecture snapshot for {}: {}", project, exc)
        return False
    return True


@dataclass(frozen=True)
class Trend:
    """The change between the earliest and latest snapshot in a window."""

    first: Snapshot
    last: Snapshot
    count: int

    @property
    def propagation_delta(self) -> float:
        return self.last.propagation_cost - self.first.propagation_cost

    @property
    def core_delta(self) -> float:
        return self.last.core_size - self.first.core_size

    @property
    def coverage_changed(self) -> bool:
        """Whether the graph itself changed size enough to explain the movement.

        A 10% swing in module count is enough that a metric shift cannot be attributed to
        the architecture alone — a language whose extraction improved adds real
        dependencies that were always there. Reporting "propagation cost rose" in that
        case would be a confident wrong answer, which is the failure this project keeps
        removing.
        """
        if not self.first.modules:
            return self.last.modules > 0
        return abs(self.last.modules - self.first.modules) / self.first.modules > 0.1

    @property
    def direction(self) -> str:
        """``"worse"``, ``"better"`` or ``"flat"`` — or ``"unclear"`` under coverage drift."""
        if self.coverage_changed:
            return "unclear"
        if abs(self.propagation_delta) < 0.005:
            return "flat"
        return "worse" if self.propagation_delta > 0 else "better"


def trend(snapshots: list[Snapshot], *, window: int = 10) -> Trend | None:
    """Compare the newest snapshot against the oldest in the last *window* runs.

    ``None`` with fewer than two snapshots: one point is not a trend, and drawing a line
    through it would invent a direction the data does not contain.
    """
    recent = snapshots[-window:]
    if len(recent) < 2:
        return None
    return Trend(first=recent[0], last=recent[-1], count=len(recent))
