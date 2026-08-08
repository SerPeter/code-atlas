"""Deterministic graph layout, computed server-side.

Pure geometry — no graph client, no framework, no randomness. The determinism is the
requirement, not an implementation detail: a force simulation that settles in the
browser places the same graph differently on every reload, which destroys the one thing
a map is good for. A reader should recognise the shape of their codebase, notice that a
module moved between subsystems, and be right about it.

The layout is *clustered circular*: communities are placed around a ring, and each
community's modules around a smaller ring of their own. It is not a force layout and
does not try to be — it optimises for "which subsystem is this in, and what does it
reach across", which is the question the map answers, rather than for minimal edge
crossings.
"""

from __future__ import annotations

import math

# Communities sit on this ring; members orbit within their own community.
_CLUSTER_RING = 100.0
_MIN_CLUSTER_RADIUS = 8.0
# Fan-out per member, so a 40-module community does not overlap itself.
_MEMBER_SPACING = 3.2


def cluster_positions(sizes: list[int], *, ring: float = _CLUSTER_RING) -> list[tuple[float, float]]:
    """Centre of each community, spaced around a ring.

    Larger communities are placed first and therefore land at stable angles, so adding a
    small new subsystem does not rotate the whole picture.
    """
    if not sizes:
        return []
    if len(sizes) == 1:
        return [(0.0, 0.0)]

    step = 2 * math.pi / len(sizes)
    return [(ring * math.cos(i * step), ring * math.sin(i * step)) for i in range(len(sizes))]


def cluster_radius(size: int) -> float:
    """How far a community's members orbit its centre.

    Grows with the square root of membership: area, not radius, should scale with the
    module count, or a large subsystem swamps the ring it sits on.
    """
    return max(_MIN_CLUSTER_RADIUS, _MEMBER_SPACING * math.sqrt(max(size, 1)) * 2.0)


def layout_communities(communities: list[list[str]], *, ring: float = _CLUSTER_RING) -> dict[str, tuple[float, float]]:
    """Assign every module an ``(x, y)``, grouped by community.

    A single-member community still gets its slot on the ring rather than being dropped
    to the centre — an isolated module is a real finding, and hiding it in the middle of
    the map reads as "unremarkable".
    """
    centres = cluster_positions([len(c) for c in communities], ring=ring)
    positions: dict[str, tuple[float, float]] = {}

    for (cx, cy), members in zip(centres, communities, strict=True):
        if not members:
            continue
        if len(members) == 1:
            positions[members[0]] = (cx, cy)
            continue
        radius = cluster_radius(len(members))
        step = 2 * math.pi / len(members)
        for i, member in enumerate(members):
            positions[member] = (cx + radius * math.cos(i * step), cy + radius * math.sin(i * step))

    return positions


def node_size(degree: int, *, minimum: float = 3.0, maximum: float = 14.0) -> float:
    """Marker size from a module's degree, compressed logarithmically.

    Linear sizing lets one hub module dwarf everything else into invisibility; the log
    keeps a 200-edge module visibly bigger than a 20-edge one without erasing the rest.
    """
    return min(maximum, minimum + math.log1p(max(degree, 0)) * 2.2)
