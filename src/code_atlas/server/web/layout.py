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


# ---------------------------------------------------------------------------
# Force-directed layout (ATL-123)
# ---------------------------------------------------------------------------

# Fixed, so the same graph lays out identically on every request. The determinism is the
# requirement; the particular value is arbitrary.
_SEED = 20260809
_ITERATIONS = 300


def force_layout(
    nodes: list[str],
    edges: dict[tuple[str, str], float],
    *,
    iterations: int = _ITERATIONS,
    seed: int = _SEED,
) -> dict[str, tuple[float, float]]:
    """Fruchterman-Reingold, vectorised, seeded.

    Position becomes a function of the edges: coupled modules pull together, unrelated
    ones drift apart. That is the whole point — the previous clustered-circular layout
    placed communities on a fixed ring, so every node landed in a band (measured on the
    real graph: 0 of 126 nodes inside radius 60) and distance meant nothing.

    Determinism was the reason that ring existed. It is preserved here by seeding the
    initial positions and running a fixed iteration count, which is what makes a map
    comparable to the one you looked at last week. An *unseeded* force layout would not
    be; a seeded one is, and it also carries information.

    Repulsion is O(n^2) per iteration, done in numpy rather than Python loops — at the
    1500-node cap that is 2.25M pair distances per step, which is a few milliseconds
    vectorised and minutes interpreted.
    """
    import numpy as np  # noqa: PLC0415  # heavy import, only needed when a map is drawn

    count = len(nodes)
    if count == 0:
        return {}
    if count == 1:
        return {nodes[0]: (0.0, 0.0)}

    index = {name: i for i, name in enumerate(nodes)}
    rng = np.random.default_rng(seed)
    # Seeded ring start rather than pure noise: a deterministic, well-spread opening
    # converges faster and avoids the coincident-point singularity at step one.
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    pos = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 100.0
    pos += rng.normal(0.0, 2.0, size=(count, 2))

    src, dst, strength = [], [], []
    # Sorted, because np.add.at accumulates in array order and float addition is not
    # associative — the same graph with its edges in a different dict order would
    # otherwise settle a hair differently, and dict order here comes from query results.
    for (a, b), weight in sorted(edges.items()):
        if a in index and b in index and a != b:
            src.append(index[a])
            dst.append(index[b])
            # Log-compressed: raw weights span 0.0027 to 126.79 on the real graph, and a
            # 47,000x pull would collapse the heavy pairs onto one point.
            strength.append(1.0 + np.log1p(max(weight, 0.0)))
    src_i = np.asarray(src, dtype=np.intp)
    dst_i = np.asarray(dst, dtype=np.intp)
    pull = np.asarray(strength, dtype=float)

    area = 1.0
    k = np.sqrt(area / count)
    temperature = 0.1
    cooling = temperature / (iterations + 1)
    scale = 200.0
    pos /= scale  # work in unit space, rescale at the end

    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        distance = np.linalg.norm(delta, axis=-1)
        np.clip(distance, 0.01, None, out=distance)

        # Everything pushes everything, inversely with distance.
        repulsion = (k * k) / distance
        np.fill_diagonal(repulsion, 0.0)
        displacement = np.einsum("ijk,ij->ik", delta, repulsion / distance)

        if src_i.size:
            edge_delta = pos[src_i] - pos[dst_i]
            edge_dist = np.linalg.norm(edge_delta, axis=-1)
            np.clip(edge_dist, 0.01, None, out=edge_dist)
            attraction = (edge_dist / k) * pull
            force = edge_delta * (attraction / edge_dist)[:, None]
            np.add.at(displacement, src_i, -force)
            np.add.at(displacement, dst_i, force)

        length = np.linalg.norm(displacement, axis=-1)
        np.clip(length, 0.01, None, out=length)
        step = np.minimum(length, temperature) / length
        pos += displacement * step[:, None]
        temperature -= cooling

    # Normalise into a stable box so the client never has to guess a viewport.
    span = pos.max(axis=0) - pos.min(axis=0)
    span[span == 0] = 1.0
    pos = (pos - pos.min(axis=0)) / span * 2.0 - 1.0
    pos *= scale

    return {name: (float(pos[i, 0]), float(pos[i, 1])) for name, i in index.items()}
