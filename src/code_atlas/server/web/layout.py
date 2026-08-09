"""Deterministic force layout — the v1.1 design's own algorithm, run server-side.

The design computes positions client-side in ``atlas-data.js``; this is that function,
constant for constant (seed 7 LCG, ``k = sqrt(w*h/n)``, repulsion ``1.1*k^2/d^2``,
attraction ``(d - 0.55k)/d * 0.09 * (0.5 + w/6)``, centring 0.006, damping 0.82,
step clamp +-20, pad 60, normalise-and-centre), vectorised with numpy. Running it on
the server keeps the layout identical on every reload — a reader should recognise the
shape of their codebase — and keeps the client free of an O(n²) loop.

Output is in the canvas's own 0-1000 space (``S`` in map.js), so the client places a
node at ``x/10 %`` with no further mapping.
"""

from __future__ import annotations

# The canvas coordinate space, shared with map.js.
SPACE = 1000.0
_PAD = 60.0
_LCG_MOD = 2147483648
# Edge weights reach the attraction term as (0.5 + w/6); the design's mock carries
# w ∈ 1..3 and the service scales real aggregates into the same band.


def _lcg(seed: int) -> int:
    """One step of the design's own linear congruential generator."""
    return (seed * 1103515245 + 12345) % _LCG_MOD


def iteration_count(node_count: int) -> int:
    """Each iteration is O(n²); a bigger graph gets proportionally fewer passes."""
    return max(110, min(420, round(420 * (40 / max(1, node_count)) ** 0.7)))


# Pairs per repulsion block: bounds the temporaries to ~64MB however many nodes the
# graph has. The full-scope entity map is the case that matters — an unchunked pair
# matrix at ~7,000 nodes is ~750MB before the first iteration finishes.
_BLOCK_PAIRS = 2_000_000


def force_layout(  # noqa: PLR0915  # one simulation, kept whole so it stays comparable to the design's
    nodes: list[str],
    edges: dict[tuple[str, str], float],
    *,
    width: float = SPACE,
    height: float = SPACE,
    _block: int | None = None,
) -> dict[str, tuple[float, float]]:
    """Position every node; a function of the edges and nothing else.

    ``_block`` overrides the repulsion block size — a test hook proving the chunked
    accumulation produces the same picture as the one-shot matrix.
    """
    import numpy as np  # noqa: PLC0415  # heavy import, only needed when a map is drawn

    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: (width / 2, height / 2)}
    block = _block or max(1, _BLOCK_PAIRS // n)

    index = {name: i for i, name in enumerate(nodes)}

    # Ring start with the design's LCG jitter, reproduced exactly.
    seed = 7
    jitter = np.empty((n, 2))
    for i in range(n):
        seed = _lcg(seed)
        jitter[i, 0] = seed / _LCG_MOD * 8
        seed = _lcg(seed)
        jitter[i, 1] = seed / _LCG_MOD * 8
    angles = np.arange(n) / n * 2 * np.pi
    pos = np.stack(
        [
            width / 2 + np.cos(angles) * (width / 3) + jitter[:, 0],
            height / 2 + np.sin(angles) * (height / 3) + jitter[:, 1],
        ],
        axis=1,
    )
    vel = np.zeros_like(pos)

    src, dst, wgt = [], [], []
    # Sorted, because float accumulation is order-sensitive and dict order here comes
    # from query results — the same graph must settle identically on every request.
    for (a, b), weight in sorted(edges.items()):
        if a in index and b in index and a != b:
            src.append(index[a])
            dst.append(index[b])
            wgt.append(weight)
    src_i = np.asarray(src, dtype=np.intp)
    dst_i = np.asarray(dst, dtype=np.intp)
    pull = np.asarray(wgt, dtype=float)

    k = float(np.sqrt((width * height) / n))
    iterations = iteration_count(n)

    # The design's constants are tuned for its ~40-node mock, while the total
    # repulsion a peripheral node feels is scale-invariant (n pairs x k^2/d^2 with
    # k^2 = area/n). Left alone, every leaf's equilibrium orbit lands outside the
    # canvas at a thousand nodes and the normalise-to-fill step pins that shell to
    # the border. Above the design's own scale, springs stiffen with sqrt(n/40) to
    # keep pace with repulsion; repulsion and the spring rest length grow with the
    # fourth root, so clusters breathe instead of knotting; and gravity grows
    # fastest, holding the halo despite the stronger push.
    sim_scale = max(1.0, float(np.sqrt(n / 40)))
    breathe = float(np.sqrt(sim_scale))
    stiffness = 0.09 * sim_scale
    repulsion_c = k * k * 1.1 * breathe
    rest = k * 0.55 * breathe
    gravity = 0.006 * sim_scale * breathe
    # A node with no edge has no spring to hold it; only gravity answers repulsion,
    # so it gets more — otherwise the edge-less settle in a ring past everything.
    connected = np.zeros(n, dtype=bool)
    connected[src_i] = True
    connected[dst_i] = True
    gravity_of = np.where(connected, gravity, gravity * 4.0)

    for it in range(iterations):
        t = 1 - it / iterations

        # Repulsion in row blocks: identical arithmetic to the full pair matrix (each
        # row still sums over every column in order), bounded temporaries.
        for start in range(0, n, block):
            rows = pos[start : start + block]
            delta = rows[:, None, :] - pos[None, :, :]
            d2 = np.einsum("ijk,ijk->ij", delta, delta)
            np.clip(d2, 0.01, None, out=d2)
            repulsion = repulsion_c / d2
            self_idx = np.arange(rows.shape[0])
            repulsion[self_idx, start + self_idx] = 0.0  # a node does not push itself
            vel[start : start + block] += np.einsum("ijk,ij->ik", delta, repulsion)

        if src_i.size:
            edge_delta = pos[dst_i] - pos[src_i]
            dist = np.linalg.norm(edge_delta, axis=-1)
            np.clip(dist, 0.01, None, out=dist)
            f = ((dist - rest) / dist) * stiffness * (0.5 + pull / 6)
            force = edge_delta * f[:, None]
            np.add.at(vel, src_i, force)
            np.add.at(vel, dst_i, -force)

        vel[:, 0] += (width / 2 - pos[:, 0]) * gravity_of
        vel[:, 1] += (height / 2 - pos[:, 1]) * gravity_of
        pos += np.clip(vel * t, -20, 20)
        vel *= 0.82

    # Normalise into the padded box and centre — per-axis scale, as the design does.
    low = pos.min(axis=0)
    high = pos.max(axis=0)
    span = np.maximum(high - low, 1.0)
    scale = np.array([(width - _PAD * 2) / span[0], (height - _PAD * 2) / span[1]])
    offset = np.array(
        [
            _PAD + (width - _PAD * 2 - span[0] * scale[0]) / 2,
            _PAD + (height - _PAD * 2 - span[1] * scale[1]) / 2,
        ]
    )
    pos = (pos - low) * scale + offset

    return {name: (float(pos[i, 0]), float(pos[i, 1])) for name, i in index.items()}
