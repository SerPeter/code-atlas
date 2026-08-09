"""Deterministic force layout — the v1.1 design's own algorithm, run server-side.

The design computes positions client-side in ``atlas-data.js``; this is that function,
constant for constant (seed 7 LCG, ``k = sqrt(w*h/n)``, repulsion ``1.1*k^2/d^2``,
attraction ``(d - 0.55k)/d * 0.09 * (0.5 + w/6)``, centring 0.006, damping 0.82,
step clamp +-20, pad 60, normalise-and-centre), vectorised with numpy. Running it on
the server keeps the layout identical on every reload — a reader should recognise the
shape of their codebase — and keeps the client free of an O(n²) loop.

Output is in the canvas's own 0-1000 space (``S`` in map.js); the client maps it onto
the viewport with one uniform scale, centred, so the settled shape keeps its aspect
ratio at any window size.
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

    # Mass-weighted repulsion, ForceAtlas-style: a hub carries its degree, so two
    # dense cluster cores push each other apart instead of parking side by side and
    # letting their leaf fans overlap. Normalised to mean 1 — the weighting shifts
    # repulsion from leaves to hubs without raising the total force budget, which
    # is what the sim_scale corrections above balanced against gravity. Left raw,
    # the mean-squared surplus blew every equilibrium orbit past the canvas and the
    # normalise step pinned a ring of leaves to the border.
    deg = np.zeros(n)
    np.add.at(deg, src_i, 1.0)
    np.add.at(deg, dst_i, 1.0)
    mass = 1.0 + np.sqrt(deg) / 2.0
    mass /= mass.mean()

    for it in range(iterations):
        t = 1 - it / iterations

        # Repulsion in row blocks: identical arithmetic to the full pair matrix (each
        # row still sums over every column in order), bounded temporaries.
        for start in range(0, n, block):
            rows = pos[start : start + block]
            delta = rows[:, None, :] - pos[None, :, :]
            d2 = np.einsum("ijk,ijk->ij", delta, delta)
            np.clip(d2, 0.01, None, out=d2)
            repulsion = (repulsion_c / d2) * mass[start : start + block, None] * mass[None, :]
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

    # Normalise into the padded box and centre. Uniform scale, deliberately not the
    # design's per-axis stretch: stretching each axis independently to fill squeezes
    # every cloud into a rectangle. The settled aspect ratio is information.
    low = pos.min(axis=0)
    high = pos.max(axis=0)
    span = np.maximum(high - low, 1.0)
    factor = min((width - _PAD * 2) / span[0], (height - _PAD * 2) / span[1])
    offset = np.array(
        [
            _PAD + (width - _PAD * 2 - span[0] * factor) / 2,
            _PAD + (height - _PAD * 2 - span[1] * factor) / 2,
        ]
    )
    pos = (pos - low) * factor + offset

    return {name: (float(pos[i, 0]), float(pos[i, 1])) for name, i in index.items()}


# Share of the usable canvas area handed to blob interiors; the rest is the
# whitespace between communities, which is the structure being drawn.
_BLOB_FILL = 0.62
_BLOB_GAP = 12.0
_BLOB_MIN_R = 26.0


def _push_apart(centres, radii, gap: float) -> bool:  # numpy arrays, typed loosely to keep the import lazy
    """One de-overlap sweep: push intersecting circle pairs apart, symmetric and
    in deterministic order. A coincident pair separates along an angle derived
    from its indices rather than a random draw — the layout must settle
    identically on every request. Returns whether anything moved."""
    import numpy as np  # noqa: PLC0415  # heavy import, only needed when a map is drawn

    n = len(radii)
    moved = False
    for i in range(n):
        for j in range(i + 1, n):
            delta = centres[j] - centres[i]
            d = float(np.hypot(delta[0], delta[1]))
            need = float(radii[i] + radii[j]) + gap
            if d >= need:
                continue
            if d < 1e-6:
                angle = (i * n + j) * 2.4
                delta = np.array([np.cos(angle), np.sin(angle)])
                d = 1.0
            push = delta / d * (need - d) / 2
            centres[i] -= push
            centres[j] += push
            moved = True
    return moved


def _pack(centres, radii, gap: float, springs) -> None:  # numpy arrays, typed loosely to keep the import lazy
    """Pack circles compactly: linked pairs pull toward touching, everything
    contracts gently, and a de-overlap sweep runs last in every round.

    Push-only relaxation was not enough: every push cascade inflates the
    packing's extent, the fit-to-canvas step then shrinks it all uniformly, and
    the picture ends up dense blobs adrift in outsized whitespace — with linked
    blobs as far apart as strangers, since nothing ever pulled them back. The
    contraction decays over the rounds and a final push-only pass certifies
    that no two circles intersect, whatever the springs wanted.

    *springs* is an iterable of ``(i, j, w)`` with ``w`` in the 1..3 band.
    """
    import numpy as np  # noqa: PLC0415  # heavy import, only needed when a map is drawn

    for it in range(160):
        t = 1 - it / 160
        centroid = centres.mean(axis=0)
        centres += (centroid - centres) * (0.04 * t)
        # Springs stay sequential like the sweep below — simultaneous spring
        # application was measured to degrade the packing the same way a Jacobi
        # sweep did. At ~300 meta springs the loop costs nothing.
        for i, j, w in springs:
            delta = centres[j] - centres[i]
            d = float(np.hypot(delta[0], delta[1]))
            want = float(radii[i] + radii[j]) + gap
            if d <= want or d < 1e-6:
                continue
            step = min((d - want) * (0.10 + 0.06 * (w - 1.0)), 40.0)
            pull = delta / d * step
            centres[i] += pull
            centres[j] -= pull
        # The de-overlap sweep stays sequential (Gauss-Seidel): a Jacobi
        # vectorisation was measured to wreck the packing — simultaneous summed
        # pushes overshoot on a heavily-overlapped start where the sequential
        # sweep converges calmly, and every downstream stage inherits the
        # damage. At ~50 meta-circles the Python loop is cheap anyway.
        _push_apart(centres, radii, gap)
    for _ in range(80):
        if not _push_apart(centres, radii, gap):
            break


def _declump(pts, centre, max_r: float, min_dist: float, rounds: int = 60) -> None:  # numpy arrays
    """Spread one community's members to a minimum pairwise distance, inside its
    territory. Jacobi rounds: every pair closer than *min_dist* pushes apart
    symmetrically, then everything clamps back within *max_r* of the centre.
    Fast local spreading between refinement turns; the global pass afterwards is
    the one that also sees foreign marks.
    """
    import numpy as np  # noqa: PLC0415  # heavy import, only needed when a map is drawn

    n = len(pts)
    if n < 2:
        return
    # Exact duplicates cannot push apart (no direction); separate them first,
    # each against its run's original value so a whole stack separates — an
    # adjacent-only comparison nudged alternating members and left the rest
    # coincident forever. Sorted scan, not the O(n²) pair loop — profiled,
    # that loop was most of this function's cost.
    order_ = np.lexsort((pts[:, 1], pts[:, 0]))
    base = pts[order_].copy()
    run_start = 0
    for k in range(1, n):
        if abs(base[k, 0] - base[run_start, 0]) > 1e-9 or abs(base[k, 1] - base[run_start, 1]) > 1e-9:
            run_start = k
            continue
        j = int(order_[k])
        pts[j] += np.array([np.cos(j * 2.4), np.sin(j * 2.4)]) * 0.5

    # Same sweep amortisation as _space_globally: one dense candidate harvest
    # buys several sparse rounds.
    sweep_every = 8
    upper = np.triu_indices(n, k=1)
    ai = bi = np.empty(0, dtype=np.intp)
    for it in range(rounds):
        if it % sweep_every == 0:
            delta = pts[:, None, :] - pts[None, :, :]
            d2 = np.einsum("ijk,ijk->ij", delta, delta)
            cand = np.zeros_like(d2, dtype=bool)
            cand[upper] = d2[upper] < (min_dist * 2.0) ** 2
            ai, bi = np.nonzero(cand)
            if len(ai) == 0:
                break  # a fresh sweep with no candidates is a converged state
        dvec = pts[ai] - pts[bi]
        d = np.sqrt(np.einsum("ij,ij->i", dvec, dvec))
        np.maximum(d, 1e-6, out=d)
        # delta*(md/d - 1) = unit*(md - d): the push magnitude is self-bounded
        # by md however small d gets, so no upper clip is needed — one was tried
        # and it silently zeroed the push for near-coincident pairs.
        coeff = min_dist / d
        coeff -= 1.0
        coeff *= 0.35
        np.clip(coeff, 0.0, None, out=coeff)
        if float(coeff.max()) > 0.0:
            push = dvec * coeff[:, None]
            acc = np.zeros_like(pts)
            np.add.at(acc, ai, push)
            np.add.at(acc, bi, -push)
            pts += acc
        off = pts - centre
        r = np.sqrt(np.einsum("ij,ij->i", off, off))
        out = r > max_r
        if bool(out.any()):
            pts[out] = centre + off[out] / r[out, None] * max_r


def _space_globally(pts, pair_min, node_centre, node_reach, rounds: int) -> None:  # numpy
    """The last word on spacing: every pair of marks — same community or not —
    keeps its minimum distance, inside its territory reach.

    A per-community pass cannot see a foreign mark, so two blobs spilling into
    the same strait overlapped each other's nodes freely; both constraints hold
    inside every round: pairwise minimum, own-territory reach. There is
    deliberately no foreign-disc exclusion — it existed once, and its
    projection arcs carved concave bites into neighbouring silhouettes while
    whole swarms still mingled in the whitespace beyond the discs it guarded.
    Interpenetration is allowed; the cross-community pairwise floor is what
    keeps mingled marks readable.

    Almost every entry of the full pair matrix is dead work — only pairs within
    a couple of minimum-distances of each other can ever push. One dense sweep
    harvests those candidates (with double the violation radius as margin for
    drift) and buys the next several rounds, which then run on the sparse pair
    list: profiled, this pass was over half the whole layout, and the sweep
    amortisation removes ~90% of it. A pair that drifts into range mid-window
    is caught at the next sweep — rounds are corrective, not load-bearing.
    """
    import numpy as np  # noqa: PLC0415  # heavy import, only needed when a map is drawn

    p = pts.astype(np.float32)
    pm = pair_min.astype(np.float32)
    # Exact duplicates have no push direction (delta is zero); nodes clamped to
    # the same rim point are exactly that. Separate them deterministically first.
    order_ = np.lexsort((p[:, 1], p[:, 0]))
    base = p[order_].copy()
    run_start = 0
    for k in range(1, len(order_)):
        if abs(base[k, 0] - base[run_start, 0]) > 1e-6 or abs(base[k, 1] - base[run_start, 1]) > 1e-6:
            run_start = k
            continue
        j = int(order_[k])
        p[j] += np.array([np.cos(j * 2.4), np.sin(j * 2.4)], dtype=np.float32) * 0.7

    sweep_every = 8
    upper = np.triu_indices(len(p), k=1)
    ai = bi = np.empty(0, dtype=np.intp)
    pm_s = np.empty(0, dtype=np.float32)
    for it in range(rounds):
        if it % sweep_every == 0:
            delta = p[:, None, :] - p[None, :, :]
            d2 = np.einsum("ijk,ijk->ij", delta, delta)
            cand = np.zeros_like(d2, dtype=bool)
            cand[upper] = d2[upper] < (pm[upper] * 2.0) ** 2
            ai, bi = np.nonzero(cand)
            pm_s = pm[ai, bi]
            if len(ai) == 0:
                break  # a fresh sweep with no candidates is a converged state
        dvec = p[ai] - p[bi]
        d = np.sqrt(np.einsum("ij,ij->i", dvec, dvec))
        np.maximum(d, 1e-6, out=d)
        # 0.425*(pm/d - 1) folds the unit vector and the overlap into one
        # coefficient; the clip zeroes every candidate already at distance.
        # Self-bounded — no upper clip, it would zero the push exactly where
        # delta is tiny.
        coeff = pm_s / d
        coeff -= 1.0
        coeff *= 0.425
        np.clip(coeff, 0.0, None, out=coeff)
        if float(coeff.max()) > 0.0:
            push = dvec * coeff[:, None]
            acc = np.zeros_like(p)
            np.add.at(acc, ai, push)
            np.add.at(acc, bi, -push)
            p += acc
        off = p - node_centre
        r = np.sqrt(np.einsum("ij,ij->i", off, off))
        out = r > node_reach
        if bool(out.any()):
            p[out] = (node_centre[out] + off[out] / r[out, None] * node_reach[out, None]).astype(np.float32)
    pts[:] = p.astype(pts.dtype)


def _affinity_refine(pts, src_i, dst_i, weight, node_centre, node_reach, rounds: int) -> None:  # numpy arrays
    """Every edge pulls its endpoints — cross-community edges included.

    The blob machinery quantised the physics: aggregate traffic placed the blobs,
    and a node's own edges stopped meaning anything for its position. But the
    per-edge pull is signal at node grain — the same evidence the partition was
    computed from — so it returns here as a refinement: nodes with external
    traffic drift toward their partner blob and the blob's shape grows out of its
    actual dependencies. Containment is the territory reach alone: a node may
    stray up to *node_reach* from home — into the whitespace and into a
    neighbouring community's ground, where colour keeps membership readable. A
    hard own-disc clamp was tried first (crowds piled on a rim arc no spacing
    pass could spread, every blob a featureless circle), then a foreign-disc
    exclusion (its projection arcs carved concave bites into silhouettes while
    swarms still mingled beyond the discs) — both lost to the simple reach.
    Spacing is not this pass's job — the global pass that follows restores it —
    so the spring needs no repulsion term.

    The weak home gravity is the pull's counterweight. A peripheral community's
    partners all lie inward, so without it every member drifts the same way:
    the outer half of the territory empties and the swarm radially compresses
    into a squashed dome. Gravity cancels that bulk migration while nodes with
    real external traffic still overcome it — shape stays differential, not a
    net translation of the whole swarm.
    """
    import numpy as np  # noqa: PLC0415  # heavy import, only needed when a map is drawn

    if len(src_i) == 0:
        return
    for _ in range(rounds):
        acc = np.zeros_like(pts)
        delta = pts[dst_i] - pts[src_i]
        f = delta * (0.002 * weight)[:, None]
        np.add.at(acc, src_i, f)
        np.add.at(acc, dst_i, -f)
        pts += np.clip(acc, -8.0, 8.0)
        pts += (node_centre - pts) * 0.005
        off = pts - node_centre
        r = np.sqrt(np.einsum("ij,ij->i", off, off))
        out = r > node_reach
        if bool(out.any()):
            pts[out] = node_centre[out] + off[out] / r[out, None] * node_reach[out, None]


def _components(members: list[str], pairs: set[tuple[str, str]]) -> list[list[str]]:
    """Connected components of a community's sub-graph, largest first."""
    adj: dict[str, list[str]] = {m: [] for m in members}
    for a, b in sorted(pairs):
        if a in adj and b in adj:
            adj[a].append(b)
            adj[b].append(a)
    seen: set[str] = set()
    comps: list[list[str]] = []
    for m in members:  # members arrive sorted, so component discovery is deterministic
        if m in seen:
            continue
        comp: list[str] = []
        stack = [m]
        seen.add(m)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(sorted(comp))
    comps.sort(key=lambda c: (-len(c), c[0]))
    return comps


def clustered_layout(  # noqa: PLR0912, PLR0915  # one two-level layout, kept whole like force_layout above
    nodes: list[str],
    edges: dict[tuple[str, str], float],
    community_of: dict[str, int],
    *,
    width: float = SPACE,
    height: float = SPACE,
) -> dict[str, tuple[float, float]]:
    """Two-level layout: every community is a blob, and blobs cannot overlap.

    ``force_layout`` answers "where does each node sit among its springs", but at
    thousands of nodes one simulation is space-filling: density equalises,
    communities interpenetrate, and the cloud reads as a rectangle of static.
    Here the structure is drawn first. Communities become weighted meta-nodes
    (radius from member share), the meta simulation places them by their
    cross-community traffic, and a deterministic packing settles them — linked
    blobs at touching distance, none intersecting. Each community then runs its
    own small simulation and is scaled into its blob. The whitespace between
    blobs is the point, not waste — but it is earned by absent edges, not by
    push cascades.
    """
    import numpy as np  # noqa: PLC0415  # heavy import, only needed when a map is drawn

    if not nodes:
        return {}
    groups: dict[int, list[str]] = {}
    for name in sorted(nodes):
        groups.setdefault(community_of.get(name, -1), []).append(name)
    if len(groups) == 1:
        return force_layout(nodes, edges, width=width, height=height)

    # Blob radii from member share: equal density across communities, floored so
    # a two-node community is still a visible place and not a speck.
    total = len(nodes)
    usable = (width - _PAD * 2) * (height - _PAD * 2)
    order = sorted(groups)
    radius = {
        cid: max(_BLOB_MIN_R, float(np.sqrt(_BLOB_FILL * usable * len(groups[cid]) / total / np.pi))) for cid in order
    }

    # The meta graph: communities linked by their aggregate cross-community
    # traffic, scaled into the same 1..3 pull band the node edges use.
    meta_edges: dict[tuple[str, str], float] = {}
    for (a, b), w in edges.items():
        ca = community_of.get(a, -1)
        cb = community_of.get(b, -1)
        if ca == cb:
            continue
        key = (str(min(ca, cb)), str(max(ca, cb)))
        meta_edges[key] = meta_edges.get(key, 0.0) + w
    if meta_edges:
        top = max(meta_edges.values())
        meta_edges = {k: 1.0 + 2.0 * (w / top) for k, w in meta_edges.items()}
    centres_by_name = force_layout([str(cid) for cid in order], meta_edges, width=width, height=height)
    centres = np.array([centres_by_name[str(cid)] for cid in order])
    radii = np.array([radius[cid] for cid in order])

    # The same traffic that seeded the meta simulation becomes packing springs:
    # blobs that exchange calls sit at touching distance, strangers only as far
    # as the packing pushes them.
    at = {str(cid): i for i, cid in enumerate(order)}
    springs = sorted((at[a], at[b], w) for (a, b), w in meta_edges.items())
    _pack(centres, radii, _BLOB_GAP, springs)

    # Each community's interior is placed by connected component, because one
    # simulation over a fragmentary sub-graph has the same pathology the top
    # level had: disconnected pieces repel each other to the box shell and the
    # blob renders as a square outline. Every multi-node component runs its own
    # simulation — the sub-graph sees the constants force_layout is tuned for —
    # inside a mini-circle sized by member share, the circles pack on a
    # phyllotaxis spiral and relax until disjoint, and singleton members fill
    # the remaining ring as a spiral swarm. Cross-community edges are not springs
    # in these component sims — they return, per edge, in the refinement below.
    golden = 2.399963229728653
    positions: dict[str, tuple[float, float]] = {}
    for idx, cid in enumerate(order):
        members = groups[cid]
        local = {
            (a, b): w
            for (a, b), w in edges.items()
            if community_of.get(a, -1) == cid and community_of.get(b, -1) == cid
        }
        cx, cy = centres[idx]
        blob_r = float(radii[idx])
        comps = _components(members, set(local))
        multis = [c for c in comps if len(c) > 1]
        singles = [c[0] for c in comps if len(c) == 1]
        if multis:
            share = [len(c) / len(members) for c in multis]
            sub_r = np.array([max(6.0, blob_r * 0.92 * float(np.sqrt(s))) for s in share])
            sub_c = np.empty((len(multis), 2))
            cum = 0.0
            for k, s in enumerate(share):
                ring = blob_r * 0.9 * float(np.sqrt(cum))  # cumulative-area spiral: big components centre first
                sub_c[k] = (cx + ring * np.cos(k * golden), cy + ring * np.sin(k * golden))
                cum += s
            _pack(sub_c, sub_r, 4.0, ())
            for k, comp in enumerate(multis):
                comp_set = set(comp)
                placed = force_layout(
                    comp, {p: w for p, w in local.items() if p[0] in comp_set}, width=width, height=height
                )
                side = sub_r[k] * 1.35
                for name, (x, y) in placed.items():
                    positions[name] = (
                        float(sub_c[k][0]) + (x / width - 0.5) * side,
                        float(sub_c[k][1]) + (y / height - 0.5) * side,
                    )
        inner = 0.62 if multis else 0.0
        for k, name in enumerate(singles):
            r = blob_r * (inner + (0.97 - inner) * float(np.sqrt((k + 0.5) / len(singles))))
            positions[name] = (
                cx + r * float(np.cos(k * golden)),
                cy + r * float(np.sin(k * golden)),
            )

    # Refinement: the per-edge pull comes back, contained. Alternating with the
    # de-clump lets both promises hold at once — the pull decides *where in the
    # blob* a node sits (external traffic drifts it to the rim facing its
    # partner, giving the blob a shape), the de-clump decides *how close* any
    # two marks may get, and running the de-clump last makes spacing the final
    # word. min-dist asks for ~64% disc coverage, leaving room for anisotropy.
    names = sorted(positions)
    at_n = {name: i for i, name in enumerate(names)}
    pts = np.array([positions[name] for name in names])
    at_c = {cid: i for i, cid in enumerate(order)}
    node_comm_idx = np.array([at_c[community_of.get(name, -1)] for name in names])
    node_centre = centres[node_comm_idx]
    # Territory, not disc: a member may stray past its blob into the whitespace —
    # which is where the lobes come from — never into a foreign blob.
    node_reach = radii[node_comm_idx] * 1.35
    src, dst, wgt = [], [], []
    for (a, b), w in sorted(edges.items()):
        if a in at_n and b in at_n and a != b:
            src.append(at_n[a])
            dst.append(at_n[b])
            wgt.append(w)
    src_i = np.asarray(src, dtype=np.intp)
    dst_i = np.asarray(dst, dtype=np.intp)
    wgt_a = np.asarray(wgt, dtype=float)
    # Pairwise minimum for the spacing passes: a community's own spacing inside
    # its blob, and a fixed floor between marks of different communities sharing
    # a strait — a bridge node reads as a neighbour, never as an overlap.
    md_of_comm = np.array(
        [1.6 * float(radii[i]) / float(np.sqrt(max(1, len(groups[cid])))) for i, cid in enumerate(order)]
    )
    md = md_of_comm[node_comm_idx]
    same = node_comm_idx[:, None] == node_comm_idx[None, :]
    pair_min = np.where(same, np.minimum(md[:, None], md[None, :]), 11.0)
    member_rows = {
        cid: np.asarray([at_n[m] for m in groups[cid]], dtype=np.intp) for cid in order if len(groups[cid]) > 3
    }

    def _declump_all(rounds: int) -> None:
        for idx, cid in enumerate(order):
            rows = member_rows.get(cid)
            if rows is None:
                continue
            sub = pts[rows]
            _declump(sub, centres[idx], float(radii[idx]) * 1.45, float(md_of_comm[idx]), rounds=rounds)
            pts[rows] = sub

    # Finely interleaved, because coarse alternation loses both ways: a long
    # refine piles half a community onto one rim arc — a heap no single spacing
    # pass can spread back across the disc — while a long spacing pass erases
    # the drift. The cheap per-community de-clump does the fast local spreading
    # between turns (measured: dropping it doubled the residual overlap); the
    # global pass — the only one that also sees foreign marks — is the last word.
    for _ in range(4):
        _affinity_refine(pts, src_i, dst_i, wgt_a, node_centre, node_reach, 25)
        _declump_all(30)
    _space_globally(pts, pair_min, node_centre, node_reach, 90)
    positions = {name: (float(pts[i, 0]), float(pts[i, 1])) for i, name in enumerate(names)}

    # One uniform normalise into the padded box, same rule as force_layout: fill
    # without stretching, because the settled aspect ratio is information.
    pos = np.array([positions[name] for name in sorted(positions)])
    low = pos.min(axis=0)
    high = pos.max(axis=0)
    span = np.maximum(high - low, 1.0)
    factor = min((width - _PAD * 2) / span[0], (height - _PAD * 2) / span[1])
    offset = np.array(
        [
            _PAD + (width - _PAD * 2 - span[0] * factor) / 2,
            _PAD + (height - _PAD * 2 - span[1] * factor) / 2,
        ]
    )
    return {
        name: (
            float((positions[name][0] - low[0]) * factor + offset[0]),
            float((positions[name][1] - low[1]) * factor + offset[1]),
        )
        for name in positions
    }
