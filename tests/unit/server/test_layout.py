"""The server-side force layout — the design's own algorithm, so the properties it
promises are the ones under test: deterministic, bounded, and a function of the edges.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from code_atlas.server.web.layout import SPACE, clustered_layout, force_layout, iteration_count


def _grid(n: int) -> list[str]:
    return [f"m{i}" for i in range(n)]


class TestDeterminism:
    def test_the_same_graph_lays_out_identically_every_time(self):
        """A layout that settles differently per request destroys recognisability."""
        nodes = _grid(12)
        edges = {("m0", "m1"): 2.0, ("m1", "m2"): 1.5, ("m3", "m4"): 3.0}

        assert force_layout(nodes, edges) == force_layout(nodes, edges)

    def test_edge_dict_order_does_not_move_the_map(self):
        """Dict order comes from query results; float accumulation must not see it."""
        nodes = _grid(6)
        forward = {("m0", "m1"): 2.0, ("m2", "m3"): 1.0, ("m4", "m5"): 3.0}
        backward = dict(reversed(list(forward.items())))

        assert force_layout(nodes, forward) == force_layout(nodes, backward)


class TestBounds:
    def test_every_position_lands_inside_the_canvas_space(self):
        positions = force_layout(_grid(20), {("m0", "m1"): 2.0, ("m5", "m6"): 1.0})

        for x, y in positions.values():
            assert 0.0 <= x <= SPACE
            assert 0.0 <= y <= SPACE

    def test_the_empty_graph_is_empty(self):
        assert force_layout([], {}) == {}

    def test_a_single_node_sits_in_the_centre(self):
        assert force_layout(["only"], {}) == {"only": (SPACE / 2, SPACE / 2)}


class TestChunkedRepulsion:
    def test_row_blocks_match_the_one_shot_matrix(self):
        """Chunking exists for memory, not for a different picture: each row still
        sums over every column in order, so the result is bit-identical."""
        nodes = _grid(9)
        edges = {("m0", "m1"): 2.0, ("m2", "m7"): 1.0, ("m4", "m5"): 3.0}

        assert force_layout(nodes, edges) == force_layout(nodes, edges, _block=2)


class TestEdgesDrivePosition:
    def test_connected_nodes_sit_closer_than_strangers(self):
        """Position is a function of the edges — the whole point of replacing the ring."""
        nodes = _grid(10)
        edges = {("m0", "m1"): 3.0}
        positions = force_layout(nodes, edges)

        def dist(a: str, b: str) -> float:
            (ax, ay), (bx, by) = positions[a], positions[b]
            return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

        coupled = dist("m0", "m1")
        strangers = [dist("m0", f"m{i}") for i in range(2, 10)]
        assert coupled < sum(strangers) / len(strangers)

    def test_nodes_do_not_collapse_onto_one_point(self):
        positions = force_layout(_grid(8), {("m0", "m1"): 1.0})

        assert len({(round(x), round(y)) for x, y in positions.values()}) > 1


class TestLargeGraphBalance:
    def test_edge_less_nodes_do_not_form_the_outer_shell(self):
        """With absolute spring constants, everything unheld drifts to a ring past
        the connected mass and the normalise step pins it to the border. Scaled
        gravity must keep the edge-less INSIDE the connected hull, not beyond it."""
        hub_and_spokes = {("m0", f"m{i}"): 2.0 for i in range(1, 60)}
        nodes = _grid(60) + [f"lone{i}" for i in range(20)]

        positions = force_layout(nodes, hub_and_spokes)

        def radius(name: str) -> float:
            x, y = positions[name]
            return ((x - SPACE / 2) ** 2 + (y - SPACE / 2) ** 2) ** 0.5

        lone_max = max(radius(f"lone{i}") for i in range(20))
        connected_max = max(radius(f"m{i}") for i in range(60))
        assert lone_max <= connected_max, (
            f"edge-less nodes reach radius {lone_max:.0f}, past the connected {connected_max:.0f}"
        )


class TestClusteredLayout:
    """Two-level layout: the promise is separation — blobs must not interpenetrate."""

    @staticmethod
    def _two_cliques() -> tuple[list[str], dict[tuple[str, str], float], dict[str, int]]:
        """Two hub-shaped communities — a chain would drag single-file toward the
        cross edge under the per-edge pull and stretch into the strait, which is
        lobe behaviour, not mixing. Hubs keep each community compact so the
        centroid assertion tests what it means to."""
        nodes = [f"a{i}" for i in range(10)] + [f"b{i}" for i in range(10)]
        edges: dict[tuple[str, str], float] = {}
        for i in range(1, 10):
            edges["a0", f"a{i}"] = 2.0
            edges["b0", f"b{i}"] = 2.0
        edges["a0", "b0"] = 1.0  # one cross-community edge holds the halves together
        community = {n: 0 if n.startswith("a") else 1 for n in nodes}
        return nodes, edges, community

    def test_deterministic(self):
        nodes, edges, community = self._two_cliques()

        assert clustered_layout(nodes, edges, community) == clustered_layout(nodes, edges, community)

    def test_communities_do_not_interpenetrate(self):
        """Every a-node is nearer its own centroid than the other community's —
        the single-simulation failure was exactly the two clouds mixing."""
        nodes, edges, community = self._two_cliques()
        positions = clustered_layout(nodes, edges, community)

        def centroid(prefix: str) -> tuple[float, float]:
            pts = [positions[n] for n in nodes if n.startswith(prefix)]
            return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

        ca, cb = centroid("a"), centroid("b")
        for n, (x, y) in positions.items():
            own, other = (ca, cb) if n.startswith("a") else (cb, ca)
            d_own = ((x - own[0]) ** 2 + (y - own[1]) ** 2) ** 0.5
            d_other = ((x - other[0]) ** 2 + (y - other[1]) ** 2) ** 0.5
            assert d_own < d_other, f"{n} sits in the other community's blob"

    def test_positions_stay_inside_the_canvas(self):
        nodes, edges, community = self._two_cliques()

        for x, y in clustered_layout(nodes, edges, community).values():
            assert 0.0 <= x <= SPACE
            assert 0.0 <= y <= SPACE

    def test_linked_communities_end_up_adjacent(self):
        """The packing must spend whitespace on absent edges, not on push-cascade
        inflation — heavily linked blobs were drifting as far apart as unlinked
        ones because the relaxation only ever pushed. Communities a and e start
        far apart on the meta ring (ids 0 and 4 of 6); only the spring brings
        them to touching distance."""
        from itertools import combinations

        names = ["a", "b", "c", "d", "e", "f"]
        nodes = [f"{p}{i}" for p in names for i in range(8)]
        edges: dict[tuple[str, str], float] = {}
        for p in names:
            for i in range(7):
                edges[f"{p}{i}", f"{p}{i + 1}"] = 2.0
        for i in range(8):  # only a and e share traffic
            edges[f"a{i}", f"e{i}"] = 3.0
        community = {n: names.index(n[0]) for n in nodes}

        positions = clustered_layout(nodes, edges, community)

        def centroid(prefix: str) -> tuple[float, float]:
            pts = [positions[n] for n in nodes if n.startswith(prefix)]
            return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

        cents = {p: centroid(p) for p in names}

        def dist(p: str, q: str) -> float:
            return ((cents[p][0] - cents[q][0]) ** 2 + (cents[p][1] - cents[q][1]) ** 2) ** 0.5

        pair_dists = sorted(dist(p, q) for p, q in combinations(names, 2))
        assert dist("a", "e") <= pair_dists[0] * 1.1, "the linked pair is not among the touching pairs"
        assert dist("a", "e") <= pair_dists[-1] * 0.6, "no whitespace left between unlinked blobs to compare against"

    def test_edge_less_members_fill_a_disc_not_a_shell(self):
        """A community of loners must render as a round swarm — the simulation
        rings them against its box and the blob drew as a square outline."""
        nodes = [f"a{i}" for i in range(6)] + [f"x{i}" for i in range(30)]
        edges = {(f"a{i}", f"a{i + 1}"): 2.0 for i in range(5)}
        community = {n: 0 if n.startswith("a") else 1 for n in nodes}

        positions = clustered_layout(nodes, edges, community)
        pts = [positions[n] for n in nodes if n.startswith("x")]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        dists = sorted(((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5 for p in pts)
        assert dists[0] < 0.45 * dists[-1], "innermost loner sits far from centre: a ring, not a disc"

    def test_members_keep_a_minimum_spacing_inside_their_blob(self):
        """A hub's leaves all settle at rest length, so blobs rendered as one
        opaque mark: 93% of nodes overlapped their nearest neighbour on the real
        graph. The de-clump pass owes every member breathing room while keeping
        it inside its blob — separation must not be the casualty."""
        import numpy as np

        from code_atlas.server.web.layout import _declump

        pts = np.zeros((24, 2)) + 500.0  # worst case: every member coincident
        centre = np.array([500.0, 500.0])
        _declump(pts, centre, max_r=120.0, min_dist=40.0)

        for i in range(len(pts)):
            assert float(np.hypot(*(pts[i] - centre))) <= 120.0 + 1e-6, "pushed outside the blob disc"
            for j in range(i + 1, len(pts)):
                d = float(np.hypot(*(pts[i] - pts[j])))
                assert d >= 40.0 * 0.6, f"pair ({i},{j}) still clumped at {d:.1f}"

    def test_one_community_degenerates_to_the_plain_layout(self):
        nodes = _grid(8)
        edges = {("m0", "m1"): 2.0, ("m2", "m3"): 1.0}
        community = dict.fromkeys(nodes, 0)

        assert clustered_layout(nodes, edges, community) == force_layout(nodes, edges)


class TestIterationBudget:
    def test_bigger_graphs_get_fewer_passes(self):
        """Each iteration is O(n²); the budget keeps the level switch instant."""
        assert iteration_count(40) >= iteration_count(240) >= iteration_count(1500)

    def test_the_budget_is_clamped_at_both_ends(self):
        assert iteration_count(1) == 420
        assert iteration_count(100_000) == 110
