"""The server-side force layout — the design's own algorithm, so the properties it
promises are the ones under test: deterministic, bounded, and a function of the edges.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from code_atlas.server.web.layout import SPACE, force_layout, iteration_count


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


class TestIterationBudget:
    def test_bigger_graphs_get_fewer_passes(self):
        """Each iteration is O(n²); the budget keeps the level switch instant."""
        assert iteration_count(40) >= iteration_count(240) >= iteration_count(1500)

    def test_the_budget_is_clamped_at_both_ends(self):
        assert iteration_count(1) == 420
        assert iteration_count(100_000) == 110
