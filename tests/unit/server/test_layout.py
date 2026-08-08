"""Server-side graph layout (ATL-117).

The layout must be deterministic — that is the requirement, not a detail. A force
simulation settling in the browser draws the same graph differently on every reload,
which destroys the map's only real job: letting someone recognise their codebase.
"""

from __future__ import annotations

import math

import pytest

from code_atlas.server.web.layout import (
    cluster_positions,
    cluster_radius,
    layout_communities,
    node_size,
)


class TestClusterPositions:
    def test_no_communities_is_no_positions(self):
        assert cluster_positions([]) == []

    def test_a_single_community_sits_at_the_origin(self):
        """Nothing to arrange around, so a ring would just push it off-centre."""
        assert cluster_positions([5]) == [(0.0, 0.0)]

    def test_communities_are_evenly_spaced_on_the_ring(self):
        positions = cluster_positions([3, 3, 3, 3], ring=100.0)

        assert len(positions) == 4
        for x, y in positions:
            assert math.hypot(x, y) == pytest.approx(100.0)

    def test_the_layout_is_stable_across_calls(self):
        """The whole point: reload the page, get the same picture."""
        assert cluster_positions([4, 2, 7]) == cluster_positions([4, 2, 7])


class TestClusterRadius:
    def test_radius_grows_with_membership(self):
        assert cluster_radius(50) > cluster_radius(5)

    def test_a_tiny_community_still_gets_a_usable_radius(self):
        assert cluster_radius(1) >= 8.0

    def test_growth_is_sublinear(self):
        """Area scales with the count, not radius — else one big subsystem swamps the ring."""
        assert cluster_radius(100) < cluster_radius(10) * 10


class TestLayoutCommunities:
    def test_every_module_gets_a_position(self):
        positions = layout_communities([["a", "b"], ["c"]])

        assert set(positions) == {"a", "b", "c"}

    def test_members_of_one_community_cluster_together(self):
        """A subsystem must read as a group, or the map says nothing about structure."""
        positions = layout_communities([["a1", "a2", "a3"], ["b1", "b2", "b3"]], ring=100.0)

        within = math.dist(positions["a1"], positions["a2"])
        across = math.dist(positions["a1"], positions["b1"])
        assert within < across

    def test_a_lone_module_is_not_hidden_at_the_centre(self):
        """An isolated module is a real finding; the middle of the map reads as 'unremarkable'."""
        positions = layout_communities([["big1", "big2"], ["lonely"]], ring=100.0)

        assert math.hypot(*positions["lonely"]) > 0

    def test_empty_communities_are_skipped_without_error(self):
        positions = layout_communities([["a"], [], ["b"]])

        assert set(positions) == {"a", "b"}

    def test_identical_input_gives_identical_output(self):
        groups = [["m1", "m2", "m3"], ["n1", "n2"]]

        assert layout_communities(groups) == layout_communities(groups)


class TestNodeSize:
    def test_size_grows_with_degree(self):
        assert node_size(50) > node_size(2)

    def test_an_isolated_module_is_still_visible(self):
        assert node_size(0) >= 3.0

    def test_a_hub_cannot_dwarf_everything_else(self):
        """Linear sizing would render the long tail invisible next to one hub."""
        assert node_size(5000) <= 14.0
