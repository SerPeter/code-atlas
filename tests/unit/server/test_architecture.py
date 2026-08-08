"""Architecture-health metrics, checked against hand-worked graphs (ATL-119).

Every expected value here is derived by hand rather than recorded from a run, because a
metric that only agrees with its own last output cannot tell you the implementation is
right — and these numbers are the ones a human will use to decide whether a codebase is
decaying.
"""

from __future__ import annotations

import pytest

from code_atlas.server.architecture import (
    analyse,
    dsm_order,
    gini,
    propagation_cost,
    strongly_connected_components,
)


class TestStronglyConnectedComponents:
    def test_a_dag_has_only_singletons(self):
        nodes = ["a", "b", "c"]
        edges = [("a", "b"), ("b", "c")]

        components = strongly_connected_components(nodes, edges)

        assert sorted(components) == [["a"], ["b"], ["c"]]

    def test_a_cycle_is_one_component(self):
        nodes = ["a", "b", "c"]
        edges = [("a", "b"), ("b", "c"), ("c", "a")]

        components = strongly_connected_components(nodes, edges)

        assert components == [["a", "b", "c"]]

    def test_a_cycle_with_a_tail(self):
        """`d` depends on the cycle but is not in it."""
        nodes = ["a", "b", "c", "d"]
        edges = [("a", "b"), ("b", "c"), ("c", "a"), ("d", "a")]

        components = strongly_connected_components(nodes, edges)

        assert sorted(components, key=len, reverse=True) == [["a", "b", "c"], ["d"]]

    def test_two_independent_cycles(self):
        nodes = ["a", "b", "x", "y"]
        edges = [("a", "b"), ("b", "a"), ("x", "y"), ("y", "x")]

        components = strongly_connected_components(nodes, edges)

        assert sorted(components) == [["a", "b"], ["x", "y"]]

    def test_a_deep_chain_does_not_exhaust_the_stack(self):
        """Iterative by design — recursion would die on a long dependency chain."""
        nodes = [f"m{i}" for i in range(5000)]
        edges = [(f"m{i}", f"m{i + 1}") for i in range(4999)]

        components = strongly_connected_components(nodes, edges)

        assert len(components) == 5000


class TestPropagationCost:
    def test_independent_modules_cost_nothing(self):
        assert propagation_cost(["a", "b", "c"], []) == 0.0

    def test_everything_reaching_everything_costs_one(self):
        nodes = ["a", "b", "c"]
        edges = [(x, y) for x in nodes for y in nodes if x != y]

        assert propagation_cost(nodes, edges) == 1.0

    def test_a_chain_is_computed_by_hand(self):
        """a→b→c: a reaches 2, b reaches 1, c reaches 0. 3/(3*2) = 0.5."""
        assert propagation_cost(["a", "b", "c"], [("a", "b"), ("b", "c")]) == pytest.approx(0.5)

    def test_a_cycle_reaches_everything_but_itself(self):
        """a→b→c→a: each reaches the other 2. 6/(3*2) = 1.0."""
        assert propagation_cost(["a", "b", "c"], [("a", "b"), ("b", "c"), ("c", "a")]) == pytest.approx(1.0)

    def test_a_single_module_is_zero_not_undefined(self):
        assert propagation_cost(["a"], []) == 0.0

    def test_self_reference_does_not_inflate_the_score(self):
        """Excluding self is what lets a decoupled system score a true 0.0."""
        assert propagation_cost(["a", "b"], [("a", "a"), ("b", "b")]) == 0.0


class TestGini:
    def test_even_distribution_is_zero(self):
        assert gini([3.0, 3.0, 3.0]) == pytest.approx(0.0)

    def test_all_zero_is_zero_not_a_division_error(self):
        assert gini([0.0, 0.0]) == 0.0

    def test_empty_is_zero(self):
        assert gini([]) == 0.0

    def test_one_module_holding_everything_approaches_one(self):
        """A god-module: 9 modules depended on by nobody, one by everything."""
        concentrated = gini([0.0] * 9 + [100.0])
        even = gini([10.0] * 10)

        assert concentrated > 0.85
        assert concentrated > even


class TestDsmOrder:
    def test_dependencies_come_before_dependents(self):
        """`c` is imported by `b` is imported by `a` — so c, b, a.

        This ordering is the whole point of the matrix: with it, every mark in a layered
        architecture falls below the diagonal and a mark above it is a cycle.
        """
        order = dsm_order(["a", "b", "c"], [("a", "b"), ("b", "c")])

        assert order == ["c", "b", "a"]

    def test_every_module_appears_exactly_once(self):
        nodes = ["a", "b", "c", "d"]
        edges = [("a", "b"), ("b", "c"), ("c", "a"), ("d", "a")]

        order = dsm_order(nodes, edges)

        assert sorted(order) == sorted(nodes)

    def test_a_fully_cyclic_graph_still_lists_everything(self):
        """A cycle cannot be ordered internally, but nothing may be dropped."""
        nodes = ["a", "b", "c"]
        order = dsm_order(nodes, [("a", "b"), ("b", "c"), ("c", "a")])

        assert sorted(order) == nodes


class TestAnalyse:
    def test_a_clean_layered_project(self):
        nodes = ["app", "service", "repo"]
        edges = [("app", "service"), ("service", "repo")]

        metrics = analyse(nodes, edges)

        assert metrics.module_count == 3
        assert metrics.largest_cycle == 1, "a DAG's largest cyclic group is a single module"
        assert metrics.core_size == pytest.approx(1 / 3)
        assert metrics.cycles == ()
        assert metrics.propagation_cost == pytest.approx(0.5)

    def test_a_ball_of_mud(self):
        """Everything reaches everything: core size 1.0, propagation cost 1.0."""
        nodes = ["a", "b", "c", "d"]
        edges = [(x, y) for x in nodes for y in nodes if x != y]

        metrics = analyse(nodes, edges)

        assert metrics.core_size == 1.0
        assert metrics.propagation_cost == 1.0
        assert metrics.largest_cycle == 4
        assert len(metrics.cycles) == 1

    def test_self_edges_are_dropped_rather_than_counted_as_cycles(self):
        """A module importing itself is a parse artifact, not an architectural cycle."""
        metrics = analyse(["a", "b"], [("a", "a"), ("a", "b")])

        assert metrics.cycles == ()
        assert metrics.edge_count == 1

    def test_duplicate_edges_are_counted_once(self):
        metrics = analyse(["a", "b"], [("a", "b"), ("a", "b")])

        assert metrics.edge_count == 1

    def test_coverage_defaults_to_the_module_count_but_can_be_narrowed(self):
        """The caveat exists so a number over partial extraction cannot read as complete."""
        full = analyse(["a", "b"], [])
        partial = analyse(["a", "b"], [], covered_modules=1, caveats=("C++ extraction at 0.69",))

        assert full.covered_modules == 2
        assert partial.covered_modules == 1
        assert partial.caveats == ("C++ extraction at 0.69",)

    def test_an_empty_project_does_not_divide_by_zero(self):
        metrics = analyse([], [])

        assert metrics.module_count == 0
        assert metrics.propagation_cost == 0.0
        assert metrics.core_size == 0.0
