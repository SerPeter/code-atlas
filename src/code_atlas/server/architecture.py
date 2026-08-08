"""Architecture-health metrics over a module dependency graph.

Pure functions on an edge list — no graph client, no I/O, no framework. That is
deliberate: these are the numbers a human will use to decide whether a codebase is
decaying, so they need to be checkable against hand-worked examples rather than only
against a live database.

The question these answer is "is this becoming a big ball of mud", which is a question
about *trajectory*, so every metric is designed to be recorded per index run and
compared over time rather than read once.

**Propagation cost** is the headline because it has published reference points.
MacCormack, Rusnak and Baldwin (2006) measured pre-refactor Mozilla at ~17%, the
refactored version at ~2%, and Linux at ~0.3%. A number with an outside anchor beats one
that only means something relative to itself.

**Core size** is the fraction of modules in the largest cyclic group. A healthy
architecture is close to a DAG, so its largest cycle is one module; a ball of mud has a
single giant strongly-connected component that everything else hangs off.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Cycle:
    """A strongly-connected group of modules — everything here reaches everything else."""

    members: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass(frozen=True)
class ArchitectureMetrics:
    """The mud report for one project.

    ``covered_modules`` is not decoration. A propagation cost computed over a graph
    whose C++ extraction sits at 0.690 (ATL-096) is a **lower bound**, and a confident
    "8% — you are fine" over partial data is exactly the failure this project spends its
    effort eliminating. Every consumer of these numbers is expected to show it.
    """

    module_count: int
    edge_count: int
    propagation_cost: float
    core_size: float
    largest_cycle: int
    cycles: tuple[Cycle, ...] = ()
    fan_in_gini: float = 0.0
    order: tuple[str, ...] = ()
    covered_modules: int = 0
    caveats: tuple[str, ...] = field(default_factory=tuple)


def strongly_connected_components(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    """Tarjan's SCC, iteratively.

    Iterative rather than recursive on purpose: a deep dependency chain in a large
    monorepo would otherwise hit the interpreter's stack limit, and this runs against
    whatever the user happens to have indexed.

    Returns every component, including the singletons — a module that is in no cycle is
    a component of size one, and dropping those would make ``core_size`` meaningless.
    """
    adjacency: dict[str, list[str]] = {n: [] for n in nodes}
    for src, dst in edges:
        if src in adjacency and dst in adjacency:
            adjacency[src].append(dst)

    index_of: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    components: list[list[str]] = []
    counter = 0

    for root in nodes:
        if root in index_of:
            continue
        # (node, iterator position) — an explicit frame stack standing in for recursion.
        work: list[tuple[str, int]] = [(root, 0)]
        index_of[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)

        while work:
            node, child_idx = work[-1]
            neighbours = adjacency[node]
            if child_idx < len(neighbours):
                work[-1] = (node, child_idx + 1)
                child = neighbours[child_idx]
                if child not in index_of:
                    index_of[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, 0))
                elif child in on_stack:
                    low[node] = min(low[node], index_of[child])
                continue

            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index_of[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(sorted(component))

    return components


def propagation_cost(nodes: list[str], edges: list[tuple[str, str]]) -> float:
    """Mean fraction of the system reachable from a module, following dependencies.

    "If I change one thing at random, what share of the codebase could be affected?"
    Computed as the density of the transitive closure — for each module, how many others
    it can reach, averaged and normalised by the module count.

    Self-reachability is excluded from the numerator and the denominator, so a system of
    fully independent modules scores 0.0 and one where everything reaches everything
    else scores 1.0. Including self would put a floor of 1/N on a perfect score, which
    makes small projects look worse than large ones for no reason.
    """
    if len(nodes) < 2:
        return 0.0

    adjacency: dict[str, list[str]] = {n: [] for n in nodes}
    for src, dst in edges:
        if src in adjacency and dst in adjacency:
            adjacency[src].append(dst)

    total_reachable = 0
    for start in nodes:
        seen: set[str] = {start}
        queue = deque(adjacency[start])
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            queue.extend(adjacency[node])
        total_reachable += len(seen) - 1  # exclude self

    return total_reachable / (len(nodes) * (len(nodes) - 1))


def gini(values: list[float]) -> float:
    """Concentration of *values*, 0.0 (even) to 1.0 (one holds everything).

    Applied to fan-in it answers "is there a god-module?" — a handful of modules that
    everything depends on will push this toward 1.0 while the mean fan-in stays
    unremarkable, which is why the average alone hides them.
    """
    if not values or all(v == 0 for v in values):
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    cumulative = sum((i + 1) * v for i, v in enumerate(ordered))
    return (2 * cumulative) / (n * sum(ordered)) - (n + 1) / n


def dsm_order(nodes: list[str], edges: list[tuple[str, str]]) -> list[str]:
    """Order modules so dependencies fall *below* the diagonal wherever possible.

    This ordering is what makes the matrix readable: in a layered architecture every
    mark sits below the diagonal, and a mark **above** it is a cycle. Without a
    dependency-respecting sort the same graph looks like uniform noise at any level of
    health, which is the whole reason a raw adjacency matrix is not a useful picture.

    Cycles cannot be ordered internally — that is what a cycle means — so components are
    condensed, the condensation (a DAG) is topologically sorted, and members within a
    component are emitted in a stable alphabetical order.
    """
    components = strongly_connected_components(nodes, edges)
    component_of: dict[str, int] = {}
    for i, component in enumerate(components):
        for member in component:
            component_of[member] = i

    # Edges are REVERSED here, and that inversion is the whole ordering.
    #
    # An edge (src, dst) means "src depends on dst". A mark sits at (row=src, col=dst),
    # so for it to fall *below* the diagonal src must come AFTER dst — dependencies
    # first, dependents last. Sorting along the edges as given produces exactly the
    # mirror image, where a clean layered architecture fills the upper triangle and
    # looks, at a glance, like the cyclic mess it is not.
    successors: dict[int, set[int]] = {i: set() for i in range(len(components))}
    in_degree: dict[int, int] = dict.fromkeys(range(len(components)), 0)
    for src, dst in edges:
        dependent, dependency = component_of.get(src), component_of.get(dst)
        if dependent is None or dependency is None or dependent == dependency:
            continue
        if dependent in successors[dependency]:
            continue
        successors[dependency].add(dependent)
        in_degree[dependent] += 1

    # Start from what depends on nothing: the foundation of the matrix.
    ready = sorted((i for i, d in in_degree.items() if d == 0), key=lambda i: components[i][0])
    ordered: list[str] = []
    queue = deque(ready)
    seen_components: set[int] = set()
    while queue:
        current = queue.popleft()
        if current in seen_components:
            continue
        seen_components.add(current)
        ordered.extend(components[current])
        for nxt in sorted(successors[current], key=lambda i: components[i][0]):
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    # Anything left sits in a cycle the condensation could not linearise; append it
    # rather than dropping it, so the matrix always shows every module.
    for i, component in enumerate(components):
        if i not in seen_components:
            ordered.extend(component)
    return ordered


def analyse(
    nodes: list[str],
    edges: list[tuple[str, str]],
    *,
    covered_modules: int | None = None,
    caveats: tuple[str, ...] = (),
) -> ArchitectureMetrics:
    """Compute the full mud report for a module graph."""
    unique_nodes = sorted(set(nodes))
    unique_edges = sorted({(a, b) for a, b in edges if a != b})

    components = strongly_connected_components(unique_nodes, unique_edges)
    # Filtered first and sorted in place: `sorted(..., key=len)` resolves the element type
    # from `len` and widens it to Sized, losing list[str].
    cyclic = [c for c in components if len(c) > 1]
    cyclic.sort(key=len, reverse=True)
    largest = len(cyclic[0]) if cyclic else 1

    fan_in: dict[str, float] = dict.fromkeys(unique_nodes, 0.0)
    for _src, dst in unique_edges:
        if dst in fan_in:
            fan_in[dst] += 1

    return ArchitectureMetrics(
        module_count=len(unique_nodes),
        edge_count=len(unique_edges),
        propagation_cost=propagation_cost(unique_nodes, unique_edges),
        core_size=largest / len(unique_nodes) if unique_nodes else 0.0,
        largest_cycle=largest,
        cycles=tuple(Cycle(members=tuple(c)) for c in cyclic),
        fan_in_gini=gini(list(fan_in.values())),
        order=tuple(dsm_order(unique_nodes, unique_edges)),
        covered_modules=covered_modules if covered_modules is not None else len(unique_nodes),
        caveats=caveats,
    )
