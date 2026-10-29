"""Weighted graphs and the MaxCut objective.

A *cut* splits the nodes of a graph into two groups; its value is the total weight
of the edges that cross between the groups. MaxCut asks for the split with the
largest value. It is NP-hard in general, which makes it the standard test problem
for quantum optimisation algorithms such as QAOA.

Node assignments follow Qiskit's little-endian convention: in basis state ``k``,
node ``i`` is in group ``(k >> i) & 1``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

Edge = tuple[int, int, float]

#: Brute force and statevector simulation both cost 2**n, so keep n small.
MAX_NODES = 20
#: Cut values within this relative distance of the maximum count as optimal. It only
#: absorbs floating-point rounding (e.g. 0.1 + 0.2 vs 0.3); genuinely different weights,
#: even 1.0 vs 1.000001, are told apart.
OPTIMAL_RTOL = 1e-9


@dataclass(frozen=True)
class Graph:
    """An undirected graph with non-negative edge weights and nodes ``0 .. num_nodes - 1``."""

    num_nodes: int
    edges: tuple[Edge, ...]

    def __post_init__(self) -> None:
        if not _is_integer(self.num_nodes):
            raise ValueError(f"num_nodes must be an integer, got {self.num_nodes!r}")
        if not 1 <= self.num_nodes <= MAX_NODES:
            raise ValueError(f"num_nodes must be between 1 and {MAX_NODES}, got {self.num_nodes}")
        seen: set[frozenset[int]] = set()
        for i, j, weight in self.edges:
            if not (_is_integer(i) and _is_integer(j)):
                raise ValueError(f"node labels must be integers, got edge ({i!r}, {j!r})")
            if isinstance(weight, bool) or not isinstance(weight, Real):
                raise ValueError(f"edge ({i}, {j}) has a non-numeric weight {weight!r}")
            if not (0 <= i < self.num_nodes and 0 <= j < self.num_nodes):
                raise ValueError(f"edge ({i}, {j}) refers to a node outside 0..{self.num_nodes - 1}")
            if i == j:
                raise ValueError(f"self-loop on node {i} is not allowed")
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"edge ({i}, {j}) needs a finite, non-negative weight, got {weight}")
            key = frozenset((i, j))
            if key in seen:
                raise ValueError(f"duplicate edge ({i}, {j})")
            seen.add(key)

    @classmethod
    def from_edges(
        cls,
        num_nodes: int,
        edges: Iterable[tuple[int, int] | tuple[int, int, float]],
    ) -> Graph:
        """Build a graph from ``(i, j)`` or ``(i, j, weight)`` tuples; weight defaults to 1."""
        normalised: list[Edge] = []
        for edge in edges:
            if len(edge) == 2:
                i, j = edge
                weight = 1.0
            elif len(edge) == 3:
                i, j, weight = edge
            else:
                raise ValueError(f"edge must have 2 or 3 entries, got {edge!r}")
            if not (_is_integer(i) and _is_integer(j)):
                # Converting first would silently turn e.g. 0.9 into node 0
                raise ValueError(f"node labels must be integers, got edge {edge!r}")
            if isinstance(weight, bool) or not isinstance(weight, Real):
                raise ValueError(f"weight must be a number, got {weight!r}")
            normalised.append((int(i), int(j), float(weight)))
        return cls(num_nodes, tuple(normalised))

    @property
    def total_weight(self) -> float:
        """Sum of all edge weights. A uniformly random cut has half this value on average."""
        return sum(weight for _, _, weight in self.edges)


def ring(num_nodes: int) -> Graph:
    """Cycle graph 0-1-2-...-(n-1)-0 with unit weights."""
    if num_nodes < 3:
        raise ValueError("a ring needs at least 3 nodes")
    return Graph.from_edges(num_nodes, [(i, (i + 1) % num_nodes) for i in range(num_nodes)])


def example_graph() -> Graph:
    """The 6-node, 9-edge, 3-regular *prism* graph used in the demo (max cut = 7)."""
    return Graph.from_edges(
        6,
        [
            (0, 1), (1, 2), (2, 0),  # top triangle
            (3, 4), (4, 5), (5, 3),  # bottom triangle
            (0, 3), (1, 4), (2, 5),  # vertical edges
        ],
    )  # fmt: skip


def random_graph(num_nodes: int, edge_probability: float, seed: int | None = None) -> Graph:
    """Erdős–Rényi graph: each possible edge is included independently with the given probability."""
    if not 0.0 <= edge_probability <= 1.0:
        raise ValueError("edge_probability must be between 0 and 1")
    rng = np.random.default_rng(seed)
    edges = [
        (i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes) if rng.random() < edge_probability
    ]
    return Graph.from_edges(num_nodes, edges)


def cut_value(graph: Graph, assignment: Sequence[int]) -> float:
    """Total weight of edges whose endpoints are in different groups."""
    if len(assignment) != graph.num_nodes:
        raise ValueError(f"assignment has {len(assignment)} entries, graph has {graph.num_nodes} nodes")
    if any(side not in (0, 1) for side in assignment):
        raise ValueError("assignment entries must be 0 or 1")
    return sum(weight for i, j, weight in graph.edges if assignment[i] != assignment[j])


def cut_values(graph: Graph) -> np.ndarray:
    """Cut value of every one of the 2**n assignments, indexed by basis state (little-endian)."""
    states = np.arange(2**graph.num_nodes)
    values = np.zeros(states.shape, dtype=float)
    for i, j, weight in graph.edges:
        crosses = ((states >> i) ^ (states >> j)) & 1
        values += weight * crosses
    return values


def assignment_from_index(index: int, num_nodes: int) -> list[int]:
    """Group of each node in basis state ``index``."""
    return [(index >> i) & 1 for i in range(num_nodes)]


def assignment_from_bitstring(bits: str) -> list[int]:
    """Convert a Qiskit counts key (highest qubit first) into a per-node assignment."""
    if not bits or set(bits) - {"0", "1"}:
        raise ValueError(f"not a bitstring: {bits!r}")
    return [int(b) for b in reversed(bits)]


def brute_force_max_cut(graph: Graph) -> tuple[float, list[int]]:
    """Exact maximum cut and one assignment that achieves it, by checking all 2**n splits."""
    values = cut_values(graph)
    best = int(np.argmax(values))
    return float(values[best]), assignment_from_index(best, graph.num_nodes)


def optimal_mask(values: np.ndarray, max_cut: float, rtol: float = OPTIMAL_RTOL) -> np.ndarray:
    """Which entries of ``values`` are maximum cuts, allowing only for rounding error."""
    if not 0 <= rtol < 1e-3:
        raise ValueError(f"rtol must be in [0, 1e-3), got {rtol}")
    return values >= max_cut - rtol * abs(max_cut)


def _is_integer(value: object) -> bool:
    """True for Python and NumPy integers, but not for booleans or whole-number floats."""
    return isinstance(value, Integral) and not isinstance(value, bool)
