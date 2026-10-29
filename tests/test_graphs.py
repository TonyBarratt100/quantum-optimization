import itertools

import numpy as np
import pytest

from qaoa_maxcut.graphs import (
    Graph,
    assignment_from_bitstring,
    assignment_from_index,
    brute_force_max_cut,
    cut_value,
    cut_values,
    example_graph,
    random_graph,
    ring,
)


def test_from_edges_defaults_weight_to_one():
    graph = Graph.from_edges(3, [(0, 1), (1, 2, 2.5)])
    assert graph.edges == ((0, 1, 1.0), (1, 2, 2.5))
    assert graph.total_weight == pytest.approx(3.5)


@pytest.mark.parametrize(
    ("num_nodes", "edges", "message"),
    [
        (0, [], "num_nodes"),
        (3, [(0, 3)], "outside"),
        (3, [(1, 1)], "self-loop"),
        (3, [(0, 1), (1, 0)], "duplicate"),
        (3, [(0, 1, float("nan"))], "non-negative"),
        (3, [(0, 1, -1.0)], "non-negative"),
        (3, [(0,)], "2 or 3 entries"),
    ],
)
def test_invalid_graphs_are_rejected(num_nodes, edges, message):
    with pytest.raises(ValueError, match=message):
        Graph.from_edges(num_nodes, edges)


def test_cut_value_counts_only_crossing_edges():
    graph = Graph.from_edges(3, [(0, 1, 2.0), (1, 2, 3.0), (0, 2, 5.0)])
    assert cut_value(graph, [0, 0, 0]) == 0
    assert cut_value(graph, [0, 1, 0]) == pytest.approx(5.0)  # edges 0-1 and 1-2
    assert cut_value(graph, [1, 1, 0]) == pytest.approx(8.0)  # edges 1-2 and 0-2


def test_cut_value_validates_assignment():
    graph = ring(4)
    with pytest.raises(ValueError, match="entries"):
        cut_value(graph, [0, 1])
    with pytest.raises(ValueError, match="0 or 1"):
        cut_value(graph, [0, 1, 2, 0])


def test_cut_values_matches_cut_value_for_every_state():
    graph = random_graph(6, 0.6, seed=11)
    values = cut_values(graph)
    for index in range(2**graph.num_nodes):
        assert values[index] == pytest.approx(cut_value(graph, assignment_from_index(index, 6)))


def test_cut_is_unchanged_by_swapping_the_two_groups():
    values = cut_values(example_graph())
    everything = 2**6 - 1
    assert np.allclose(values, values[everything ^ np.arange(2**6)])


def test_bitstring_is_read_little_endian():
    # Qiskit prints qubit 0 last, so "001" means only node 0 is in group 1
    assert assignment_from_bitstring("001") == [1, 0, 0]
    with pytest.raises(ValueError):
        assignment_from_bitstring("012")


@pytest.mark.parametrize(("num_nodes", "expected"), [(4, 4), (5, 4), (6, 6), (7, 6)])
def test_ring_max_cut_is_n_for_even_and_n_minus_one_for_odd(num_nodes, expected):
    value, assignment = brute_force_max_cut(ring(num_nodes))
    assert value == expected
    assert cut_value(ring(num_nodes), assignment) == expected


def test_example_graph_max_cut_is_seven():
    graph = example_graph()
    value, assignment = brute_force_max_cut(graph)
    assert value == 7
    # Independent check without the vectorised code path
    assert value == max(cut_value(graph, list(a)) for a in itertools.product((0, 1), repeat=6))
    assert cut_value(graph, assignment) == 7


def test_random_graph_is_reproducible():
    assert random_graph(8, 0.5, seed=3) == random_graph(8, 0.5, seed=3)
    assert random_graph(5, 1.0).edges == Graph.from_edges(5, itertools.combinations(range(5), 2)).edges
