import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qaoa_maxcut.graphs import Graph, brute_force_max_cut, cut_values, example_graph, random_graph, ring
from qaoa_maxcut.qaoa import (
    best_sampled_cut,
    bind_angles,
    expected_cut,
    optimize,
    qaoa_circuit,
    qaoa_statevector,
    sample,
    state_probabilities,
)


@pytest.fixture(scope="module")
def prism_p3():
    """Optimising p=3 takes about a second, so share one result across tests."""
    return optimize(example_graph(), layers=3, restarts=5, seed=1)


def test_circuit_has_expected_shape_and_parameters():
    graph = example_graph()
    circuit = qaoa_circuit(graph, layers=2)
    assert circuit.num_qubits == 6
    assert circuit.num_parameters == 4
    ops = circuit.count_ops()
    assert ops["rzz"] == 2 * len(graph.edges)
    assert ops["rx"] == 2 * graph.num_nodes


def test_invalid_layer_counts_are_rejected():
    with pytest.raises(ValueError, match="layers"):
        qaoa_circuit(ring(4), layers=0)
    with pytest.raises(ValueError, match="2 gammas"):
        bind_angles(qaoa_circuit(ring(4), layers=2), [0.1], [0.2])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_numpy_simulation_matches_qiskit_circuit(seed):
    """The fast simulator used by the optimiser must prepare the same state as the circuit."""
    rng = np.random.default_rng(seed)
    graph = Graph.from_edges(
        5, [(0, 1, 1.0), (1, 2, 0.5), (2, 3, 2.0), (3, 4, 1.0), (4, 0, 1.5), (0, 2, 1.0)]
    )
    gammas, betas = rng.uniform(0, np.pi, 2), rng.uniform(0, np.pi / 2, 2)
    fast = qaoa_statevector(graph, gammas, betas)
    exact = Statevector(bind_angles(qaoa_circuit(graph, 2), gammas, betas)).data
    assert abs(np.vdot(fast, exact)) ** 2 == pytest.approx(1.0)  # equal up to a global phase


def test_zero_angles_give_the_random_guess_average():
    # With all angles 0 the state stays |+>^n: every cut equally likely, so <C> = W / 2
    graph = random_graph(6, 0.5, seed=4)
    assert expected_cut(graph, [0.0], [0.0]) == pytest.approx(graph.total_weight / 2)


def test_probabilities_are_normalised():
    probs = state_probabilities(example_graph(), [0.4, 0.9], [0.3, 0.1])
    assert probs.sum() == pytest.approx(1.0)
    assert np.all(probs >= 0)


def test_p1_on_a_ring_reaches_the_known_optimum_of_three_quarters():
    """Farhi et al. (2014): on a long even ring, p=1 QAOA achieves exactly <C> = 3/4 of the max cut."""
    result = optimize(ring(8), layers=1, restarts=5, seed=0)
    assert result.max_cut == 8
    assert result.approximation_ratio == pytest.approx(0.75, abs=1e-6)


def test_more_layers_do_not_get_worse(prism_p3):
    graph = example_graph()
    p1 = optimize(graph, layers=1, restarts=5, seed=1)
    p2 = optimize(graph, layers=2, restarts=5, seed=1)
    assert p1.approximation_ratio < p2.approximation_ratio < prism_p3.approximation_ratio
    assert p1.approximation_ratio > 0.5 * graph.total_weight / p1.max_cut  # beats random


def test_optimize_is_reproducible_with_a_seed():
    first = optimize(ring(6), layers=1, restarts=3, seed=42)
    second = optimize(ring(6), layers=1, restarts=3, seed=42)
    assert first == second


def test_result_reports_consistent_numbers(prism_p3):
    assert prism_p3.expected_cut == pytest.approx(
        expected_cut(example_graph(), prism_p3.gammas, prism_p3.betas)
    )
    assert prism_p3.max_cut == 7
    assert 0.9 < prism_p3.optimal_probability <= 1.0


def test_sampling_finds_the_maximum_cut(prism_p3):
    graph = example_graph()
    counts = sample(graph, prism_p3, shots=500, seed=3)
    assert sum(counts.values()) == 500
    value, _ = best_sampled_cut(graph, counts)
    assert value == brute_force_max_cut(graph)[0]
    # Sampled frequency of optimal cuts should match the exact probability within noise
    hits = sum(n for bits, n in counts.items() if best_sampled_cut(graph, {bits: 1})[0] == 7)
    assert hits / 500 == pytest.approx(prism_p3.optimal_probability, abs=0.05)


def test_sampled_bitstrings_map_to_the_right_nodes():
    """A star with weighted spokes has a unique max cut (centre vs the rest), so a
    reversed bit order would put the wrong node alone and fail this test."""
    graph = Graph.from_edges(4, [(0, 1, 1.0), (0, 2, 2.0), (0, 3, 3.0)])
    result = optimize(graph, layers=2, restarts=5, seed=0)
    value, assignment = best_sampled_cut(graph, sample(graph, result, shots=300, seed=0))
    assert value == pytest.approx(6.0)
    assert assignment in ([1, 0, 0, 0], [0, 1, 1, 1])


def test_optimize_rejects_bad_input():
    with pytest.raises(ValueError, match="positive weight"):
        optimize(Graph.from_edges(3, []), layers=1)
    with pytest.raises(ValueError, match="positive weight"):
        optimize(Graph.from_edges(2, [(0, 1, 0.0)]), layers=1)
    with pytest.raises(ValueError, match="restarts"):
        optimize(ring(4), layers=1, restarts=0)
    with pytest.raises(ValueError, match="layers"):
        optimize(ring(4), layers=0)


def test_nearly_equal_weights_are_not_counted_as_optimal():
    """Only the true maximum cuts count, not cuts that are merely close to it."""
    graph = Graph.from_edges(3, [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.000001)])
    result = optimize(graph, layers=1, restarts=2, seed=0)
    probs = state_probabilities(graph, result.gammas, result.betas)
    values = cut_values(graph)
    exact = probs[values == values.max()].sum()
    assert result.optimal_probability == pytest.approx(exact)
    assert result.optimal_probability < 0.9  # it was wrongly ~1.0 with np.isclose


def test_best_sampled_cut_ignores_outcomes_that_were_never_observed():
    graph = Graph.from_edges(2, [(0, 1)])
    assert best_sampled_cut(graph, {"00": 10, "01": 0}) == (0, [0, 0])
    with pytest.raises(ValueError, match="no outcome"):
        best_sampled_cut(graph, {"01": 0})
    with pytest.raises(ValueError, match="no outcome"):
        best_sampled_cut(graph, {})
    with pytest.raises(ValueError, match="non-negative"):
        best_sampled_cut(graph, {"00": 5, "01": -1})
    with pytest.raises(ValueError, match="2 bits"):
        best_sampled_cut(graph, {"001": 5})


def test_result_reports_optimizer_convergence(prism_p3):
    assert prism_p3.converged
    assert 1 <= prism_p3.restarts_converged <= 5
    assert prism_p3.optimizer_message


def test_edgeless_graphs_and_non_finite_angles_are_rejected():
    with pytest.raises(ValueError, match="positive weight"):
        qaoa_circuit(Graph.from_edges(3, []), layers=1)
    with pytest.raises(ValueError, match="finite"):
        qaoa_statevector(ring(4), [float("nan")], [0.1])
    with pytest.raises(ValueError, match="finite"):
        bind_angles(qaoa_circuit(ring(4), layers=1), [float("inf")], [0.1])
    with pytest.raises(ValueError, match="maxiter"):
        optimize(ring(4), layers=1, maxiter=0)
