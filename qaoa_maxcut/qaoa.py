"""QAOA (Quantum Approximate Optimisation Algorithm) for MaxCut, in Qiskit.

QAOA prepares the state

    |gamma, beta> = U_B(beta_p) U_C(gamma_p) ... U_B(beta_1) U_C(gamma_1) |+>^n

where U_C(gamma) = exp(-i gamma C) applies a phase proportional to each basis
state's cut value C, and U_B(beta) = exp(-i beta sum_k X_k) mixes the basis states.
A classical optimiser tunes the 2p angles to maximise the expected cut <C>.
Measuring the final state then gives good cuts with high probability.

For an edge (i, j) with weight w, C contains w (1 - Z_i Z_j) / 2. Up to a global
phase, exp(-i gamma w (1 - Z_i Z_j) / 2) = RZZ(-gamma w), and exp(-i beta X) = RX(2 beta).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from .graphs import Graph, assignment_from_bitstring, brute_force_max_cut, cut_value, cut_values


@dataclass(frozen=True)
class QAOAResult:
    """Outcome of optimising the QAOA angles for one graph and one number of layers."""

    layers: int
    gammas: tuple[float, ...]
    betas: tuple[float, ...]
    expected_cut: float
    max_cut: float
    optimal_probability: float
    evaluations: int

    @property
    def approximation_ratio(self) -> float:
        """Expected cut divided by the true maximum cut (1.0 is perfect)."""
        return self.expected_cut / self.max_cut


def qaoa_circuit(graph: Graph, layers: int, measure: bool = False) -> QuantumCircuit:
    """Parameterised QAOA circuit with ``layers`` cost/mixer pairs.

    The circuit's parameters are ``gamma[0..p-1]`` and ``beta[0..p-1]``; bind them with
    :func:`bind_angles`.
    """
    if layers < 1:
        raise ValueError(f"layers must be at least 1, got {layers}")
    gammas = ParameterVector("gamma", layers)
    betas = ParameterVector("beta", layers)

    qc = QuantumCircuit(graph.num_nodes, name=f"qaoa_p{layers}")
    qc.h(range(graph.num_nodes))
    for layer in range(layers):
        for i, j, weight in graph.edges:
            qc.rzz(-gammas[layer] * weight, i, j)
        qc.rx(2 * betas[layer], range(graph.num_nodes))
    if measure:
        qc.measure_all()
    return qc


def bind_angles(circuit: QuantumCircuit, gammas: Sequence[float], betas: Sequence[float]) -> QuantumCircuit:
    """Return a copy of a :func:`qaoa_circuit` with concrete angles."""
    layers = _layer_count(circuit)
    if len(gammas) != layers or len(betas) != layers:
        raise ValueError(f"expected {layers} gammas and {layers} betas, got {len(gammas)} and {len(betas)}")
    angles = [float(a) for a in (*gammas, *betas)]
    return circuit.assign_parameters(dict(zip(_ordered_parameters(circuit), angles, strict=True)))


def qaoa_statevector(graph: Graph, gammas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
    """Final QAOA state, computed directly in NumPy (little-endian amplitudes).

    This is the same state :func:`qaoa_circuit` prepares (up to a global phase), but it
    skips binding and simulating a circuit, which makes it several times faster
    inside the optimiser loop. The tests check that the two agree.
    """
    if len(gammas) != len(betas):
        raise ValueError(f"got {len(gammas)} gammas but {len(betas)} betas")
    if len(gammas) == 0:
        raise ValueError("need at least one layer of angles")
    return _evolve(cut_values(graph), graph.num_nodes, gammas, betas)


def state_probabilities(graph: Graph, gammas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
    """Exact probability of each basis state (little-endian) for the given angles."""
    return np.abs(qaoa_statevector(graph, gammas, betas)) ** 2


def expected_cut(graph: Graph, gammas: Sequence[float], betas: Sequence[float]) -> float:
    """Exact expectation of the cut value, <gamma, beta| C |gamma, beta>."""
    return float(state_probabilities(graph, gammas, betas) @ cut_values(graph))


def optimize(
    graph: Graph,
    layers: int,
    restarts: int = 10,
    seed: int | None = None,
    maxiter: int = 1000,
) -> QAOAResult:
    """Find angles that maximise the expected cut, using L-BFGS-B from several random starts.

    QAOA's energy landscape has many local optima, so the best of ``restarts`` runs is
    returned. Starting angles are drawn from gamma in [0, pi) and beta in [0, pi/2);
    shifting any beta by pi/2 only swaps the two groups, so that range loses
    nothing. Objective evaluations use the fast NumPy simulation, not the circuit.
    """
    if layers < 1:
        raise ValueError(f"layers must be at least 1, got {layers}")
    if restarts < 1:
        raise ValueError(f"restarts must be at least 1, got {restarts}")
    if graph.total_weight == 0:
        raise ValueError("graph has no edges with positive weight, so every cut is 0")

    values = cut_values(graph)
    max_cut, _ = brute_force_max_cut(graph)
    optimal = np.isclose(values, max_cut)
    rng = np.random.default_rng(seed)
    evaluations = 0

    def probabilities(x: np.ndarray) -> np.ndarray:
        return np.abs(_evolve(values, graph.num_nodes, x[:layers], x[layers:])) ** 2

    def negative_expected_cut(x: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        return -float(probabilities(x) @ values)

    runs = []
    for _ in range(restarts):
        start = np.concatenate([rng.uniform(0, np.pi, layers), rng.uniform(0, np.pi / 2, layers)])
        runs.append(minimize(negative_expected_cut, start, method="L-BFGS-B", options={"maxiter": maxiter}))
    best = min(runs, key=lambda run: run.fun)
    best_x, best_value = best.x, -float(best.fun)

    return QAOAResult(
        layers=layers,
        gammas=tuple(float(g) for g in best_x[:layers]),
        betas=tuple(float(b) for b in best_x[layers:]),
        expected_cut=float(best_value),
        max_cut=max_cut,
        optimal_probability=float(probabilities(best_x)[optimal].sum()),
        evaluations=evaluations,
    )


def sample(graph: Graph, result: QAOAResult, shots: int = 1000, seed: int | None = None) -> dict[str, int]:
    """Measure the optimised circuit ``shots`` times on AerSimulator; returns Qiskit counts."""
    if shots < 1:
        raise ValueError(f"shots must be at least 1, got {shots}")
    circuit = bind_angles(qaoa_circuit(graph, result.layers, measure=True), result.gammas, result.betas)
    simulator = AerSimulator(seed_simulator=seed)
    job = simulator.run(transpile(circuit, simulator), shots=shots)
    return job.result().get_counts()


def best_sampled_cut(graph: Graph, counts: dict[str, int]) -> tuple[float, list[int]]:
    """The best cut found among the measured bitstrings."""
    if not counts:
        raise ValueError("counts is empty")
    assignments = (assignment_from_bitstring(bits) for bits in counts)
    best = max(assignments, key=lambda a: cut_value(graph, a))
    return cut_value(graph, best), best


def _evolve(
    values: np.ndarray, num_qubits: int, gammas: Sequence[float], betas: Sequence[float]
) -> np.ndarray:
    """Apply the QAOA layers to |+>^n, given the precomputed cut value of every basis state."""
    state = np.full(2**num_qubits, 2 ** (-num_qubits / 2), dtype=complex)  # |+>^n
    for gamma, beta in zip(gammas, betas, strict=True):
        state = state * np.exp(-1j * gamma * values)  # U_C: diagonal phase
        state = _apply_rx_to_all(state, num_qubits, 2 * beta)  # U_B: RX(2 beta) on every qubit
    return state


def _apply_rx_to_all(state: np.ndarray, num_qubits: int, theta: float) -> np.ndarray:
    """Apply RX(theta) = [[cos, -i sin], [-i sin, cos]](theta / 2) to every qubit."""
    c, s = np.cos(theta / 2), -1j * np.sin(theta / 2)
    for qubit in range(num_qubits):
        # Axis 1 of this view is the qubit's bit: index = high * 2**(q+1) + bit * 2**q + low
        view = state.reshape(-1, 2, 2**qubit)
        zero, one = view[:, 0, :], view[:, 1, :]
        state = np.stack([c * zero + s * one, s * zero + c * one], axis=1).reshape(-1)
    return state


def _ordered_parameters(circuit: QuantumCircuit) -> list[Parameter]:
    """Parameters as [gamma_0..gamma_{p-1}, beta_0..beta_{p-1}], matching the optimiser vector."""
    layers = _layer_count(circuit)
    by_name = {param.name: param for param in circuit.parameters}
    return [by_name[f"gamma[{k}]"] for k in range(layers)] + [by_name[f"beta[{k}]"] for k in range(layers)]


def _layer_count(circuit: QuantumCircuit) -> int:
    layers = sum(1 for param in circuit.parameters if param.name.startswith("gamma["))
    if layers == 0:
        raise ValueError("circuit has no QAOA parameters; build it with qaoa_circuit()")
    return layers
