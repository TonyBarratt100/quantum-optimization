"""The IBM hardware path, exercised against a fake IBM device that runs locally.

Fake backends copy a real device's qubit layout, native gates and measured noise, so this
tests exactly the code that talks to real hardware, without an account or any quota.
"""

import numpy as np
import pytest

pytest.importorskip("qiskit_ibm_runtime")

from qiskit import QuantumCircuit, transpile

from qaoa_maxcut.graphs import example_graph
from qaoa_maxcut.hardware import (
    calibration_circuits,
    compile_for,
    confusion_matrices,
    correct_readout,
    expected_error,
    is_real_hardware,
    load_backend,
    measured_qubits,
    run_on_backend,
)
from qaoa_maxcut.qaoa import bind_angles, optimize, qaoa_circuit

GRAPH = example_graph()


@pytest.fixture(scope="module")
def backend():
    return load_backend("fake_marrakesh")


@pytest.fixture(scope="module")
def circuits():
    built = {}
    for layers in (1, 2):
        result = optimize(GRAPH, layers, restarts=3, seed=7)
        built[layers] = bind_angles(qaoa_circuit(GRAPH, layers, measure=True), result.gammas, result.betas)
    return built


@pytest.fixture(scope="module")
def run(backend, circuits):
    return run_on_backend(circuits, backend, shots=2000, seed=1)


def test_readout_correction_exactly_undoes_known_errors():
    # Different, asymmetric errors per bit, so a bit-order mix-up would be caught
    bit0 = np.array([[0.95, 0.10], [0.05, 0.90]])
    bit1 = np.array([[0.80, 0.02], [0.20, 0.98]])
    ideal = np.array([0.5, 0.0, 0.1, 0.4])  # indexed by basis state, bit 0 least significant
    noisy = np.kron(bit1, bit0) @ ideal
    counts = {format(k, "02b"): round(p * 1e6) for k, p in enumerate(noisy)}
    corrected = correct_readout(counts, [bit0, bit1])
    assert corrected["00"] == pytest.approx(0.5, abs=1e-5)
    assert corrected["10"] == pytest.approx(0.1, abs=1e-5)
    assert corrected["11"] == pytest.approx(0.4, abs=1e-5)
    assert sum(corrected.values()) == pytest.approx(1)


def test_confusion_matrices_read_each_qubits_flips():
    zeros = {"00": 90, "01": 10}  # the qubit measured into bit 0 flipped 10% of the time
    ones = {"11": 80, "01": 20}  # the qubit measured into bit 1 flipped 20% of the time
    matrices = confusion_matrices(zeros, ones, qubits=[7, 3])
    assert matrices[7] == pytest.approx(np.array([[0.9, 0.0], [0.1, 1.0]]))
    assert matrices[3] == pytest.approx(np.array([[1.0, 0.2], [0.0, 0.8]]))


def test_calibration_circuits_measure_the_given_qubits_in_order():
    zeros, ones = calibration_circuits([5, 2], num_device_qubits=8)
    assert measured_qubits(zeros) == (5, 2)
    assert measured_qubits(ones) == (5, 2)
    assert ones.count_ops()["x"] == 2
    assert "x" not in zeros.count_ops()


def test_compile_keeps_the_lowest_expected_error_version(backend, circuits):
    compiled = compile_for(circuits[1], backend, attempts=4)
    alternatives = [
        transpile(circuits[1], backend=backend, optimization_level=3, seed_transpiler=s) for s in range(4)
    ]
    assert expected_error(compiled, backend) == pytest.approx(
        min(expected_error(c, backend) for c in alternatives)
    )
    assert set(compiled.count_ops()) <= {"rz", "sx", "x", "cz", "measure", "barrier"}


def test_expected_error_counts_readout():
    backend = load_backend("fake_marrakesh")
    circuit = QuantumCircuit(backend.num_qubits, 1)
    circuit.measure(0, 0)
    assert expected_error(circuit, backend) > 0


def test_run_returns_corrected_distributions_per_depth(run):
    assert run.backend == "fake_marrakesh"
    assert [s.layers for s in run.samples] == [1, 2]
    for sample in run.samples:
        assert sum(sample.counts.values()) == 2000
        assert sum(sample.corrected.values()) == pytest.approx(1)
        assert len(set(sample.measured_qubits)) == GRAPH.num_nodes
    assert run.samples[0].two_qubit_gates < run.samples[1].two_qubit_gates


def test_backend_names():
    assert not is_real_hardware("fake_marrakesh")
    assert is_real_hardware("ibm_marrakesh") and is_real_hardware("least-busy")
    with pytest.raises(ValueError, match="no fake backend"):
        load_backend("fake_nowhere")


def test_run_rejects_bad_input(backend, circuits):
    with pytest.raises(ValueError, match="shots"):
        run_on_backend(circuits, backend, shots=0)
    with pytest.raises(ValueError, match="no circuits"):
        run_on_backend({}, backend, shots=100)
