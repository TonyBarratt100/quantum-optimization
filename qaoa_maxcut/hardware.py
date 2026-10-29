"""Sample optimised QAOA circuits on an IBM quantum computer, real or fake.

QAOA is a hybrid algorithm: a classical optimiser tunes the angles, and the quantum
computer prepares and measures the resulting state. Here the tuning is done exactly on
the laptop (see :func:`qaoa_maxcut.qaoa.optimize`), so the quantum computer is only
asked for the final measurements. That keeps the quota needed to a few seconds.

Everything goes in one job:

* two **readout calibration** circuits, all |0> and all |1>, on every physical qubit the
  QAOA circuits measure, so each measured bit can be corrected for its own qubit's
  readout error; and
* one QAOA circuit per depth.

The prism graph contains triangles, which IBM's heavy-hex chips do not, so the circuits
need SWAP routing. Qiskit's transpiler is run with several seeds and the version with
the lowest expected error, using the device's published error rates, is kept.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile

#: Transpiler seeds tried per circuit; the lowest expected-error result is kept.
LAYOUT_ATTEMPTS = 8


@dataclass(frozen=True)
class HardwareSample:
    """Measurements of one QAOA circuit on a device."""

    layers: int
    counts: Mapping[str, int]
    corrected: Mapping[str, float]
    measured_qubits: tuple[int, ...]
    two_qubit_gates: int


@dataclass(frozen=True)
class HardwareRun:
    """Everything one hardware job returned."""

    backend: str
    job_id: str | None
    samples: tuple[HardwareSample, ...]


def load_backend(name: str) -> Any:
    """``"least-busy"``, a real device such as ``"ibm_marrakesh"``, or a local fake such as
    ``"fake_marrakesh"`` (no account needed, no quota used)."""
    if name.startswith("fake_"):
        from qiskit_ibm_runtime import fake_provider

        class_name = "Fake" + "".join(part.capitalize() for part in name.removeprefix("fake_").split("_"))
        backend_class = getattr(fake_provider, class_name, None)
        if backend_class is None:
            raise ValueError(f"no fake backend called {name!r} (looked for fake_provider.{class_name})")
        return backend_class()
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()
    if name == "least-busy":
        return service.least_busy(operational=True, simulator=False)
    return service.backend(name)


def is_real_hardware(name: str) -> bool:
    """Whether this backend name uses IBM Quantum time."""
    return not name.startswith("fake_")


def compile_for(circuit: QuantumCircuit, backend: Any, attempts: int = LAYOUT_ATTEMPTS) -> QuantumCircuit:
    """The lowest-expected-error device version of ``circuit`` out of ``attempts`` transpiler seeds."""
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")
    candidates = [
        transpile(circuit, backend=backend, optimization_level=3, seed_transpiler=seed)
        for seed in range(attempts)
    ]
    return min(candidates, key=lambda candidate: expected_error(candidate, backend))


def expected_error(circuit: QuantumCircuit, backend: Any) -> float:
    """Sum of -log(1 - error) over every gate and measurement, from the device's reported errors.

    This approximates -log(probability that nothing goes wrong), so lower is better.
    RZ gates are free on IBM hardware and report no error.
    """
    target = backend.target
    total = 0.0
    for instruction in circuit.data:
        name = instruction.operation.name
        if name in ("barrier", "delay") or name not in target.operation_names:
            continue
        qubits = tuple(circuit.find_bit(q).index for q in instruction.qubits)
        properties = target[name].get(qubits)
        error = getattr(properties, "error", None) or 0.0
        total += -math.log(1.0 - min(max(float(error), 0.0), 1.0 - 1e-9))
    return total


def measured_qubits(circuit: QuantumCircuit) -> tuple[int, ...]:
    """The physical qubit measured into each classical bit, in bit order."""
    mapping: dict[int, int] = {}
    for instruction in circuit.data:
        if instruction.operation.name == "measure":
            qubit = circuit.find_bit(instruction.qubits[0]).index
            clbit = circuit.find_bit(instruction.clbits[0]).index
            mapping[clbit] = qubit
    if sorted(mapping) != list(range(circuit.num_clbits)):
        raise ValueError("every classical bit must be measured exactly once")
    return tuple(mapping[c] for c in range(circuit.num_clbits))


def calibration_circuits(
    qubits: Sequence[int], num_device_qubits: int
) -> tuple[QuantumCircuit, QuantumCircuit]:
    """All |0> and all |1> on the given physical qubits, measured in that order."""
    zeros = QuantumCircuit(num_device_qubits)
    zeros.add_register(ClassicalRegister(len(qubits), "meas"))
    ones = zeros.copy()
    for bit, qubit in enumerate(qubits):
        ones.x(qubit)
        zeros.measure(qubit, bit)
        ones.measure(qubit, bit)
    return zeros, ones


def confusion_matrices(
    zeros: Mapping[str, int], ones: Mapping[str, int], qubits: Sequence[int]
) -> dict[int, np.ndarray]:
    """Per-qubit 2x2 readout confusion matrices [[P(0|0), P(0|1)], [P(1|0), P(1|1)]]."""
    width = len(qubits)
    total0, total1 = sum(zeros.values()), sum(ones.values())
    matrices = {}
    for bit, qubit in enumerate(qubits):
        position = width - 1 - bit  # Qiskit prints bit 0 last
        flip0 = sum(n for key, n in zeros.items() if key[position] == "1") / total0
        flip1 = sum(n for key, n in ones.items() if key[position] == "0") / total1
        matrices[qubit] = np.array([[1 - flip0, flip1], [flip0, 1 - flip1]])
    return matrices


def correct_readout(counts: Mapping[str, int], matrices: Sequence[np.ndarray]) -> dict[str, float]:
    """Undo independent per-bit readout error, then project onto valid probabilities."""
    width = len(matrices)
    vector = np.zeros(2**width)
    for key, n in counts.items():
        vector[int(key, 2)] += n
    vector /= vector.sum()
    for bit, matrix in enumerate(matrices):
        if abs(np.linalg.det(matrix)) < 1e-9:
            raise ValueError(f"bit {bit}: readout too noisy to correct")
        view = vector.reshape(-1, 2, 2**bit)
        vector = np.einsum("ij,ajb->aib", np.linalg.inv(matrix), view).reshape(-1)
    probabilities = _project_to_simplex(vector)
    return {format(k, f"0{width}b"): float(p) for k, p in enumerate(probabilities) if p > 1e-12}


def run_on_backend(
    circuits: Mapping[int, QuantumCircuit], backend: Any, shots: int, seed: int = 0
) -> HardwareRun:
    """Compile, calibrate and sample measured QAOA circuits, keyed by depth, in one job."""
    from qiskit_ibm_runtime.executor_sampler import Sampler  # optional dependency

    if isinstance(shots, bool) or not isinstance(shots, int) or shots < 1:
        raise ValueError(f"shots must be a positive integer, got {shots!r}")
    if not circuits:
        raise ValueError("no circuits to run")
    depths = list(circuits)
    compiled = [compile_for(circuits[depth], backend) for depth in depths]
    layouts = [measured_qubits(circuit) for circuit in compiled]
    used = sorted({qubit for layout in layouts for qubit in layout})
    batch = [*calibration_circuits(used, backend.num_qubits), *compiled]

    sampler = Sampler(mode=backend)
    sampler.options.twirling.enable_gates = False
    sampler.options.twirling.enable_measure = False
    sampler.options.dynamical_decoupling.enable = False
    sampler.options.simulator.seed_simulator = seed  # only used by fake backends
    job = sampler.run(batch, shots=shots)
    result = job.result()
    counts = [dict(result[i].data.meas.get_counts()) for i in range(len(batch))]

    matrices = confusion_matrices(counts[0], counts[1], used)
    samples = []
    for depth, isa, layout, raw in zip(depths, compiled, layouts, counts[2:], strict=True):
        samples.append(
            HardwareSample(
                layers=depth,
                counts=raw,
                corrected=correct_readout(raw, [matrices[q] for q in layout]),
                measured_qubits=layout,
                two_qubit_gates=sum(1 for i in isa.data if i.operation.num_qubits == 2),
            )
        )
    job_id = job.job_id() if callable(getattr(job, "job_id", None)) else None
    return HardwareRun(backend=str(backend.name), job_id=job_id, samples=tuple(samples))


def _project_to_simplex(vector: np.ndarray) -> np.ndarray:
    """Closest probability distribution in Euclidean distance (Duchi et al., 2008)."""
    v = vector - vector.max()
    u = np.sort(v)[::-1]
    cumulative = np.cumsum(u) - 1.0
    last = np.nonzero(u - cumulative / np.arange(1, v.size + 1) > 0)[0][-1]
    return np.maximum(v - cumulative[last] / (last + 1), 0.0)
