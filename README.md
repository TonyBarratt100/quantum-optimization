# quantum-optimization

[![tests](https://github.com/TonyBarratt100/quantum-optimization/actions/workflows/tests.yml/badge.svg)](https://github.com/TonyBarratt100/quantum-optimization/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Quantum optimisation in [Qiskit](https://www.ibm.com/quantum/qiskit): the **Quantum
Approximate Optimisation Algorithm (QAOA)** applied to **MaxCut**, prepared for the IBM
Qiskit Fall Fest hackathon in Amsterdam (20–21 November 2026). It is the companion to
[quantum-walkers](https://github.com/TonyBarratt100/quantum-walkers).

**MaxCut:** split the nodes of a graph into two groups so that as many edges as possible
run between the groups. It is NP-hard in general, which makes it the standard test problem
for quantum optimisation.

**QAOA:** a short circuit with two kinds of layer. A *cost* layer gives each candidate
split a phase proportional to its cut value. A *mixer* layer lets the candidates interfere.
A classical optimiser tunes the layer angles so that good cuts interfere constructively, and
measuring the circuit then returns good cuts with high probability. More layers (*p*) give
better answers.

![QAOA on the prism graph](qaoa_maxcut.png)

## What's here

| Path | What it does |
|---|---|
| [`qaoa_maxcut/graphs.py`](qaoa_maxcut/graphs.py) | Weighted graphs, cut values, brute-force MaxCut, and example graphs |
| [`qaoa_maxcut/qaoa.py`](qaoa_maxcut/qaoa.py) | The QAOA circuit, a fast NumPy simulation of it, the angle optimiser, and sampling on `AerSimulator` |
| [`qaoa_maxcut/__main__.py`](qaoa_maxcut/__main__.py) | Command-line demo that prints the results table and saves the plot |
| [`tests/`](tests) | pytest checks covering the graph maths, the physics, and the demo |

## Results

The demo graph is the 6-node, 9-edge *prism* (two triangles joined by three edges). Its
maximum cut is 7. A random split cuts 4.5 edges on average.

`python -m qaoa_maxcut`, 10 optimiser restarts and 1,000 shots per depth:

| p | Expected cut | Approximation ratio | P(max cut), exact | P(max cut), sampled |
|---|---|---|---|---|
| 1 | 5.94 | 0.848 | 0.401 | 0.376 |
| 2 | 6.60 | 0.943 | 0.721 | 0.707 |
| 3 | 6.84 | 0.977 | 0.933 | 0.922 |

With three layers, about 93% of measurements return an optimal cut. The sampled
frequencies agree with the exact statevector values to within shot noise.

As a physics check, the tests reproduce a known result from the original QAOA paper
([Farhi, Goldstone and Gutmann, 2014](https://arxiv.org/abs/1411.4028)): on a ring with an even
number of nodes, *p* = 1 reaches exactly 3/4 of the maximum cut.

## How it works

- **Encoding:** one qubit per node. Measuring qubit *i* as 0 or 1 puts node *i* in
  group 0 or 1. Bit order follows Qiskit's little-endian convention.
- **Start:** a Hadamard on every qubit, so every split is equally likely.
- **Cost layer, angle γ:** an `RZZ(−γw)` gate on each edge of weight *w*. This applies
  exp(−iγC), where C is the cut value.
- **Mixer layer, angle β:** an `RX(2β)` gate on every qubit.
- **Optimiser:** SciPy's L-BFGS-B, started from several random angles, maximises the exact
  expected cut. The objective uses a NumPy simulation of the same state, which is several
  times faster than simulating the circuit. A test checks that the two give the same state.
- **Sampling:** the optimised circuit is run on `AerSimulator`, and the best measured
  bitstring is the answer.

## Getting started

You need Python 3.11 or later (qiskit-aer requires it). CI runs on 3.11 to 3.14.

```bash
git clone https://github.com/TonyBarratt100/quantum-optimization.git
cd quantum-optimization
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt   # or requirements.txt for just the code
```

Run the demo and the checks:

```bash
python -m qaoa_maxcut                   # results table, and saves qaoa_maxcut.png
python -m qaoa_maxcut --layers 1 2 3 4  # try other depths (see --help)
pytest -q                               # run the tests
ruff check . && ruff format --check .   # lint and formatting
```

Use it from Python:

```python
from qaoa_maxcut import Graph, best_sampled_cut, optimize, sample

graph = Graph.from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0, 2.0)])  # optional 3rd value = weight
result = optimize(graph, layers=2, seed=1)
print(result.approximation_ratio)
print(best_sampled_cut(graph, sample(graph, result, shots=1000, seed=1)))
```

## Coding standards

- Package layout with type hints and docstrings on every public function
- Input validation with clear error messages, and seeds for reproducible runs
- [Ruff](https://docs.astral.sh/ruff/) lint and formatting rules in `pyproject.toml`
- GitHub Actions runs lint, the tests on Python 3.11–3.14, and the demo on every push

## Next steps

- Compare with the classical Goemans–Williamson algorithm (guaranteed ratio 0.878)
- Weighted and random graphs at larger sizes, and warm-starting deeper circuits from
  shallower ones
- Add a noise model, and run on IBM Quantum hardware via `qiskit-ibm-runtime`

## Licence

[MIT](LICENSE)
