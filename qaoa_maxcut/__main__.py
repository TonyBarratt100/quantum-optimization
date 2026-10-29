"""Command-line demo: solve MaxCut on the 6-node prism graph with QAOA.

python -m qaoa_maxcut                            # p = 1, 2, 3; saves qaoa_maxcut.png
python -m qaoa_maxcut --layers 1 2 3 4 --shots 2000 --no-plot
python -m qaoa_maxcut --backend fake_marrakesh   # a local copy of an IBM device's noise, free
python -m qaoa_maxcut --backend least-busy       # a real IBM quantum computer (uses quota)
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .graphs import (
    Graph,
    assignment_from_bitstring,
    brute_force_max_cut,
    cut_value,
    cut_values,
    example_graph,
    optimal_mask,
)
from .qaoa import (
    QAOAResult,
    best_sampled_cut,
    bind_angles,
    optimize,
    qaoa_circuit,
    sample,
    state_probabilities,
)

#: Default shots: generous on the free simulator, frugal with real hardware time.
SIMULATOR_SHOTS, HARDWARE_SHOTS = 1000, 4000


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m qaoa_maxcut", description=__doc__.splitlines()[0])
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3], help="QAOA depths p to try")
    parser.add_argument("--restarts", type=int, default=10, help="random starts per depth")
    parser.add_argument(
        "--shots",
        type=int,
        help=f"measurements per depth (default {SIMULATOR_SHOTS:,} simulated, {HARDWARE_SHOTS:,} on IBM)",
    )
    parser.add_argument(
        "--backend",
        default="simulator",
        help="'simulator' (default), a fake IBM device such as 'fake_marrakesh', "
        "a real one such as 'ibm_marrakesh', or 'least-busy'",
    )
    parser.add_argument("--yes", action="store_true", help="skip the confirmation before using IBM quota")
    parser.add_argument("--seed", type=int, default=7, help="seed for reproducible results")
    parser.add_argument("--plot", type=Path, default=Path("qaoa_maxcut.png"), help="where to save the plot")
    parser.add_argument("--no-plot", action="store_true", help="skip the plot")
    args = parser.parse_args(argv)
    if min(args.layers) < 1:
        parser.error("--layers values must be at least 1")
    if args.restarts < 1:
        parser.error("--restarts must be at least 1")
    args.on_ibm = args.backend != "simulator"
    if args.shots is None:
        args.shots = HARDWARE_SHOTS if args.on_ibm else SIMULATOR_SHOTS
    if args.shots < 1:
        parser.error("--shots must be at least 1")
    if args.on_ibm and args.shots < 100:
        parser.error("--shots must be at least 100 on IBM backends, to calibrate readout")
    if args.on_ibm and max(args.layers) > 5:
        parser.error("--layers above 5 make circuits too deep for today's hardware")
    if not args.no_plot and not args.plot.parent.is_dir():
        parser.error(f"--plot: folder {args.plot.parent} does not exist")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.on_ibm:
        return run_on_ibm(args)
    graph = example_graph()
    max_cut, best_assignment = brute_force_max_cut(graph)
    print(f"Prism graph: {graph.num_nodes} nodes, {len(graph.edges)} edges")
    print(f"Maximum cut (brute force): {max_cut:g}   random guess averages {graph.total_weight / 2:g}\n")

    columns = ("p", "<cut>", "ratio", "P(max) exact", "P(max) sampled", "best sampled")
    widths = (2, 6, 6, 12, 14, 12)
    header = "  ".join(f"{name:>{width}}" for name, width in zip(columns, widths, strict=True))
    print(header)
    print("-" * len(header))
    results: list[QAOAResult] = []
    for layers in args.layers:
        result = optimize(graph, layers, restarts=args.restarts, seed=args.seed)
        counts = sample(graph, result, shots=args.shots, seed=args.seed)
        best_cut, _ = best_sampled_cut(graph, counts)
        sampled_cuts = np.array([cut_value(graph, assignment_from_bitstring(bits)) for bits in counts])
        shot_counts = np.array(list(counts.values()))
        hits = shot_counts[optimal_mask(sampled_cuts, max_cut)].sum() / args.shots
        print(
            f"{layers:>2}  {result.expected_cut:>6.3f}  {result.approximation_ratio:>6.3f}  "
            f"{result.optimal_probability:>12.3f}  {hits:>14.3f}  {best_cut:>12g}"
        )
        results.append(result)

    if not args.no_plot:
        plot(graph, best_assignment, results, args.plot, positions=PRISM_LAYOUT)
        print(f"\nSaved {args.plot}")
    return 0


def run_on_ibm(args: argparse.Namespace) -> int:
    """Optimise the angles exactly here, then measure the circuits on an IBM device."""
    from .hardware import is_real_hardware, load_backend, run_on_backend

    try:
        backend = load_backend(args.backend)
    except ImportError as error:
        raise SystemExit('error: IBM backends need the extra package: pip install -e ".[ibm]"') from error
    except Exception as error:  # account, network or unknown-backend problems from IBM's client
        raise SystemExit(f"error: could not get IBM backend {args.backend!r}: {error}") from error

    graph = example_graph()
    max_cut, best_assignment = brute_force_max_cut(graph)
    results = {
        layers: optimize(graph, layers, restarts=args.restarts, seed=args.seed) for layers in args.layers
    }
    circuits = {
        layers: bind_angles(qaoa_circuit(graph, layers, measure=True), r.gammas, r.betas)
        for layers, r in results.items()
    }
    if is_real_hardware(args.backend) and not args.yes:
        print(
            f"This sends one job of {len(circuits) + 2} circuits x {args.shots:,} shots to {backend.name} "
            "(2 readout calibrations + 1 per depth), using some of your IBM Quantum time "
            "(typically well under a minute)."
        )
        if input("Continue? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Cancelled: nothing was sent to IBM.")
            return 1

    try:
        run = run_on_backend(circuits, backend, shots=args.shots, seed=args.seed)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error

    kind = "real IBM quantum computer" if is_real_hardware(args.backend) else "local fake, real device noise"
    print(f"Prism graph: {graph.num_nodes} nodes, {len(graph.edges)} edges, maximum cut {max_cut:g}")
    print(f"Backend: {run.backend} ({kind}); angles optimised exactly on this computer")
    if run.job_id and is_real_hardware(args.backend):
        print(f"IBM job: {run.job_id}")
    print()
    columns = ("p", "2q gates", "P(max) exact", "P(max) raw", "P(max) corrected", "best found")
    widths = (2, 8, 12, 10, 16, 10)
    header = "  ".join(f"{name:>{width}}" for name, width in zip(columns, widths, strict=True))
    print(header)
    print("-" * len(header))
    measured: dict[int, Mapping[str, float]] = {}
    for sample_ in run.samples:
        exact = results[sample_.layers].optimal_probability
        raw = _optimal_share(graph, sample_.counts, max_cut)
        corrected = _optimal_share(graph, sample_.corrected, max_cut)
        best, _ = best_sampled_cut(graph, sample_.counts)
        print(
            f"{sample_.layers:>2}  {sample_.two_qubit_gates:>8}  {exact:>12.3f}  {raw:>10.3f}  "
            f"{corrected:>16.3f}  {best:>10g}"
        )
        measured[sample_.layers] = sample_.corrected
    print("\nP(max) is the share of measurements that found a maximum cut. 'corrected' undoes each")
    print("measured qubit's readout error, using calibration circuits run in the same job.")

    if not args.no_plot:
        plot(
            graph,
            best_assignment,
            list(results.values()),
            args.plot,
            positions=PRISM_LAYOUT,
            measured=measured,
            source=f"measured on {run.backend}, readout-corrected",
        )
        print(f"\nSaved {args.plot}")
    return 0


def _optimal_share(graph: Graph, distribution: Mapping[str, float], max_cut: float) -> float:
    """Fraction of the distribution (counts or probabilities) on maximum cuts."""
    keys = list(distribution)
    cuts = np.array([cut_value(graph, assignment_from_bitstring(bits)) for bits in keys])
    weights = np.array([distribution[bits] for bits in keys], dtype=float)
    return float(weights[optimal_mask(cuts, max_cut)].sum() / weights.sum())


#: Draw the prism as an outer triangle (0, 1, 2) around an inner one (3, 4, 5).
PRISM_LAYOUT = np.array(
    [[np.cos(a), np.sin(a)] for a in np.pi / 2 + np.arange(3) * 2 * np.pi / 3]
    + [[0.45 * np.cos(a), 0.45 * np.sin(a)] for a in np.pi / 2 + np.arange(3) * 2 * np.pi / 3]
)


def circular_layout(num_nodes: int) -> np.ndarray:
    angles = np.pi / 2 + np.arange(num_nodes) * 2 * np.pi / num_nodes
    return np.column_stack([np.cos(angles), np.sin(angles)])


def plot(
    graph: Graph,
    assignment: Sequence[int],
    results: Sequence[QAOAResult],
    path: Path,
    positions: np.ndarray | None = None,
    measured: Mapping[int, Mapping[str, float]] | None = None,
    source: str = "exact",
) -> None:
    """Left: the graph with a maximum cut coloured in. Right: distribution of cut values per depth.

    The distributions are the exact simulated ones unless ``measured`` gives, per depth,
    measured probabilities keyed by bitstring.
    """
    import matplotlib.pyplot as plt  # imported here so the table works without a display

    fig, (ax_graph, ax_dist) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1, 1.5]})

    xy = circular_layout(graph.num_nodes) if positions is None else positions
    for i, j, _ in graph.edges:
        crosses = assignment[i] != assignment[j]
        ax_graph.plot(
            *xy[[i, j]].T,
            color="tab:red" if crosses else "lightgray",
            linewidth=2.5 if crosses else 1.5,
            zorder=1,
        )
    colours = ["#333333" if side else "#bbbbbb" for side in assignment]
    ax_graph.scatter(*xy.T, s=600, c=colours, edgecolors="black", zorder=2)
    for node, (x, y) in enumerate(xy):
        ax_graph.text(
            x,
            y,
            str(node),
            ha="center",
            va="center",
            color="white" if assignment[node] else "black",
            fontweight="bold",
        )
    ax_graph.set_title(f"A maximum cut: {cut_value(graph, assignment):g} edges (red) cross")
    ax_graph.set_aspect("equal")
    ax_graph.margins(0.15)
    ax_graph.axis("off")

    values = cut_values(graph)
    levels = np.unique(values)
    series: dict[str, np.ndarray] = {"random guess": np.full(values.shape, 1 / values.size)}
    for result in results:
        probs: np.ndarray
        if measured is not None and result.layers in measured:
            probs = np.zeros(values.size)
            for bits, p in measured[result.layers].items():
                probs[int(bits, 2)] += p
        else:
            probs = state_probabilities(graph, result.gammas, result.betas)
        series[f"QAOA p={result.layers}"] = probs
    width = 0.8 / len(series)
    slots = np.arange(len(levels))  # one slot per possible cut value, evenly spaced
    for k, (label, distribution) in enumerate(series.items()):
        totals = [distribution[values == level].sum() for level in levels]
        ax_dist.bar(
            slots + (k - (len(series) - 1) / 2) * width,
            totals,
            width,
            label=label,
            color="#999999" if k == 0 else None,
        )
    ax_dist.set_xlabel("cut value")
    ax_dist.set_ylabel("probability")
    ax_dist.set_title("Deeper circuits concentrate on the best cut" if source == "exact" else source)
    ax_dist.set_xticks(slots, [f"{level:g}" for level in levels])
    ax_dist.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
