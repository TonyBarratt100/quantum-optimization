"""Command-line demo: solve MaxCut on the 6-node prism graph with QAOA.

python -m qaoa_maxcut                   # p = 1, 2, 3; saves qaoa_maxcut.png
python -m qaoa_maxcut --layers 1 2 3 4 --shots 2000 --no-plot
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
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
from .qaoa import QAOAResult, best_sampled_cut, optimize, sample, state_probabilities


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m qaoa_maxcut", description=__doc__.splitlines()[0])
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3], help="QAOA depths p to try")
    parser.add_argument("--restarts", type=int, default=10, help="random starts per depth")
    parser.add_argument("--shots", type=int, default=1000, help="measurements per depth")
    parser.add_argument("--seed", type=int, default=7, help="seed for reproducible results")
    parser.add_argument("--plot", type=Path, default=Path("qaoa_maxcut.png"), help="where to save the plot")
    parser.add_argument("--no-plot", action="store_true", help="skip the plot")
    args = parser.parse_args(argv)
    if min(args.layers) < 1:
        parser.error("--layers values must be at least 1")
    if args.restarts < 1:
        parser.error("--restarts must be at least 1")
    if args.shots < 1:
        parser.error("--shots must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
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
) -> None:
    """Left: the graph with a maximum cut coloured in. Right: distribution of cut values per depth."""
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
    series = {"random guess": np.full(values.shape, 1 / values.size)}
    for result in results:
        series[f"QAOA p={result.layers}"] = state_probabilities(graph, result.gammas, result.betas)
    width = 0.8 / len(series)
    slots = np.arange(len(levels))  # one slot per possible cut value, evenly spaced
    for k, (label, probs) in enumerate(series.items()):
        totals = [probs[values == level].sum() for level in levels]
        ax_dist.bar(
            slots + (k - (len(series) - 1) / 2) * width,
            totals,
            width,
            label=label,
            color="#999999" if k == 0 else None,
        )
    ax_dist.set_xlabel("cut value")
    ax_dist.set_ylabel("probability")
    ax_dist.set_title("Deeper circuits concentrate on the best cut")
    ax_dist.set_xticks(slots, [f"{level:g}" for level in levels])
    ax_dist.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
