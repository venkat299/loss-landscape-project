"""
Generate a multigrid summary figure for Appendix A.

This script aggregates per-run metrics across datasets, architectures,
activations, and optimizers and visualizes mean test accuracy as a
heatmap grid for each dataset. The resulting figure can be referenced
from Appendix A in the final report.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from project.reports.markdown import RunSummary, _load_run_summaries


def _group_summaries_by_dataset(
    summaries: Sequence[RunSummary],
) -> Dict[str, List[RunSummary]]:
    """
    Group run summaries by dataset name.

    Args:
        summaries (Sequence[RunSummary]): Collection of run summaries.

    Returns:
        Dict[str, List[RunSummary]]: Mapping from dataset name to runs.
    """
    grouped: Dict[str, List[RunSummary]] = {}
    for s in summaries:
        grouped.setdefault(s.dataset, []).append(s)
    return grouped


def generate_appendix_a_multigrid(
    experiments_root: Path,
    output_path: Path,
) -> None:
    """
    Generate a multigrid heatmap summarizing Appendix A metrics.

    Rows correspond to architectures and columns to (activation, optimizer)
    combinations; each dataset is plotted as a separate panel.

    Args:
        experiments_root (Path): Root directory containing experiment runs.
        output_path (Path): Destination path for the generated figure.

    Returns:
        None
    """
    summaries = _load_run_summaries(experiments_root)
    if not summaries:
        raise ValueError(f"No run summaries found under {experiments_root}")

    by_dataset = _group_summaries_by_dataset(summaries)
    datasets = sorted(by_dataset.keys())
    architectures = sorted({s.architecture for s in summaries})
    activations = sorted({s.activation for s in summaries})
    optimizers = sorted({s.optimizer for s in summaries})

    combos: List[Tuple[str, str]] = []
    col_labels: List[str] = []
    for act in activations:
        for opt in optimizers:
            combos.append((act, opt))
            col_labels.append(f"{act}\n{opt}")

    n_datasets = len(datasets)
    fig, axes = plt.subplots(
        1,
        n_datasets,
        figsize=(4 * n_datasets, 4),
        squeeze=False,
    )

    for idx, dataset in enumerate(datasets):
        ax = axes[0, idx]
        ds_summaries = by_dataset[dataset]
        grid = np.full((len(architectures), len(combos)), np.nan, dtype=float)

        for i, arch in enumerate(architectures):
            for j, (act, opt) in enumerate(combos):
                values: List[float] = [
                    s.final_test_accuracy
                    for s in ds_summaries
                    if s.architecture == arch and s.activation == act and s.optimizer == opt
                ]
                if values:
                    grid[i, j] = float(sum(values) / len(values))

        im = ax.imshow(grid, vmin=0.8, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_title(dataset)
        ax.set_xticks(range(len(combos)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(len(architectures)))
        ax.set_yticklabels(architectures, fontsize=6)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Test accuracy")

    fig.suptitle("Appendix A Multigrid: Test Accuracy by Architecture / Activation / Optimizer")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the multigrid figure generator.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate multigrid summary figure for Appendix A.",
    )
    parser.add_argument(
        "--experiments-root",
        type=str,
        default="reports/experiments",
        help="Root directory containing experiment runs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/figures/appendix_a_multigrid.png",
        help="Output path for the multigrid figure.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point for CLI execution of Appendix A multigrid figure generation.

    Returns:
        None
    """
    args = _parse_args()
    experiments_root = Path(args.experiments_root).resolve()
    output_path = Path(args.output).resolve()
    generate_appendix_a_multigrid(experiments_root, output_path)


if __name__ == "__main__":
    main()

