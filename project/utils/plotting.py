"""
Plotting utilities for loss landscape experiments.

This module provides small, reusable helpers for:
    - line plots (e.g. interpolation, random 1D slices)
    - 3D surface + contour plots (e.g. 2D slices, PCA grids)
    - histograms (e.g. sharpness distributions)
    - stem plots (e.g. Hessian eigenvalues)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (required for 3D projection)


def _ensure_parent_dir(path: Path) -> None:
    """
    Ensure that the parent directory of a file path exists.

    Args:
        path (Path): Target file path.

    Returns:
        None
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_line(
    x: Sequence[float],
    y: Sequence[float],
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """
    Create a simple line plot and save it to disk.

    Args:
        x (Sequence[float]): Horizontal axis values.
        y (Sequence[float]): Vertical axis values.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Plot title.
        output_path (Path): Path to save the figure.

    Returns:
        None
    """
    _ensure_parent_dir(output_path)
    fig, ax = plt.subplots()
    ax.plot(x, y, marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_line_multi(
    x: Sequence[float],
    ys: Sequence[Sequence[float]],
    labels: Sequence[str],
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """
    Create a multi-line plot and save it to disk.

    Args:
        x (Sequence[float]): Shared x-axis values.
        ys (Sequence[Sequence[float]]): Collection of y-series.
        labels (Sequence[str]): Labels for each series.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Plot title.
        output_path (Path): Path to save the figure.

    Returns:
        None
    """
    _ensure_parent_dir(output_path)
    fig, ax = plt.subplots()
    for series, label in zip(ys, labels):
        ax.plot(x, series, marker="o", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_surface_and_contour(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    title: str,
    output_surface_path: Path,
    output_contour_path: Path,
) -> None:
    """
    Create 3D surface and 2D contour plots from grid data.

    Args:
        x_grid (np.ndarray): Grid of x-values (N, M).
        y_grid (np.ndarray): Grid of y-values (N, M).
        z_grid (np.ndarray): Grid of z-values (N, M).
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        zlabel (str): Label for z-axis.
        title (str): Plot title.
        output_surface_path (Path): Path for the 3D surface figure.
        output_contour_path (Path): Path for the 2D contour figure.

    Returns:
        None
    """
    _ensure_parent_dir(output_surface_path)
    _ensure_parent_dir(output_contour_path)

    # 3D surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        cmap=cm.viridis,
        linewidth=0,
        antialiased=True,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    fig.tight_layout()
    fig.savefig(output_surface_path)
    plt.close(fig)

    # 2D contour
    fig2, ax2 = plt.subplots()
    contour = ax2.contourf(x_grid, y_grid, z_grid, cmap=cm.viridis)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.set_title(title)
    fig2.colorbar(contour)
    fig2.tight_layout()
    fig2.savefig(output_contour_path)
    plt.close(fig2)


def plot_histogram(
    values: Sequence[float],
    bins: int,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """
    Plot a histogram of values.

    Args:
        values (Sequence[float]): Data values to histogram.
        bins (int): Number of bins.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        title (str): Plot title.
        output_path (Path): Path to save the figure.

    Returns:
        None
    """
    _ensure_parent_dir(output_path)
    fig, ax = plt.subplots()
    ax.hist(values, bins=bins, alpha=0.8, edgecolor="black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_stem(
    x: Sequence[float],
    y: Sequence[float],
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """
    Create a stem plot (useful for spectra such as Hessian eigenvalues).

    Args:
        x (Sequence[float]): x-axis positions.
        y (Sequence[float]): y-axis values.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        title (str): Plot title.
        output_path (Path): Path to save the figure.

    Returns:
        None
    """
    _ensure_parent_dir(output_path)
    fig, ax = plt.subplots()
    # ``use_line_collection`` is deprecated/removed in newer Matplotlib
    # versions, so we rely on default behavior for compatibility.
    markerline, stemlines, baseline = ax.stem(x, y)
    markerline.set_markerfacecolor("blue")
    stemlines.set_linewidth(1.5)
    baseline.set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_trajectory_2d(
    coords: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """
    Plot a 2D trajectory (e.g. PCA path) with arrows indicating direction.

    Args:
        coords (np.ndarray): Array of shape (T, 2) with trajectory points.
        xlabel (str): x-axis label.
        ylabel (str): y-axis label.
        title (str): Plot title.
        output_path (Path): Path to save the figure.

    Returns:
        None
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (T, 2).")

    _ensure_parent_dir(output_path)
    fig, ax = plt.subplots()
    ax.plot(coords[:, 0], coords[:, 1], "-o", markersize=3)
    for i in range(len(coords) - 1):
        ax.arrow(
            coords[i, 0],
            coords[i, 1],
            coords[i + 1, 0] - coords[i, 0],
            coords[i + 1, 1] - coords[i, 1],
            head_width=0.05,
            head_length=0.08,
            length_includes_head=True,
            fc="red",
            ec="red",
            alpha=0.7,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
