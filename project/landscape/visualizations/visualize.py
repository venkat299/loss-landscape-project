"""
Visualization utilities for loss landscape probes.

This module glues together numerical outputs from the probing modules
and plotting helpers in ``project.utils.plotting`` to produce figures
saved under ``reports/figures/<experiment_group>/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

from project.utils.plotting import (
    plot_histogram,
    plot_line,
    plot_line_multi,
    plot_stem,
    plot_surface_and_contour,
    plot_trajectory_2d,
)


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array on CPU.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        np.ndarray: Converted NumPy array.
    """
    return tensor.detach().cpu().numpy()


def save_interpolation_plots(
    results: Dict[str, torch.Tensor],
    output_dir: Path,
    prefix: str = "interp",
) -> None:
    """
    Save interpolation loss and accuracy curves.

    Args:
        results (Dict[str, torch.Tensor]): Output of
            ``linear_interpolation_curve``.
        output_dir (Path): Directory where figures will be saved.
        prefix (str): File name prefix for plots.

    Returns:
        None
    """
    alphas = _tensor_to_numpy(results["alphas"])
    train_loss = _tensor_to_numpy(results["train_loss"])
    test_loss = _tensor_to_numpy(results["test_loss"])
    train_acc = _tensor_to_numpy(results["train_accuracy"])
    test_acc = _tensor_to_numpy(results["test_accuracy"])

    plot_line_multi(
        x=alphas,
        ys=[train_loss, test_loss],
        labels=["train", "test"],
        xlabel="alpha",
        ylabel="loss",
        title="Interpolation loss",
        output_path=output_dir / f"{prefix}_loss.png",
    )

    plot_line_multi(
        x=alphas,
        ys=[train_acc, test_acc],
        labels=["train", "test"],
        xlabel="alpha",
        ylabel="accuracy",
        title="Interpolation accuracy",
        output_path=output_dir / f"{prefix}_accuracy.png",
    )


def save_random_slice_plots(
    slice_1d: Dict[str, torch.Tensor],
    slice_2d: Dict[str, torch.Tensor],
    output_dir: Path,
    prefix: str = "slice",
) -> None:
    """
    Save 1D and 2D random slice plots.

    Args:
        slice_1d (Dict[str, torch.Tensor]): Output of
            ``random_1d_loss_slice``.
        slice_2d (Dict[str, torch.Tensor]): Output of
            ``random_2d_loss_surface``.
        output_dir (Path): Directory where figures will be saved.
        prefix (str): File name prefix.

    Returns:
        None
    """
    # 1D slices
    alphas_1d = _tensor_to_numpy(slice_1d["alphas"])
    train_loss_1d = _tensor_to_numpy(slice_1d["train_loss"])
    test_loss_1d = _tensor_to_numpy(slice_1d["test_loss"])

    plot_line_multi(
        x=alphas_1d,
        ys=[train_loss_1d, test_loss_1d],
        labels=["train", "test"],
        xlabel="alpha",
        ylabel="loss",
        title="1D random direction loss slice",
        output_path=output_dir / f"{prefix}_1d.png",
    )

    # 2D slices
    alpha_grid = _tensor_to_numpy(slice_2d["alpha_grid"])
    beta_grid = _tensor_to_numpy(slice_2d["beta_grid"])
    train_loss_2d = _tensor_to_numpy(slice_2d["train_loss"])

    plot_surface_and_contour(
        x_grid=alpha_grid,
        y_grid=beta_grid,
        z_grid=train_loss_2d,
        xlabel="alpha",
        ylabel="beta",
        zlabel="train loss",
        title="2D random direction train loss surface",
        output_surface_path=output_dir / f"{prefix}_2d_surface.png",
        output_contour_path=output_dir / f"{prefix}_2d_contour.png",
    )


def save_hessian_spectrum_plot(
    eigenvalues: torch.Tensor,
    output_dir: Path,
    prefix: str = "hessian",
) -> None:
    """
    Save a stem plot of Hessian eigenvalues.

    Args:
        eigenvalues (torch.Tensor): Top-k eigenvalues from Hessian spectrum
            estimation.
        output_dir (Path): Directory where the figure will be saved.
        prefix (str): File name prefix.

    Returns:
        None
    """
    eig_vals = _tensor_to_numpy(eigenvalues)
    indices = np.arange(len(eig_vals))

    plot_stem(
        x=indices,
        y=eig_vals,
        xlabel="index",
        ylabel="eigenvalue",
        title="Top-k Hessian eigenvalues",
        output_path=output_dir / f"{prefix}_spectrum.png",
    )


def save_sharpness_histogram(
    loss_increases: torch.Tensor,
    output_dir: Path,
    prefix: str = "sharpness",
    bins: int = 30,
) -> None:
    """
    Save a histogram of sharpness-induced loss increases.

    Args:
        loss_increases (torch.Tensor): 1D tensor of loss increases returned
            by ``epsilon_sharpness``.
        output_dir (Path): Directory where the figure will be saved.
        prefix (str): File name prefix.
        bins (int): Number of histogram bins.

    Returns:
        None
    """
    values = _tensor_to_numpy(loss_increases)
    plot_histogram(
        values=values,
        bins=bins,
        xlabel="loss increase",
        ylabel="count",
        title="Sharpness loss increase distribution",
        output_path=output_dir / f"{prefix}_hist.png",
    )


def save_pca_plots(
    trajectory_coords: torch.Tensor,
    loss_grid: Dict[str, torch.Tensor],
    output_dir: Path,
    prefix: str = "pca",
) -> None:
    """
    Save PCA trajectory and PCA-plane loss surface plots.

    Args:
        trajectory_coords (torch.Tensor): Tensor of shape (T, 2) containing
            PCA coordinates of the training trajectory.
        loss_grid (Dict[str, torch.Tensor]): Output dictionary from
            ``loss_over_pca_grid``.
        output_dir (Path): Directory where the figures will be saved.
        prefix (str): File name prefix.

    Returns:
        None
    """
    coords_np = _tensor_to_numpy(trajectory_coords)
    plot_trajectory_2d(
        coords=coords_np,
        xlabel="PC1",
        ylabel="PC2",
        title="Training trajectory in PCA plane",
        output_path=output_dir / f"{prefix}_trajectory.png",
    )

    alpha_grid = _tensor_to_numpy(loss_grid["alpha_grid"])
    beta_grid = _tensor_to_numpy(loss_grid["beta_grid"])
    train_loss = _tensor_to_numpy(loss_grid["train_loss"])

    plot_surface_and_contour(
        x_grid=alpha_grid,
        y_grid=beta_grid,
        z_grid=train_loss,
        xlabel="PC1 coeff",
        ylabel="PC2 coeff",
        zlabel="train loss",
        title="Loss over PCA plane",
        output_surface_path=output_dir / f"{prefix}_surface.png",
        output_contour_path=output_dir / f"{prefix}_contour.png",
    )


def save_connectivity_plots(
    results: Dict[str, torch.Tensor | float],
    output_dir: Path,
    prefix: str = "connectivity",
) -> None:
    """
    Save linear connectivity loss curves with annotated barrier heights.

    Args:
        results (Dict[str, torch.Tensor | float]): Output dictionary from
            ``linear_connectivity_barrier`` containing loss curves and
            barrier heights.
        output_dir (Path): Directory where figures will be saved.
        prefix (str): File name prefix for plots.

    Returns:
        None
    """
    alphas = _tensor_to_numpy(results["alphas"])  # type: ignore[arg-type]
    train_loss = _tensor_to_numpy(results["train_loss"])  # type: ignore[arg-type]
    test_loss = _tensor_to_numpy(results["test_loss"])  # type: ignore[arg-type]

    train_barrier = float(results["train_barrier"])  # type: ignore[arg-type]
    test_barrier = float(results["test_barrier"])  # type: ignore[arg-type]

    train_label = f"train (barrier={train_barrier:.4f})"
    test_label = f"test (barrier={test_barrier:.4f})"

    plot_line_multi(
        x=alphas,
        ys=[train_loss, test_loss],
        labels=[train_label, test_label],
        xlabel="alpha",
        ylabel="loss",
        title="Linear mode connectivity",
        output_path=output_dir / f"{prefix}_loss.png",
    )


def save_regularization_interpolation_surface(
    alphas: torch.Tensor,
    weight_decays: Sequence[float],
    loss_grid: torch.Tensor,
    output_dir: Path,
    title: str,
    prefix: str = "regularization_interp",
) -> None:
    """
    Save a 3D surface showing interpolation loss as a function of
    interpolation parameter and weight decay.

    Args:
        alphas (torch.Tensor): 1D tensor of interpolation coefficients
            between initialization and final weights.
        weight_decays (Sequence[float]): Sequence of weight decay values.
        loss_grid (torch.Tensor): 2D tensor of shape
            (len(weight_decays), len(alphas)) containing loss values.
        output_dir (Path): Directory where figures will be saved.
        title (str): Plot title describing the configuration.
        prefix (str): File name prefix for output plots.

    Returns:
        None
    """
    alpha_np = _tensor_to_numpy(alphas)
    loss_np = _tensor_to_numpy(loss_grid)
    weight_np = np.asarray(weight_decays, dtype=float)

    alpha_grid, weight_grid = np.meshgrid(alpha_np, weight_np)

    surface_path = output_dir / f"{prefix}_surface.png"
    contour_path = output_dir / f"{prefix}_contour.png"

    plot_surface_and_contour(
        x_grid=alpha_grid,
        y_grid=weight_grid,
        z_grid=loss_np,
        xlabel="alpha (init→final)",
        ylabel="weight_decay",
        zlabel="train loss",
        title=title,
        output_surface_path=surface_path,
        output_contour_path=contour_path,
    )


def save_regularization_random_slice_surfaces(
    alpha_grid: torch.Tensor,
    beta_grid: torch.Tensor,
    weight_decays: Sequence[float],
    loss_volume: torch.Tensor,
    output_dir: Path,
    title_prefix: str = "2D random slice train loss",
    filename_prefix: str = "regularization_slice",
) -> None:
    """
    Save 3D surfaces for 2D random slices across multiple weight decays.

    For each weight decay value, this function creates a 3D surface and
    2D contour plot over the (alpha, beta) plane using the corresponding
    slice from ``loss_volume``.

    Args:
        alpha_grid (torch.Tensor): Grid of alpha values (N, N).
        beta_grid (torch.Tensor): Grid of beta values (N, N).
        weight_decays (Sequence[float]): Sequence of weight decay values in
            the same order as the first dimension of ``loss_volume``.
        loss_volume (torch.Tensor): Tensor of shape
            (len(weight_decays), N, N) with train loss values.
        output_dir (Path): Directory where figures will be saved.
        title_prefix (str): Prefix for the plot title.
        filename_prefix (str): Prefix for output file names.

    Returns:
        None
    """
    alpha_np = _tensor_to_numpy(alpha_grid)
    beta_np = _tensor_to_numpy(beta_grid)
    loss_np = _tensor_to_numpy(loss_volume)

    num_decays = min(len(weight_decays), loss_np.shape[0])
    for idx in range(num_decays):
        wd = float(weight_decays[idx])
        z_grid = loss_np[idx]

        surface_path = output_dir / f"{filename_prefix}_wd{idx}_surface.png"
        contour_path = output_dir / f"{filename_prefix}_wd{idx}_contour.png"

        title = f"{title_prefix} (weight_decay={wd:.1e})"

        plot_surface_and_contour(
            x_grid=alpha_np,
            y_grid=beta_np,
            z_grid=z_grid,
            xlabel="alpha",
            ylabel="beta",
            zlabel="train loss",
            title=title,
            output_surface_path=surface_path,
            output_contour_path=contour_path,
        )
