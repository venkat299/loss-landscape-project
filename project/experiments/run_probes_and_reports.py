"""
Run all landscape probes and generate Markdown reports.

This script expects that training runs have already been completed and
stored under an experiments root directory (see
``project.experiments.run_full_matrix``). For each run it:

    1. Regenerates the synthetic dataset from saved configs.
    2. Loads initial and final checkpoints for the trained model.
    3. Runs landscape probes:
        - linear interpolation (init → final)
        - 1D and 2D random direction slices
        - Hessian spectrum (top-k eigenvalues + trace)
        - ε-sharpness metric
    4. Performs connectivity tests among seeds for each configuration.
    5. Computes PCA projections using checkpoint collections.
    6. Saves figures under ``reports/figures/...``.
    7. Generates Markdown reports under ``reports/``.

Usage (example):
    uv run python -m project.experiments.run_probes_and_reports \\
        --experiments-root reports/experiments \\
        --figures-root reports/figures \\
        --reports-root reports
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from project.data import (
    generate_circles_dataset,
    generate_gaussian_clusters_dataset,
    generate_moons_dataset,
    generate_xor_dataset,
)
from project.experiments import evaluate_model
from project.landscape.connectivity import (
    apply_neuron_permutation_1layer,
    compute_neuron_permutation_1layer,
    linear_connectivity_barrier,
)
from project.landscape.common import flatten_parameters, get_parameter_tensors
from project.landscape.hessian import estimate_hessian_spectrum
from project.landscape.interpolation import linear_interpolation_curve
from project.landscape.pca import (
    collect_parameter_trajectory,
    compute_pca_components,
    loss_over_pca_grid,
    project_trajectory_to_pca,
)
from project.landscape.random_slice import (
    random_1d_loss_slice,
    random_2d_loss_surface,
)
from project.landscape.sharpness import epsilon_sharpness
from project.landscape.visualizations import (
    save_connectivity_plots,
    save_hessian_spectrum_plot,
    save_interpolation_plots,
    save_pca_plots,
    save_random_slice_plots,
    save_regularization_interpolation_surface,
    save_regularization_random_slice_surfaces,
    save_sharpness_histogram,
)
from project.models import MLPClassifier, build_mlp_from_config
from project.reports import generate_study_reports
from project.utils.configs import (
    DatasetConfig,
    HessianConfig,
    ModelConfig,
    PCAConfig,
    SharpnessConfig,
    SliceConfig,
    TrainingConfig,
)
from project.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """
    Configuration for a single trained run reconstructed from disk.

    Args:
        dataset (DatasetConfig): Dataset configuration.
        model (ModelConfig): MLP architecture configuration.
        training (TrainingConfig): Training configuration.
        seed (int): Random seed used for training.
        run_dir (Path): Directory containing checkpoints and metrics.
    """

    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    seed: int
    run_dir: Path


def _load_run_configs(experiments_root: Path) -> List[RunConfig]:
    """
    Discover and reconstruct run configurations from an experiments root.

    Args:
        experiments_root (Path): Root directory containing training runs.

    Returns:
        List[RunConfig]: List of reconstructed run configurations.
    """
    run_configs: List[RunConfig] = []

    if not experiments_root.exists():
        return run_configs

    for summary_path in experiments_root.rglob("summary.json"):
        run_dir = summary_path.parent
        with summary_path.open("r", encoding="utf-8") as f:
            summary_list = json.load(f)
        if not summary_list:
            continue
        summary = summary_list[0]

        dataset_cfg = DatasetConfig(
            name=summary["dataset"]["name"],
            n_train=int(summary["dataset"]["n_train"]),
            n_test=int(summary["dataset"]["n_test"]),
            noise=float(summary["dataset"]["noise"]),
            n_classes=int(summary["dataset"]["n_classes"]),
            seed=int(summary["dataset"]["seed"]),
        )

        model_cfg = ModelConfig(
            input_dim=int(summary["model"]["input_dim"]),
            hidden_layers=int(summary["model"]["hidden_layers"]),
            hidden_size=int(summary["model"]["hidden_size"]),
            output_dim=int(summary["model"]["output_dim"]),
            activation=summary["model"]["activation"],
            bias=bool(summary["model"]["bias"]),
            dropout=float(summary["model"]["dropout"]),
        )

        training_cfg = TrainingConfig(
            optimizer=summary["training"]["optimizer"],
            learning_rate=float(summary["training"]["learning_rate"]),
            momentum=float(summary["training"]["momentum"]),
            weight_decay=float(summary["training"]["weight_decay"]),
            batch_size=int(summary["training"]["batch_size"]),
            epochs=int(summary["training"]["epochs"]),
            lr_step_size=int(summary["training"]["lr_step_size"]),
            lr_gamma=float(summary["training"]["lr_gamma"]),
            device=summary["training"]["device"],
            seed=int(summary["training"]["seed"]),
            checkpoint_dir=run_dir / "checkpoints",
            log_interval=int(summary["training"]["log_interval"]),
        )

        seed = int(summary["seed"])

        run_configs.append(
            RunConfig(
                dataset=dataset_cfg,
                model=model_cfg,
                training=training_cfg,
                seed=seed,
                run_dir=run_dir,
            )
        )

    return run_configs


def _generate_dataset(config: DatasetConfig) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Regenerate a synthetic dataset from a DatasetConfig.

    Args:
        config (DatasetConfig): Dataset configuration.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]:
            Training and test feature/label tensors.
    """
    if config.name == "moons":
        return generate_moons_dataset(config)
    if config.name == "circles":
        return generate_circles_dataset(config)
    if config.name == "xor":
        return generate_xor_dataset(config)
    if config.name == "gaussians":
        return generate_gaussian_clusters_dataset(config)
    raise ValueError(f"Unsupported dataset type: {config.name}")


def _make_dataloaders(
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and test DataLoaders from tensors.

    Args:
        x_train (Tensor): Training features.
        y_train (Tensor): Training labels.
        x_test (Tensor): Test features.
        y_test (Tensor): Test labels.
        batch_size (int): Batch size.

    Returns:
        Tuple[DataLoader, DataLoader]: Train and test DataLoaders.
    """
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def _load_model_from_checkpoint(
    model_cfg: ModelConfig,
    checkpoint_path: Path,
    device: torch.device,
) -> MLPClassifier:
    """
    Construct an MLP model and load weights from a checkpoint.

    This helper uses ``weights_only=False`` when calling ``torch.load``.
    The checkpoints are produced by this project itself, so this is safe
    and avoids issues with PyTorch's default ``weights_only=True`` strict
    unpickling behavior in newer versions.

    Args:
        model_cfg (ModelConfig): Model configuration.
        checkpoint_path (Path): Path to the checkpoint file.
        device (torch.device): Device on which to place the model.

    Returns:
        MLPClassifier: Model instance with loaded weights.
    """
    model = build_mlp_from_config(model_cfg)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    return model.to(device)


def _run_single_run_probes(
    run: RunConfig,
    figures_root: Path,
    device: torch.device,
    interp_points: int,
    slice_1d_points: int,
    slice_1d_radius: float,
    slice_2d_points: int,
    slice_2d_radius: float,
    hessian_cfg: HessianConfig,
    sharpness_cfg: SharpnessConfig,
    pca_cfg: PCAConfig,
) -> None:
    """
    Run per-run landscape probes and save figures.

    Args:
        run (RunConfig): Run configuration.
        figures_root (Path): Root directory for figures.
        device (torch.device): Computation device.
        interp_points (int): Number of interpolation points.
        slice_1d_points (int): Number of 1D slice points.
        slice_1d_radius (float): Radius for 1D slice.
        slice_2d_points (int): Number of 2D slice points.
        slice_2d_radius (float): Radius for 2D slice.
        hessian_cfg (HessianConfig): Hessian configuration.
        sharpness_cfg (SharpnessConfig): Sharpness configuration.
        pca_cfg (PCAConfig): PCA configuration (for grid resolution).

    Returns:
        None
    """
    _ = pca_cfg  # PCA is handled at the group level; config kept for parity.

    set_global_seed(run.seed)

    x_train, y_train, x_test, y_test = _generate_dataset(run.dataset)
    train_loader, test_loader = _make_dataloaders(
        x_train, y_train, x_test, y_test, batch_size=run.training.batch_size
    )

    init_ckpt = run.training.checkpoint_dir / "init_epoch0.pt"
    final_ckpt = run.training.checkpoint_dir / f"final_epoch{run.training.epochs}.pt"

    interp_dir = (
        figures_root
        / f"dataset={run.dataset.name}"
        / f"arch={run.model.hidden_layers}x{run.model.hidden_size}"
        / f"act={run.model.activation}"
        / f"opt={run.training.optimizer}"
        / f"seed={run.seed}"
        / "interpolation"
    )

    # Interpolation: initial → final.
    model_init = _load_model_from_checkpoint(run.model, init_ckpt, device)
    model_final = _load_model_from_checkpoint(run.model, final_ckpt, device)

    interp_results = linear_interpolation_curve(
        model_a=model_init,
        model_b=model_final,
        train_loader=train_loader,
        test_loader=test_loader,
        num_points=interp_points,
        device=device,
        normalize_per_layer=True,
    )
    save_interpolation_plots(interp_results, output_dir=interp_dir, prefix="init_final")

    # Path-based flatness metrics along the interpolation.
    interp_stats = {
        "num_points": int(interp_results["train_loss"].numel()),
        "train_min": float(interp_results["train_loss"].min().item()),
        "train_max": float(interp_results["train_loss"].max().item()),
        "train_mean": float(interp_results["train_loss"].mean().item()),
        "test_min": float(interp_results["test_loss"].min().item()),
        "test_max": float(interp_results["test_loss"].max().item()),
        "test_mean": float(interp_results["test_loss"].mean().item()),
    }
    interp_stats_path = interp_dir / "interp_stats.json"
    interp_stats_path.parent.mkdir(parents=True, exist_ok=True)
    with interp_stats_path.open("w", encoding="utf-8") as f_interp:
        json.dump(interp_stats, f_interp, indent=2)

    # Random slices (1D and 2D) around the final model.
    slice_cfg_1d = SliceConfig(num_points=slice_1d_points, radius=slice_1d_radius)
    slice_cfg_2d = SliceConfig(num_points=slice_2d_points, radius=slice_2d_radius)

    slice_dir = interp_dir.parent / "random_slice"

    slice_1d = random_1d_loss_slice(
        model=model_final,
        train_loader=train_loader,
        test_loader=test_loader,
        config=slice_cfg_1d,
        device=device,
        seed=run.seed * 10 + 1,
    )
    slice_2d = random_2d_loss_surface(
        model=model_final,
        train_loader=train_loader,
        test_loader=test_loader,
        config=slice_cfg_2d,
        device=device,
        seed=run.seed * 10 + 2,
    )
    save_random_slice_plots(slice_1d, slice_2d, output_dir=slice_dir, prefix="random")

    # Hessian spectrum (top-k eigenvalues + trace estimate).
    hessian_dir = interp_dir.parent / "hessian"
    eigenvalues, trace_estimate = estimate_hessian_spectrum(
        model=model_final,
        dataloader=train_loader,
        config=hessian_cfg,
        device=device,
    )
    save_hessian_spectrum_plot(eigenvalues, output_dir=hessian_dir, prefix="hessian")

    spectrum_json = {
        "eigenvalues": eigenvalues.tolist(),
        "trace_estimate": trace_estimate,
    }
    hessian_json_path = hessian_dir / "spectrum.json"
    hessian_json_path.parent.mkdir(parents=True, exist_ok=True)
    with hessian_json_path.open("w", encoding="utf-8") as f_spec:
        json.dump(spectrum_json, f_spec, indent=2)

    # Sharpness / flatness metric.
    sharpness_dir = interp_dir.parent / "sharpness"
    sharp_value, increases = epsilon_sharpness(
        model=model_final,
        dataloader=train_loader,
        config=sharpness_cfg,
        device=device,
        seed=run.seed * 10 + 3,
    )
    save_sharpness_histogram(increases, output_dir=sharpness_dir, prefix="sharpness")

    sharp_json = {
        "epsilon": sharpness_cfg.epsilon,
        "num_samples": sharpness_cfg.num_samples,
        "sharpness": sharp_value,
    }
    sharp_json_path = sharpness_dir / "sharpness.json"
    with sharp_json_path.open("w", encoding="utf-8") as f_sharp:
        json.dump(sharp_json, f_sharp, indent=2)

    # PAC-Bayes-style metric: distance from initialization in parameter space.
    init_params = get_parameter_tensors(model_init)
    final_params = get_parameter_tensors(model_final)
    diff_flat = flatten_parameters([fb - fa for fa, fb in zip(init_params, final_params)])
    weight_distance = float(diff_flat.norm().item())

    criterion = nn.CrossEntropyLoss()
    final_train_loss, _ = evaluate_model(model_final, train_loader, criterion, device)

    run_root = run.training.checkpoint_dir.parent
    pac_path = run_root / "pacbayes_style.json"
    pac_metrics = {
        "weight_l2_distance_from_init": weight_distance,
        "final_train_loss": final_train_loss,
    }
    with pac_path.open("w", encoding="utf-8") as f_pac:
        json.dump(pac_metrics, f_pac, indent=2)


def _group_runs_by_configuration(
    runs: Sequence[RunConfig],
) -> Dict[Tuple[str, int, int, str, str], List[RunConfig]]:
    """
    Group runs by (dataset, hidden_layers, hidden_size, activation, optimizer).

    Args:
        runs (Sequence[RunConfig]): List of run configurations.

    Returns:
        Dict[Tuple[str, int, int, str, str], List[RunConfig]]:
            Mapping from configuration key to runs (different seeds).
    """
    groups: Dict[Tuple[str, int, int, str, str], List[RunConfig]] = {}
    for r in runs:
        key = (
            r.dataset.name,
            r.model.hidden_layers,
            r.model.hidden_size,
            r.model.activation,
            r.training.optimizer,
        )
        groups.setdefault(key, []).append(r)
    return groups


def _group_runs_by_base_configuration(
    runs: Sequence[RunConfig],
) -> Dict[Tuple[str, int, int, str, str], List[RunConfig]]:
    """
    Group runs by dataset, architecture, activation, and optimizer,
    ignoring regularization strength.

    Args:
        runs (Sequence[RunConfig]): List of run configurations.

    Returns:
        Dict[Tuple[str, int, int, str, str], List[RunConfig]]:
            Mapping from configuration key to runs across weight decays.
    """
    return _group_runs_by_configuration(runs)


def _compute_interpolation_curve_for_run(
    run: RunConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_points: int,
) -> Dict[str, torch.Tensor]:
    """
    Compute an initialization-to-final interpolation curve for a single run.

    Args:
        run (RunConfig): Run configuration containing paths and hyperparameters.
        train_loader (DataLoader): Training DataLoader.
        test_loader (DataLoader): Test DataLoader.
        device (torch.device): Computation device.
        num_points (int): Number of interpolation points.

    Returns:
        Dict[str, torch.Tensor]: Output dictionary from
        ``linear_interpolation_curve`` restricted to the current run.
    """
    init_ckpt = run.training.checkpoint_dir / "init_epoch0.pt"
    final_ckpt = run.training.checkpoint_dir / f"final_epoch{run.training.epochs}.pt"

    model_init = _load_model_from_checkpoint(run.model, init_ckpt, device)
    model_final = _load_model_from_checkpoint(run.model, final_ckpt, device)

    interp_results = linear_interpolation_curve(
        model_a=model_init,
        model_b=model_final,
        train_loader=train_loader,
        test_loader=test_loader,
        num_points=num_points,
        device=device,
        normalize_per_layer=True,
    )
    return interp_results


def _run_regularization_interpolation_surfaces(
    runs: Sequence[RunConfig],
    figures_root: Path,
    device: torch.device,
    num_points: int,
) -> None:
    """
    Generate 3D interpolation loss surfaces as a function of weight decay.

    This analysis groups runs that share the same dataset, architecture,
    activation, and optimizer but differ in L2 weight decay. For each group,
    it computes interpolation curves between initialization and final weights
    and aggregates them into a surface over (alpha, weight_decay).

    Args:
        runs (Sequence[RunConfig]): All reconstructed run configurations.
        figures_root (Path): Root directory for figures.
        device (torch.device): Computation device.
        num_points (int): Number of interpolation points along alpha.

    Returns:
        None
    """
    base_groups = _group_runs_by_base_configuration(runs)

    for key, group_runs in base_groups.items():
        if not group_runs:
            continue

        # Collect distinct weight decay values in this group.
        weight_decays = sorted({float(r.training.weight_decay) for r in group_runs})
        if len(weight_decays) <= 1:
            # No variation in regularization; nothing to visualize.
            continue

        dataset_name, hidden_layers, hidden_size, activation, optimizer = key

        # Regenerate dataset once per group.
        dataset_cfg = group_runs[0].dataset
        x_train, y_train, x_test, y_test = _generate_dataset(dataset_cfg)
        train_loader, test_loader = _make_dataloaders(
            x_train, y_train, x_test, y_test, batch_size=group_runs[0].training.batch_size
        )

        # Organize runs by weight decay and aggregate interpolation curves.
        curves_by_decay: Dict[float, List[torch.Tensor]] = {wd: [] for wd in weight_decays}
        alphas_ref: torch.Tensor | None = None

        for run in group_runs:
            interp_results = _compute_interpolation_curve_for_run(
                run=run,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                num_points=num_points,
            )

            alphas = interp_results["alphas"]
            train_loss = interp_results["train_loss"]

            if alphas_ref is None:
                alphas_ref = alphas.detach().cpu()
            else:
                if alphas_ref.shape != alphas.shape or not torch.allclose(
                    alphas_ref, alphas.detach().cpu()
                ):
                    logger.warning(
                        "Skipping run at %s due to mismatched interpolation grid.",
                        run.run_dir,
                    )
                    continue

            weight_decay = float(run.training.weight_decay)
            curves_by_decay.setdefault(weight_decay, []).append(train_loss.detach().cpu())

        if alphas_ref is None:
            continue

        # Build loss grid: one row per weight decay, averaged over seeds.
        num_decays = len(weight_decays)
        num_alphas = int(alphas_ref.numel())
        loss_grid = torch.zeros(num_decays, num_alphas)

        for idx, wd in enumerate(weight_decays):
            curves = curves_by_decay.get(wd, [])
            if not curves:
                logger.warning(
                    "No interpolation curves collected for weight_decay=%f in group %s.",
                    wd,
                    key,
                )
                continue
            stacked = torch.stack(curves, dim=0)
            loss_grid[idx] = stacked.mean(dim=0)

        group_dir = (
            figures_root
            / f"dataset={dataset_name}"
            / f"arch={hidden_layers}x{hidden_size}"
            / f"act={activation}"
            / f"opt={optimizer}"
        )
        reg_dir = group_dir / "regularization"

        surface_path = reg_dir / "regularization_interp_surface.png"
        if surface_path.exists():
            continue

        title = (
            f"Interpolation loss vs weight decay\n"
            f"dataset={dataset_name}, arch={hidden_layers}x{hidden_size}, "
            f"act={activation}, opt={optimizer}"
        )
        reg_dir.mkdir(parents=True, exist_ok=True)
        save_regularization_interpolation_surface(
            alphas=alphas_ref,
            weight_decays=weight_decays,
            loss_grid=loss_grid,
            output_dir=reg_dir,
            title=title,
            prefix="regularization_interp",
        )


def _run_regularization_random_slice_surfaces(
    runs: Sequence[RunConfig],
    figures_root: Path,
    device: torch.device,
    num_points: int,
    radius: float,
) -> None:
    """
    Generate 3D random-slice loss surfaces as a function of weight decay.

    For each (dataset, architecture, activation, optimizer) group with
    multiple weight decay values, this computes 2D random direction
    loss surfaces around the final models and aggregates them across
    seeds to produce per-weight-decay surfaces.

    Args:
        runs (Sequence[RunConfig]): All reconstructed run configurations.
        figures_root (Path): Root directory for figures.
        device (torch.device): Computation device.
        num_points (int): Number of grid points per axis for slices.
        radius (float): Radius for random direction exploration.

    Returns:
        None
    """
    base_groups = _group_runs_by_base_configuration(runs)

    for key, group_runs in base_groups.items():
        if not group_runs:
            continue

        weight_decays = sorted({float(r.training.weight_decay) for r in group_runs})
        if len(weight_decays) <= 1:
            continue

        dataset_name, hidden_layers, hidden_size, activation, optimizer = key

        dataset_cfg = group_runs[0].dataset
        x_train, y_train, x_test, y_test = _generate_dataset(dataset_cfg)
        train_loader, test_loader = _make_dataloaders(
            x_train, y_train, x_test, y_test, batch_size=group_runs[0].training.batch_size
        )

        slice_cfg = SliceConfig(num_points=num_points, radius=radius)

        surfaces_by_decay: Dict[float, List[torch.Tensor]] = {wd: [] for wd in weight_decays}
        alpha_grid_ref: torch.Tensor | None = None
        beta_grid_ref: torch.Tensor | None = None

        for run in group_runs:
            ckpt = run.training.checkpoint_dir / f"final_epoch{run.training.epochs}.pt"
            model_final = _load_model_from_checkpoint(run.model, ckpt, device)

            slice_result = random_2d_loss_surface(
                model=model_final,
                train_loader=train_loader,
                test_loader=test_loader,
                config=slice_cfg,
                device=device,
                seed=0,
            )

            alpha_grid = slice_result["alpha_grid"]
            beta_grid = slice_result["beta_grid"]
            train_surface = slice_result["train_loss"]

            if alpha_grid_ref is None or beta_grid_ref is None:
                alpha_grid_ref = alpha_grid.detach().cpu()
                beta_grid_ref = beta_grid.detach().cpu()
            else:
                if (
                    alpha_grid_ref.shape != alpha_grid.shape
                    or beta_grid_ref.shape != beta_grid.shape
                    or not torch.allclose(alpha_grid_ref, alpha_grid.detach().cpu())
                    or not torch.allclose(beta_grid_ref, beta_grid.detach().cpu())
                ):
                    logger.warning(
                        "Skipping run at %s due to mismatched random slice grid.",
                        run.run_dir,
                    )
                    continue

            wd = float(run.training.weight_decay)
            surfaces_by_decay.setdefault(wd, []).append(train_surface.detach().cpu())

        if alpha_grid_ref is None or beta_grid_ref is None:
            continue

        num_decays = len(weight_decays)
        height, width = alpha_grid_ref.shape
        loss_volume = torch.zeros(num_decays, height, width)

        for idx, wd in enumerate(weight_decays):
            surfaces = surfaces_by_decay.get(wd, [])
            if not surfaces:
                logger.warning(
                    "No random slice surfaces collected for weight_decay=%f in group %s.",
                    wd,
                    key,
                )
                continue
            stacked = torch.stack(surfaces, dim=0)
            loss_volume[idx] = stacked.mean(dim=0)

        group_dir = (
            figures_root
            / f"dataset={dataset_name}"
            / f"arch={hidden_layers}x{hidden_size}"
            / f"act={activation}"
            / f"opt={optimizer}"
        )
        reg_dir = group_dir / "regularization"

        marker = reg_dir / "regularization_slice_wd0_surface.png"
        if marker.exists():
            continue

        reg_dir.mkdir(parents=True, exist_ok=True)
        title_prefix = (
            f"2D random slice train loss\n"
            f"dataset={dataset_name}, arch={hidden_layers}x{hidden_size}, "
            f"act={activation}, opt={optimizer}"
        )
        save_regularization_random_slice_surfaces(
            alpha_grid=alpha_grid_ref,
            beta_grid=beta_grid_ref,
            weight_decays=weight_decays,
            loss_volume=loss_volume,
            output_dir=reg_dir,
            title_prefix=title_prefix,
            filename_prefix="regularization_slice",
        )


def _run_connectivity_and_pca_for_group(
    group_key: Tuple[str, int, int, str, str],
    runs: Sequence[RunConfig],
    figures_root: Path,
    experiments_root: Path,
    device: torch.device,
    connectivity_points: int,
    pca_cfg: PCAConfig,
) -> None:
    """
    Run connectivity tests and PCA projections for a configuration group.

    Args:
        group_key (Tuple[str, int, int, str, str]): Group identifier.
        runs (Sequence[RunConfig]): Runs sharing the same configuration.
        figures_root (Path): Root directory for figures.
        experiments_root (Path): Root directory for experiments (unused, kept for parity).
        device (torch.device): Computation device.
        connectivity_points (int): Number of points for linear connectivity paths.
        pca_cfg (PCAConfig): PCA grid configuration.

    Returns:
        None
    """
    _ = experiments_root

    if not runs:
        return

    dataset_name, hidden_layers, hidden_size, activation, optimizer = group_key

    # Regenerate dataset once per group.
    dataset_cfg = runs[0].dataset
    x_train, y_train, x_test, y_test = _generate_dataset(dataset_cfg)
    train_loader, test_loader = _make_dataloaders(
        x_train, y_train, x_test, y_test, batch_size=runs[0].training.batch_size
    )

    group_dir = (
        figures_root
        / f"dataset={dataset_name}"
        / f"arch={hidden_layers}x{hidden_size}"
        / f"act={activation}"
        / f"opt={optimizer}"
    )

    # Connectivity tests among seeds.
    connectivity_dir = group_dir / "connectivity"
    num_runs = len(runs)
    for i in range(num_runs):
        for j in range(i + 1, num_runs):
            run_i = runs[i]
            run_j = runs[j]

            ckpt_i = run_i.training.checkpoint_dir / f"final_epoch{run_i.training.epochs}.pt"
            ckpt_j = run_j.training.checkpoint_dir / f"final_epoch{run_j.training.epochs}.pt"

            model_i = _load_model_from_checkpoint(run_i.model, ckpt_i, device)
            model_j = _load_model_from_checkpoint(run_j.model, ckpt_j, device)

            # Apply simple neuron permutation alignment for 1-hidden-layer models.
            if run_i.model.hidden_layers == 1:
                perm = compute_neuron_permutation_1layer(model_i, model_j)
                model_j = apply_neuron_permutation_1layer(model_j, perm)

            conn_results = linear_connectivity_barrier(
                model_a=model_i,
                model_b=model_j,
                train_loader=train_loader,
                test_loader=test_loader,
                num_points=connectivity_points,
                device=device,
                normalize_per_layer=True,
            )

            pair_dir = connectivity_dir / f"seed={run_i.seed}_to_seed={run_j.seed}"
            save_connectivity_plots(conn_results, output_dir=pair_dir, prefix="linear")

            # Save a JSON summary of barrier heights and path-based flatness metrics.
            train_losses_path = conn_results["train_loss"]
            test_losses_path = conn_results["test_loss"]
            barriers = {
                "train_barrier": float(conn_results["train_barrier"]),
                "test_barrier": float(conn_results["test_barrier"]),
                "train_min": float(train_losses_path.min().item()),
                "train_max": float(train_losses_path.max().item()),
                "train_mean": float(train_losses_path.mean().item()),
                "test_min": float(test_losses_path.min().item()),
                "test_max": float(test_losses_path.max().item()),
                "test_mean": float(test_losses_path.mean().item()),
            }
            pair_dir.mkdir(parents=True, exist_ok=True)
            with (pair_dir / "barriers.json").open("w", encoding="utf-8") as f_bar:
                json.dump(barriers, f_bar, indent=2)

    # PCA trajectory across checkpoints for this configuration.
    # Collect initial and final checkpoints from all runs.
    checkpoint_paths: List[Path] = []
    for r in runs:
        checkpoint_paths.append(r.training.checkpoint_dir / "init_epoch0.pt")
        checkpoint_paths.append(r.training.checkpoint_dir / f"final_epoch{r.training.epochs}.pt")

    base_model = build_mlp_from_config(runs[0].model).to(device)
    weight_matrix = collect_parameter_trajectory(
        model=base_model,
        checkpoint_paths=checkpoint_paths,
        device=device,
    )

    mean_vec, components, _ = compute_pca_components(weight_matrix, num_components=pca_cfg.num_components)
    trajectory_coords = project_trajectory_to_pca(weight_matrix, mean_vec, components[:, :2])

    # Choose the first final checkpoint as reference for PCA loss grid.
    reference_ckpt = runs[0].training.checkpoint_dir / f"final_epoch{runs[0].training.epochs}.pt"
    base_model = _load_model_from_checkpoint(runs[0].model, reference_ckpt, device)

    from project.landscape.common import clone_parameters  # Local import to avoid circularity at module level.

    base_params = clone_parameters(base_model)

    loss_grid = loss_over_pca_grid(
        model=base_model,
        base_params=base_params,
        mean_vector=mean_vec,
        components=components[:, :2],
        train_loader=train_loader,
        test_loader=test_loader,
        config=pca_cfg,
        device=device,
    )

    pca_dir = group_dir / "pca"
    save_pca_plots(trajectory_coords, loss_grid, output_dir=pca_dir, prefix="pca")


def run_all_probes(args: argparse.Namespace) -> None:
    """
    Execute all probes and report generation based on CLI arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    experiments_root = Path(args.experiments_root).resolve()
    figures_root = Path(args.figures_root).resolve()
    reports_root = Path(args.reports_root).resolve()

    device = torch.device(args.device)

    run_configs = _load_run_configs(experiments_root)
    if not run_configs:
        logger.warning("No run configurations found under %s", experiments_root)
        return

    logger.info("Discovered %d runs; launching landscape probes.", len(run_configs))

    hessian_cfg = HessianConfig(
        top_k=args.hessian_top_k,
        num_trace_samples=args.hessian_trace_samples,
        num_power_iterations=args.hessian_power_iterations,
    )
    sharpness_cfg = SharpnessConfig(
        epsilon=args.sharpness_epsilon,
        num_samples=args.sharpness_samples,
        normalize_per_layer=True,
    )
    pca_cfg = PCAConfig(
        num_components=2,
        grid_size=args.pca_grid_size,
        grid_radius=args.pca_grid_radius,
    )

    # Per-run probes with progress bar and idempotent skipping.
    per_run_progress = tqdm(
        total=len(run_configs),
        desc="Per-run probes",
        unit="run",
    )
    for run in run_configs:
        # Use existence of Hessian spectrum JSON as a completion marker.
        hessian_dir = (
            figures_root
            / f"dataset={run.dataset.name}"
            / f"arch={run.model.hidden_layers}x{run.model.hidden_size}"
            / f"act={run.model.activation}"
            / f"opt={run.training.optimizer}"
            / f"seed={run.seed}"
            / "hessian"
        )
        hessian_json = hessian_dir / "spectrum.json"
        if hessian_json.exists():
            per_run_progress.update(1)
            continue

        _run_single_run_probes(
            run=run,
            figures_root=figures_root,
            device=device,
            interp_points=args.interp_points,
            slice_1d_points=args.slice_1d_points,
            slice_1d_radius=args.slice_1d_radius,
            slice_2d_points=args.slice_2d_points,
            slice_2d_radius=args.slice_2d_radius,
            hessian_cfg=hessian_cfg,
            sharpness_cfg=sharpness_cfg,
            pca_cfg=pca_cfg,
        )
        per_run_progress.update(1)
    per_run_progress.close()

    # Group-level connectivity and PCA with progress bar and idempotent skipping.
    groups = _group_runs_by_configuration(run_configs)
    group_keys = list(groups.keys())
    group_progress = tqdm(
        total=len(group_keys),
        desc="Group-level connectivity/PCA",
        unit="config",
    )
    for key in group_keys:
        runs = groups[key]
        dataset_name, hidden_layers, hidden_size, activation, optimizer = key
        group_dir = (
            figures_root
            / f"dataset={dataset_name}"
            / f"arch={hidden_layers}x{hidden_size}"
            / f"act={activation}"
            / f"opt={optimizer}"
        )
        pca_surface = group_dir / "pca" / "pca_surface.png"
        if pca_surface.exists():
            group_progress.update(1)
            continue

        _run_connectivity_and_pca_for_group(
            group_key=key,
            runs=runs,
            figures_root=figures_root,
            experiments_root=experiments_root,
            device=device,
            connectivity_points=args.connectivity_points,
            pca_cfg=pca_cfg,
        )
        group_progress.update(1)
    group_progress.close()

    # Regularization study: interpolation surfaces vs weight decay.
    _run_regularization_interpolation_surfaces(
        runs=run_configs,
        figures_root=figures_root,
        device=device,
        num_points=args.interp_points,
    )

    _run_regularization_random_slice_surfaces(
        runs=run_configs,
        figures_root=figures_root,
        device=device,
        num_points=args.slice_2d_points,
        radius=args.slice_2d_radius,
    )

    # Markdown reports.
    generate_study_reports(experiments_root=experiments_root, reports_root=reports_root)

    logger.info("All probes and report generation completed.")


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the probe-and-report script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run all loss landscape probes and generate Markdown reports.",
    )
    parser.add_argument(
        "--experiments-root",
        type=str,
        default="reports/experiments",
        help="Root directory containing training experiment outputs.",
    )
    parser.add_argument(
        "--figures-root",
        type=str,
        default="reports/figures",
        help="Root directory where figures will be stored.",
    )
    parser.add_argument(
        "--reports-root",
        type=str,
        default="reports",
        help="Root directory for Markdown reports.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device (e.g., 'cpu' or 'cuda').",
    )
    parser.add_argument(
        "--interp-points",
        type=int,
        default=50,
        help="Number of points for interpolation curves.",
    )
    parser.add_argument(
        "--slice-1d-points",
        type=int,
        default=41,
        help="Number of points for 1D random direction slices.",
    )
    parser.add_argument(
        "--slice-1d-radius",
        type=float,
        default=0.5,
        help="Radius for 1D random direction slices.",
    )
    parser.add_argument(
        "--slice-2d-points",
        type=int,
        default=41,
        help="Number of grid points per axis for 2D random direction slices.",
    )
    parser.add_argument(
        "--slice-2d-radius",
        type=float,
        default=0.5,
        help="Radius for 2D random direction slices.",
    )
    parser.add_argument(
        "--hessian-top-k",
        type=int,
        default=10,
        help="Number of top Hessian eigenvalues to estimate.",
    )
    parser.add_argument(
        "--hessian-trace-samples",
        type=int,
        default=10,
        help="Number of Hutchinson samples for Hessian trace estimation.",
    )
    parser.add_argument(
        "--hessian-power-iterations",
        type=int,
        default=20,
        help="Number of power iterations per eigenvalue.",
    )
    parser.add_argument(
        "--sharpness-epsilon",
        type=float,
        default=1e-2,
        help="Radius epsilon for sharpness perturbations.",
    )
    parser.add_argument(
        "--sharpness-samples",
        type=int,
        default=32,
        help="Number of random perturbations for sharpness estimation.",
    )
    parser.add_argument(
        "--connectivity-points",
        type=int,
        default=41,
        help="Number of points along linear connectivity paths.",
    )
    parser.add_argument(
        "--pca-grid-size",
        type=int,
        default=25,
        help="Grid size for PCA-plane loss evaluation.",
    )
    parser.add_argument(
        "--pca-grid-radius",
        type=float,
        default=2.0,
        help="Radius in PCA coefficient space for the loss grid.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point for CLI execution of the probe-and-report pipeline.

    Returns:
        None
    """
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    run_all_probes(args)


if __name__ == "__main__":
    main()
