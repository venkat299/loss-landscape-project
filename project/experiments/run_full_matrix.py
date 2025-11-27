"""
Run the full experiment matrix over architectures, activations, optimizers, and datasets.

This script:
    - generates synthetic datasets (moons by default, circles/XOR optional)
    - builds MLP models for the specified architecture variants
    - trains each configuration with SGD and Adam for multiple seeds
    - saves checkpoints and per-epoch metrics for downstream landscape analysis

Usage (example):
    uv run python -m project.experiments.run_full_matrix --output-root reports/experiments
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from project.data import (
    generate_circles_dataset,
    generate_moons_dataset,
    generate_xor_dataset,
)
from project.experiments import train_model
from project.models import MLPClassifier, build_mlp_from_config, get_predefined_model_config
from project.utils.configs import DatasetConfig, ExperimentConfig, TrainingConfig

logger = logging.getLogger(__name__)


def _build_dataloaders(
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
        batch_size (int): Mini-batch size.

    Returns:
        Tuple[DataLoader, DataLoader]: Train and test DataLoaders.
    """
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def _generate_dataset(config: DatasetConfig) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Generate a synthetic dataset based on the DatasetConfig.

    Args:
        config (DatasetConfig): Dataset configuration.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]:
            Normalized training and test splits.
    """
    if config.name == "moons":
        return generate_moons_dataset(config)
    if config.name == "circles":
        return generate_circles_dataset(config)
    if config.name == "xor":
        return generate_xor_dataset(config)
    raise ValueError(f"Unsupported dataset type for matrix experiment: {config.name}")


def _build_experiment_configs(
    architectures: Sequence[str],
    activations: Sequence[str],
    optimizers: Sequence[str],
    datasets: Sequence[str],
    seeds: Sequence[int],
    n_train: int,
    n_test: int,
    noise: float,
    device: str,
    base_lr: float,
    batch_size: int,
    epochs: int,
    lr_step_size: int,
    lr_gamma: float,
    momentum: float,
    weight_decay: float,
    output_root: Path,
    data_seed: int,
) -> List[ExperimentConfig]:
    """
    Construct a list of ExperimentConfig objects covering the requested matrix.

    Args:
        architectures (Sequence[str]): Architecture variant identifiers.
        activations (Sequence[str]): Activation function names.
        optimizers (Sequence[str]): Optimizer names.
        datasets (Sequence[str]): Dataset names.
        seeds (Sequence[int]): Training seeds to use for each combination.
        n_train (int): Number of training examples.
        n_test (int): Number of test examples.
        noise (float): Input noise level.
        device (str): Device identifier (e.g., "cpu" or "cuda").
        base_lr (float): Base learning rate.
        batch_size (int): Mini-batch size.
        epochs (int): Number of training epochs.
        lr_step_size (int): Learning rate step size.
        lr_gamma (float): Learning rate decay factor.
        momentum (float): Momentum for SGD.
        weight_decay (float): Weight decay coefficient.
        output_root (Path): Root directory for checkpoints and metrics.
        data_seed (int): Seed for dataset generation.

    Returns:
        List[ExperimentConfig]: List of experiment configurations.
    """
    configs: List[ExperimentConfig] = []
    input_dim = 2
    n_classes = 2

    for dataset_name in datasets:
        dataset_cfg = DatasetConfig(
            name=dataset_name,  # type: ignore[arg-type]
            n_train=n_train,
            n_test=n_test,
            noise=noise,
            n_classes=n_classes,
            seed=data_seed,
        )

        for arch in architectures:
            for activation in activations:
                model_cfg = get_predefined_model_config(
                    variant=arch,  # type: ignore[arg-type]
                    input_dim=input_dim,
                    output_dim=n_classes,
                    activation=activation,  # type: ignore[arg-type]
                )

                for optimizer_name in optimizers:
                    run_dir = (
                        output_root
                        / f"dataset={dataset_name}"
                        / f"arch={arch}"
                        / f"act={activation}"
                        / f"opt={optimizer_name}"
                    )

                    training_cfg = TrainingConfig(
                        optimizer=optimizer_name,  # type: ignore[arg-type]
                        learning_rate=base_lr,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        batch_size=batch_size,
                        epochs=epochs,
                        lr_step_size=lr_step_size,
                        lr_gamma=lr_gamma,
                        device=device,
                        seed=0,  # placeholder, overridden per seed
                        checkpoint_dir=run_dir / "checkpoints",
                        log_interval=10,
                    )

                    exp_cfg = ExperimentConfig(
                        dataset=dataset_cfg,
                        model=model_cfg,
                        training=training_cfg,
                        seeds=list(seeds),
                    )
                    configs.append(exp_cfg)

    return configs


def _save_metrics(
    metrics: List[Dict],
    output_path: Path,
) -> None:
    """
    Save per-epoch metrics as JSON.

    Args:
        metrics (List[Dict]): List of metric dictionaries.
        output_path (Path): File path to write JSON metrics to.

    Returns:
        None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def run_full_matrix(args: argparse.Namespace) -> None:
    """
    Execute the full experiment matrix based on CLI arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    architectures = [
        "medium",         # 2-layer, 100 units
        "deep-small",     # 4-layer, 100 units
        "shallow-wide",   # 1-layer, 500 units
        "shallow-small",  # 1-layer, 50 units
        "deep-large",     # 4-layer, 250 units
    ]
    activations = ["relu", "tanh", "gelu"]
    optimizers = ["sgd", "adam"]

    datasets: List[str] = ["moons"]
    if args.include_circles:
        datasets.append("circles")
    if args.include_xor:
        datasets.append("xor")

    experiment_configs = _build_experiment_configs(
        architectures=architectures,
        activations=activations,
        optimizers=optimizers,
        datasets=datasets,
        seeds=list(args.seeds),
        n_train=args.n_train,
        n_test=args.n_test,
        noise=args.noise,
        device=args.device,
        base_lr=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        output_root=output_root,
        data_seed=args.data_seed,
    )

    total_runs = len(experiment_configs) * len(args.seeds)
    progress = tqdm(total=total_runs, desc="Experiment matrix", unit="run")

    for exp_cfg in experiment_configs:
        x_train, y_train, x_test, y_test = _generate_dataset(exp_cfg.dataset)

        for seed in exp_cfg.seeds:
            training_cfg = exp_cfg.training
            training_cfg.seed = seed

            run_dir = Path(training_cfg.checkpoint_dir).parent / f"seed={seed}"
            training_cfg.checkpoint_dir = run_dir / "checkpoints"

            metrics_path = run_dir / "metrics.json"

            model: MLPClassifier = build_mlp_from_config(exp_cfg.model)

            trained_model, metrics = train_model(
                model=model,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                config=training_cfg,
                checkpoint_epochs=[],
            )
            _ = trained_model  # model is saved via checkpoints; not used further here.

            _save_metrics(metrics, metrics_path)

            # Prepare a JSON-serializable summary dictionary. In particular,
            # convert ``checkpoint_dir`` (a Path) to a string.
            dataset_dict = asdict(exp_cfg.dataset)
            model_dict = asdict(exp_cfg.model)
            training_dict = asdict(training_cfg)
            if "checkpoint_dir" in training_dict:
                training_dict["checkpoint_dir"] = str(training_cfg.checkpoint_dir)

            summary = {
                "dataset": dataset_dict,
                "model": model_dict,
                "training": training_dict,
                "seed": seed,
            }
            summary_path = run_dir / "summary.json"
            _save_metrics([summary], summary_path)

            progress.update(1)

    progress.close()
    logger.info("Completed %d runs in the experiment matrix.", total_runs)


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the experiment matrix script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run full loss landscape experiment matrix.")
    parser.add_argument(
        "--output-root",
        type=str,
        default="reports/experiments",
        help="Root directory for checkpoints and metrics.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device (e.g., 'cpu' or 'cuda').",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Training seeds to use for each configuration.",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=2048,
        help="Number of training samples per dataset.",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=512,
        help="Number of test samples per dataset.",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.1,
        help="Input noise level for synthetic datasets.",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=42,
        help="Seed for dataset generation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Base learning rate for optimizers.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini-batch size for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr-step-size",
        type=int,
        default=20,
        help="Epoch interval for learning-rate step decay.",
    )
    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=0.1,
        help="Multiplicative factor for learning-rate decay.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD optimizer.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="L2 weight decay coefficient.",
    )
    parser.add_argument(
        "--include-circles",
        action="store_true",
        help="Include the circles dataset in the experiment matrix.",
    )
    parser.add_argument(
        "--include-xor",
        action="store_true",
        help="Include the XOR-like dataset in the experiment matrix.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point for CLI execution of the full experiment matrix.

    Returns:
        None
    """
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    run_full_matrix(args)


if __name__ == "__main__":
    main()
