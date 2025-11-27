"""
Example experiment: apply loss landscape probes to CNN and residual MLP architectures.

This script demonstrates that the existing training and probing
pipeline applies to non-MLP architectures (ConvNet and ResidualMLP).
It uses the moons dataset as a simple testbed.

Usage:
    uv run python -m project.experiments.run_cnn_resnet_example
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from project.data import generate_moons_dataset
from project.experiments import evaluate_model, train_model
from project.landscape.interpolation import linear_interpolation_curve
from project.landscape.random_slice import random_1d_loss_slice
from project.models.cnn import CNNConfig, build_cnn_from_config
from project.models.resnet import ResidualMLPConfig, build_resnet_from_config
from project.utils.configs import DatasetConfig, SliceConfig, TrainingConfig
from project.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


def _make_loaders_from_tensors(
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(0)

    # Dataset: moons
    dataset_cfg = DatasetConfig(
        name="moons",
        n_train=1024,
        n_test=256,
        noise=0.1,
        n_classes=2,
        seed=0,
    )
    x_train, y_train, x_test, y_test = generate_moons_dataset(dataset_cfg)

    # For the CNN, reshape 2D points into a 1x2x1 "image" tensor so that
    # 2x2 pooling is well-defined.
    def _to_cnn_input(x: Tensor) -> Tensor:
        return x.view(x.size(0), 1, 2, 1)

    # Training config (shared).
    base_checkpoint_dir = Path("reports/experiments_cnn_resnet")
    base_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = TrainingConfig(
        optimizer="adam",
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.0,
        batch_size=128,
        epochs=20,
        lr_step_size=10,
        lr_gamma=0.1,
        device=str(device),
        seed=0,
        checkpoint_dir=base_checkpoint_dir / "cnn" / "checkpoints",
        log_interval=10,
    )

    # 1) Train ConvNet
    cnn_config = CNNConfig(in_channels=1, num_classes=2, hidden_channels=8, num_blocks=1)
    cnn_model = build_cnn_from_config(cnn_config)
    train_cfg.checkpoint_dir = base_checkpoint_dir / "cnn" / "checkpoints"

    # Ensure classifier is initialized by running a single forward pass
    # before saving the initial checkpoint inside the training loop.
    _ = cnn_model(_to_cnn_input(x_train[:1]))

    cnn_model, _ = train_model(
        model=cnn_model,
        x_train=_to_cnn_input(x_train),
        y_train=y_train,
        x_test=_to_cnn_input(x_test),
        y_test=y_test,
        config=train_cfg,
        checkpoint_epochs=[],
    )

    # Simple interpolation sanity check between initial and final CNN weights
    cnn_train_loader, cnn_test_loader = _make_loaders_from_tensors(
        _to_cnn_input(x_train), y_train, _to_cnn_input(x_test), y_test, batch_size=256
    )

    # Load initial and final CNN states into separate model instances to ensure
    # that parameter lists match and include the classifier.
    init_cnn = build_cnn_from_config(cnn_config).to(device)
    _ = init_cnn(_to_cnn_input(x_train[:1].to(device)))
    init_state = torch.load(
        train_cfg.checkpoint_dir / "init_epoch0.pt",
        map_location=device,
        weights_only=False,
    )
    init_cnn.load_state_dict(init_state["model_state_dict"])

    final_cnn = build_cnn_from_config(cnn_config).to(device)
    _ = final_cnn(_to_cnn_input(x_train[:1].to(device)))
    final_state = torch.load(
        train_cfg.checkpoint_dir / "final_epoch20.pt",
        map_location=device,
        weights_only=False,
    )
    final_cnn.load_state_dict(final_state["model_state_dict"])

    cnn_interp = linear_interpolation_curve(
        model_a=init_cnn,
        model_b=final_cnn,
        train_loader=cnn_train_loader,
        test_loader=cnn_test_loader,
        num_points=25,
        device=device,
        normalize_per_layer=True,
    )
    logger.info(
        "CNN interpolation: train_loss[0]=%.4f, train_loss[-1]=%.4f",
        float(cnn_interp["train_loss"][0].item()),
        float(cnn_interp["train_loss"][-1].item()),
    )

    # 2) Train Residual MLP
    resnet_cfg = ResidualMLPConfig(input_dim=2, hidden_dim=64, num_blocks=3, num_classes=2)
    resnet_model = build_resnet_from_config(resnet_cfg)
    train_cfg.checkpoint_dir = base_checkpoint_dir / "resnet" / "checkpoints"

    resnet_model, _ = train_model(
        model=resnet_model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        config=train_cfg,
        checkpoint_epochs=[],
    )

    # Run a 1D random slice around the residual MLP solution.
    resnet_train_loader, resnet_test_loader = _make_loaders_from_tensors(
        x_train, y_train, x_test, y_test, batch_size=256
    )
    slice_cfg = SliceConfig(num_points=41, radius=0.5)
    slice_result = random_1d_loss_slice(
        model=resnet_model,
        train_loader=resnet_train_loader,
        test_loader=resnet_test_loader,
        config=slice_cfg,
        device=device,
        seed=0,
    )
    logger.info(
        "Residual MLP 1D slice: min_train_loss=%.4f, max_train_loss=%.4f",
        float(slice_result["train_loss"].min().item()),
        float(slice_result["train_loss"].max().item()),
    )

    logger.info("CNN and residual MLP example probes completed.")


if __name__ == "__main__":
    main()
