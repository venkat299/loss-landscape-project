"""
Example experiment: apply loss landscape probes to CNN and residual MLP architectures.

This script demonstrates that the existing training and probing
pipeline applies to non-MLP architectures (ConvNet and ResidualMLP).
It uses the moons dataset as a simple testbed and also includes a
higher-dimensional synthetic Gaussian mixture to probe more complex
loss surfaces.

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

from project.data import generate_moons_dataset, normalize_splits
from project.experiments import evaluate_model, train_model
from project.landscape.interpolation import linear_interpolation_curve
from project.landscape.random_slice import random_1d_loss_slice, random_2d_loss_surface
from project.landscape.visualizations import save_interpolation_plots, save_random_slice_plots
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
    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def _generate_highdim_gaussian_mixture(
    n_train: int,
    n_test: int,
    input_dim: int,
    n_classes: int,
    seed: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Generate a higher-dimensional Gaussian mixture dataset.

    Args:
        n_train (int): Number of training samples.
        n_test (int): Number of test samples.
        input_dim (int): Feature dimensionality.
        n_classes (int): Number of classes.
        seed (int): Random seed.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: Normalized train/test splits.
    """
    set_global_seed(seed)

    total = n_train + n_test
    n_per_class = total // n_classes

    data_list = []
    labels_list = []
    for k in range(n_classes):
        mean = torch.zeros(input_dim)
        mean[0] = 3.0 * (k - (n_classes - 1) / 2.0)
        cov_scale = 0.5
        samples = torch.randn(n_per_class, input_dim) * cov_scale + mean
        data_list.append(samples)
        labels_list.append(torch.full((n_per_class,), k, dtype=torch.int64))

    data = torch.cat(data_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    if data.size(0) < total:
        deficit = total - data.size(0)
        extra = torch.randn(deficit, input_dim) * 0.5 + data_list[0][0]
        data = torch.cat([data, extra], dim=0)
        labels = torch.cat([labels, torch.zeros(deficit, dtype=torch.int64)], dim=0)

    indices = torch.randperm(total)
    data = data[indices]
    labels = labels[indices]

    x_train = data[:n_train]
    y_train = labels[:n_train]
    x_test = data[n_train:total]
    y_test = labels[n_train:total]

    x_train_norm, x_test_norm, _, _ = normalize_splits(x_train, x_test)
    return x_train_norm, y_train, x_test_norm, y_test


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(0)

    # Dataset: moons (2D)
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

    figures_root = Path("reports/figures/experiments_cnn_resnet")

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

    # Save CNN interpolation plots.
    cnn_interp_dir = figures_root / "cnn" / "interpolation"
    save_interpolation_plots(
        results=cnn_interp,
        output_dir=cnn_interp_dir,
        prefix="cnn_interp",
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

    # Run random slices around the residual MLP solution.
    resnet_train_loader, resnet_test_loader = _make_loaders_from_tensors(
        x_train, y_train, x_test, y_test, batch_size=256
    )
    slice_cfg_1d = SliceConfig(num_points=41, radius=0.5)
    slice_cfg_2d = SliceConfig(num_points=41, radius=0.5)

    slice_1d = random_1d_loss_slice(
        model=resnet_model,
        train_loader=resnet_train_loader,
        test_loader=resnet_test_loader,
        config=slice_cfg_1d,
        device=device,
        seed=0,
    )
    slice_2d = random_2d_loss_surface(
        model=resnet_model,
        train_loader=resnet_train_loader,
        test_loader=resnet_test_loader,
        config=slice_cfg_2d,
        device=device,
        seed=1,
    )

    resnet_slice_dir = figures_root / "resnet" / "random_slice"
    save_random_slice_plots(
        slice_1d=slice_1d,
        slice_2d=slice_2d,
        output_dir=resnet_slice_dir,
        prefix="resnet_random",
    )

    logger.info(
        "Residual MLP 1D slice: min_train_loss=%.4f, max_train_loss=%.4f",
        float(slice_1d["train_loss"].min().item()),
        float(slice_1d["train_loss"].max().item()),
    )
    logger.info("CNN and residual MLP example probes on moons completed.")

    # High-dimensional synthetic Gaussian mixture (e.g. 16x16 images).
    highdim_input_dim = 16 * 16
    highdim_n_classes = 3
    x_train_hd, y_train_hd, x_test_hd, y_test_hd = _generate_highdim_gaussian_mixture(
        n_train=2048,
        n_test=512,
        input_dim=highdim_input_dim,
        n_classes=highdim_n_classes,
        seed=1,
    )

    def _to_highdim_image(x: Tensor) -> Tensor:
        return x.view(x.size(0), 1, 16, 16)

    # 3) High-dimensional ConvNet (deeper/wider).
    hd_cnn_cfg = CNNConfig(
        in_channels=1,
        num_classes=highdim_n_classes,
        hidden_channels=16,
        num_blocks=3,
    )
    hd_cnn = build_cnn_from_config(hd_cnn_cfg)
    hd_cnn_train_cfg = TrainingConfig(
        optimizer="adam",
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.0,
        batch_size=128,
        epochs=20,
        lr_step_size=10,
        lr_gamma=0.1,
        device=str(device),
        seed=1,
        checkpoint_dir=base_checkpoint_dir / "highdim_cnn" / "checkpoints",
        log_interval=10,
    )
    _ = hd_cnn(_to_highdim_image(x_train_hd[:1]))
    hd_cnn, _ = train_model(
        model=hd_cnn,
        x_train=_to_highdim_image(x_train_hd),
        y_train=y_train_hd,
        x_test=_to_highdim_image(x_test_hd),
        y_test=y_test_hd,
        config=hd_cnn_train_cfg,
        checkpoint_epochs=[],
    )

    hd_cnn_train_loader, hd_cnn_test_loader = _make_loaders_from_tensors(
        _to_highdim_image(x_train_hd),
        y_train_hd,
        _to_highdim_image(x_test_hd),
        y_test_hd,
        batch_size=256,
    )

    init_hd_cnn = build_cnn_from_config(hd_cnn_cfg).to(device)
    _ = init_hd_cnn(_to_highdim_image(x_train_hd[:1].to(device)))
    init_hd_state = torch.load(
        hd_cnn_train_cfg.checkpoint_dir / "init_epoch0.pt",
        map_location=device,
        weights_only=False,
    )
    init_hd_cnn.load_state_dict(init_hd_state["model_state_dict"])

    final_hd_cnn = build_cnn_from_config(hd_cnn_cfg).to(device)
    _ = final_hd_cnn(_to_highdim_image(x_train_hd[:1].to(device)))
    final_hd_state = torch.load(
        hd_cnn_train_cfg.checkpoint_dir / "final_epoch20.pt",
        map_location=device,
        weights_only=False,
    )
    final_hd_cnn.load_state_dict(final_hd_state["model_state_dict"])

    hd_cnn_interp = linear_interpolation_curve(
        model_a=init_hd_cnn,
        model_b=final_hd_cnn,
        train_loader=hd_cnn_train_loader,
        test_loader=hd_cnn_test_loader,
        num_points=25,
        device=device,
        normalize_per_layer=True,
    )
    hd_cnn_interp_dir = figures_root / "highdim" / "cnn" / "interpolation"
    save_interpolation_plots(
        results=hd_cnn_interp,
        output_dir=hd_cnn_interp_dir,
        prefix="cnn_highdim_interp",
    )
    logger.info(
        "High-dim CNN interpolation: train_loss[0]=%.4f, train_loss[-1]=%.4f",
        float(hd_cnn_interp["train_loss"][0].item()),
        float(hd_cnn_interp["train_loss"][-1].item()),
    )

    # 4) High-dimensional residual MLP (deeper/wider).
    hd_resnet_cfg = ResidualMLPConfig(
        input_dim=highdim_input_dim,
        hidden_dim=256,
        num_blocks=6,
        num_classes=highdim_n_classes,
    )
    hd_resnet = build_resnet_from_config(hd_resnet_cfg)
    hd_resnet_train_cfg = TrainingConfig(
        optimizer="adam",
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.0,
        batch_size=128,
        epochs=20,
        lr_step_size=10,
        lr_gamma=0.1,
        device=str(device),
        seed=2,
        checkpoint_dir=base_checkpoint_dir / "highdim_resnet" / "checkpoints",
        log_interval=10,
    )

    hd_resnet, _ = train_model(
        model=hd_resnet,
        x_train=x_train_hd,
        y_train=y_train_hd,
        x_test=x_test_hd,
        y_test=y_test_hd,
        config=hd_resnet_train_cfg,
        checkpoint_epochs=[],
    )

    hd_resnet_train_loader, hd_resnet_test_loader = _make_loaders_from_tensors(
        x_train_hd,
        y_train_hd,
        x_test_hd,
        y_test_hd,
        batch_size=256,
    )

    hd_slice_cfg_1d = SliceConfig(num_points=41, radius=0.5)
    hd_slice_cfg_2d = SliceConfig(num_points=41, radius=0.5)
    hd_slice_1d = random_1d_loss_slice(
        model=hd_resnet,
        train_loader=hd_resnet_train_loader,
        test_loader=hd_resnet_test_loader,
        config=hd_slice_cfg_1d,
        device=device,
        seed=3,
    )
    hd_slice_2d = random_2d_loss_surface(
        model=hd_resnet,
        train_loader=hd_resnet_train_loader,
        test_loader=hd_resnet_test_loader,
        config=hd_slice_cfg_2d,
        device=device,
        seed=4,
    )

    hd_resnet_slice_dir = figures_root / "highdim" / "resnet" / "random_slice"
    save_random_slice_plots(
        slice_1d=hd_slice_1d,
        slice_2d=hd_slice_2d,
        output_dir=hd_resnet_slice_dir,
        prefix="resnet_highdim_random",
    )
    logger.info(
        "High-dim residual MLP 1D slice: min_train_loss=%.4f, max_train_loss=%.4f",
        float(hd_slice_1d["train_loss"].min().item()),
        float(hd_slice_1d["train_loss"].max().item()),
    )

    logger.info("High-dimensional CNN and residual MLP probes completed.")


if __name__ == "__main__":
    main()
