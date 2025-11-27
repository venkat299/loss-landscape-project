"""
Random direction loss landscape slices.

This module implements 1D and 2D loss evaluations along random
directions in parameter space, with per-layer direction normalization.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from project.experiments import evaluate_model
from project.landscape.common import (
    ParameterList,
    add_scaled_direction,
    assign_parameters,
    clone_parameters,
    flatten_parameters,
    normalize_direction_per_layer,
    random_direction_like,
    unflatten_vector,
)
from project.utils.configs import SliceConfig
from project.utils.seed import set_global_seed


def _evaluate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """
    Compute average loss on a dataset.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): DataLoader for the dataset.
        device (torch.device): Computation device.

    Returns:
        float: Average loss value.
    """
    criterion = nn.CrossEntropyLoss()
    loss, _ = evaluate_model(model, dataloader, criterion, device)
    return loss


def random_1d_loss_slice(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: SliceConfig,
    device: torch.device,
    seed: int,
) -> Dict[str, Tensor]:
    """
    Compute a 1D random direction loss slice.

    Args:
        model (nn.Module): Base model at reference parameters θ.
        train_loader (DataLoader): Training dataset loader.
        test_loader (DataLoader): Test dataset loader.
        config (SliceConfig): Slice configuration (num_points, radius, etc.).
        device (torch.device): Computation device.
        seed (int): Random seed controlling direction sampling.

    Returns:
        Dict[str, Tensor]: Dictionary containing:
            - ``"alphas"``: step values in [−radius, radius] (num_points,)
            - ``"train_loss"``: training loss along the slice (num_points,)
            - ``"test_loss"``: test loss along the slice (num_points,)
    """
    set_global_seed(seed)

    working_model = copy.deepcopy(model).to(device)
    base_params = clone_parameters(working_model)

    direction = random_direction_like(working_model)
    if config.normalize_per_layer:
        direction = normalize_direction_per_layer(base_params, direction)

    alphas = torch.linspace(-config.radius, config.radius, steps=config.num_points)
    train_losses: List[float] = []
    test_losses: List[float] = []

    for alpha in alphas:
        alpha_scalar = float(alpha.item())
        params_alpha = add_scaled_direction(base_params, direction, alpha_scalar)
        assign_parameters(working_model, params_alpha)

        train_loss = _evaluate_loss(working_model, train_loader, device)
        test_loss = _evaluate_loss(working_model, test_loader, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

    return {
        "alphas": alphas,
        "train_loss": torch.tensor(train_losses),
        "test_loss": torch.tensor(test_losses),
    }


def random_2d_loss_surface(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: SliceConfig,
    device: torch.device,
    seed: int,
) -> Dict[str, Tensor]:
    """
    Compute a 2D random direction loss surface.

    Args:
        model (nn.Module): Base model at reference parameters θ.
        train_loader (DataLoader): Training dataset loader.
        test_loader (DataLoader): Test dataset loader.
        config (SliceConfig): Slice configuration (num_points, radius, etc.).
        device (torch.device): Computation device.
        seed (int): Random seed controlling direction sampling.

    Returns:
        Dict[str, Tensor]: Dictionary containing:
            - ``"alpha_grid"``: grid of α values (num_points, num_points)
            - ``"beta_grid"``: grid of β values (num_points, num_points)
            - ``"train_loss"``: training loss surface (num_points, num_points)
            - ``"test_loss"``: test loss surface (num_points, num_points)
    """
    set_global_seed(seed)

    working_model = copy.deepcopy(model).to(device)
    base_params = clone_parameters(working_model)

    # Sample two random directions.
    dir1 = random_direction_like(working_model)
    dir2 = random_direction_like(working_model)

    if config.normalize_per_layer:
        dir1 = normalize_direction_per_layer(base_params, dir1)
        dir2 = normalize_direction_per_layer(base_params, dir2)

    # Orthonormalize directions in flattened space (Gram–Schmidt).
    dir1_flat = flatten_parameters(dir1)
    dir2_flat = flatten_parameters(dir2)

    dir1_flat = dir1_flat / dir1_flat.norm()
    dir2_flat = dir2_flat - torch.dot(dir2_flat, dir1_flat) * dir1_flat
    dir2_flat = dir2_flat / dir2_flat.norm()

    dir1 = unflatten_vector(dir1_flat, base_params)
    dir2 = unflatten_vector(dir2_flat, base_params)

    alpha_vals = torch.linspace(-config.radius, config.radius, steps=config.num_points)
    beta_vals = torch.linspace(-config.radius, config.radius, steps=config.num_points)
    alpha_grid, beta_grid = torch.meshgrid(alpha_vals, beta_vals, indexing="ij")

    train_surface = torch.zeros_like(alpha_grid)
    test_surface = torch.zeros_like(beta_grid)

    for i in range(config.num_points):
        for j in range(config.num_points):
            alpha = float(alpha_grid[i, j].item())
            beta = float(beta_grid[i, j].item())

            params_alpha_beta: ParameterList = []
            for base, d1, d2 in zip(base_params, dir1, dir2):
                params_alpha_beta.append(base + alpha * d1 + beta * d2)

            assign_parameters(working_model, params_alpha_beta)

            train_loss = _evaluate_loss(working_model, train_loader, device)
            test_loss = _evaluate_loss(working_model, test_loader, device)

            train_surface[i, j] = train_loss
            test_surface[i, j] = test_loss

    return {
        "alpha_grid": alpha_grid,
        "beta_grid": beta_grid,
        "train_loss": train_surface,
        "test_loss": test_surface,
    }
