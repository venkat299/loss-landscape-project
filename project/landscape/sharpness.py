"""
Sharpness and flatness metrics for loss landscapes.

This module implements an ε-sharpness metric based on sampling random
perturbations within a ball around the current parameters, using
per-layer normalized perturbation directions.
"""

from __future__ import annotations

import copy
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from project.experiments import evaluate_model
from project.landscape.common import (
    ParameterList,
    add_scaled_direction,
    assign_parameters,
    clone_parameters,
    normalize_direction_per_layer,
    random_direction_like,
)
from project.utils.configs import SharpnessConfig
from project.utils.seed import set_global_seed


def _compute_loss(
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


def epsilon_sharpness(
    model: nn.Module,
    dataloader: DataLoader,
    config: SharpnessConfig,
    device: torch.device,
    seed: int,
) -> Tuple[float, Tensor]:
    """
    Estimate ε-sharpness via random perturbations around current parameters.

    The procedure:
        1. Compute baseline loss L(θ).
        2. For each sample, draw a random direction d, normalize it
           per layer, then scale by ε: Δθ = ε * d.
        3. Evaluate perturbed loss L(θ + Δθ).
        4. Record loss increases L(θ + Δθ) − L(θ).

    The ε-sharpness is defined as the maximum observed loss increase,
    and the full distribution of increases is returned for histogramming.

    Args:
        model (nn.Module): Model at parameters θ.
        dataloader (DataLoader): Dataset loader used to evaluate loss.
        config (SharpnessConfig): Sharpness configuration (ε, num_samples).
        device (torch.device): Computation device.
        seed (int): Random seed controlling perturbation sampling.

    Returns:
        Tuple[float, Tensor]:
            - ε-sharpness value (maximum loss increase),
            - 1D tensor of all loss increases for histogramming.
    """
    set_global_seed(seed)

    working_model = copy.deepcopy(model).to(device)
    base_params = clone_parameters(working_model)

    base_loss = _compute_loss(working_model, dataloader, device)

    increases = torch.zeros(config.num_samples)

    for idx in range(config.num_samples):
        direction = random_direction_like(working_model)
        if config.normalize_per_layer:
            direction = normalize_direction_per_layer(base_params, direction)

        scaled_params = add_scaled_direction(base_params, direction, config.epsilon)
        assign_parameters(working_model, scaled_params)

        perturbed_loss = _compute_loss(working_model, dataloader, device)
        increases[idx] = max(0.0, perturbed_loss - base_loss)

    sharpness_value = float(increases.max().item())
    return sharpness_value, increases

