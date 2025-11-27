"""
Linear interpolation between model parameter vectors.

This module implements loss evaluation along a straight-line path
between two trained models, optionally using per-layer normalized
directions as described in the project specification.
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
    normalize_direction_per_layer,
    parameter_difference,
)


def _evaluate_loss_and_accuracy(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """
    Evaluate training and test loss/accuracy for a model.

    Args:
        model (nn.Module): Model to evaluate.
        train_loader (DataLoader): DataLoader for the training set.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Computation device.

    Returns:
        Tuple[float, float, float, float]:
            Training loss, training accuracy, test loss, test accuracy.
    """
    criterion = nn.CrossEntropyLoss()
    train_loss, train_acc = evaluate_model(model, train_loader, criterion, device)
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    return train_loss, train_acc, test_loss, test_acc


def _build_direction(
    params_a: ParameterList,
    params_b: ParameterList,
    normalize_per_layer: bool,
) -> ParameterList:
    """
    Construct an interpolation direction between two parameter vectors.

    Args:
        params_a (ParameterList): Starting parameters.
        params_b (ParameterList): Target parameters.
        normalize_per_layer (bool): Whether to apply per-layer
            normalization to the direction.

    Returns:
        ParameterList: Direction tensors.
    """
    delta = parameter_difference(params_b, params_a)
    if normalize_per_layer:
        return normalize_direction_per_layer(params_a, delta)
    return delta


def linear_interpolation_curve(
    model_a: nn.Module,
    model_b: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_points: int,
    device: torch.device,
    normalize_per_layer: bool = True,
) -> Dict[str, Tensor]:
    """
    Evaluate loss along a linear interpolation between two models.

    Args:
        model_a (nn.Module): Starting model (θ_A).
        model_b (nn.Module): Target model (θ_B).
        train_loader (DataLoader): Training dataset loader.
        test_loader (DataLoader): Test dataset loader.
        num_points (int): Number of interpolation points α ∈ [0, 1].
        device (torch.device): Device for computation.
        normalize_per_layer (bool): If True, use per-layer normalized
            directions; otherwise perform simple linear interpolation.

    Returns:
        Dict[str, Tensor]: Dictionary containing:
            - ``"alphas"``: interpolation coefficients (num_points,)
            - ``"train_loss"``: training loss values (num_points,)
            - ``"test_loss"``: test loss values (num_points,)
            - ``"train_accuracy"``: training accuracies (num_points,)
            - ``"test_accuracy"``: test accuracies (num_points,)
    """
    working_model = copy.deepcopy(model_a).to(device)
    model_b = model_b.to(device)

    params_a = clone_parameters(working_model)
    params_b = clone_parameters(model_b)

    direction = _build_direction(params_a, params_b, normalize_per_layer=normalize_per_layer)

    alphas = torch.linspace(0.0, 1.0, steps=num_points, device=device)
    train_losses: List[float] = []
    test_losses: List[float] = []
    train_accs: List[float] = []
    test_accs: List[float] = []

    for alpha in alphas:
        alpha_scalar = float(alpha.item())
        params_alpha = add_scaled_direction(params_a, direction, alpha_scalar)
        assign_parameters(working_model, params_alpha)

        train_loss, train_acc, test_loss, test_acc = _evaluate_loss_and_accuracy(
            working_model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    return {
        "alphas": alphas.detach().cpu(),
        "train_loss": torch.tensor(train_losses),
        "test_loss": torch.tensor(test_losses),
        "train_accuracy": torch.tensor(train_accs),
        "test_accuracy": torch.tensor(test_accs),
    }

