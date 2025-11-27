"""
Mode connectivity utilities for neural network loss landscapes.

This module provides:
    - linear interpolation barrier height between two modes
    - simple neuron permutation alignment for 1-hidden-layer MLPs
    - optional quadratic Bézier path evaluation given a control model
"""

from __future__ import annotations

import copy
from typing import Dict, List, Literal, Tuple

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


def _compute_endpoint_barrier(loss_curve: Tensor) -> float:
    """
    Compute the linear barrier height from a loss curve.

    Barrier height is defined as:

        max_alpha L(alpha) − max(L(0), L(1))

    where ``alpha`` indexes interpolation between the endpoints.

    Args:
        loss_curve (Tensor): 1D tensor of loss values along a path.

    Returns:
        float: Non-negative barrier height.
    """
    if loss_curve.numel() == 0:
        return 0.0
    max_along_path = float(loss_curve.max().item())
    end_max = float(torch.max(loss_curve[0], loss_curve[-1]).item())
    barrier = max(0.0, max_along_path - end_max)
    return barrier


def _evaluate_losses(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate training and test loss for a model.

    Args:
        model (nn.Module): Model to evaluate.
        train_loader (DataLoader): Training dataset loader.
        test_loader (DataLoader): Test dataset loader.
        device (torch.device): Computation device.

    Returns:
        Tuple[float, float]: Training loss and test loss.
    """
    criterion = nn.CrossEntropyLoss()
    train_loss, _ = evaluate_model(model, train_loader, criterion, device)
    test_loss, _ = evaluate_model(model, test_loader, criterion, device)
    return train_loss, test_loss


def _build_direction(
    params_a: ParameterList,
    params_b: ParameterList,
    normalize_per_layer: bool,
) -> ParameterList:
    """
    Construct a direction vector between two parameter configurations.

    Args:
        params_a (ParameterList): Starting parameters.
        params_b (ParameterList): Target parameters.
        normalize_per_layer (bool): Whether to apply per-layer normalization.

    Returns:
        ParameterList: Direction tensors.
    """
    delta = parameter_difference(params_b, params_a)
    if normalize_per_layer:
        return normalize_direction_per_layer(params_a, delta)
    return delta


def linear_connectivity_barrier(
    model_a: nn.Module,
    model_b: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_points: int,
    device: torch.device,
    normalize_per_layer: bool = True,
) -> Dict[str, Tensor | float]:
    """
    Compute linear mode connectivity metrics between two models.

    Args:
        model_a (nn.Module): Starting model.
        model_b (nn.Module): Target model.
        train_loader (DataLoader): Training dataset loader.
        test_loader (DataLoader): Test dataset loader.
        num_points (int): Number of interpolation points α ∈ [0, 1].
        device (torch.device): Computation device.
        normalize_per_layer (bool): Whether to use per-layer normalized
            directions when interpolating in parameter space.

    Returns:
        Dict[str, Tensor | float]: Dictionary containing:
            - ``"alphas"`` (Tensor): interpolation coefficients
            - ``"train_loss"`` (Tensor): train loss along path
            - ``"test_loss"`` (Tensor): test loss along path
            - ``"train_barrier"`` (float): train loss barrier height
            - ``"test_barrier"`` (float): test loss barrier height
    """
    working_model = copy.deepcopy(model_a).to(device)
    model_b = model_b.to(device)

    params_a = clone_parameters(working_model)
    params_b = clone_parameters(model_b)
    direction = _build_direction(params_a, params_b, normalize_per_layer=normalize_per_layer)

    alphas = torch.linspace(0.0, 1.0, steps=num_points, device=device)
    train_losses: List[float] = []
    test_losses: List[float] = []

    for alpha in alphas:
        alpha_scalar = float(alpha.item())
        params_alpha = add_scaled_direction(params_a, direction, alpha_scalar)
        assign_parameters(working_model, params_alpha)

        train_loss, test_loss = _evaluate_losses(
            working_model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    train_loss_tensor = torch.tensor(train_losses)
    test_loss_tensor = torch.tensor(test_losses)

    train_barrier = _compute_endpoint_barrier(train_loss_tensor)
    test_barrier = _compute_endpoint_barrier(test_loss_tensor)

    return {
        "alphas": alphas.detach().cpu(),
        "train_loss": train_loss_tensor,
        "test_loss": test_loss_tensor,
        "train_barrier": train_barrier,
        "test_barrier": test_barrier,
    }


def _get_single_hidden_layer_weights(
    model: nn.Module,
) -> Tuple[nn.Linear, nn.Linear]:
    """
    Extract first hidden and output linear layers for a 1-hidden-layer MLP.

    Args:
        model (nn.Module): Model expected to follow the 1-hidden-layer MLP
            structure defined in this project.

    Returns:
        Tuple[nn.Linear, nn.Linear]: Input-to-hidden and hidden-to-output layers.
    """
    linear_layers: List[nn.Linear] = [
        module for module in model.modules() if isinstance(module, nn.Linear)
    ]
    if len(linear_layers) != 2:
        raise ValueError("Permutation alignment is only supported for 1-hidden-layer MLPs.")
    return linear_layers[0], linear_layers[1]


def compute_neuron_permutation_1layer(
    model_a: nn.Module,
    model_b: nn.Module,
    distance: Literal["l2"] = "l2",
) -> Tensor:
    """
    Compute a simple hidden-unit permutation aligning two 1-layer MLPs.

    A greedy matching is used based on the L2 distance between concatenated
    incoming and outgoing weight vectors for each hidden unit.

    Args:
        model_a (nn.Module): Reference model.
        model_b (nn.Module): Model to be permuted.
        distance (Literal["l2"]): Distance metric identifier (currently only
            ``"l2"`` is supported).

    Returns:
        Tensor: Long tensor of shape (H,) where ``perm[i]`` gives the index
        in ``model_b`` corresponding to hidden unit ``i`` in ``model_a``.
    """
    if distance != "l2":
        raise ValueError(f"Unsupported distance metric: {distance}")

    in_a, out_a = _get_single_hidden_layer_weights(model_a)
    in_b, out_b = _get_single_hidden_layer_weights(model_b)

    hidden_a = in_a.weight.shape[0]
    hidden_b = in_b.weight.shape[0]
    if hidden_a != hidden_b:
        raise ValueError("Hidden sizes of the two models must match for permutation alignment.")

    h = hidden_a

    def _hidden_representation(
        in_layer: nn.Linear,
        out_layer: nn.Linear,
    ) -> Tensor:
        incoming = in_layer.weight  # (H, in_features)
        outgoing = out_layer.weight.transpose(0, 1)  # (H, out_features)
        return torch.cat([incoming, outgoing], dim=1)  # (H, in+out)

    rep_a = _hidden_representation(in_a, out_a)  # (H, F)
    rep_b = _hidden_representation(in_b, out_b)  # (H, F)

    # Compute pairwise squared distances.
    # cost[i, j] = ||rep_a[i] - rep_b[j]||^2
    rep_a_exp = rep_a.unsqueeze(1)  # (H, 1, F)
    rep_b_exp = rep_b.unsqueeze(0)  # (1, H, F)
    diff = rep_a_exp - rep_b_exp
    cost = torch.sum(diff * diff, dim=2)  # (H, H)

    perm = torch.full((h,), -1, dtype=torch.long)
    used_b = torch.zeros(h, dtype=torch.bool)

    for _ in range(h):
        masked_cost = cost.clone()
        masked_cost[perm != -1] = float("inf")
        masked_cost[:, used_b] = float("inf")

        index = torch.argmin(masked_cost)
        i = int(index // h)
        j = int(index % h)
        perm[i] = j
        used_b[j] = True

    if (perm < 0).any():
        raise RuntimeError("Failed to construct a full hidden-unit permutation.")

    return perm


def apply_neuron_permutation_1layer(
    model: nn.Module,
    permutation: Tensor,
) -> nn.Module:
    """
    Apply a hidden-unit permutation to a 1-hidden-layer MLP.

    Args:
        model (nn.Module): Model whose hidden units will be permuted.
        permutation (Tensor): Long tensor of shape (H,) specifying the
            new ordering of hidden units.

    Returns:
        nn.Module: New model instance with permuted hidden layer weights.
    """
    permuted_model = copy.deepcopy(model)
    in_layer, out_layer = _get_single_hidden_layer_weights(permuted_model)

    perm = permutation.to(in_layer.weight.device)

    # Permute rows of input-to-hidden weights and biases.
    in_layer.weight.data = in_layer.weight.data[perm, :]
    if in_layer.bias is not None:
        in_layer.bias.data = in_layer.bias.data[perm]

    # Permute columns of hidden-to-output weights.
    out_layer.weight.data = out_layer.weight.data[:, perm]

    return permuted_model


def quadratic_bezier_connectivity(
    model_a: nn.Module,
    model_b: nn.Module,
    control_model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_points: int,
    device: torch.device,
    normalize_per_layer: bool = False,
) -> Dict[str, Tensor]:
    """
    Evaluate loss along a quadratic Bézier path between two models.

    The path is defined in parameter space as:

        θ(t) = (1−t)^2 θ_A + 2t(1−t) θ_C + t^2 θ_B

    where θ_A and θ_B are endpoint weights and θ_C is a control point
    specified by ``control_model``.

    Args:
        model_a (nn.Module): Starting model.
        model_b (nn.Module): Ending model.
        control_model (nn.Module): Control-point model.
        train_loader (DataLoader): Training dataset loader.
        test_loader (DataLoader): Test dataset loader.
        num_points (int): Number of evaluation points t ∈ [0, 1].
        device (torch.device): Computation device.
        normalize_per_layer (bool): If True, normalize the vectors
            (θ_C − θ_A) and (θ_B − θ_C) per layer before constructing
            the Bézier path.

    Returns:
        Dict[str, Tensor]: Dictionary containing:
            - ``"ts"``: parameter values along the path (num_points,)
            - ``"train_loss"``: training loss along the path (num_points,)
            - ``"test_loss"``: test loss along the path (num_points,)
    """
    working_model = copy.deepcopy(model_a).to(device)
    model_b = model_b.to(device)
    control_model = control_model.to(device)

    params_a = clone_parameters(working_model)
    params_b = clone_parameters(model_b)
    params_c = clone_parameters(control_model)

    if normalize_per_layer:
        vec_ac = parameter_difference(params_c, params_a)
        vec_cb = parameter_difference(params_b, params_c)
        vec_ac = normalize_direction_per_layer(params_a, vec_ac)
        vec_cb = normalize_direction_per_layer(params_c, vec_cb)
        params_c = add_scaled_direction(params_a, vec_ac, 1.0)
        params_b = add_scaled_direction(params_c, vec_cb, 1.0)

    ts = torch.linspace(0.0, 1.0, steps=num_points, device=device)
    train_losses: List[float] = []
    test_losses: List[float] = []

    for t in ts:
        t_scalar = float(t.item())
        one_minus_t = 1.0 - t_scalar

        params_t: ParameterList = []
        for pa, pc, pb in zip(params_a, params_c, params_b):
            theta_t = (one_minus_t ** 2) * pa + 2.0 * t_scalar * one_minus_t * pc + (t_scalar ** 2) * pb
            params_t.append(theta_t)

        assign_parameters(working_model, params_t)

        train_loss, test_loss = _evaluate_losses(
            working_model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)

    return {
        "ts": ts.detach().cpu(),
        "train_loss": torch.tensor(train_losses),
        "test_loss": torch.tensor(test_losses),
    }

