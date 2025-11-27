"""
Shared utilities for loss landscape analysis.

This module focuses on parameter-vector operations such as cloning
model parameters, assigning new parameter values, and constructing
and normalizing random directions in parameter space.
"""

from __future__ import annotations

from typing import List

import torch
from torch import Tensor, nn

ParameterList = List[Tensor]


def get_parameter_tensors(model: nn.Module) -> ParameterList:
    """
    Return a list of trainable parameter tensors for a model.

    Args:
        model (nn.Module): Model whose parameters will be accessed.

    Returns:
        ParameterList: List of tensors corresponding to trainable parameters.
    """
    return [p for p in model.parameters() if p.requires_grad]


def clone_parameters(model: nn.Module) -> ParameterList:
    """
    Clone all trainable parameters of a model.

    Args:
        model (nn.Module): Model whose parameters will be cloned.

    Returns:
        ParameterList: Detached, cloned parameter tensors.
    """
    return [p.detach().clone() for p in get_parameter_tensors(model)]


def assign_parameters(model: nn.Module, params: ParameterList) -> None:
    """
    Assign a new list of parameter tensors to a model in-place.

    Args:
        model (nn.Module): Model whose parameters will be updated.
        params (ParameterList): New parameter tensors with the same
            shapes and ordering as ``get_parameter_tensors(model)``.

    Returns:
        None
    """
    for current, new in zip(get_parameter_tensors(model), params):
        current.data.copy_(new)


def parameter_difference(params_b: ParameterList, params_a: ParameterList) -> ParameterList:
    """
    Compute the parameter-wise difference between two parameter lists.

    Args:
        params_b (ParameterList): Target parameters.
        params_a (ParameterList): Reference parameters.

    Returns:
        ParameterList: List of tensors representing ``params_b - params_a``.
    """
    if len(params_a) != len(params_b):
        raise ValueError("Parameter lists must have the same length.")
    return [pb - pa for pa, pb in zip(params_a, params_b)]


def add_scaled_direction(
    base_params: ParameterList,
    direction: ParameterList,
    alpha: float,
) -> ParameterList:
    """
    Add a scaled direction vector to a base parameter vector.

    Args:
        base_params (ParameterList): Base parameter tensors.
        direction (ParameterList): Direction tensors with the same shapes
            as ``base_params``.
        alpha (float): Scalar multiplier for the direction.

    Returns:
        ParameterList: New parameter tensors representing
        ``base_params + alpha * direction``.
    """
    if len(base_params) != len(direction):
        raise ValueError("Base parameters and direction must have the same length.")
    return [b + alpha * d for b, d in zip(base_params, direction)]


def random_direction_like(model: nn.Module) -> ParameterList:
    """
    Generate a random direction in parameter space with the same shapes
    as the model's trainable parameters.

    Args:
        model (nn.Module): Model providing the parameter shapes.

    Returns:
        ParameterList: Random tensors matching the shapes of the model
        parameters.
    """
    direction: ParameterList = []
    for param in get_parameter_tensors(model):
        direction.append(torch.randn_like(param))
    return direction


def normalize_direction_per_layer(
    reference_params: ParameterList,
    direction: ParameterList,
    eps: float = 1e-12,
) -> ParameterList:
    """
    Apply per-layer normalization to a direction vector.

    Each direction tensor is rescaled so that its norm matches the norm
    of the corresponding reference parameter tensor:

        direction[layer] *= (w.norm() / direction[layer].norm())

    Args:
        reference_params (ParameterList): Reference parameter tensors,
            typically the current model weights.
        direction (ParameterList): Direction tensors to be normalized.
        eps (float): Small constant to avoid division by zero.

    Returns:
        ParameterList: Normalized direction tensors.
    """
    if len(reference_params) != len(direction):
        raise ValueError("Reference parameters and direction must have the same length.")

    normalized: ParameterList = []
    for weight, d in zip(reference_params, direction):
        d_norm = d.norm()
        if d_norm < eps:
            normalized.append(torch.zeros_like(d))
            continue
        w_norm = weight.norm()
        scale = (w_norm / d_norm) if w_norm > eps else 1.0
        normalized.append(d * scale)
    return normalized


def flatten_parameters(params: ParameterList) -> Tensor:
    """
    Flatten a list of parameter tensors into a single 1D tensor.

    Args:
        params (ParameterList): Parameter tensors to flatten.

    Returns:
        Tensor: Concatenated 1D tensor of all parameters.
    """
    return torch.cat([p.reshape(-1) for p in params])


def unflatten_vector(vector: Tensor, template_params: ParameterList) -> ParameterList:
    """
    Unflatten a 1D vector into a list of tensors that match template shapes.

    Args:
        vector (Tensor): Flattened parameter vector.
        template_params (ParameterList): Template tensors providing shapes.

    Returns:
        ParameterList: List of tensors reshaped to match the templates.
    """
    params: ParameterList = []
    offset = 0
    for tmpl in template_params:
        numel = tmpl.numel()
        segment = vector[offset : offset + numel].reshape_as(tmpl)
        params.append(segment)
        offset += numel
    return params

