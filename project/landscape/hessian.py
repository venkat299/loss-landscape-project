"""
Hessian spectrum estimation utilities.

This module implements:
    - Hessian–vector products using autograd
    - power iteration for top-k eigenvalues
    - Hutchinson estimator for the Hessian trace
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from project.landscape.common import (
    ParameterList,
    flatten_parameters,
    get_parameter_tensors,
    unflatten_vector,
)
from project.utils.configs import HessianConfig


def _hessian_vector_product(
    model: nn.Module,
    dataloader: DataLoader,
    vector_flat: Tensor,
    device: torch.device,
) -> Tensor:
    """
    Compute the Hessian–vector product H v for the loss over a dataset.

    Args:
        model (nn.Module): Model whose loss Hessian is considered.
        dataloader (DataLoader): DataLoader providing data for the loss.
        vector_flat (Tensor): Flattened vector ``v`` with the same size
            as the concatenated parameter vector.
        device (torch.device): Computation device.

    Returns:
        Tensor: Flattened tensor representing H v.
    """
    model.eval()
    params = get_parameter_tensors(model)
    template_params: ParameterList = [p.detach() for p in params]
    vector = unflatten_vector(vector_flat, template_params)

    total_hvp: Tensor | None = None
    num_batches = 0
    criterion = nn.CrossEntropyLoss()

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        model.zero_grad(set_to_none=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        grads = torch.autograd.grad(loss, params, create_graph=True)
        grad_dot_vec = torch.zeros(1, device=device)
        for g, v in zip(grads, vector):
            grad_dot_vec = grad_dot_vec + torch.sum(g * v)

        hvp_tensors = torch.autograd.grad(grad_dot_vec, params)
        hvp_flat = flatten_parameters(list(hvp_tensors))

        if total_hvp is None:
            total_hvp = hvp_flat.detach()
        else:
            total_hvp = total_hvp + hvp_flat.detach()
        num_batches += 1

    if total_hvp is None or num_batches == 0:
        raise RuntimeError("Empty dataloader provided for Hessian–vector product.")

    return total_hvp / float(num_batches)


def _power_iteration(
    model: nn.Module,
    dataloader: DataLoader,
    config: HessianConfig,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """
    Estimate the top-k Hessian eigenvalues via power iteration.

    Args:
        model (nn.Module): Model whose Hessian is analyzed.
        dataloader (DataLoader): DataLoader providing data for the loss.
        config (HessianConfig): Configuration specifying ``top_k`` and
            ``num_power_iterations``.
        device (torch.device): Computation device.

    Returns:
        Tuple[Tensor, Tensor]:
            - Eigenvalues tensor of shape (top_k,)
            - Eigenvectors tensor of shape (top_k, D) where D is the
              total number of parameters.
    """
    params = get_parameter_tensors(model)
    template_params: ParameterList = [p.detach() for p in params]
    dim = sum(p.numel() for p in template_params)

    eigenvalues = torch.zeros(config.top_k, device=device)
    eigenvectors = torch.zeros(config.top_k, dim, device=device)

    for k in range(config.top_k):
        v = torch.randn(dim, device=device)
        v = v / v.norm()

        for _ in range(config.num_power_iterations):
            hv = _hessian_vector_product(model, dataloader, v, device=device)

            for j in range(k):
                prev_vec = eigenvectors[j]
                hv = hv - torch.dot(hv, prev_vec) * prev_vec

            norm_hv = hv.norm()
            if norm_hv <= 1e-10:
                break
            v = hv / norm_hv

        hv = _hessian_vector_product(model, dataloader, v, device=device)
        lambda_k = torch.dot(v, hv)

        eigenvalues[k] = lambda_k
        eigenvectors[k] = v

    return eigenvalues, eigenvectors


def _hutchinson_trace_estimate(
    model: nn.Module,
    dataloader: DataLoader,
    config: HessianConfig,
    device: torch.device,
) -> float:
    """
    Estimate the trace of the Hessian using Hutchinson's method.

    Args:
        model (nn.Module): Model whose Hessian trace is estimated.
        dataloader (DataLoader): DataLoader providing data for the loss.
        config (HessianConfig): Configuration specifying ``num_trace_samples``.
        device (torch.device): Computation device.

    Returns:
        float: Estimated Hessian trace.
    """
    params = get_parameter_tensors(model)
    template_params: ParameterList = [p.detach() for p in params]
    dim = sum(p.numel() for p in template_params)

    estimates: List[float] = []

    for _ in range(config.num_trace_samples):
        v = torch.empty(dim, device=device).bernoulli_(0.5).mul_(2).sub_(1)
        hv = _hessian_vector_product(model, dataloader, v, device=device)
        v_h_v = torch.dot(v, hv).item()
        estimates.append(v_h_v)

    if not estimates:
        raise RuntimeError("No Hutchinson samples were generated for trace estimation.")

    return float(sum(estimates) / len(estimates))


def estimate_hessian_spectrum(
    model: nn.Module,
    dataloader: DataLoader,
    config: HessianConfig,
    device: torch.device,
) -> Tuple[Tensor, float]:
    """
    Estimate the top-k eigenvalues and trace of the Hessian.

    Args:
        model (nn.Module): Model whose Hessian spectrum is analyzed.
        dataloader (DataLoader): DataLoader used to define the loss.
        config (HessianConfig): Hessian configuration.
        device (torch.device): Computation device.

    Returns:
        Tuple[Tensor, float]:
            - Tensor of shape (top_k,) containing approximate eigenvalues.
            - Scalar float containing the trace estimate.
    """
    model = model.to(device)
    eigenvalues, _ = _power_iteration(model, dataloader, config, device=device)
    trace_estimate = _hutchinson_trace_estimate(model, dataloader, config, device=device)
    return eigenvalues.detach().cpu(), trace_estimate

