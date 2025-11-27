"""
PCA-based projections of loss landscapes.

This module provides helpers to:
    - collect parameter vectors from checkpoints
    - compute principal components of training trajectories
    - project trajectories into a low-dimensional PCA plane
    - evaluate loss over a 2D PCA grid
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from project.experiments import evaluate_model
from project.landscape.common import (
    ParameterList,
    assign_parameters,
    clone_parameters,
    flatten_parameters,
    unflatten_vector,
)
from project.utils.configs import PCAConfig


def collect_parameter_trajectory(
    model: nn.Module,
    checkpoint_paths: Sequence[Path],
    device: torch.device,
) -> Tensor:
    """
    Collect flattened parameter vectors from a sequence of checkpoints.

    Args:
        model (nn.Module): Model instance matching the checkpoint architecture.
        checkpoint_paths (Sequence[Path]): Paths to checkpoint files saved
            by the training utilities.
        device (torch.device): Device for loading model weights.

    Returns:
        Tensor: Matrix of shape (T, D) where T is the number of checkpoints
        and D is the number of parameters.
    """
    weight_vectors: List[Tensor] = []
    for path in checkpoint_paths:
        # Use ``weights_only=False`` for compatibility with checkpoints that
        # store additional metadata (e.g. config objects) alongside weights.
        state = torch.load(path, map_location=device, weights_only=False)
        state_dict = state["model_state_dict"]
        model.load_state_dict(state_dict)
        model = model.to(device)
        params: ParameterList = clone_parameters(model)
        flat = flatten_parameters(params).detach().cpu()
        weight_vectors.append(flat)

    if not weight_vectors:
        raise ValueError("No checkpoints provided for PCA trajectory collection.")

    return torch.stack(weight_vectors, dim=0)


def compute_pca_components(weight_matrix: Tensor, num_components: int) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute PCA components for a matrix of parameter vectors.

    Args:
        weight_matrix (Tensor): Matrix of shape (T, D) containing flattened
            parameter vectors along the training trajectory.
        num_components (int): Number of principal components to retain.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            - mean_vector: Mean of the weight vectors (D,)
            - components: Principal components (D, num_components)
            - explained_variance: Variance explained by each component (num_components,)
    """
    if weight_matrix.ndim != 2:
        raise ValueError("weight_matrix must be 2D (T, D).")

    mean_vector = weight_matrix.mean(dim=0, keepdim=True)
    centered = weight_matrix - mean_vector

    # Compute SVD of centered data: centered = U S Vh
    u, s, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:num_components].T
    explained_variance = (s[:num_components] ** 2) / (centered.shape[0] - 1)

    return mean_vector.squeeze(0), components, explained_variance


def project_trajectory_to_pca(
    weight_matrix: Tensor,
    mean_vector: Tensor,
    components: Tensor,
) -> Tensor:
    """
    Project a trajectory of weight vectors into PCA coordinate space.

    Args:
        weight_matrix (Tensor): Matrix of shape (T, D) with flattened
            weight vectors.
        mean_vector (Tensor): Mean vector of shape (D,).
        components (Tensor): PCA components of shape (D, K).

    Returns:
        Tensor: Matrix of shape (T, K) containing PCA coordinates.
    """
    centered = weight_matrix - mean_vector.unsqueeze(0)
    return centered @ components


def loss_over_pca_grid(
    model: nn.Module,
    base_params: ParameterList,
    mean_vector: Tensor,
    components: Tensor,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: PCAConfig,
    device: torch.device,
) -> Dict[str, Tensor]:
    """
    Evaluate loss over a 2D PCA plane defined by the first two components.

    Args:
        model (nn.Module): Model whose parameters will be manipulated.
        base_params (ParameterList): Reference parameter tensors used to
            provide shapes for unflattening.
        mean_vector (Tensor): PCA mean vector of shape (D,).
        components (Tensor): PCA components of shape (D, K) with K>=2.
        train_loader (DataLoader): Training dataset loader.
        test_loader (DataLoader): Test dataset loader.
        config (PCAConfig): PCA configuration (grid size and radius).
        device (torch.device): Computation device.

    Returns:
        Dict[str, Tensor]: Dictionary containing:
            - ``"alpha_grid"``: PCA coefficient grid for component 1.
            - ``"beta_grid"``: PCA coefficient grid for component 2.
            - ``"train_loss"``: Training loss surface over the grid.
            - ``"test_loss"``: Test loss surface over the grid.
    """
    if components.shape[1] < 2:
        raise ValueError("At least two PCA components are required for a 2D grid.")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    comp1 = components[:, 0]
    comp2 = components[:, 1]

    alpha_vals = torch.linspace(-config.grid_radius, config.grid_radius, steps=config.grid_size)
    beta_vals = torch.linspace(-config.grid_radius, config.grid_radius, steps=config.grid_size)
    alpha_grid, beta_grid = torch.meshgrid(alpha_vals, beta_vals, indexing="ij")

    train_surface = torch.zeros_like(alpha_grid)
    test_surface = torch.zeros_like(beta_grid)

    for i in range(config.grid_size):
        for j in range(config.grid_size):
            alpha = float(alpha_grid[i, j].item())
            beta = float(beta_grid[i, j].item())

            flat_params = mean_vector + alpha * comp1 + beta * comp2
            params = unflatten_vector(flat_params, base_params)
            assign_parameters(model, params)

            train_loss, _ = evaluate_model(model, train_loader, criterion, device)
            test_loss, _ = evaluate_model(model, test_loader, criterion, device)

            train_surface[i, j] = train_loss
            test_surface[i, j] = test_loss

    return {
        "alpha_grid": alpha_grid,
        "beta_grid": beta_grid,
        "train_loss": train_surface,
        "test_loss": test_surface,
    }
