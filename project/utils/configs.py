"""
Configuration dataclasses for the Loss Landscape Geometry project.

These dataclasses centralize all hyperparameters so that modules do not
rely on hard-coded constants spread throughout the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence


@dataclass
class DatasetConfig:
    """
    Configuration for synthetic classification dataset generation.

    Args:
        name (str): Name of the dataset type (e.g. ``\"moons\"``).
        n_train (int): Number of training samples.
        n_test (int): Number of test samples.
        noise (float): Standard deviation of noise added to inputs.
        n_classes (int): Number of classes for the dataset.
        seed (int): Random seed for data generation.
    """

    name: Literal["moons", "circles", "gaussians", "xor"]
    n_train: int
    n_test: int
    noise: float
    n_classes: int
    seed: int


@dataclass
class ModelConfig:
    """
    Configuration for a fully-connected MLP classifier.

    Args:
        input_dim (int): Dimensionality of input features.
        hidden_layers (int): Number of hidden layers.
        hidden_size (int): Number of units per hidden layer.
        output_dim (int): Number of output classes.
        activation (str): Activation type: ``\"relu\"``, ``\"tanh\"``, or ``\"gelu\"``.
        bias (bool): Whether to include bias terms.
        dropout (float): Dropout probability for hidden layers.
    """

    input_dim: int
    hidden_layers: int
    hidden_size: int
    output_dim: int
    activation: Literal["relu", "tanh", "gelu"]
    bias: bool = True
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    """
    Configuration for model training loops.

    Args:
        optimizer (str): Optimizer name: ``\"sgd\"`` or ``\"adam\"``.
        learning_rate (float): Base learning rate.
        momentum (float): Momentum for SGD (ignored for Adam).
        weight_decay (float): L2 regularization coefficient.
        batch_size (int): Mini-batch size.
        epochs (int): Number of training epochs.
        lr_step_size (int): Step size (in epochs) for LR decay.
        lr_gamma (float): Multiplicative factor for LR decay.
        device (str): Device string, e.g. ``\"cpu\"`` or ``\"cuda\"``.
        seed (int): Random seed for training.
        checkpoint_dir (Path): Directory to save checkpoints.
        log_interval (int): Log statistics every N batches.
    """

    optimizer: Literal["sgd", "adam"]
    learning_rate: float
    momentum: float
    weight_decay: float
    batch_size: int
    epochs: int
    lr_step_size: int
    lr_gamma: float
    device: str
    seed: int
    checkpoint_dir: Path
    log_interval: int = 10


@dataclass
class InterpolationConfig:
    """
    Configuration for linear interpolation in parameter space.

    Args:
        num_points (int): Number of interpolation points between endpoints.
        normalize_per_layer (bool): Whether to normalize parameter
            differences per layer.
    """

    num_points: int
    normalize_per_layer: bool = True


@dataclass
class SliceConfig:
    """
    Configuration for random direction loss landscape slices.

    Args:
        num_points (int): Number of points along each direction.
        radius (float): Maximum step size radius for exploration.
        normalize_per_layer (bool): Whether to normalize directions
            per layer using filter-wise scaling.
    """

    num_points: int
    radius: float
    normalize_per_layer: bool = True


@dataclass
class HessianConfig:
    """
    Configuration for Hessian spectrum estimation.

    Args:
        top_k (int): Number of leading eigenvalues to estimate.
        num_trace_samples (int): Number of Hutchinson samples for trace.
        num_power_iterations (int): Number of iterations for power
            iteration per eigenvalue.
    """

    top_k: int
    num_trace_samples: int
    num_power_iterations: int


@dataclass
class SharpnessConfig:
    """
    Configuration for sharpness and flatness metrics.

    Args:
        epsilon (float): Radius of the perturbation ball.
        num_samples (int): Number of random perturbations to evaluate.
        normalize_per_layer (bool): Whether to normalize perturbations
            per layer before scaling by ``epsilon``.
    """

    epsilon: float
    num_samples: int
    normalize_per_layer: bool = True


@dataclass
class PCAConfig:
    """
    Configuration for PCA-based landscape projections.

    Args:
        num_components (int): Number of principal components to retain.
        grid_size (int): Size of the evaluation grid along each PCA axis.
        grid_radius (float): Radius for the PCA grid in coefficient space.
    """

    num_components: int
    grid_size: int
    grid_radius: float


@dataclass
class ExperimentConfig:
    """
    Aggregate configuration describing a single experiment run.

    Args:
        dataset (DatasetConfig): Dataset configuration.
        model (ModelConfig): Model architecture configuration.
        training (TrainingConfig): Training loop configuration.
        seeds (Sequence[int]): Random seeds to use for repeated runs.
    """

    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    seeds: Sequence[int]
