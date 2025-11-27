"""
Synthetic classification dataset generators.

This module provides small 2D datasets (moons, circles, Gaussian clusters,
and XOR-like) with train/test splits and feature normalization utilities.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from project.utils.configs import DatasetConfig
from project.utils.seed import set_global_seed


def normalize_splits(x_train: Tensor, x_test: Tensor, eps: float = 1e-8) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Normalize train and test feature tensors using training statistics.

    Args:
        x_train (Tensor): Training features of shape (N_train, D).
        x_test (Tensor): Test features of shape (N_test, D).
        eps (float): Small constant to avoid division by zero.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]:
            Normalized ``x_train`` and ``x_test``, followed by the mean and
            standard deviation used for normalization.
    """
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(eps)
    x_train_norm = (x_train - mean) / std
    x_test_norm = (x_test - mean) / std
    return x_train_norm, x_test_norm, mean.squeeze(0), std.squeeze(0)


def _split_train_test(
    data: np.ndarray,
    labels: np.ndarray,
    n_train: int,
    n_test: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Split NumPy arrays into train and test Tensor splits.

    Args:
        data (np.ndarray): Feature array of shape (N, D).
        labels (np.ndarray): Label array of shape (N,).
        n_train (int): Number of training samples to select.
        n_test (int): Number of test samples to select.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]:
            ``x_train``, ``y_train``, ``x_test``, ``y_test`` tensors.
    """
    total = n_train + n_test
    if total > data.shape[0]:
        raise ValueError(f"Requested {total} samples but only {data.shape[0]} available.")

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    indices = indices[:total]

    train_idx = indices[:n_train]
    test_idx = indices[n_train:total]

    x_train = torch.from_numpy(data[train_idx].astype(np.float32))
    x_test = torch.from_numpy(data[test_idx].astype(np.float32))
    y_train = torch.from_numpy(labels[train_idx].astype(np.int64))
    y_test = torch.from_numpy(labels[test_idx].astype(np.int64))

    return x_train, y_train, x_test, y_test


def generate_moons_dataset(config: DatasetConfig) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Generate a two-moons classification dataset.

    Args:
        config (DatasetConfig): Dataset configuration specifying sample sizes,
            noise level, and seed. ``name`` should be ``\"moons\"``.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]:
            Normalized ``x_train``, ``y_train``, ``x_test``, ``y_test``.
    """
    if config.name != "moons":
        raise ValueError(f"DatasetConfig.name must be 'moons', got {config.name}.")

    set_global_seed(config.seed)

    n_samples = config.n_train + config.n_test
    n_per_class = n_samples // 2

    angles = np.random.rand(n_per_class) * np.pi
    x_inner = np.c_[np.cos(angles), np.sin(angles)]
    x_outer = np.c_[1 - np.cos(angles), 1 - np.sin(angles) - 0.5]

    data = np.vstack([x_inner, x_outer])
    labels = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])

    if data.shape[0] < n_samples:
        # In case of odd total, pad one extra sample from class 0.
        data = np.vstack([data, x_inner[0:1]])
        labels = np.concatenate([labels, np.zeros(1)])

    data += np.random.normal(scale=config.noise, size=data.shape)

    x_train, y_train, x_test, y_test = _split_train_test(data, labels, config.n_train, config.n_test)
    x_train, x_test, _, _ = normalize_splits(x_train, x_test)
    return x_train, y_train, x_test, y_test


def generate_circles_dataset(config: DatasetConfig) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Generate a concentric circles classification dataset.

    Args:
        config (DatasetConfig): Dataset configuration specifying sample sizes,
            noise level, and seed. ``name`` should be ``\"circles\"``.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]:
            Normalized ``x_train``, ``y_train``, ``x_test``, ``y_test``.
    """
    if config.name != "circles":
        raise ValueError(f"DatasetConfig.name must be 'circles', got {config.name}.")

    set_global_seed(config.seed)

    n_samples = config.n_train + config.n_test
    n_per_class = n_samples // 2

    angles_inner = 2 * np.pi * np.random.rand(n_per_class)
    angles_outer = 2 * np.pi * np.random.rand(n_per_class)

    inner_radius = 0.5
    outer_radius = 1.5

    x_inner = np.c_[inner_radius * np.cos(angles_inner), inner_radius * np.sin(angles_inner)]
    x_outer = np.c_[outer_radius * np.cos(angles_outer), outer_radius * np.sin(angles_outer)]

    data = np.vstack([x_inner, x_outer])
    labels = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])

    if data.shape[0] < n_samples:
        data = np.vstack([data, x_inner[0:1]])
        labels = np.concatenate([labels, np.zeros(1)])

    data += np.random.normal(scale=config.noise, size=data.shape)

    x_train, y_train, x_test, y_test = _split_train_test(data, labels, config.n_train, config.n_test)
    x_train, x_test, _, _ = normalize_splits(x_train, x_test)
    return x_train, y_train, x_test, y_test


def generate_gaussian_clusters_dataset(config: DatasetConfig) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Generate a dataset composed of 2–4 Gaussian clusters.

    Args:
        config (DatasetConfig): Dataset configuration specifying sample sizes,
            noise level, number of classes (2–4), and seed. ``name`` should be
            ``\"gaussians\"``.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]:
            Normalized ``x_train``, ``y_train``, ``x_test``, ``y_test``.
    """
    if config.name != "gaussians":
        raise ValueError(f"DatasetConfig.name must be 'gaussians', got {config.name}.")
    if not (2 <= config.n_classes <= 4):
        raise ValueError(f"Gaussian clusters require 2–4 classes, got {config.n_classes}.")

    set_global_seed(config.seed)

    n_samples = config.n_train + config.n_test
    n_per_class = n_samples // config.n_classes

    means = []
    covs = []
    for k in range(config.n_classes):
        angle = 2 * np.pi * k / config.n_classes
        radius = 2.0
        center = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        means.append(center)
        covs.append(0.2 * np.eye(2))

    data_list = []
    labels_list = []
    for k in range(config.n_classes):
        samples = np.random.multivariate_normal(means[k], covs[k], size=n_per_class)
        data_list.append(samples)
        labels_list.append(np.full(n_per_class, k))

    data = np.vstack(data_list)
    labels = np.concatenate(labels_list)

    if data.shape[0] < n_samples:
        deficit = n_samples - data.shape[0]
        extra = np.random.multivariate_normal(means[0], covs[0], size=deficit)
        data = np.vstack([data, extra])
        labels = np.concatenate([labels, np.zeros(deficit)])

    data += np.random.normal(scale=config.noise, size=data.shape)

    x_train, y_train, x_test, y_test = _split_train_test(data, labels, config.n_train, config.n_test)
    x_train, x_test, _, _ = normalize_splits(x_train, x_test)
    return x_train, y_train, x_test, y_test


def generate_xor_dataset(config: DatasetConfig) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Generate an XOR-like binary classification dataset in 2D.

    Args:
        config (DatasetConfig): Dataset configuration specifying sample sizes,
            noise level, and seed. ``name`` should be ``\"xor\"``.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]:
            Normalized ``x_train``, ``y_train``, ``x_test``, ``y_test``.
    """
    if config.name != "xor":
        raise ValueError(f"DatasetConfig.name must be 'xor', got {config.name}.")

    set_global_seed(config.seed)

    n_samples = config.n_train + config.n_test
    n_per_quadrant = n_samples // 4

    centers = np.array(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ]
    )
    labels_quadrants = np.array([0, 1, 1, 0])

    data_list = []
    labels_list = []
    for idx, center in enumerate(centers):
        samples = center + np.random.normal(scale=0.3, size=(n_per_quadrant, 2))
        data_list.append(samples)
        labels_list.append(np.full(n_per_quadrant, labels_quadrants[idx]))

    data = np.vstack(data_list)
    labels = np.concatenate(labels_list)

    if data.shape[0] < n_samples:
        deficit = n_samples - data.shape[0]
        extra = centers[0] + np.random.normal(scale=0.3, size=(deficit, 2))
        data = np.vstack([data, extra])
        labels = np.concatenate([labels, np.full(deficit, labels_quadrants[0])])

    data += np.random.normal(scale=config.noise, size=data.shape)

    x_train, y_train, x_test, y_test = _split_train_test(data, labels, config.n_train, config.n_test)
    x_train, x_test, _, _ = normalize_splits(x_train, x_test)
    return x_train, y_train, x_test, y_test

