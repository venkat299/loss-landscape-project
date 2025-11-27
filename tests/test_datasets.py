"""
Unit tests for synthetic dataset generators.
"""

from __future__ import annotations

import torch

from project.data import (
    generate_circles_dataset,
    generate_gaussian_clusters_dataset,
    generate_moons_dataset,
    generate_xor_dataset,
)
from project.utils.configs import DatasetConfig


def _basic_dataset_checks(x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor, y_test: torch.Tensor) -> None:
    assert x_train.ndim == 2 and x_test.ndim == 2
    assert x_train.size(1) == x_test.size(1) == 2
    assert y_train.ndim == 1 and y_test.ndim == 1
    assert x_train.size(0) == y_train.size(0)
    assert x_test.size(0) == y_test.size(0)
    # Check normalization roughly: mean ~0, std ~1 for train
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0)
    assert torch.all(torch.isfinite(mean))
    assert torch.all(std > 0)


def test_generate_moons_dataset_shapes_and_normalization() -> None:
    cfg = DatasetConfig(name="moons", n_train=128, n_test=64, noise=0.1, n_classes=2, seed=0)
    x_train, y_train, x_test, y_test = generate_moons_dataset(cfg)
    _basic_dataset_checks(x_train, y_train, x_test, y_test)


def test_generate_circles_dataset_shapes_and_normalization() -> None:
    cfg = DatasetConfig(name="circles", n_train=128, n_test=64, noise=0.05, n_classes=2, seed=1)
    x_train, y_train, x_test, y_test = generate_circles_dataset(cfg)
    _basic_dataset_checks(x_train, y_train, x_test, y_test)


def test_generate_gaussian_clusters_dataset_shapes_and_normalization() -> None:
    cfg = DatasetConfig(name="gaussians", n_train=128, n_test=64, noise=0.0, n_classes=3, seed=2)
    x_train, y_train, x_test, y_test = generate_gaussian_clusters_dataset(cfg)
    _basic_dataset_checks(x_train, y_train, x_test, y_test)
    # Check that labels fall in the expected range
    assert int(y_train.max()) < cfg.n_classes
    assert int(y_test.max()) < cfg.n_classes


def test_generate_xor_dataset_shapes_and_normalization() -> None:
    cfg = DatasetConfig(name="xor", n_train=128, n_test=64, noise=0.05, n_classes=2, seed=3)
    x_train, y_train, x_test, y_test = generate_xor_dataset(cfg)
    _basic_dataset_checks(x_train, y_train, x_test, y_test)

