"""
Synthetic dataset generation utilities for the Loss Landscape Geometry project.

This subpackage exposes helpers to create small 2D classification datasets
used throughout the experiments.
"""

from .datasets import (
    generate_circles_dataset,
    generate_gaussian_clusters_dataset,
    generate_moons_dataset,
    generate_xor_dataset,
    normalize_splits,
)

__all__ = [
    "generate_moons_dataset",
    "generate_circles_dataset",
    "generate_gaussian_clusters_dataset",
    "generate_xor_dataset",
    "normalize_splits",
]

