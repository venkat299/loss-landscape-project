"""
Loss landscape analysis utilities.

This package implements core probing methods used throughout the
experiments, including:

    - linear interpolation between models
    - random direction loss slices
    - sharpness / flatness metrics

Additional modules (Hessian spectrum, PCA projections, connectivity)
can build on these utilities.
"""

from .interpolation import linear_interpolation_curve
from .random_slice import random_1d_loss_slice, random_2d_loss_surface
from .sharpness import epsilon_sharpness

__all__ = [
    "linear_interpolation_curve",
    "random_1d_loss_slice",
    "random_2d_loss_surface",
    "epsilon_sharpness",
]

