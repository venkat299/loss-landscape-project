"""
Training and experiment utilities for the Loss Landscape Geometry project.

This subpackage contains reusable training loops and will host
experiment driver scripts for different study dimensions (depth,
width, activations, optimizers, connectivity, and Hessian analysis).
"""

from .train_model import (
    train_model,
    evaluate_model,
)

__all__ = [
    "train_model",
    "evaluate_model",
]

