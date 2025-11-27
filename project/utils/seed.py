"""
Deterministic seeding utilities.

This module centralizes all randomness control for the project to ensure
reproducible experiments across Python, NumPy, and PyTorch.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True, cuda_deterministic: Optional[bool] = None) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed (int): Base seed value to use for all RNGs.
        deterministic (bool): If True, enable PyTorch deterministic flags
            where available. This may have performance implications.
        cuda_deterministic (Optional[bool]): Optional override for CUDA
            deterministic behavior. If None, this mirrors ``deterministic``.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if cuda_deterministic is None:
        cuda_deterministic = deterministic

    if deterministic:
        torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = not cuda_deterministic
        torch.backends.cudnn.deterministic = cuda_deterministic

