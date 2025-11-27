"""
Residual MLP-style architectures.

This module implements a simple residual MLP classifier to demonstrate
that the loss landscape probes apply to architectures with skip
connections as well.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class ResidualMLPConfig:
    """
    Configuration for a residual MLP classifier.

    Args:
        input_dim (int): Dimensionality of input features.
        hidden_dim (int): Dimensionality of hidden layers.
        num_blocks (int): Number of residual blocks.
        num_classes (int): Number of output classes.
    """

    input_dim: int
    hidden_dim: int
    num_blocks: int
    num_classes: int


class ResidualBlock(nn.Module):
    """
    Simple residual block with two linear layers and ReLU.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        out = out + residual
        out = self.activation(out)
        return out


class ResidualMLPClassifier(nn.Module):
    """
    Fully-connected classifier with residual connections.
    """

    def __init__(self, config: ResidualMLPConfig) -> None:
        super().__init__()
        self.config = config

        self.input_layer = nn.Linear(config.input_dim, config.hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock(config.hidden_dim) for _ in range(config.num_blocks)]
        )
        self.output_layer = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Input features of shape (N, input_dim).

        Returns:
            Tensor: Logits of shape (N, num_classes).
        """
        out = self.input_layer(x)
        out = self.blocks(out)
        logits = self.output_layer(out)
        return logits


def build_resnet_from_config(config: ResidualMLPConfig) -> ResidualMLPClassifier:
    """
    Build a ResidualMLPClassifier from a ResidualMLPConfig.

    Args:
        config (ResidualMLPConfig): Residual MLP configuration.

    Returns:
        ResidualMLPClassifier: Instantiated model.
    """
    return ResidualMLPClassifier(config)

