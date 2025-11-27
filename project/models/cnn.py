"""
Simple convolutional classifier architectures.

These models are designed to demonstrate that the loss landscape
analysis tools apply beyond MLPs. They can operate on 2D synthetic
inputs (reshaped) or small image datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


@dataclass
class CNNConfig:
    """
    Configuration for a simple convolutional classifier.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        hidden_channels (int): Base number of convolutional channels.
        num_blocks (int): Number of conv+pool blocks.
        activation (Literal["relu", "tanh"]): Activation function.
    """

    in_channels: int
    num_classes: int
    hidden_channels: int = 16
    num_blocks: int = 2
    activation: Literal["relu", "tanh"] = "relu"


def _get_activation(name: Literal["relu", "tanh"]) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported CNN activation: {name}")


class ConvNetClassifier(nn.Module):
    """
    Lightweight convolutional classifier.

    This architecture is intentionally small and generic so that it can
    be used with the same training and probing utilities as the MLP.
    """

    def __init__(self, config: CNNConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        in_channels = config.in_channels
        channels = config.hidden_channels
        activation = _get_activation(config.activation)

        for _ in range(config.num_blocks):
            layers.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1))
            layers.append(activation.__class__())
            # Use pooling that halves the height but preserves width, so it
            # works for very narrow inputs like 1x2 or 2x1.
            layers.append(nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
            in_channels = channels
            channels *= 2

        self.features = nn.Sequential(*layers)
        self.classifier: nn.Linear | None = None

    def _ensure_classifier(self, x: Tensor) -> None:
        if self.classifier is not None:
            return
        flattened_dim = x.view(x.size(0), -1).size(1)
        self.classifier = nn.Linear(flattened_dim, self.config.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Input of shape (N, C, H, W).

        Returns:
            Tensor: Logits of shape (N, num_classes).
        """
        features = self.features(x)
        self._ensure_classifier(features)
        assert self.classifier is not None
        logits = self.classifier(features.view(features.size(0), -1))
        return logits


def build_cnn_from_config(config: CNNConfig) -> ConvNetClassifier:
    """
    Build a ConvNetClassifier from a CNNConfig.

    Args:
        config (CNNConfig): CNN configuration.

    Returns:
        ConvNetClassifier: Instantiated model.
    """
    return ConvNetClassifier(config)
