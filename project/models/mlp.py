# Update: implemented configurable MLP classifier
# Location: project/models/mlp.py
"""
Configurable fully-connected MLP classifier.

This module implements a flexible multilayer perceptron with:
    - variable depth and width
    - configurable activation (ReLU, Tanh, GELU)
    - Xavier or He initialization depending on activation

It also exposes helpers to construct predefined architecture variants
specified in the project tasks.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from project.utils.configs import ModelConfig


def _get_activation(name: Literal["relu", "tanh", "gelu"]) -> nn.Module:
    """
    Return an activation module given its name.

    Args:
        name (Literal["relu", "tanh", "gelu"]): Activation identifier.

    Returns:
        nn.Module: Corresponding activation module.
    """
    name_lower = name.lower()
    if name_lower == "relu":
        return nn.ReLU()
    if name_lower == "tanh":
        return nn.Tanh()
    if name_lower == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def _initialize_linear(layer: nn.Linear, activation: Literal["relu", "tanh", "gelu"]) -> None:
    """
    Initialize a linear layer using Xavier or He initialization.

    Args:
        layer (nn.Linear): Linear layer to initialize.
        activation (Literal["relu", "tanh", "gelu"]): Activation used after
            this layer, which determines the initialization scheme.

    Returns:
        None
    """
    activation_lower = activation.lower()
    if activation_lower in {"relu", "gelu"}:
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    else:
        nn.init.xavier_normal_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class MLPClassifier(nn.Module):
    """
    Fully-connected multilayer perceptron classifier.

    The architecture is defined by a ``ModelConfig`` instance specifying
    input dimensionality, number of hidden layers, width, activation,
    and output dimensionality.
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialize the MLP classifier.

        Args:
            config (ModelConfig): Model configuration describing network
                depth, width, activation, and input/output sizes.
        """
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        in_features = config.input_dim

        activation = _get_activation(config.activation)

        for _ in range(config.hidden_layers):
            linear = nn.Linear(in_features, config.hidden_size, bias=config.bias)
            _initialize_linear(linear, config.activation)
            layers.append(linear)
            layers.append(activation.__class__())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(p=config.dropout))
            in_features = config.hidden_size

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(in_features, config.output_dim, bias=True)
        _initialize_linear(self.output, "tanh")

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x (Tensor): Input features of shape (batch_size, input_dim).

        Returns:
            Tensor: Output logits of shape (batch_size, output_dim).
        """
        hidden_out = self.hidden(x)
        logits = self.output(hidden_out)
        return logits


def build_mlp_from_config(config: ModelConfig) -> MLPClassifier:
    """
    Construct an ``MLPClassifier`` instance from a ``ModelConfig``.

    Args:
        config (ModelConfig): Model configuration specification.

    Returns:
        MLPClassifier: Instantiated MLP model.
    """
    return MLPClassifier(config)


def get_predefined_model_config(
    variant: Literal["shallow-small", "shallow-wide", "deep-small", "deep-large", "medium"],
    input_dim: int,
    output_dim: int,
    activation: Literal["relu", "tanh", "gelu"] = "relu",
    bias: bool = True,
    dropout: float = 0.0,
) -> ModelConfig:
    """
    Return a ``ModelConfig`` corresponding to a predefined architecture variant.

    The variants follow the project specification:
        - shallow-small: 1 hidden layer, 50 units
        - shallow-wide: 1 hidden layer, 500 units
        - deep-small: 4 hidden layers, 100 units
        - deep-large: 4 hidden layers, 250 units
        - medium: 2 hidden layers, 100 units

    Args:
        variant (Literal["shallow-small", "shallow-wide", "deep-small",
            "deep-large", "medium"]): Variant identifier.
        input_dim (int): Input feature dimensionality.
        output_dim (int): Number of output classes.
        activation (Literal["relu", "tanh", "gelu"], optional): Activation
            function to use. Defaults to ``"relu"``.
        bias (bool, optional): Whether to include bias terms. Defaults to True.
        dropout (float, optional): Dropout probability. Defaults to 0.0.

    Returns:
        ModelConfig: Configuration describing the chosen architecture.
    """
    if variant == "shallow-small":
        hidden_layers = 1
        hidden_size = 50
    elif variant == "shallow-wide":
        hidden_layers = 1
        hidden_size = 500
    elif variant == "deep-small":
        hidden_layers = 4
        hidden_size = 100
    elif variant == "deep-large":
        hidden_layers = 4
        hidden_size = 250
    elif variant == "medium":
        hidden_layers = 2
        hidden_size = 100
    else:
        raise ValueError(f"Unknown architecture variant: {variant}")

    return ModelConfig(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        hidden_size=hidden_size,
        output_dim=output_dim,
        activation=activation,
        bias=bias,
        dropout=dropout,
    )

