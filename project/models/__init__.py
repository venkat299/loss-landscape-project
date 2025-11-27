# Update: added MLP model exports
# Location: project/models/__init__.py
"""
Model definitions for the Loss Landscape Geometry project.

Currently exposes a configurable fully-connected MLP classifier and
helpers to construct predefined architecture variants.
"""

from .mlp import (
    MLPClassifier,
    build_mlp_from_config,
    get_predefined_model_config,
)

__all__ = [
    "MLPClassifier",
    "build_mlp_from_config",
    "get_predefined_model_config",
]

