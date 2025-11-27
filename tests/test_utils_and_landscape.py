"""
Unit tests for key utilities and landscape operations.

These tests are intentionally lightweight to keep runtime manageable.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from project.landscape.common import (
    clone_parameters,
    flatten_parameters,
    normalize_direction_per_layer,
    random_direction_like,
    unflatten_vector,
)
from project.landscape.hessian import estimate_hessian_spectrum
from project.landscape.interpolation import linear_interpolation_curve
from project.models import build_mlp_from_config, get_predefined_model_config
from project.utils.configs import HessianConfig
from project.utils.seed import set_global_seed


def _small_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(64, 2)
    y = torch.randint(0, 2, (64,))
    return x, y


def _make_dataloader(x: torch.Tensor, y: torch.Tensor) -> DataLoader:
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=16, shuffle=False)


def test_direction_normalization_preserves_layer_norms() -> None:
    set_global_seed(0)
    config = get_predefined_model_config(
        variant="medium",
        input_dim=2,
        output_dim=2,
        activation="relu",
    )
    model = build_mlp_from_config(config)
    params = clone_parameters(model)
    direction = random_direction_like(model)
    normalized = normalize_direction_per_layer(params, direction)

    for w, d in zip(params, normalized):
        if w.norm().item() == 0.0:
            continue
        assert torch.allclose(w.norm(), d.norm(), atol=1e-5)


def test_flatten_unflatten_roundtrip() -> None:
    set_global_seed(0)
    config = get_predefined_model_config(
        variant="medium",
        input_dim=2,
        output_dim=2,
        activation="relu",
    )
    model = build_mlp_from_config(config)
    params = clone_parameters(model)
    flat = flatten_parameters(params)
    restored = unflatten_vector(flat, params)

    for original, new in zip(params, restored):
        assert torch.allclose(original, new)


def test_hessian_spectrum_runs_on_tiny_model() -> None:
    set_global_seed(0)
    config = get_predefined_model_config(
        variant="shallow-small",
        input_dim=2,
        output_dim=2,
        activation="relu",
    )
    model = build_mlp_from_config(config)

    x, y = _small_dataset()
    loader = _make_dataloader(x, y)

    hessian_cfg = HessianConfig(top_k=2, num_trace_samples=2, num_power_iterations=3)
    eigenvalues, trace_estimate = estimate_hessian_spectrum(
        model=model,
        dataloader=loader,
        config=hessian_cfg,
        device=torch.device("cpu"),
    )

    assert eigenvalues.shape[0] == 2
    assert isinstance(trace_estimate, float)


def test_linear_interpolation_outputs_monotonic_alphas() -> None:
    set_global_seed(0)
    config = get_predefined_model_config(
        variant="medium",
        input_dim=2,
        output_dim=2,
        activation="relu",
    )
    model_a = build_mlp_from_config(config)
    model_b = build_mlp_from_config(config)

    x, y = _small_dataset()
    loader = _make_dataloader(x, y)

    results = linear_interpolation_curve(
        model_a=model_a,
        model_b=model_b,
        train_loader=loader,
        test_loader=loader,
        num_points=5,
        device=torch.device("cpu"),
        normalize_per_layer=True,
    )

    alphas = results["alphas"]
    assert torch.all(alphas[:-1] <= alphas[1:])
