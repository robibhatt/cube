"""Utilities for sparsifying multilayer perceptrons."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Iterable

import torch
import torch.nn.functional as F

from src.models.mlp import MLP


def _zero_small_weights(linear_layers: Iterable[torch.nn.Module], threshold: float) -> None:
    """Zero out weights whose absolute value falls below ``threshold``."""
    with torch.no_grad():
        for layer in linear_layers:
            weight = getattr(layer, "weight", None)
            if weight is None:
                continue
            abs_weight = weight.detach().abs()
            mask = abs_weight < threshold
            weight.data.masked_fill_(mask, 0.0)


def sparsify_mlp(model: MLP, threshold: float) -> MLP:
    """Return a copy of ``model`` with small weights zeroed out."""
    if not isinstance(model, MLP):
        raise TypeError("model must be an instance of MLP")
    if threshold <= 0:
        raise ValueError("threshold must be a positive float")

    config_copy = deepcopy(model.config)
    sparsified = MLP(config_copy)
    sparsified.load_state_dict(model.state_dict())
    _zero_small_weights(sparsified.linear_layers, threshold)
    return sparsified


def mse_diff(number_of_samples: int, mlp_a: MLP, mlp_b: MLP) -> float:
    """Estimate the mean squared error between two MLPs on hypercube samples."""
    if number_of_samples <= 0:
        raise ValueError("number_of_samples must be a positive integer")
    if not isinstance(mlp_a, MLP) or not isinstance(mlp_b, MLP):
        raise TypeError("mlp_a and mlp_b must be instances of MLP")
    if mlp_a.config.input_dim != mlp_b.config.input_dim:
        raise ValueError("mlp_a and mlp_b must have the same input dimension")

    input_dim = mlp_a.config.input_dim

    try:
        device_a = next(mlp_a.parameters()).device
    except StopIteration:  # pragma: no cover - defensive fallback
        device_a = torch.device("cpu")
    try:
        device_b = next(mlp_b.parameters()).device
    except StopIteration:  # pragma: no cover - defensive fallback
        device_b = torch.device("cpu")

    base_samples = torch.randint(0, 2, (number_of_samples, input_dim), dtype=torch.int64)
    base_samples = base_samples * 2 - 1

    dtype_a = next(mlp_a.parameters()).dtype
    dtype_b = next(mlp_b.parameters()).dtype

    inputs_a = base_samples.to(device_a, dtype=dtype_a)
    inputs_b = base_samples.to(device_b, dtype=dtype_b)

    with torch.no_grad():
        outputs_a = mlp_a(inputs_a)
        outputs_b = mlp_b(inputs_b)

    mse = F.mse_loss(outputs_a, outputs_b.to(device_a, dtype=outputs_a.dtype))
    return float(mse.detach().cpu().item())


def binary_search_sparsify_threshold(
    model: MLP,
    mse_threshold: float,
    *,
    sample_multiplier: float = 64.0,
    tolerance_ratio: float = 0.05,
    max_iterations: int = 50,
) -> float:
    """Find the largest sparsification threshold that preserves a target MSE."""

    if not isinstance(model, MLP):
        raise TypeError("model must be an instance of MLP")
    if mse_threshold <= 0:
        raise ValueError("mse_threshold must be a positive float")
    if sample_multiplier <= 0:
        raise ValueError("sample_multiplier must be positive")

    max_weight = 0.0
    for layer in model.linear_layers:
        weight = getattr(layer, "weight", None)
        if weight is None:
            continue
        layer_max = float(weight.detach().abs().max().item())
        max_weight = max(max_weight, layer_max)

    if max_weight == 0.0:
        return 0.0

    samples = max(16, int(math.ceil(sample_multiplier / mse_threshold)))
    samples = min(samples, 100_000)

    def mse_for_threshold(threshold: float) -> float:
        sparsified = sparsify_mlp(model, threshold)
        return mse_diff(samples, model, sparsified)

    low = 0.0
    high = max_weight
    upper_mse = mse_for_threshold(high)

    if upper_mse <= mse_threshold:
        for _ in range(10):
            candidate = high * 2.0
            if candidate == high:
                break
            candidate_mse = mse_for_threshold(candidate)
            if candidate_mse > mse_threshold:
                low = high
                high = candidate
                upper_mse = candidate_mse
                break
            high = candidate
            upper_mse = candidate_mse
        else:
            return high

    iterations = 0
    while iterations < max_iterations:
        if high <= 0:
            break
        relative_width = (high - low) / max(high, 1e-12)
        if relative_width <= tolerance_ratio:
            break

        mid = (low + high) / 2.0
        mid_mse = mse_for_threshold(mid)
        if mid_mse <= mse_threshold:
            low = mid
        else:
            high = mid
            upper_mse = mid_mse
        iterations += 1

    if upper_mse <= mse_threshold:
        return high

    return low
