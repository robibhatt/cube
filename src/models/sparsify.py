"""Utilities for sparsifying multilayer perceptrons."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Iterable, Optional

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


def _select_device_prefer_gpu(default: torch.device) -> torch.device:
    """Return ``cuda`` when available, otherwise ``default``."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    return default


def _generate_binary_hypercube_samples(
    number_of_samples: int,
    input_dim: int,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate ±1 samples on the vertices of a hypercube."""

    if number_of_samples <= 0:
        raise ValueError("number_of_samples must be a positive integer")
    if input_dim <= 0:
        raise ValueError("input_dim must be a positive integer")

    if device is None:
        device = torch.device("cpu")

    samples = torch.randint(
        0,
        2,
        (number_of_samples, input_dim),
        device=device,
        dtype=torch.float32,
    )
    return samples.mul_(2.0).sub_(1.0)


def _model_device_and_dtype(model: MLP) -> tuple[torch.device, torch.dtype]:
    try:
        param = next(model.parameters())
    except StopIteration:  # pragma: no cover - defensive fallback
        device = _select_device_prefer_gpu(torch.device("cpu"))
        return device, torch.float32
    return param.device, param.dtype


def mse_diff(
    number_of_samples: int,
    mlp_a: MLP,
    mlp_b: MLP,
    *,
    inputs: Optional[torch.Tensor] = None,
) -> float:
    """Estimate the mean squared error between two MLPs on shared inputs."""

    if number_of_samples <= 0:
        raise ValueError("number_of_samples must be a positive integer")
    if not isinstance(mlp_a, MLP) or not isinstance(mlp_b, MLP):
        raise TypeError("mlp_a and mlp_b must be instances of MLP")
    if mlp_a.config.input_dim != mlp_b.config.input_dim:
        raise ValueError("mlp_a and mlp_b must have the same input dimension")

    if inputs is None:
        base_samples = _generate_binary_hypercube_samples(
            number_of_samples,
            mlp_a.config.input_dim,
            device=_select_device_prefer_gpu(torch.device("cpu")),
        )
    else:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a torch.Tensor when provided")
        if inputs.ndim != 2:
            raise ValueError("inputs must be a 2D tensor")
        if inputs.shape[0] != number_of_samples:
            raise ValueError(
                "inputs must contain number_of_samples rows"
            )
        if inputs.shape[1] != mlp_a.config.input_dim:
            raise ValueError("inputs width must match the MLP input dimension")
        base_samples = inputs

    device_a, dtype_a = _model_device_and_dtype(mlp_a)
    device_b, dtype_b = _model_device_and_dtype(mlp_b)

    inputs_a = base_samples.to(device=device_a, dtype=dtype_a)
    inputs_b = base_samples.to(device=device_b, dtype=dtype_b)

    with torch.no_grad():
        outputs_a = mlp_a(inputs_a)
        outputs_b = mlp_b(inputs_b)

    mse = F.mse_loss(outputs_a, outputs_b.to(device_a, dtype=outputs_a.dtype))
    return float(mse.detach().cpu().item())


def _mse_for_threshold(
    model: MLP,
    threshold: float,
    sample_count: int,
    shared_inputs: torch.Tensor,
) -> float:
    sparsified = sparsify_mlp(model, threshold)
    return mse_diff(
        sample_count,
        model,
        sparsified,
        inputs=shared_inputs,
    )


def binary_search_sparsify_threshold(
    model: MLP,
    mse_threshold: float,
    *,
    sample_count: int,
    max_iterations: int = 50,
    relative_tolerance: float = 1e-3,
) -> float:
    """Find a sparsification threshold that respects an MSE tolerance budget."""

    if not isinstance(model, MLP):
        raise TypeError("model must be an instance of MLP")
    if mse_threshold <= 0:
        raise ValueError("mse_threshold must be a positive float")
    if sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be positive")

    max_weight = 0.0
    for layer in model.linear_layers:
        weight = getattr(layer, "weight", None)
        if weight is None:
            continue
        layer_max = float(weight.detach().abs().max().item())
        max_weight = max(max_weight, layer_max)

    if max_weight == 0.0:
        return 0.0

    sample_device = _select_device_prefer_gpu(_model_device_and_dtype(model)[0])
    shared_inputs = _generate_binary_hypercube_samples(
        sample_count,
        model.config.input_dim,
        device=sample_device,
    )

    low = 0.0
    high = max_weight
    best = 0.0

    high_mse = _mse_for_threshold(
        model,
        high,
        sample_count,
        shared_inputs,
    )
    if high_mse <= mse_threshold:
        return high

    for _ in range(max_iterations):
        mid = (low + high) / 2.0
        if math.isclose(low, mid, rel_tol=1e-12, abs_tol=1e-12):
            break
        mid_mse = _mse_for_threshold(
            model,
            mid,
            sample_count,
            shared_inputs,
        )
        if mid_mse <= mse_threshold:
            best = mid
            low = mid
        else:
            high = mid

        if (high - low) / high <= relative_tolerance:
            break

    return best
