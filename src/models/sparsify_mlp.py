"""Utilities for pruning MLP weights by thresholding."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Iterable, List

import torch
import torch.nn.functional as F

from src.models.mlp import MLP
from src.models.mlp_config import MLPConfig


def _zero_small_weights(linear_layers: Iterable[torch.nn.Module], threshold: float) -> None:
    """Zero out weights whose absolute value falls below ``threshold``.

    Parameters
    ----------
    linear_layers:
        Iterable of linear layers (expected to expose a ``weight`` tensor) whose
        weights will be modified in-place.
    threshold:
        Positive float representing the sparsification threshold.
    """
    with torch.no_grad():
        for layer in linear_layers:
            weight = getattr(layer, "weight", None)
            if weight is None:
                continue
            abs_weight = weight.detach().abs()
            mask = abs_weight < threshold
            weight.data.masked_fill_(mask, 0.0)


def sparsify_mlp(model: MLP, threshold: float) -> MLP:
    """Return a copy of ``model`` with small weights zeroed out.

    The sparsification is performed independently for every linear layer in the
    network. A new model instance is returned to avoid mutating the original
    ``model``.

    Parameters
    ----------
    model:
        The :class:`~src.models.mlp.MLP` instance to sparsify.
    threshold:
        Positive float. Weights whose absolute value is strictly smaller than
        this threshold are set to zero.

    Returns
    -------
    MLP
        A deep copy of ``model`` whose weights have been sparsified.

    Raises
    ------
    TypeError
        If ``model`` is not an instance of :class:`~src.models.mlp.MLP`.
    ValueError
        If ``threshold`` is not strictly positive.
    """
    if not isinstance(model, MLP):
        raise TypeError("model must be an instance of MLP")
    if threshold <= 0:
        raise ValueError("threshold must be a positive float")

    config_copy = deepcopy(model.config)
    sparsified = MLP(config_copy)
    sparsified.load_state_dict(model.state_dict())
    _zero_small_weights(sparsified.linear_layers, threshold)
    return sparsified


def prune(model: MLP) -> MLP:
    """Remove neurons that are not path connected to both inputs and output.

    Parameters
    ----------
    model:
        A sparsified :class:`~src.models.mlp.MLP` instance whose zero weights
        define the connectivity graph used for pruning.

    Returns
    -------
    MLP
        A new MLP containing only the neurons that participate in at least one
        path from an input neuron to the output neuron. All preserved weights
        and biases are copied from ``model``.
    """

    if not isinstance(model, MLP):
        raise TypeError("model must be an instance of MLP")

    linear_layers = list(model.linear_layers)
    if not linear_layers:
        config_copy = deepcopy(model.config)
        pruned = MLP(config_copy)
        pruned.load_state_dict(model.state_dict())
        return pruned

    input_dim = model.config.input_dim
    n_layers = len(linear_layers)
    n_hidden = n_layers - 1

    forward_masks: List[torch.Tensor] = []
    prev_mask = torch.ones(input_dim, dtype=torch.bool)
    for idx in range(n_hidden):
        weight = linear_layers[idx].weight.detach().to(torch.device("cpu"))
        if weight.shape[1] != prev_mask.numel():
            raise RuntimeError("incompatible layer dimensions while pruning")
        has_conn = weight != 0
        if prev_mask.any():
            active = has_conn[:, prev_mask].any(dim=1)
        else:
            active = torch.zeros(weight.shape[0], dtype=torch.bool)
        forward_masks.append(active)
        prev_mask = active

    backward_masks: List[torch.Tensor] = [torch.zeros_like(mask) for mask in forward_masks]
    next_mask = torch.ones(linear_layers[-1].weight.shape[0], dtype=torch.bool)
    for idx in range(n_layers - 1, 0, -1):
        weight = linear_layers[idx].weight.detach().to(torch.device("cpu"))
        if next_mask.numel() != weight.shape[0]:
            raise RuntimeError("incompatible layer dimensions while pruning")
        if next_mask.any():
            active_inputs = (weight[next_mask, :] != 0).any(dim=0)
        else:
            active_inputs = torch.zeros(weight.shape[1], dtype=torch.bool)
        if idx - 1 < len(backward_masks):
            backward_masks[idx - 1] = active_inputs
        next_mask = active_inputs

    kept_indices: List[List[int]] = []
    for fwd_mask, bwd_mask in zip(forward_masks, backward_masks):
        combined = fwd_mask & bwd_mask
        indices = torch.nonzero(combined, as_tuple=False).flatten().tolist()
        kept_indices.append(indices)

    layer_mapping = [idx for idx, indices in enumerate(kept_indices) if indices]
    filtered_indices = [kept_indices[idx] for idx in layer_mapping]
    new_hidden_dims = [len(indices) for indices in filtered_indices]

    new_config = MLPConfig(input_dim=input_dim, hidden_dims=new_hidden_dims)

    pruned_model = MLP(new_config)

    try:
        sample_param = next(model.parameters())
        device = sample_param.device
        dtype = sample_param.dtype
    except StopIteration:
        device = torch.device("cpu")
        dtype = torch.float32

    pruned_model.to(device=device, dtype=dtype)

    with torch.no_grad():
        prev_indices = list(range(input_dim))
        new_layer_position = 0
        for old_idx, indices in zip(layer_mapping, filtered_indices):
            old_layer = linear_layers[old_idx]
            new_layer = pruned_model.linear_layers[new_layer_position]

            row_idx = torch.tensor(indices, device=old_layer.weight.device)
            col_idx = torch.tensor(prev_indices, device=old_layer.weight.device)

            weight_slice = old_layer.weight.detach().index_select(0, row_idx).index_select(1, col_idx)
            new_layer.weight.data.copy_(weight_slice.to(device=device, dtype=dtype))

            if old_layer.bias is not None and new_layer.bias is not None:
                bias_slice = old_layer.bias.detach().index_select(0, row_idx)
                new_layer.bias.data.copy_(bias_slice.to(device=device, dtype=dtype))
            elif new_layer.bias is not None:
                new_layer.bias.data.zero_()

            prev_indices = indices
            new_layer_position += 1

        old_out_layer = linear_layers[-1]
        new_out_layer = pruned_model.linear_layers[-1]

        if filtered_indices:
            col_idx = torch.tensor(prev_indices, device=old_out_layer.weight.device)
            weight_slice = old_out_layer.weight.detach().index_select(1, col_idx)
        else:
            if model.config.hidden_dims:
                weight_shape = (old_out_layer.weight.shape[0], pruned_model.config.input_dim)
                weight_slice = torch.zeros(weight_shape, device=device, dtype=dtype)
            else:
                col_idx = torch.arange(pruned_model.config.input_dim, device=old_out_layer.weight.device)
                weight_slice = old_out_layer.weight.detach().index_select(1, col_idx)

        new_out_layer.weight.data.copy_(weight_slice.to(device=device, dtype=dtype))

        if old_out_layer.bias is not None and new_out_layer.bias is not None:
            new_out_layer.bias.data.copy_(old_out_layer.bias.detach().to(device=device, dtype=dtype))
        elif new_out_layer.bias is not None:
            new_out_layer.bias.data.zero_()

    return pruned_model


def mse_diff(number_of_samples: int, mlp_a: MLP, mlp_b: MLP) -> float:
    """Estimate the mean squared error between two MLPs on hypercube samples.

    Parameters
    ----------
    number_of_samples:
        Positive integer indicating how many points to draw from the
        hypercube. Each coordinate of a sample is independently chosen from
        ``{-1, 1}``.
    mlp_a, mlp_b:
        :class:`~src.models.mlp.MLP` instances that will be evaluated. Both
        networks must share the same input dimension.

    Returns
    -------
    float
        The mean squared error between the outputs of ``mlp_a`` and ``mlp_b``
        averaged over the sampled points.

    Raises
    ------
    TypeError
        If either model is not an instance of :class:`~src.models.mlp.MLP`.
    ValueError
        If ``number_of_samples`` is not positive or if the models do not share
        the same input dimension.
    """

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
    """Find the largest sparsification threshold that preserves a target MSE.

    Parameters
    ----------
    model:
        The :class:`~src.models.mlp.MLP` instance whose weights should be
        sparsified.
    mse_threshold:
        Maximum acceptable mean-squared error between the original model and
        its sparsified counterpart. Must be strictly positive.
    sample_multiplier:
        Positive constant that controls how many random samples are used when
        estimating the mean-squared error. The number of samples is computed as
        ``ceil(sample_multiplier / mse_threshold)`` and bounded below by 16 to
        provide a reasonably accurate estimate. The value can be tuned by
        callers but defaults to a conservative ``64.0``.
    tolerance_ratio:
        Relative difference between the upper and lower bounds of the binary
        search interval at which the algorithm stops and returns the best known
        feasible threshold.
    max_iterations:
        Maximum number of binary search iterations performed as an additional
        safeguard against infinite loops.

    Returns
    -------
    float
        The largest threshold ``t`` (up to the configured tolerance) such that
        sparsifying ``model`` with ``t`` produces an MLP whose mean-squared
        difference from the original model does not exceed ``mse_threshold``.

    Raises
    ------
    TypeError
        If ``model`` is not an instance of :class:`~src.models.mlp.MLP`.
    ValueError
        If ``mse_threshold`` is not strictly positive or if
        ``sample_multiplier`` is non-positive.
    """

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
        # Expand the search interval in an attempt to find a threshold that
        # violates the constraint. If this never happens we can safely return
        # the last tested value because sparsifying with larger thresholds will
        # not change the model any further.
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
