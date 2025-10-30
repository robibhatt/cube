"""Utilities for pruning MLP weights by thresholding."""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import torch
import torch.nn.functional as F

from src.models.mlp import MLP


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
            mask = weight.abs() < threshold
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

    sparsified = deepcopy(model)
    _zero_small_weights(sparsified.linear_layers, threshold)
    return sparsified


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
