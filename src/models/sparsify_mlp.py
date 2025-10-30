"""Utilities for pruning MLP weights by thresholding."""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import torch

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
