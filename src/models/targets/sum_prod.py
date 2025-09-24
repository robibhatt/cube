from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models.targets.target_function_registry import register_target_function
from src.models.targets.configs.sum_prod import SumProdTargetConfig


@register_target_function("SumProdTarget")
class SumProdTarget(nn.Module):
    """Compute a weighted sum of product terms over selected coordinates."""

    def __init__(self, config: SumProdTargetConfig):
        super().__init__()
        self.config = config
        self.input_shape = config.input_shape
        self.output_shape = config.output_shape
        self.indices_list = [list(group) for group in config.indices_list]
        self.weights = [float(weight) for weight in config.weights]
        self.normalize = bool(config.normalize)
        if len(self.indices_list) != len(self.weights):
            raise ValueError("weights length must match indices_list length")
        self.max_index = max(max(group) for group in self.indices_list)
        weights_tensor = torch.tensor(self.weights, dtype=torch.float32)
        self.register_buffer("weights_tensor", weights_tensor)
        self.scale = self._compute_normalization() if self.normalize else 1.0

    def forward(self, x: Tensor) -> Tensor:
        self._validate_input(x)
        return self._forward(x)

    def _forward(self, X: Tensor) -> Tensor:
        flat = X.reshape(*X.shape[:-len(self.input_shape)], -1)
        if flat.shape[-1] <= self.max_index:
            raise ValueError(
                "SumProdTarget expects at least "
                f"{self.max_index + 1} elements in input_shape, but got {flat.shape[-1]}"
            )
        weights = self.weights_tensor.to(flat)
        total = torch.zeros_like(flat[..., 0])
        for weight, group in zip(weights.unbind(), self.indices_list):
            total = total + weight * flat[..., group].prod(dim=-1)
        total = total * self.scale
        return total.unsqueeze(-1)

    def initialize_weights(self) -> None:  # pragma: no cover - default no-op
        """Initialize model parameters."""

    def get_base_model(self) -> SumProdTarget | None:  # pragma: no cover - default no-op
        """Return a base-width reference model if available."""
        return None

    def _compute_normalization(self) -> float:
        num_samples = 2048
        g = torch.Generator()
        g.manual_seed(0)
        X = torch.randint(0, 2, (num_samples, *self.input_shape), generator=g)
        X = X.to(torch.float32) * 2 - 1
        flat = X.view(num_samples, -1)
        total = torch.zeros(num_samples, dtype=flat.dtype)
        for weight, group in zip(self.weights_tensor.unbind(), self.indices_list):
            total = total + weight * flat[:, group].prod(dim=-1)
        var = total.var(unbiased=False)
        if var > 0:
            return float(1.0 / torch.sqrt(var))
        else:
            return 1.0

    def _validate_input(self, x: Tensor) -> None:
        expected_rank = len(self.input_shape)

        if x.ndim <= expected_rank:
            raise ValueError(
                "Input must include at least one batch dimension. "
                f"Expected tensor of rank > {expected_rank} (batch + {self.input_shape}), "
                f"but got rank {x.ndim}."
            )

        if x.shape[-expected_rank:] != self.input_shape:
            raise ValueError(
                "Input shape mismatch. Expected trailing dimensions "
                f"{self.input_shape}, but got {tuple(x.shape)}."
            )

    def __str__(self) -> str:
        return (
            "SumProdTarget("
            f"input_shape={tuple(self.input_shape)}, "
            f"indices_list={self.indices_list}, "
            f"weights={self.weights}, "
            f"normalize={self.normalize})"
        )
