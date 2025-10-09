from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Optional

import torch
import torch.nn as nn

from src.models.mlp import MLP


def _get_linear_types() -> tuple[type[nn.Module], ...]:
    """Return the linear module types supported by :class:`MLP`."""

    linear_types: list[type[nn.Module]] = [nn.Linear]
    try:
        from mup import Linear as MuLinear, MuReadout

        linear_types.extend([MuLinear, MuReadout])
    except ImportError:  # pragma: no cover - environment without μP
        pass
    return tuple(linear_types)


LINEAR_TYPES = _get_linear_types()


class FourierMlp(nn.Module):
    """Compute Fourier-weighted neuron statistics from an MLP."""

    def __init__(
        self,
        input_dim: int,
        fourier_indices: Sequence[Sequence[int]],
        mlp: MLP,
        neuron_start_index: int,
        neuron_end_index: int,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        self.input_dim = input_dim
        self.neuron_start_index = neuron_start_index
        self.neuron_end_index = neuron_end_index

        self._validate_index_range()
        self.fourier_indices = [tuple(indices) for indices in fourier_indices]
        self._validate_fourier_indices()

        if not isinstance(mlp, MLP):
            raise TypeError("mlp must be an instance of src.models.mlp.MLP")
        if mlp.config.input_dim != input_dim:
            raise ValueError("mlp input_dim must match the provided input_dim")

        self.mlp = mlp
        self._module_list = list(self.mlp.net)
        self._linear_positions = [
            idx for idx, module in enumerate(self._module_list) if isinstance(module, LINEAR_TYPES)
        ]

        if not self._linear_positions:
            raise ValueError("mlp must contain at least one linear layer")

        if len(self._linear_positions) != len(self.mlp.linear_layers):
            raise RuntimeError("Mismatch between mlp.net linear layers and mlp.linear_layers")

        self._layer_dims = [layer.out_features for layer in self.mlp.linear_layers]
        self._capture_map = self._compute_capture_map()
        self._layer_ranges = [self._compute_layer_range(dim) for dim in self._layer_dims]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _validate_index_range(self) -> None:
        if self.neuron_start_index < 0 or self.neuron_end_index < 0:
            raise ValueError("neuron indices must be non-negative")
        if self.neuron_end_index < self.neuron_start_index:
            raise ValueError("neuron_end_index must be >= neuron_start_index")

    def _validate_fourier_indices(self) -> None:
        for indices in self.fourier_indices:
            if not isinstance(indices, Iterable):
                raise TypeError("fourier_indices must be iterables of integers")
            for idx in indices:
                if not isinstance(idx, int):
                    raise TypeError("fourier index values must be integers")
                if not (0 <= idx < self.input_dim):
                    raise ValueError("fourier index values must be within input_dim range")

    def _compute_capture_map(self) -> dict[int, int]:
        capture_map: dict[int, int] = {}
        last_linear_idx = len(self._linear_positions) - 1
        for layer_idx, pos in enumerate(self._linear_positions):
            if layer_idx == last_linear_idx:
                capture_map[pos] = layer_idx
                continue

            activation_pos = self._find_post_activation_position(pos)
            capture_map[activation_pos] = layer_idx
        return capture_map

    def _find_post_activation_position(self, linear_pos: int) -> int:
        for offset in range(linear_pos + 1, len(self._module_list)):
            module = self._module_list[offset]
            if isinstance(module, LINEAR_TYPES):
                break
            if not isinstance(module, nn.Identity):
                return offset
        raise RuntimeError("Unable to locate activation following a linear layer in the mlp")

    def _compute_layer_range(self, layer_dim: int) -> tuple[int, int]:
        start = min(layer_dim, self.neuron_start_index)
        end = min(layer_dim, self.neuron_end_index)
        if end < start:
            end = start
        return start, end

    def _compute_fourier_products(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.fourier_indices:
            return None

        products = []
        for indices in self.fourier_indices:
            if len(indices) == 0:
                products.append(torch.ones(x.size(0), dtype=x.dtype, device=x.device))
            else:
                products.append(x[:, list(indices)].prod(dim=1))
        return torch.stack(products, dim=1)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> tuple[Optional[torch.Tensor], ...]:
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError("Input tensor must have shape (batch_size, input_dim)")

        fourier_products = self._compute_fourier_products(x)

        captures: list[Optional[torch.Tensor]] = [None] * len(self._layer_ranges)
        current = x

        for idx, module in enumerate(self._module_list):
            current = module(current)
            capture_idx = self._capture_map.get(idx)
            if capture_idx is not None:
                captures[capture_idx] = current

        if any(capture is None for capture in captures):
            raise RuntimeError("Failed to capture outputs for all layers")

        results: list[Optional[torch.Tensor]] = []
        for output, (start, end) in zip(captures, self._layer_ranges):
            assert output is not None  # for type checkers
            if end <= start:
                results.append(None)
                continue

            neuron_values = output[:, start:end]
            if fourier_products is None:
                results.append(neuron_values.new_empty((neuron_values.size(1), 0)))
                continue

            combined = neuron_values.unsqueeze(2) * fourier_products.unsqueeze(1)
            results.append(combined.mean(dim=0))

        return tuple(results)
