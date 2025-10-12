"""Fourier statistics for :class:`~src.models.mlp.MLP` activations."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from src.models.mlp import MLP


class FourierMlpModule(torch.nn.Module):
    """Estimate correlations between neuron activations and Fourier features.

    The module wraps an :class:`~src.models.mlp.MLP` instance and exposes a
    ``forward`` method that returns, for every linear layer in the network, the
    average correlation between selected neurons and a collection of Fourier
    basis functions.  Correlations are computed over the mini-batch presented in
    the forward pass.
    """

    def __init__(
        self,
        *,
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
        self.mlp = mlp
        self.neuron_start_index = max(0, neuron_start_index)
        self.neuron_end_index = max(self.neuron_start_index, neuron_end_index)

        self.fourier_indices = [tuple(indices) for indices in fourier_indices]
        for indices in self.fourier_indices:
            for idx in indices:
                if not (0 <= idx < self.input_dim):
                    raise ValueError("fourier index values must be within input_dim range")

        if mlp.config.input_dim != input_dim:
            raise ValueError("mlp input_dim must match the provided input_dim")

        self._module_sequence = list(self.mlp.net)
        self._capture_positions = self._infer_capture_positions()
        self._layer_ranges = self._compute_layer_ranges()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _infer_capture_positions(self) -> list[int]:
        """Locate the module outputs that should be captured for each layer."""

        positions: list[int] = []
        for layer_idx, layer in enumerate(self.mlp.linear_layers):
            try:
                linear_pos = self._module_sequence.index(layer)
            except ValueError as exc:  # pragma: no cover - unexpected mismatch
                raise RuntimeError("MLP linear layer not found in sequential net") from exc

            if layer_idx == len(self.mlp.linear_layers) - 1:
                positions.append(linear_pos)
                continue

            if linear_pos + 1 >= len(self._module_sequence):
                raise RuntimeError("Expected activation after hidden linear layer")
            positions.append(linear_pos + 1)
        return positions

    def _compute_layer_ranges(self) -> list[tuple[int, int]]:
        """Determine neuron slices to include for each layer."""

        ranges: list[tuple[int, int]] = []
        for layer in self.mlp.linear_layers:
            dim = layer.out_features
            start = min(dim, self.neuron_start_index)
            end = min(dim, self.neuron_end_index)
            if end < start:
                end = start
            ranges.append((start, end))
        return ranges

    def _compute_fourier_products(self, x: torch.Tensor) -> torch.Tensor | None:
        if not self.fourier_indices:
            return None

        products = []
        for indices in self.fourier_indices:
            if not indices:
                products.append(torch.ones(x.size(0), dtype=x.dtype, device=x.device))
            else:
                products.append(x[:, list(indices)].prod(dim=1))
        return torch.stack(products, dim=1)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """Return Fourier correlations for each MLP layer.

        Parameters
        ----------
        x:
            A batch of inputs with shape ``(batch_size, input_dim)``.

        Returns
        -------
        tuple of tensors
            One entry per linear layer in the wrapped MLP.  Each tensor contains
            the mean correlation between the selected neurons of that layer and
            the configured Fourier basis functions.  Layers whose neuron slice is
            empty return ``None``.
        """

        if x.ndim != 2 or x.size(1) != self.input_dim:
            raise ValueError("Input tensor must have shape (batch_size, input_dim)")

        fourier_products = self._compute_fourier_products(x)

        activations: list[torch.Tensor] = []
        current = x
        for module in self._module_sequence:
            current = module(current)
            activations.append(current)

        outputs: list[torch.Tensor | None] = []
        for capture_pos, (start, end) in zip(self._capture_positions, self._layer_ranges):
            layer_values = activations[capture_pos]
            if end <= start:
                outputs.append(None)
                continue

            neuron_values = layer_values[:, start:end]
            if fourier_products is None:
                outputs.append(neuron_values.new_empty((neuron_values.size(1), 0)))
                continue

            combined = neuron_values.unsqueeze(2) * fourier_products.unsqueeze(1)
            outputs.append(combined.mean(dim=0))

        return tuple(outputs)
