"""Utility functions for working with :mod:`src.models.mlp`."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from mup import Linear as MuLinear, MuReadout
from src.models.mlp import MLP

def visualize(mlp: MLP, path: Path, threshold: float = 0.001) -> None:
    """Visualize the network architecture and save as a PNG image."""

    path.mkdir(parents=True, exist_ok=True)
    save_path = path / "visualization.png"

    layer_dims = [
        mlp.config.input_dim,
        *mlp.config.hidden_dims,
        mlp.config.output_dim,
    ]

    num_layers = len(layer_dims)

    y_spacing = 1.0
    fig_height = num_layers * y_spacing
    fig_width = 2 * fig_height

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    orders: list[list[int]] = []
    for _, dim in enumerate(layer_dims):
        order = list(range(dim))
        orders.append(order)

    active: list[list[bool]] = []
    for l_idx, dim in enumerate(layer_dims):
        flags = [False] * dim
        if l_idx > 0:
            incoming = mlp.linear_layers[l_idx - 1].weight.detach().abs().cpu()
            if incoming.numel() > 0:
                flags_in = (incoming >= threshold).any(dim=1).tolist()
                for i, flag in enumerate(flags_in):
                    flags[i] = flags[i] or bool(flag)
        if l_idx < len(mlp.linear_layers):
            outgoing = mlp.linear_layers[l_idx].weight.detach().abs().cpu()
            if outgoing.numel() > 0:
                flags_out = (outgoing >= threshold).any(dim=0).tolist()
                for i, flag in enumerate(flags_out):
                    flags[i] = flags[i] or bool(flag)
        active.append(flags)

    positions: list[dict[int, tuple[float, float]]] = []
    for l_idx, dim in enumerate(layer_dims):
        y = (num_layers - l_idx - 1) * y_spacing
        order = orders[l_idx]
        flags = active[l_idx]

        pos_dict: dict[int, tuple[float, float]] = {}

        layer_span = fig_width if dim > 1 else 0.0
        start_x = -layer_span / 2.0

        if l_idx == 0:
            spacing = layer_span / (dim - 1) if dim > 1 else 0.0
            for j, idx in enumerate(order):
                x = start_x + j * spacing
                pos_dict[idx] = (x, y)
                ax.scatter(x, y, s=50, color="black")
            positions.append(pos_dict)
            continue

        active_count = sum(1 for i in order if flags[i])
        inactive_count = dim - active_count

        if active_count and inactive_count:
            active_span = layer_span / 3.0
            inactive_span = layer_span * 2.0 / 3.0
        elif active_count:
            active_span = layer_span
            inactive_span = 0.0
        else:
            active_span = 0.0
            inactive_span = layer_span

        start_inactive = start_x + active_span

        active_spacing = (
            active_span / (active_count - 1) if active_count > 1 else 0.0
        )
        inactive_spacing = (
            inactive_span / (inactive_count - 1) if inactive_count > 1 else 0.0
        )

        next_active_x = start_x
        next_inactive_x = start_inactive

        for idx in order:
            if flags[idx]:
                if active_count > 1:
                    x = next_active_x
                    next_active_x += active_spacing
                else:
                    x = start_x + active_span / 2.0
            else:
                if inactive_count > 1:
                    x = next_inactive_x
                    next_inactive_x += inactive_spacing
                else:
                    x = start_inactive + inactive_span / 2.0
            pos_dict[idx] = (x, y)
            ax.scatter(x, y, s=50, color="black")

        positions.append(pos_dict)

    for l_idx, layer in enumerate(mlp.linear_layers):
        weights = layer.weight.detach().abs().cpu()
        if weights.numel() == 0:
            continue
        max_w = weights.max().item()
        if max_w <= 0:
            continue
        for out_idx in range(weights.shape[0]):
            for in_idx in range(weights.shape[1]):
                w = weights[out_idx, in_idx].item()
                if w < threshold:
                    continue
                norm = w / max_w
                color = (1.0, 1.0 - norm, 1.0 - norm)
                x1, y1 = positions[l_idx][in_idx]
                x2, y2 = positions[l_idx + 1][out_idx]
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=1.0)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
