"""Graph serialization utilities for trained MLP models."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn

from mup import MuReadout

from src.data.cube_distribution import CubeDistribution
from src.data.cube_distribution_config import CubeDistributionConfig
from src.models.mlp import MLP


@dataclass(frozen=True)
class NodeKey:
    """Identifier for a neuron within the network."""

    layer_index: int
    neuron_index: int

    def filename(self) -> str:
        return f"layer_{self.layer_index:02d}_neuron_{self.neuron_index:03d}.json"

    def activation_filename(self) -> str:
        return f"layer_{self.layer_index:02d}_neuron_{self.neuron_index:03d}_activations.csv"


class MlpActivationGraph:
    """Construct and serialize an activation graph derived from a trained MLP.

    Parameters
    ----------
    mlp:
        The trained multi-layer perceptron.
    eps:
        Threshold for determining whether an edge exists between two neurons.
        We treat a connection as present when the absolute weight is greater
        than or equal to ``eps``.
    output_dir:
        Directory in which a subdirectory for this graph will be created.
    graph_name:
        Optional name for the created subdirectory.  When omitted, a name based
        on the current timestamp is used.
    """

    DEFAULT_SAMPLE_SIZE = 1024

    def __init__(
        self,
        mlp: MLP,
        eps: float,
        output_dir: Path | str,
        *,
        graph_name: Optional[str] = None,
        cube_distribution_config: Optional[CubeDistributionConfig] = None,
        sample_size: Optional[int] = None,
        sample_seed: Optional[int] = None,
    ) -> None:
        self.mlp = mlp
        self.eps = float(eps)
        self.output_root = Path(output_dir)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.graph_dir = self._create_graph_directory(graph_name)

        self.linear_layers = list(self.mlp.linear_layers)
        if not self.linear_layers:
            raise ValueError("The provided MLP does not contain any linear layers")

        self.activation = nn.ReLU()
        self.weight_tensors = [layer.weight.detach().cpu().clone() for layer in self.linear_layers]
        self.bias_tensors = [
            layer.bias.detach().cpu().clone() if layer.bias is not None else None
            for layer in self.linear_layers
        ]
        self.input_dim = self.mlp.config.input_dim
        self.num_layers = len(self.linear_layers)
        self.sample_size = int(sample_size) if sample_size is not None else self.DEFAULT_SAMPLE_SIZE
        if self.sample_size <= 0:
            raise ValueError("sample_size must be a positive integer")
        self.sample_seed = int(sample_seed) if sample_seed is not None else 0
        self.device = self.weight_tensors[0].device
        self.cube_distribution: Optional[CubeDistribution] = None
        if cube_distribution_config is not None:
            self.cube_distribution = CubeDistribution(cube_distribution_config, device=self.device)

        # μP models apply an input scaling before the readout linear layer.
        # Persist the per-layer scale so that our standalone forward pass
        # mirrors the live module exactly, regardless of the current width.
        self.layer_input_scales = [
            float(layer.output_mult / layer.width_mult()) if isinstance(layer, MuReadout) else 1.0
            for layer in self.linear_layers
        ]

        self.layer_connections: List[Dict[int, Set[int]]] = []
        self.layer_ancestors: List[Dict[int, Set[int]]] = []

        self._build_connections()
        self._compute_ancestors()
        self._serialize_node_activations()

    # ------------------------------------------------------------------
    # directory helpers
    # ------------------------------------------------------------------
    def _create_graph_directory(self, graph_name: Optional[str]) -> Path:
        base_name = graph_name or f"mlp_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        candidate = self.output_root / base_name
        counter = 1
        while candidate.exists():
            candidate = self.output_root / f"{base_name}_{counter}"
            counter += 1
        candidate.mkdir(parents=False, exist_ok=False)
        return candidate

    # ------------------------------------------------------------------
    # graph construction
    # ------------------------------------------------------------------
    def _build_connections(self) -> None:
        prev_width = self.input_dim
        for layer_idx, weight in enumerate(self.weight_tensors, start=1):
            layer_connections: Dict[int, Set[int]] = {}
            for neuron_idx in range(weight.size(0)):
                connections: Set[int] = set()
                for parent_idx in range(prev_width):
                    w = weight[neuron_idx, parent_idx].item()
                    if abs(w) >= self.eps:
                        connections.add(parent_idx)
                layer_connections[neuron_idx] = connections
            self.layer_connections.append(layer_connections)
            prev_width = weight.size(0)

    def _compute_ancestors(self) -> None:
        for layer_idx, layer_connections in enumerate(self.layer_connections):
            if layer_idx == 0:
                ancestors = {
                    neuron_idx: set(parents)
                    for neuron_idx, parents in layer_connections.items()
                }
            else:
                prev_ancestors = self.layer_ancestors[layer_idx - 1]
                ancestors = {}
                for neuron_idx, parents in layer_connections.items():
                    ancestor_set: Set[int] = set()
                    for parent in parents:
                        parent_anc = prev_ancestors.get(parent)
                        if parent_anc is None:
                            continue
                        ancestor_set.update(parent_anc)
                    ancestors[neuron_idx] = ancestor_set
            self.layer_ancestors.append(ancestors)

    # ------------------------------------------------------------------
    # evaluation and serialization
    # ------------------------------------------------------------------
    def _serialize_node_activations(self) -> None:
        dtype = self.weight_tensors[0].dtype
        sample_inputs = self._draw_input_sample(dtype=dtype)
        with torch.no_grad():
            layer_outputs = self._forward(sample_inputs)
        activation_stats = self._collect_activation_statistics(sample_inputs, layer_outputs)

        for layer_idx, weight in enumerate(self.weight_tensors, start=1):
            layer_dir = self.graph_dir / f"layer_{layer_idx:02d}"
            layer_dir.mkdir(parents=True, exist_ok=True)

            for neuron_idx in range(weight.size(0)):
                ancestors = sorted(self.layer_ancestors[layer_idx - 1].get(neuron_idx, set()))
                if not ancestors:
                    continue
                parent_connections = sorted(self.layer_connections[layer_idx - 1][neuron_idx])
                stats = activation_stats[layer_idx - 1].get(neuron_idx, {})
                if not stats:
                    continue

                csv_path = layer_dir / NodeKey(layer_idx, neuron_idx).activation_filename()
                fieldnames = [str(idx) for idx in ancestors] + ["activation"]
                with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for assignment in sorted(stats):
                        row = {str(idx): int(value) for idx, value in zip(ancestors, assignment)}
                        row["activation"] = stats[assignment]
                        writer.writerow(row)

                node_data = {
                    "layer_index": layer_idx,
                    "neuron_index": neuron_idx,
                    "parents": parent_connections,
                    "ancestors": ancestors,
                    "activations_csv": csv_path.name,
                }
                node_file = layer_dir / NodeKey(layer_idx, neuron_idx).filename()
                with open(node_file, "w", encoding="utf-8") as f:
                    json.dump(node_data, f, indent=2)

    def _draw_input_sample(self, *, dtype: torch.dtype) -> torch.Tensor:
        if self.cube_distribution is not None:
            inputs, _ = self.cube_distribution.base_sample(self.sample_size, self.sample_seed)
            return inputs.to(dtype=dtype).cpu()

        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.sample_seed)
        raw = torch.randint(
            0,
            2,
            (self.sample_size, self.input_dim),
            dtype=torch.int64,
            generator=generator,
            device=self.device,
        )
        inputs = raw * 2 - 1
        return inputs.to(dtype=dtype, device="cpu")

    def _collect_activation_statistics(
        self,
        sample_inputs: torch.Tensor,
        layer_outputs: Sequence[torch.Tensor],
    ) -> List[Dict[int, Dict[Tuple[int, ...], float]]]:
        input_assignments = sample_inputs.to(torch.int8)
        stats_per_layer: List[Dict[int, Dict[Tuple[int, ...], float]]] = []

        for layer_idx, activations in enumerate(layer_outputs, start=1):
            activations_cpu = activations.detach().cpu()
            layer_stats: Dict[int, Dict[Tuple[int, ...], float]] = {}
            for neuron_idx in range(activations_cpu.size(1)):
                ancestors = sorted(self.layer_ancestors[layer_idx - 1].get(neuron_idx, set()))
                if not ancestors:
                    continue
                ancestor_values = input_assignments[:, ancestors]
                if ancestor_values.numel() == 0:
                    continue

                unique_assignments, inverse_indices = torch.unique(
                    ancestor_values,
                    dim=0,
                    return_inverse=True,
                )

                values = activations_cpu[:, neuron_idx].to(torch.float64)
                sums = torch.zeros(unique_assignments.size(0), dtype=torch.float64)
                counts = torch.zeros(unique_assignments.size(0), dtype=torch.int64)
                sums.scatter_add_(0, inverse_indices, values)
                ones = torch.ones_like(inverse_indices, dtype=torch.int64)
                counts.scatter_add_(0, inverse_indices, ones)

                neuron_stats: Dict[Tuple[int, ...], float] = {}
                for idx in range(unique_assignments.size(0)):
                    count = int(counts[idx].item())
                    if count == 0:
                        continue
                    assignment = tuple(int(v.item()) for v in unique_assignments[idx])
                    mean_value = float((sums[idx] / count).item())
                    neuron_stats[assignment] = mean_value

                if neuron_stats:
                    layer_stats[neuron_idx] = neuron_stats

            stats_per_layer.append(layer_stats)

        return stats_per_layer

    def _forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        if inputs.dim() == 1:
            outputs = self._forward(inputs.unsqueeze(0))
            return [tensor.squeeze(0) for tensor in outputs]
        if inputs.dim() != 2:
            raise ValueError("Expected inputs to be a 1D or 2D tensor")

        activations: List[torch.Tensor] = []
        prev = inputs
        for layer_idx, (weight, bias, input_scale) in enumerate(
            zip(self.weight_tensors, self.bias_tensors, self.layer_input_scales),
            start=1,
        ):
            pre_activation = prev @ weight.t()
            if input_scale != 1.0:
                pre_activation = pre_activation * input_scale
            if bias is not None:
                pre_activation = pre_activation + bias
            if layer_idx == self.num_layers:
                activations.append(pre_activation)
                prev = pre_activation
            else:
                activated = self.activation(pre_activation)
                activations.append(activated)
                prev = activated
        return activations
