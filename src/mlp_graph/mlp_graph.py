"""Graph serialization utilities for trained MLP models."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import torch

from src.models.activations import ACTIVATION_MAP
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

    def __init__(
        self,
        mlp: MLP,
        eps: float,
        output_dir: Path | str,
        *,
        graph_name: Optional[str] = None,
    ) -> None:
        self.mlp = mlp
        self.eps = float(eps)
        self.output_root = Path(output_dir)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.graph_dir = self._create_graph_directory(graph_name)

        self.linear_layers = list(self.mlp.linear_layers)
        if not self.linear_layers:
            raise ValueError("The provided MLP does not contain any linear layers")

        self.activation = ACTIVATION_MAP[self.mlp.config.activation]()
        self.weight_tensors = [layer.weight.detach().cpu().clone() for layer in self.linear_layers]
        self.bias_tensors = [
            layer.bias.detach().cpu().clone() if layer.bias is not None else None
            for layer in self.linear_layers
        ]
        self.input_dim = self.mlp.config.input_dim
        self.num_layers = len(self.linear_layers)
        self.activation_batch_size = 1024

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
        with torch.no_grad():
            for layer_idx, (weight, bias) in enumerate(
                zip(self.weight_tensors, self.bias_tensors),
                start=1,
            ):
                layer_dir = self.graph_dir / f"layer_{layer_idx:02d}"
                layer_dir.mkdir(parents=True, exist_ok=True)

                for neuron_idx in range(weight.size(0)):
                    ancestors = sorted(
                        self.layer_ancestors[layer_idx - 1].get(neuron_idx, set())
                    )
                    if not ancestors:
                        continue
                    parents = sorted(self.layer_connections[layer_idx - 1][neuron_idx])
                    activations_file = self._evaluate_node(
                        layer_idx,
                        neuron_idx,
                        ancestors,
                        dtype=dtype,
                        layer_dir=layer_dir,
                    )
                    node_data = {
                        "layer_index": layer_idx,
                        "neuron_index": neuron_idx,
                        "parents": parents,
                        "ancestors": ancestors,
                        "activations_csv": activations_file,
                    }
                    node_file = layer_dir / NodeKey(layer_idx, neuron_idx).filename()
                    with open(node_file, "w", encoding="utf-8") as f:
                        json.dump(node_data, f, indent=2)

    def _evaluate_node(
        self,
        layer_index: int,
        neuron_index: int,
        ancestors: Sequence[int],
        *,
        dtype: torch.dtype,
        layer_dir: Path,
    ) -> str:
        csv_path = layer_dir / NodeKey(layer_index, neuron_index).activation_filename()
        fieldnames = [str(idx) for idx in ancestors] + ["activation"]
        non_ancestors = self._non_ancestor_indices(ancestors)

        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for assignment in product([-1, 1], repeat=len(ancestors)):
                averaged_value = self._average_activation(
                    layer_index,
                    neuron_index,
                    ancestors,
                    assignment,
                    non_ancestors,
                    dtype=dtype,
                )
                row = {str(idx): int(value) for idx, value in zip(ancestors, assignment)}
                row["activation"] = averaged_value
                writer.writerow(row)
        return csv_path.name

    def _average_activation(
        self,
        layer_index: int,
        neuron_index: int,
        ancestors: Sequence[int],
        assignment: Sequence[int],
        non_ancestors: Sequence[int],
        *,
        dtype: torch.dtype,
    ) -> float:
        input_batch = self._sample_input_batch(ancestors, assignment, non_ancestors, dtype=dtype)
        layer_outputs = self._forward(input_batch)
        target_values = layer_outputs[layer_index - 1][:, neuron_index]
        return float(target_values.mean().item())

    def _non_ancestor_indices(self, ancestors: Iterable[int]) -> List[int]:
        ancestor_set = set(ancestors)
        return [idx for idx in range(self.input_dim) if idx not in ancestor_set]

    def _sample_input_batch(
        self,
        ancestors: Sequence[int],
        assignment: Sequence[int],
        non_ancestors: Sequence[int],
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch = torch.zeros((self.activation_batch_size, self.input_dim), dtype=dtype)
        for idx, value in zip(ancestors, assignment):
            batch[:, idx] = float(value)
        if non_ancestors:
            random_signs = torch.randint(
                0,
                2,
                (self.activation_batch_size, len(non_ancestors)),
            )
            random_signs = random_signs.to(dtype=dtype) * 2 - 1
            batch[:, non_ancestors] = random_signs
        return batch

    def _forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        if inputs.dim() == 1:
            outputs = self._forward(inputs.unsqueeze(0))
            return [tensor.squeeze(0) for tensor in outputs]
        if inputs.dim() != 2:
            raise ValueError("Expected inputs to be a 1D or 2D tensor")

        activations: List[torch.Tensor] = []
        prev = inputs
        for layer_idx, (weight, bias) in enumerate(
            zip(self.weight_tensors, self.bias_tensors),
            start=1,
        ):
            pre_activation = prev @ weight.t()
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
