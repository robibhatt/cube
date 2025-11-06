"""Utilities for pruning and visualizing sparsified multilayer perceptrons."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from itertools import product
import random
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import torch
import torch.nn.functional as F

from src.models.mlp import MLP
from src.models.mlp_config import MLPConfig


MAX_EXACT_ANCESTOR_COUNT = 7
MAX_ANCESTOR_ASSIGNMENT_SAMPLES = 128
MAX_NEURONS_PER_LAYER = 64


def _node_basename(layer_index: int, neuron_index: int) -> str:
    return f"layer_{layer_index:02d}_neuron_{neuron_index:03d}"


def _assignment_from_index(index: int, length: int) -> Tuple[int, ...]:
    return tuple(
        1 if (index >> shift) & 1 else -1 for shift in reversed(range(length))
    )


def _sample_binary_assignments(
    num_variables: int, max_samples: int, rng: random.Random
) -> List[Tuple[int, ...]]:
    total_combinations = 1 << num_variables
    sample_count = min(max_samples, total_combinations)
    if sample_count <= 0:
        return []

    if sample_count == total_combinations:
        return [_assignment_from_index(idx, num_variables) for idx in range(sample_count)]

    assignments: Set[Tuple[int, ...]] = set()
    while len(assignments) < sample_count:
        choice = rng.randrange(total_combinations)
        assignments.add(_assignment_from_index(choice, num_variables))

    return sorted(assignments)


def prune(model: MLP) -> Tuple[MLP, Set[int]]:
    """Remove neurons that are not path connected to both inputs and output."""

    if not isinstance(model, MLP):
        raise TypeError("model must be an instance of MLP")

    linear_layers = list(model.linear_layers)
    if not linear_layers:
        config_copy = deepcopy(model.config)
        setattr(config_copy, "exact_base_shapes", True)
        pruned = MLP(config_copy)
        pruned.load_state_dict(model.state_dict())
        return pruned, set()

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

    new_config = deepcopy(model.config)
    new_config.input_dim = input_dim
    new_config.hidden_dims = new_hidden_dims
    setattr(new_config, "exact_base_shapes", True)

    pruned_model = MLP(new_config)

    output_scale: float | None = None
    try:
        original_readout = linear_layers[-1]
    except IndexError:
        original_readout = None
    if original_readout is not None and hasattr(original_readout, "width_mult"):
        width_mult = float(original_readout.width_mult())
        if width_mult != 0.0 and hasattr(original_readout, "output_mult"):
            output_scale = float(original_readout.output_mult) / width_mult

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

        if (
            output_scale is not None
            and hasattr(new_out_layer, "width_mult")
            and hasattr(new_out_layer, "output_mult")
        ):
            new_width_mult = float(new_out_layer.width_mult())
            if new_width_mult != 0.0:
                new_out_layer.output_mult = output_scale * new_width_mult

    connected_inputs: Set[int]
    if forward_masks:
        first_layer_weight = linear_layers[0].weight.detach().to(torch.device("cpu"))
        first_combined = forward_masks[0] & backward_masks[0]
        if first_combined.any():
            active_inputs = (first_layer_weight[first_combined, :] != 0).any(dim=0)
        else:
            active_inputs = torch.zeros(first_layer_weight.shape[1], dtype=torch.bool)
    else:
        output_weight = linear_layers[0].weight.detach().to(torch.device("cpu"))
        active_inputs = (output_weight != 0).any(dim=0)

    connected_inputs = {idx for idx, is_active in enumerate(active_inputs.tolist()) if is_active}

    return pruned_model, connected_inputs


def visualize_pruned_mlp(
    pruned_mlp: MLP,
    active_inputs: Iterable[int],
    output_dir: Path | str,
) -> None:
    """Serialize connectivity and activations for a pruned MLP."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    linear_layers = list(pruned_mlp.linear_layers)
    if not linear_layers:
        return

    def _effective_readout_scale(layer: torch.nn.Module) -> float:
        """Return the scale factor that maps μP readout weights to real outputs."""

        width_mult_fn = getattr(layer, "width_mult", None)
        if callable(width_mult_fn):
            width_mult = float(width_mult_fn())
            if width_mult != 0.0:
                output_mult = getattr(layer, "output_mult", 1.0)
                if isinstance(output_mult, torch.Tensor):
                    output_mult = float(output_mult.detach().cpu())
                else:
                    output_mult = float(output_mult)
                return output_mult / width_mult
        return 1.0

    weight_tensors: List[torch.Tensor] = []
    bias_tensors: List[torch.Tensor | None] = []

    for idx, layer in enumerate(linear_layers):
        weight = layer.weight.detach().cpu().clone()

        if idx == len(linear_layers) - 1:
            scale = _effective_readout_scale(layer)
            if scale != 1.0:
                weight.mul_(scale)

        bias = layer.bias.detach().cpu().clone() if layer.bias is not None else None

        weight_tensors.append(weight)
        bias_tensors.append(bias)

    input_dim = pruned_mlp.config.input_dim
    dtype = weight_tensors[0].dtype
    active_input_set: Set[int] = {int(idx) for idx in active_inputs}

    layer_connections: List[Dict[int, List[int]]] = []
    layer_ancestors: List[Dict[int, Set[int]]] = []

    for layer_idx, weight in enumerate(weight_tensors, start=1):
        layer_conn: Dict[int, List[int]] = {}
        layer_anc: Dict[int, Set[int]] = {}
        for neuron_idx in range(weight.size(0)):
            parent_indices = [
                int(parent)
                for parent in torch.nonzero(weight[neuron_idx], as_tuple=False)
                .flatten()
                .tolist()
            ]
            layer_conn[neuron_idx] = parent_indices

            if layer_idx == 1:
                candidates = set(parent_indices)
                if active_input_set:
                    ancestors = candidates & active_input_set
                else:
                    ancestors = candidates
                if not ancestors and parent_indices:
                    ancestors = candidates
            else:
                prev_ancestors = layer_ancestors[layer_idx - 2]
                ancestors = set()
                for parent in parent_indices:
                    ancestors.update(prev_ancestors.get(parent, set()))
            layer_anc[neuron_idx] = ancestors
        layer_connections.append(layer_conn)
        layer_ancestors.append(layer_anc)

    def forward(inputs: torch.Tensor) -> List[torch.Tensor]:
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        if inputs.dim() != 2:
            raise ValueError("Expected inputs to be a 1D or 2D tensor")

        prev = inputs.to(dtype=dtype)
        activations: List[torch.Tensor] = []
        for layer_idx, (weight, bias) in enumerate(zip(weight_tensors, bias_tensors), start=1):
            pre_act = F.linear(prev, weight, bias)
            if layer_idx == len(weight_tensors):
                activations.append(pre_act)
                prev = pre_act
            else:
                post_act = F.relu(pre_act)
                activations.append(post_act)
                prev = post_act
        return [tensor.squeeze(0) for tensor in activations]

    for layer_idx, weight in enumerate(weight_tensors, start=1):
        layer_dir = output_root / f"layer_{layer_idx:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        neuron_indices = list(range(weight.size(0)))
        if len(neuron_indices) > MAX_NEURONS_PER_LAYER:
            neuron_rng = random.Random(f"layer-{layer_idx}-neuron-sample")
            neuron_indices = sorted(
                neuron_rng.sample(neuron_indices, MAX_NEURONS_PER_LAYER)
            )

        for neuron_idx in neuron_indices:
            parents = layer_connections[layer_idx - 1][neuron_idx]
            ancestors = sorted(layer_ancestors[layer_idx - 1][neuron_idx])

            csv_path = layer_dir / f"{_node_basename(layer_idx, neuron_idx)}_activations.csv"

            if ancestors:
                fieldnames = [str(idx) for idx in ancestors] + ["activation"]
                if len(ancestors) >= MAX_EXACT_ANCESTOR_COUNT:
                    assignment_rng = random.Random(
                        f"layer-{layer_idx}-neuron-{neuron_idx}-assignments"
                    )
                    assignments = _sample_binary_assignments(
                        len(ancestors),
                        MAX_ANCESTOR_ASSIGNMENT_SAMPLES,
                        assignment_rng,
                    )
                else:
                    assignments = list(product([-1, 1], repeat=len(ancestors)))
            else:
                fieldnames = ["activation"]
                assignments = [tuple()]

            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for assignment in assignments:
                    input_vector = torch.zeros(input_dim, dtype=dtype)
                    for value, ancestor_idx in zip(assignment, ancestors):
                        input_vector[ancestor_idx] = float(value)

                    with torch.no_grad():
                        activations = forward(input_vector)
                        activation_value = float(activations[layer_idx - 1][neuron_idx].item())

                    row = {str(idx): int(val) for idx, val in zip(ancestors, assignment)}
                    row["activation"] = activation_value
                    writer.writerow(row)

            node_data = {
                "layer_index": layer_idx,
                "neuron_index": neuron_idx,
                "parents": parents,
                "ancestors": ancestors,
                "activations_csv": csv_path.name,
            }

            node_file = layer_dir / f"{_node_basename(layer_idx, neuron_idx)}.json"
            with open(node_file, "w", encoding="utf-8") as f:
                json.dump(node_data, f, indent=2)
