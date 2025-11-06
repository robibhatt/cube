from __future__ import annotations

import csv
from pathlib import Path

import torch

from src.models.mlp import MLP
from src.models.mlp_config import MLPConfig
from src.models.pruned_mlp import prune, visualize_pruned_mlp


def _load_activation_rows(csv_path: Path) -> list[dict[str, float]]:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, float]] = []
        for row in reader:
            rows.append({key: float(value) for key, value in row.items()})
    return rows


def test_visualize_pruned_mlp_emits_scaled_readout(tmp_path):
    config = MLPConfig(input_dim=2, hidden_dims=[3])
    model = MLP(config)

    with torch.no_grad():
        for layer in model.linear_layers:
            layer.weight.fill_(0.25)
            layer.bias.fill_(0.1)
        # Introduce a non-trivial μP scaling so the readout needs adjustment.
        model.linear_layers[-1].output_mult = torch.tensor(0.5)

    pruned_model, active_inputs = prune(model)

    output_dir = tmp_path / "viz"
    visualize_pruned_mlp(pruned_model, active_inputs, output_dir)

    final_layer_dir = output_dir / "layer_02"
    csv_files = sorted(final_layer_dir.glob("*_activations.csv"))
    assert csv_files, "expected activation CSV files for the readout layer"

    dtype = pruned_model.linear_layers[0].weight.dtype

    for csv_path in csv_files:
        rows = _load_activation_rows(csv_path)
        assert rows, "activation CSV should contain rows"

        for row in rows:
            activation = row.pop("activation")
            input_vector = torch.zeros(pruned_model.config.input_dim, dtype=dtype)
            for index_str, value in row.items():
                input_vector[int(index_str)] = value

            with torch.no_grad():
                expected = pruned_model(input_vector.unsqueeze(0)).squeeze(0).squeeze(-1)

            assert torch.isclose(torch.tensor(activation, dtype=dtype), expected, atol=1e-6)
