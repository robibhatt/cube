import csv

import torch
import pytest

from src.models.mlp import MLP
from src.models.mlp_config import MLPConfig
from src.mlp_graph.mlp_graph import MlpActivationGraph, NodeKey


def _build_mlp(input_dim: int = 4, hidden_dim: int = 8, output_dim: int = 1) -> MLP:
    cfg = MLPConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=[hidden_dim],
        start_activation=False,
        end_activation=False,
    )
    model = MLP(cfg)
    model.eval()
    return model


def test_graph_replays_mlp_forward_with_mup_scaling(tmp_path):
    torch.manual_seed(42)
    mlp = _build_mlp()
    graph = MlpActivationGraph(mlp, eps=1.0, output_dir=tmp_path)

    inputs = torch.randn(16, mlp.config.input_dim)
    with torch.no_grad():
        expected = mlp(inputs)
    graph_outputs = graph._forward(inputs)[-1]

    assert torch.allclose(graph_outputs, expected, atol=1e-6)


def test_readout_scale_matches_layer_metadata(tmp_path):
    mlp = _build_mlp()
    graph = MlpActivationGraph(mlp, eps=1.0, output_dir=tmp_path)

    readout = mlp.linear_layers[-1]
    expected_scale = pytest.approx(readout.output_mult / readout.width_mult())
    assert graph.layer_input_scales[-1] == expected_scale


def test_activation_csv_matches_sample_average(tmp_path):
    mlp = _build_mlp(input_dim=2, hidden_dim=1, output_dim=1)
    with torch.no_grad():
        first_layer = mlp.linear_layers[0]
        first_layer.weight.copy_(torch.tensor([[1.0, 1.0]]))
        first_layer.bias.zero_()
        readout = mlp.linear_layers[1]
        readout.weight.copy_(torch.tensor([[1.0]]))
        readout.bias.zero_()

    graph = MlpActivationGraph(
        mlp,
        eps=0.0,
        output_dir=tmp_path,
        sample_size=32,
        sample_seed=123,
    )

    generator = torch.Generator(device=torch.device("cpu"))
    generator.manual_seed(123)
    raw = torch.randint(0, 2, (32, mlp.config.input_dim), generator=generator, dtype=torch.int64)
    sample_inputs = (raw * 2 - 1).to(torch.float32)

    with torch.no_grad():
        hidden_activations = graph._forward(sample_inputs)[0][:, 0]

    expected_means = {}
    for coords, value in zip(sample_inputs.int().tolist(), hidden_activations.tolist()):
        key = tuple(int(v) for v in coords)
        expected_means.setdefault(key, []).append(value)
    expected_means = {k: sum(vals) / len(vals) for k, vals in expected_means.items()}

    layer_dir = graph.graph_dir / "layer_01"
    csv_path = layer_dir / NodeKey(1, 0).activation_filename()
    with csv_path.open() as csvfile:
        reader = csv.DictReader(csvfile)
        observed = {
            (int(row["0"]), int(row["1"])): float(row["activation"])
            for row in reader
        }

    assert observed.keys() == expected_means.keys()
    for assignment, mean_value in expected_means.items():
        assert observed[assignment] == pytest.approx(mean_value)
