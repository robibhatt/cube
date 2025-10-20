import torch
import pytest

from src.models.mlp import MLP
from src.models.mlp_config import MLPConfig
from src.mlp_graph.mlp_graph import MlpActivationGraph


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
