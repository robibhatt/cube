from dataclasses import replace

import pytest
import torch
from mup import Linear as MuLinear, MuReadout

import src.models.bootstrap  # noqa: F401
from src.models.mlp import MLP
from src.models.mlp_config import MLPConfig
from src.models.mlp_utils import export_neuron_input_gradients
from src.data.noisy_data_provider import NoisyProvider
from src.data.cube_distribution import CubeDistribution
from src.data.cube_distribution_config import CubeDistributionConfig


@pytest.fixture
def basic_config():
    """Basic MLP config with 2 hidden layers."""
    return MLPConfig(
        input_dim=3,
        hidden_dims=[4, 2],
        activation="relu",
        output_dim=1,
        start_activation=False,
        end_activation=False
    )


@pytest.fixture
def model(basic_config):
    """Basic MLP model with 2 hidden layers."""
    return MLP(basic_config)


def test_initialization(basic_config):
    """Model should process input and produce output with configured dimensions."""
    model = MLP(basic_config)

    x = torch.randn(5, basic_config.input_dim)
    y = model(x)

    assert y.shape == (5, basic_config.output_dim)


def test_forward_pass(model):
    """Test that forward pass works and produces correct shapes."""
    batch_size = 5
    x = torch.randn(batch_size, 3)
    y = model(x)
    
    assert y.shape == (batch_size, 1)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


def test_different_activations():
    """Test that different activation functions work."""
    activations = ["relu", "tanh", "sigmoid", "quadratic"]

    for activation in activations:
        config = MLPConfig(
            input_dim=2,
            hidden_dims=[3],
            activation=activation,
            output_dim=1,
            start_activation=False,
            end_activation=False,
        )
        model = MLP(config)

        # Test forward pass
        x = torch.randn(4, 2)
        y = model(x)

        assert y.shape == (4, 1)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()


def test_invalid_activation():
    """Test that invalid activation raises error."""
    config = MLPConfig(
        input_dim=2,
        hidden_dims=[3],
        activation="invalid_activation",
        output_dim=1,
        start_activation=False,
        end_activation=False,
    )

    with pytest.raises(ValueError, match="Unsupported activation"):
        MLP(config)




def test_start_and_end_activation_flags():
    """Verify the optional start‐ and end‐activation layers are inserted correctly."""
    # --- start_activation=True, end_activation=False --------------------------
    cfg_start = MLPConfig(
        input_dim=3,
        hidden_dims=[4, 2],
        activation="relu",
        output_dim=1,
        start_activation=True,
        end_activation=False,
    )
    m_start = MLP(cfg_start)

    # Forward pass should respect configured dimensions
    assert m_start(torch.randn(2, cfg_start.input_dim)).shape == (
        2,
        cfg_start.output_dim,
    )

    # ``end_activation=True`` is no longer supported under μP-only mode.
    cfg_end = MLPConfig(
        input_dim=3,
        hidden_dims=[4, 2],
        activation="relu",
        output_dim=1,
        start_activation=False,
        end_activation=True,
    )
    with pytest.raises(ValueError):
        MLP(cfg_end)

    cfg_both = MLPConfig(
        input_dim=3,
        hidden_dims=[4, 2],
        activation="relu",
        output_dim=1,
        start_activation=True,
        end_activation=True,
    )
    with pytest.raises(ValueError):
        MLP(cfg_both)


def test_mup_initialization_uses_mup_layers(basic_config):
    """MLP should use μP-aware layers by default."""
    model = MLP(basic_config)

    # first hidden layer should be MuLinear and last layer MuReadout
    assert isinstance(model.layers[0 if not basic_config.start_activation else 1], MuLinear)
    assert isinstance(model.layers[-1], MuReadout)


def test_mup_get_base_model(basic_config):
    """``get_base_model`` should return a base-width ``MLP`` configured for μP."""
    model = MLP(basic_config)
    base = model.get_base_model()
    assert isinstance(base, MLP)
    assert base.mup is True
    assert base.config.hidden_dims == [64] * len(basic_config.hidden_dims)


def test_mup_disallows_end_activation(basic_config):
    """MuP mode should reject ``end_activation=True``."""
    cfg = MLPConfig(
        input_dim=basic_config.input_dim,
        hidden_dims=basic_config.hidden_dims,
        activation=basic_config.activation,
        output_dim=basic_config.output_dim,
        start_activation=basic_config.start_activation,
        end_activation=True,
    )
    with pytest.raises(ValueError):
        MLP(cfg)


def test_bias_flag_controls_bias_params():
    """``bias=False`` removes bias terms while the default keeps them."""
    cfg_no_bias = MLPConfig(
        input_dim=3,
        hidden_dims=[4],
        activation="relu",
        output_dim=1,
        start_activation=False,
        end_activation=False,
        bias=False,
    )
    model_no_bias = MLP(cfg_no_bias)
    linear_layers_no_bias = [
        l for l in model_no_bias.layers if isinstance(l, (MuLinear, MuReadout))
    ]
    assert linear_layers_no_bias, "No linear layers found"
    assert all(layer.bias is None for layer in linear_layers_no_bias)

    # default ``bias=True`` should retain bias parameters
    cfg_bias = replace(cfg_no_bias, bias=True)
    model_bias = MLP(cfg_bias)
    linear_layers_bias = [
        l for l in model_bias.layers if isinstance(l, (MuLinear, MuReadout))
    ]
    assert all(layer.bias is not None for layer in linear_layers_bias)


def test_export_neuron_input_gradients_layer_numbering(tmp_path):
    """Layer indices in the exported CSV should start at 1."""
    cfg = MLPConfig(
        input_dim=2,
        hidden_dims=[3],
        activation="relu",
        output_dim=1,
        start_activation=False,
        end_activation=False,
    )
    model = MLP(cfg)

    dist_cfg = CubeDistributionConfig(
        input_dim=2,
        indices_list=[[0]],
        weights=[0.0],
        normalize=False,
        noise_mean=0.0,
        noise_std=0.0,
    )
    distribution = CubeDistribution(dist_cfg, torch.device("cpu"))
    provider = NoisyProvider(
        distribution,
        seed=0,
        dataset_size=4,
        batch_size=4,
    )
    out_path = tmp_path / "grads.csv"
    export_neuron_input_gradients(model, provider, out_path)

    import csv

    with out_path.open() as f:
        rows = list(csv.DictReader(f))
    layers = {int(row["layer"]) for row in rows}
    assert layers == {1, 2}


