import pytest
import torch
from mup import Linear as MuLinear, MuReadout

from src.models.mlp import MLP
from src.models.mlp_config import MLPConfig


@pytest.fixture
def basic_config():
    """Basic MLP config with 2 hidden layers."""
    return MLPConfig(
        input_dim=3,
        hidden_dims=[4, 2],
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


def test_start_and_end_activation_flags():
    """Verify the optional start‐ and end‐activation layers are inserted correctly."""
    # --- start_activation=True, end_activation=False --------------------------
    cfg_start = MLPConfig(
        input_dim=3,
        hidden_dims=[4, 2],
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
        output_dim=1,
        start_activation=False,
        end_activation=True,
    )
    with pytest.raises(ValueError):
        MLP(cfg_end)

    cfg_both = MLPConfig(
        input_dim=3,
        hidden_dims=[4, 2],
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
        output_dim=basic_config.output_dim,
        start_activation=basic_config.start_activation,
        end_activation=True,
    )
    with pytest.raises(ValueError):
        MLP(cfg)


def test_linear_layers_include_bias_parameters(basic_config):
    """Every μP linear layer should include a bias parameter."""
    model = MLP(basic_config)
    linear_layers = [
        layer for layer in model.layers if isinstance(layer, (MuLinear, MuReadout))
    ]
    assert linear_layers, "No linear layers found"
    assert all(layer.bias is not None for layer in linear_layers)


