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

def test_mup_initialization_uses_mup_layers(basic_config):
    """MLP should use μP-aware layers by default."""
    model = MLP(basic_config)

    # first hidden layer should be MuLinear and last layer MuReadout
    assert isinstance(model.layers[0], MuLinear)
    assert isinstance(model.layers[-1], MuReadout)


def test_mup_get_base_model(basic_config):
    """``get_base_model`` should return a base-width ``MLP`` configured for μP."""
    model = MLP(basic_config)
    base = model.get_base_model()
    assert isinstance(base, MLP)
    assert base.mup is True
    assert base.config.hidden_dims == [64] * len(basic_config.hidden_dims)


def test_mup_get_base_model_respects_exact_base_shapes():
    config = MLPConfig(input_dim=5, hidden_dims=[3, 7, 5], exact_base_shapes=True)
    model = MLP(config)

    base = model.get_base_model()

    assert isinstance(base, MLP)
    assert base.config.hidden_dims == config.hidden_dims


def test_linear_layers_include_bias_parameters(basic_config):
    """Every μP linear layer should include a bias parameter."""
    model = MLP(basic_config)
    linear_layers = [
        layer for layer in model.layers if isinstance(layer, (MuLinear, MuReadout))
    ]
    assert linear_layers, "No linear layers found"
    assert all(layer.bias is not None for layer in linear_layers)


def test_mup_sanity_scale_guardrail_toggle():
    """Scale guardrail can be bypassed when explicitly requested."""
    config = MLPConfig(input_dim=2, hidden_dims=[2, 2, 2])
    model = MLP(config)

    with torch.no_grad():
        model.linear_layers[0].weight.fill_(0.0)
        pattern = torch.tensor(
            [[1.0, -1.0], [-1.0, 1.0]],
            dtype=model.linear_layers[1].weight.dtype,
            device=model.linear_layers[1].weight.device,
        )
        model.linear_layers[1].weight.copy_(pattern)
        model.linear_layers[2].weight.copy_(pattern)

    with pytest.raises(RuntimeError):
        model._mup_post_init_sanity_check(skip_scale_check=False)

    # Should not raise when the scale check is disabled
    model._mup_post_init_sanity_check(skip_scale_check=True)


def test_exact_base_shapes_skip_scale_guardrail(monkeypatch):
    """Custom base shapes should skip the scale guardrail during init."""

    config = MLPConfig(input_dim=4, hidden_dims=[5, 3], exact_base_shapes=True)

    calls: dict[str, bool] = {}
    original = MLP._mup_post_init_sanity_check

    def spy(self, **kwargs):
        calls.update(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(MLP, "_mup_post_init_sanity_check", spy)

    MLP(config)

    assert calls.get("skip_scale_check") is True


