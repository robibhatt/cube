import pytest
import torch
from pathlib import Path
from src.models.architectures.model_factory import create_model
import pytest
import torch

import src.models.bootstrap  # noqa: F401
from src.models.architectures.model_factory import create_model
from src.models.architectures.configs.mlp import MLPConfig


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
    return create_model(basic_config)


def test_initialization(basic_config):
    """Model factory should produce a working model for the given config."""
    model = create_model(basic_config)

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
            end_activation=False
        )
        model = create_model(config)

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
        end_activation=False
    )

    with pytest.raises(ValueError, match="Unsupported activation"):
        create_model(config)

