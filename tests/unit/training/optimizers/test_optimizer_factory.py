import pytest

import src.models.bootstrap  # noqa: F401
from src.models.mlp import MLP
from src.training.optimizers.optimizer_factory import create_optimizer


@pytest.fixture
def model(mlp_config):
    """Basic MLP model with 2 hidden layers."""
    return MLP(mlp_config)


@pytest.fixture
def optimizer(adam_config, mlp_config):
    """Optimizer instance bound to a freshly constructed MLP."""
    model = MLP(mlp_config)
    return create_optimizer(adam_config, model)


def test_initialization(mlp_config):
    """Test that model initializes with correct architecture."""
    model = MLP(mlp_config)

    assert len(model.net) == 5
    assert model.net[0].in_features == 3
    assert model.net[0].out_features == 4
    assert model.net[2].in_features == 4
    assert model.net[2].out_features == 2
    assert model.net[4].in_features == 2
    assert model.net[4].out_features == 1


def test_initialization_2(mlp_config, adam_config):
    """Test that the optimizer picks up the learning rate from the config."""
    model = MLP(mlp_config)
    optimizer = create_optimizer(adam_config, model)

    assert optimizer.config.lr == pytest.approx(2.5)
