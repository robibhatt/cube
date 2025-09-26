import pytest
from src.training.optimizers.configs.sgd import SgdConfig
from src.models.mlp_config import MLPConfig
from src.training.optimizers.configs.adam import AdamConfig


@pytest.fixture
def mlp_config():
    return MLPConfig(
        input_dim=3,
        hidden_dims=[4, 2],
        activation="relu",
        output_dim=1,
        start_activation=False,
        end_activation=False,
    )


@pytest.fixture
def adam_config():
    return AdamConfig(lr=2.5)

@pytest.fixture
def sgd_config():
    return SgdConfig(lr=2.5)
