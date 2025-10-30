import pytest
from src.training.sgd_config import SgdConfig
from src.models.mlp_config import MLPConfig


@pytest.fixture
def mlp_config():
    return MLPConfig(
        input_dim=3,
        hidden_dims=[4, 2],
    )


@pytest.fixture
def sgd_config():
    return SgdConfig(lr=2.5)
