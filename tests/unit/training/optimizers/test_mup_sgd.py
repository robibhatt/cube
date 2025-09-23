import pytest
import torch
import torch.nn.functional as F

import src.models.bootstrap  # noqa: F401
from src.models.architectures.model_factory import create_model
from src.training.optimizers.optimizer_factory import create_optimizer
from src.training.optimizers.configs.sgd import SgdConfig


@pytest.fixture
def model(mlp_config):
    return create_model(mlp_config)


def test_step_updates_parameters(model):
    optimizer = create_optimizer(SgdConfig(lr=0.01, mup=True), model)
    x = torch.randn(8, model.config.input_dim)
    y = torch.randn(8, 1)

    before = [p.detach().clone() for p in model.parameters()]

    loss = F.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()

    after = list(model.parameters())
    changed = [not torch.allclose(a, b) for a, b in zip(before, after)]
    assert any(changed), "MuSGD step did not modify model parameters!"
