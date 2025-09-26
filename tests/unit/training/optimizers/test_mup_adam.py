import pytest
import torch
import torch.nn.functional as F
import pytest
import torch
import torch.nn.functional as F

import src.models.bootstrap  # noqa: F401
from src.models.mlp import MLP
from src.training.optimizers.optimizer_factory import create_optimizer
from src.training.optimizers.configs.adam import AdamConfig


@pytest.fixture
def model(mlp_config):
    return MLP(mlp_config)


def test_step_updates_parameters(model):
    optimizer = create_optimizer(AdamConfig(lr=0.01, mup=True), model)
    x = torch.randn(8, model.config.input_dim)
    y = torch.randn(8, 1)

    before = [p.detach().clone() for p in model.parameters()]

    loss = F.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()

    after = list(model.parameters())
    changed = [not torch.allclose(a, b) for a, b in zip(before, after)]
    assert any(changed), "MuAdam step did not modify model parameters!"
def test_accepts_non_mup_model(mlp_config):
    non_mup_model = MLP(mlp_config)
    cfg = AdamConfig(lr=0.01, mup=True)
    # Should not raise an AssertionError or any exception
    optimizer = create_optimizer(cfg, non_mup_model)
    assert optimizer is not None
