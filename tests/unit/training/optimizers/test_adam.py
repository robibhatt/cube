import pytest
import torch
import torch.nn.functional as F

import src.models.bootstrap  # noqa: F401
from src.models.architectures.mlp import MLP
from src.training.optimizers.optimizer_factory import create_optimizer
from src.training.optimizers.configs.adam import AdamConfig



# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def model(mlp_config):
    return MLP(mlp_config)


@pytest.fixture
def optimizer(model, adam_config):
    return create_optimizer(adam_config, model)


# -----------------------------------------------------------------------------
# Existing sanity checks
# -----------------------------------------------------------------------------

def test_initialization(mlp_config):
    """Verify the MLP layer shapes are as expected."""
    model = MLP(mlp_config)

    # Net layout: Linear‑ReLU‑Linear‑ReLU‑Linear
    assert len(model.net) == 5
    assert model.net[0].in_features == 3 and model.net[0].out_features == 4
    assert model.net[2].in_features == 4 and model.net[2].out_features == 2
    assert model.net[4].in_features == 2 and model.net[4].out_features == 1


def test_initialization_2(mlp_config, adam_config):
    model = MLP(mlp_config)
    optimizer = create_optimizer(adam_config, model)
    assert optimizer.config.lr == pytest.approx(2.5)


# -----------------------------------------------------------------------------
# Added tests
# -----------------------------------------------------------------------------

def test_step_updates_parameters(mlp_config, adam_config):
    """After a backward/step cycle at least one parameter tensor should change."""
    torch.manual_seed(0)
    model = MLP(mlp_config)
    optimizer = create_optimizer(adam_config, model)

    x = torch.randn(8, mlp_config.input_dim)
    y = torch.randn(8, 1)

    before = [p.detach().clone() for p in model.parameters()]

    loss = F.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()

    after = list(model.parameters())
    changed = [not torch.allclose(a, b) for a, b in zip(before, after)]
    assert any(changed), "Adam.step() did not modify model parameters!"


def test_lr_is_passed_through(mlp_config):
    """Ensure the LR specified in the config reaches the underlying torch optimiser."""
    cfg = AdamConfig(lr=0.123)
    opt = create_optimizer(cfg, MLP(mlp_config))
    assert isinstance(opt.stepper, torch.optim.Adam)
    assert opt.stepper.param_groups[0]["lr"] == pytest.approx(0.123)


def test_weight_decay_is_passed_through(mlp_config):
    """Ensure weight decay reaches the torch optimiser."""
    cfg = AdamConfig(lr=0.01, weight_decay=0.2)
    opt = create_optimizer(cfg, MLP(mlp_config))
    assert isinstance(opt.stepper, torch.optim.Adam)
    assert opt.stepper.param_groups[0]["weight_decay"] == pytest.approx(0.2)

    # ensure bias parameters receive no weight decay
    bias_param = next(
        p for n, p in opt.model.named_parameters() if n.endswith("bias")
    )
    group = next(
        pg for pg in opt.stepper.param_groups if any(bias_param is p for p in pg["params"])
    )
    assert group["weight_decay"] == pytest.approx(0.0)

