import torch
import pytest

import src.models.bootstrap  # noqa: F401
from src.models.mlp import MLP
from src.training.sgd import Sgd
from src.training.sgd_config import SgdConfig


def test_sgd_instantiation(mlp_config):
    opt = Sgd(SgdConfig(lr=0.1, weight_decay=0.01), MLP(mlp_config))
    assert isinstance(opt.stepper, torch.optim.SGD)
    assert opt.stepper.defaults["momentum"] == 0
    assert opt.stepper.defaults["weight_decay"] == pytest.approx(0.01)

    # ensure biases are excluded from weight decay
    bias_param = next(
        p for n, p in opt.model.named_parameters() if n.endswith("bias")
    )
    group = next(
        pg for pg in opt.stepper.param_groups if any(bias_param is p for p in pg["params"])
    )
    assert group["weight_decay"] == pytest.approx(0.0)
