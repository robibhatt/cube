import torch.nn as nn

from src.training.optimizers.optimizer_factory import create_optimizer
from src.training.optimizers.configs.adam import AdamConfig


class NoParamModel(nn.Module):
    def forward(self, x):  # pragma: no cover - simple passthrough
        return x


def test_optimizer_with_no_parameters_is_noop():
    model = NoParamModel()
    optimizer = create_optimizer(AdamConfig(lr=0.1), model)

    # Should be able to call zero_grad and step without errors
    optimizer.stepper.zero_grad()
    optimizer.step()

    assert optimizer.stepper.state_dict() == {}
