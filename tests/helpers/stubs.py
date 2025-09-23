from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.optim import Optimizer as TorchOptimizer

from src.data.joint_distributions.joint_distribution import JointDistribution
from src.data.joint_distributions.configs.base import JointDistributionConfig
from src.data.joint_distributions.joint_distribution_registry import (
    register_joint_distribution,
)
from src.data.joint_distributions.configs.joint_distribution_config_registry import (
    register_joint_distribution_config,
)
from dataclasses import dataclass, field
from dataclasses_json import config

class StubModel(nn.Module):
    """Tiny nn.Module whose forward pass is configurable."""
    def __init__(self, forward_fn=None):
        super().__init__()
        self.forward_fn = forward_fn or (lambda x: torch.zeros(x.size(0), 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.forward_fn(x)


class StubTargetFunction(nn.Module):
    """Tiny target function with a configurable forward pass."""

    def __init__(self, forward_fn=None):
        super().__init__()
        self.forward_fn = forward_fn or (lambda x: torch.zeros(x.size(0), 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.forward_fn(x)


class _StubStepper:
    def __init__(self):
        self.param_groups = [{"lr": 0.0}]
        self._state = {}
        self._step_count = 0

    def zero_grad(self):
        pass

    def step(self):
        self._step_count += 1

    def state_dict(self):
        return {"step_count": self._step_count}

    def load_state_dict(self, state):
        self._step_count = state.get("step_count", 0)


class StubOptimizer:
    """Lightweight optimizer exposing a ``step`` via ``stepper`` attribute."""
    def __init__(self, model: nn.Module):
        self.model = model
        self.config = None
        self.stepper = _StubStepper()

    def initialize(self):
        pass

    def step(self):
        self.stepper.step()


@register_joint_distribution("StubJointDistribution")
class StubJointDistribution(JointDistribution):
    """Joint distribution returning predetermined tensors."""

    @dataclass
    @register_joint_distribution_config("StubJointDistribution")
    class _Config(JointDistributionConfig):
        X: torch.Tensor = field(
            metadata=config(encoder=lambda t: t.tolist(), decoder=lambda v: torch.tensor(v))
        )
        y: torch.Tensor = field(
            metadata=config(encoder=lambda t: t.tolist(), decoder=lambda v: torch.tensor(v))
        )
        input_shape: torch.Size = field(init=False)
        output_shape: torch.Size = field(init=False)

        def __post_init__(self) -> None:  # type: ignore[override]
            self.input_shape = self.X.shape[1:]
            self.output_shape = self.y.shape[1:]
            self.distribution_type = "StubJointDistribution"

    def __init__(self, config: _Config, device: torch.device):
        assert config.X.size(0) == config.y.size(0)
        super().__init__(config, device)
        self._X = config.X.float().to(device)
        self._y = config.y.float().to(device)

    def sample(self, n_samples: int, seed: int):
        if n_samples > self._X.size(0):
            raise ValueError("Requested more samples than available")
        return (
            self._X[:n_samples].clone(),
            self._y[:n_samples].clone(),
        )

    def __str__(self) -> str:
        return "StubJointDistribution"

    def preferred_provider(self) -> str:
        """Always use the in-memory :class:`TensorDataProvider`."""
        return "TensorDataProvider"

    # Methods required by ``JointDistribution``
    def base_sample(self, n_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sample(n_samples, seed)

    def forward(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = X.size(0)
        repeat_factor = -(-batch_size // self._y.size(0))
        y = self._y.repeat(repeat_factor, *([1] * (self._y.dim() - 1)))
        return X, y[:batch_size].clone()

    def forward_X(
        self, X: torch.Tensor
    ) -> torch.Tensor:
        return X

