import torch
from dataclasses import dataclass, field
from typing import Tuple


from src.data.joint_distributions import (
    register_joint_distribution,
    create_joint_distribution,
)
from src.data.joint_distributions.joint_distribution import JointDistribution
from src.data.joint_distributions.configs.base import JointDistributionConfig



def test_register_and_create_joint_distribution():
    @register_joint_distribution("DummyJoint")
    class DummyJoint(JointDistribution):
        @dataclass
        class _Config(JointDistributionConfig):
            input_dim: int = field(default=1)

            def __post_init__(self) -> None:
                if self.input_dim <= 0:
                    raise ValueError("input_dim must be positive")
                self.distribution_type = "DummyJoint"

        def __init__(self, config: _Config, device: torch.device) -> None:
            super().__init__(config, device)

        def sample(self, n_samples: int, seed: int):
            X = torch.zeros(n_samples, self.input_dim, device=self.device)
            y = torch.zeros(n_samples, self.output_dim, device=self.device)
            return X, y

        def __str__(self) -> str:
            return "DummyJoint"

        def base_sample(self, n_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.sample(n_samples, seed)

        def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            return X, torch.zeros(X.size(0), 1, device=self.device)

        def forward_X(self, X: torch.Tensor) -> torch.Tensor:
            return X

        def preferred_provider(self) -> str:
            return "TensorDataProvider"

    device = torch.device("cpu")
    cfg = DummyJoint._Config()
    joint = create_joint_distribution(cfg, device)
    assert isinstance(joint, DummyJoint)
    assert joint.device == device
