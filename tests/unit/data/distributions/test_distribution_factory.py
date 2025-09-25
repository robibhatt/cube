import pytest
import torch

from src.data.joint_distributions import create_joint_distribution
from src.data.joint_distributions.configs.gaussian import GaussianConfig
from src.data.joint_distributions.configs.base import JointDistributionConfig
from dataclasses import dataclass, field


def test_create_distribution_gaussian():
    cfg = GaussianConfig(input_dim=2, mean=0.0, std=1.0)
    dist = create_joint_distribution(cfg, torch.device("cpu"))
    assert dist.config.distribution_type == "Gaussian"
    assert dist.mean.device == torch.device("cpu")


@dataclass
class DummyConfig(JointDistributionConfig):
    input_dim: int = field(default=1)

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.distribution_type = "Nonexistent"


def test_create_distribution_invalid_type():
    cfg = DummyConfig(input_dim=2)
    with pytest.raises(ValueError):
        create_joint_distribution(cfg, torch.device("cpu"))
