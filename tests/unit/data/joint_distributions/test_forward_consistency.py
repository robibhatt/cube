import pytest
import torch

from src.data.joint_distributions import create_joint_distribution
from src.data.joint_distributions.configs.gaussian import GaussianConfig
from src.data.joint_distributions.configs.mapped_joint_distribution import (
    MappedJointDistributionConfig,
)

from tests.helpers.stubs import StubJointDistribution
from tests.unit.data.conftest import DummyJointDistribution
from src.models.targets.configs.sum_prod import SumProdTargetConfig


def test_gaussian_forward_matches_forward_X():
    cfg = GaussianConfig(input_dim=2, mean=0.0, std=1.0)
    dist = create_joint_distribution(cfg, torch.device("cpu"))
    X, _ = dist.sample(5, seed=0)
    X_fwd, _ = dist.forward(X)
    X_fx = dist.forward_X(X)
    assert torch.allclose(X_fwd, X_fx)


def test_mapped_forward_matches_forward_X():
    base_cfg = GaussianConfig(input_dim=2, mean=0.0, std=1.0)
    target_cfg = SumProdTargetConfig(
        input_shape=torch.Size([2]),
        indices_list=[[0], [1]],
        weights=[1.0, 1.0],
        normalize=False,
    )
    cfg = MappedJointDistributionConfig(
        distribution_config=base_cfg,
        target_function_config=target_cfg,
    )
    dist = create_joint_distribution(cfg, torch.device("cpu"))
    X, _ = dist.sample(3, seed=0)
    X_fwd, _ = dist.forward(X)
    X_fx = dist.forward_X(X)
    assert torch.allclose(X_fwd, X_fx)
