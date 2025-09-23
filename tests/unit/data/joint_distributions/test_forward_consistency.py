import torch
import pytest

from src.data.joint_distributions import create_joint_distribution
from src.data.joint_distributions.configs.gaussian import GaussianConfig
from src.data.joint_distributions.configs.mapped_joint_distribution import (
    MappedJointDistributionConfig,
)
from src.data.joint_distributions.configs.representor_distribution import (
    RepresentorDistributionConfig,
)

from tests.helpers.stubs import StubJointDistribution
from tests.unit.data.conftest import (
    DummyJointDistribution,
    AddOneNoiseDistributionConfig,
)
from conftest import LinearTargetFunctionConfig


def test_gaussian_forward_matches_forward_X():
    cfg = GaussianConfig(input_shape=torch.Size([2]), mean=0.0, std=1.0)
    dist = create_joint_distribution(cfg, torch.device("cpu"))
    X, _ = dist.sample(5, seed=0)
    X_fwd, _ = dist.forward(X)
    X_fx = dist.forward_X(X)
    assert torch.allclose(X_fwd, X_fx)


def test_mapped_forward_matches_forward_X():
    base_cfg = GaussianConfig(input_shape=torch.Size([2]), mean=0.0, std=1.0)
    target_cfg = LinearTargetFunctionConfig(input_shape=torch.Size([2]))
    cfg = MappedJointDistributionConfig(
        distribution_config=base_cfg,
        target_function_config=target_cfg,
    )
    dist = create_joint_distribution(cfg, torch.device("cpu"))
    X, _ = dist.sample(3, seed=0)
    X_fwd, _ = dist.forward(X)
    X_fx = dist.forward_X(X)
    assert torch.allclose(X_fwd, X_fx)


def test_representor_forward_matches_forward_X(trained_trainer):
    cfg = RepresentorDistributionConfig(
        base_distribution_config=trained_trainer.config.joint_distribution_config,
        model_config=trained_trainer.config.model_config,
        checkpoint_dir=trained_trainer.checkpoint_dir,
        from_rep=0,
        to_rep=1,
    )
    dist = create_joint_distribution(cfg, torch.device("cpu"))
    X, _ = dist.base_sample(3, seed=0)
    X_fwd, _ = dist.forward(X)
    X_fx = dist.forward_X(X)
    assert torch.allclose(X_fwd, X_fx)
