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
from conftest import LinearTargetFunctionConfig


@pytest.mark.parametrize("dist_name", ["gaussian", "mapped", "representor"])
def test_sample_equivalence_base_forward(dist_name, model_representor, trained_trainer):
    device = torch.device("cpu")
    seed = 123
    n = 4

    if dist_name == "gaussian":
        cfg = GaussianConfig(input_shape=torch.Size([2]), mean=0.0, std=1.0)
        dist = create_joint_distribution(cfg, device)
    elif dist_name == "mapped":
        base_cfg = GaussianConfig(input_shape=torch.Size([2]), mean=0.0, std=1.0)
        target_cfg = LinearTargetFunctionConfig(input_shape=torch.Size([2]))
        cfg = MappedJointDistributionConfig(
            distribution_config=base_cfg,
            target_function_config=target_cfg,
        )
        dist = create_joint_distribution(cfg, device)
    elif dist_name == "representor":
        cfg = RepresentorDistributionConfig(
            base_distribution_config=trained_trainer.config.joint_distribution_config,
            model_config=trained_trainer.config.model_config,
            checkpoint_dir=trained_trainer.checkpoint_dir,
            from_rep=0,
            to_rep=1,
        )
        dist = create_joint_distribution(cfg, device)
    else:
        raise ValueError(f"Unknown distribution {dist_name}")

    X_base, _ = dist.base_sample(n, seed=seed)
    X_fwd, y_fwd = dist.forward(X_base)

    X_direct, y_direct = dist.sample(n, seed=seed)

    assert torch.allclose(X_fwd, X_direct)
    assert torch.allclose(y_fwd, y_direct)
