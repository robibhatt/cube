import torch
import pytest

from src.data.joint_distributions import create_joint_distribution
from src.data.joint_distributions.configs.gaussian import GaussianConfig
from src.data.joint_distributions.configs.mapped_joint_distribution import (
    MappedJointDistributionConfig,
)

from tests.helpers.stubs import StubJointDistribution
from src.models.targets.configs.sum_prod import SumProdTargetConfig


@pytest.mark.parametrize("dist_name", ["gaussian", "mapped"])
def test_sample_equivalence_base_forward(dist_name, trained_trainer):
    device = torch.device("cpu")
    seed = 123
    n = 4

    if dist_name == "gaussian":
        cfg = GaussianConfig(input_shape=torch.Size([2]), mean=0.0, std=1.0)
        dist = create_joint_distribution(cfg, device)
    elif dist_name == "mapped":
        base_cfg = GaussianConfig(input_shape=torch.Size([2]), mean=0.0, std=1.0)
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
        dist = create_joint_distribution(cfg, device)
    else:
        raise ValueError(f"Unknown distribution {dist_name}")

    X_base, _ = dist.base_sample(n, seed=seed)
    X_fwd, y_fwd = dist.forward(X_base)

    X_direct, y_direct = dist.sample(n, seed=seed)

    assert torch.allclose(X_fwd, X_direct)
    assert torch.allclose(y_fwd, y_direct)
