import pytest
import torch

from src.data.joint_distributions import create_joint_distribution
from src.data.joint_distributions.configs.gaussian import GaussianConfig
from src.data.joint_distributions.configs.mapped_joint_distribution import (
    MappedJointDistributionConfig,
)
from src.data.joint_distributions.configs.cube_distribution import (
    CubeDistributionConfig,
)
from src.models.targets.configs.sum_prod import SumProdTargetConfig


def test_gaussian_variance_zero():
    cfg = GaussianConfig(input_dim=2, mean=0.0, std=1.0)
    dist = create_joint_distribution(cfg, torch.device("cpu"))
    var = dist.average_output_variance(n_samples=1000, seed=0)
    assert var == pytest.approx(0.0, abs=1e-6)


def test_mapped_linear_variance_matches_dimension():
    base_cfg = GaussianConfig(input_dim=3, mean=0.0, std=1.0)
    target_cfg = SumProdTargetConfig(
        input_shape=torch.Size([3]),
        indices_list=[[0], [1], [2]],
        weights=[1.0, 1.0, 1.0],
        normalize=False,
    )
    cfg = MappedJointDistributionConfig(
        distribution_config=base_cfg,
        target_function_config=target_cfg,
    )
    dist = create_joint_distribution(cfg, torch.device("cpu"))
    var = dist.average_output_variance(n_samples=5000, seed=1)
    assert var == pytest.approx(3.0, abs=0.2)


def test_cube_distribution_variance_zero():
    cube_cfg = CubeDistributionConfig(
        input_dim=2,
        indices_list=[[0]],
        weights=[0.0],
        normalize=False,
        noise_mean=1.0,
        noise_std=0.0,
    )
    dist = create_joint_distribution(cube_cfg, torch.device("cpu"))
    var = dist.average_output_variance(n_samples=1000, seed=0)
    assert var == pytest.approx(0.0, abs=1e-6)

def test_average_output_variance_matches_empirical_mse():
    base_cfg = GaussianConfig(input_dim=3, mean=0.0, std=1.0)
    target_cfg = SumProdTargetConfig(
        input_shape=torch.Size([3]),
        indices_list=[[0], [1], [2]],
        weights=[1.0, 1.0, 1.0],
        normalize=False,
    )
    cfg = MappedJointDistributionConfig(
        distribution_config=base_cfg,
        target_function_config=target_cfg,
    )
    dist = create_joint_distribution(cfg, torch.device("cpu"))

    n = 1000
    seed = 42
    _, y = dist.sample(n, seed=seed)
    y_bar = y.mean(dim=0)
    mse = ((y - y_bar).pow(2).reshape(n, -1).sum(dim=1).mean()).item()

    var = dist.average_output_variance(n_samples=n, seed=seed)

    assert mse == pytest.approx(var, abs=1e-6)
