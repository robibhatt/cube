import torch
from src.data.joint_distributions import create_joint_distribution
from src.data.joint_distributions.configs.noisy_distribution import NoisyDistributionConfig
from src.models.targets.configs.sum_prod import SumProdTargetConfig
from tests.unit.data.conftest import DummyJointDistribution


def test_noisy_distribution_construct_and_sample():
    cfg = NoisyDistributionConfig(
        base_distribution_config=DummyJointDistribution._Config(),
        target_function_config=SumProdTargetConfig(
            input_shape=torch.Size([2]),
            indices_list=[[0]],
            weights=[1.0],
            normalize=False,
        ),
        noise_mean=1.0,
        noise_std=0.0,
    )
    dist = create_joint_distribution(cfg, torch.device("cpu"))

    x, y = dist.sample(3, seed=0)
    assert x.shape == (3, *dist.input_shape)
    assert y.shape == (3, *dist.output_shape)
