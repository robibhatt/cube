import torch
from src.data.joint_distributions import create_joint_distribution
from src.data.joint_distributions.configs.noisy_distribution import NoisyDistributionConfig
from tests.unit.data.conftest import DummyJointDistribution, AddOneNoiseDistributionConfig


def test_noisy_distribution_construct_and_sample():
    cfg = NoisyDistributionConfig(
        base_distribution_config=DummyJointDistribution._Config(),
        noise_distribution_config=AddOneNoiseDistributionConfig(),
    )
    dist = create_joint_distribution(cfg, torch.device("cpu"))

    x, y = dist.sample(3, seed=0)
    assert x.shape == (3, *dist.input_shape)
    assert y.shape == (3, *dist.output_shape)
