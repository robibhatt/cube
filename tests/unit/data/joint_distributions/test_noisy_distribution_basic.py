import torch
from src.data.joint_distributions import create_joint_distribution
from src.data.joint_distributions.configs.noisy_distribution import NoisyDistributionConfig


def test_noisy_distribution_construct_and_sample():
    cfg = NoisyDistributionConfig(
        input_dim=2,
        indices_list=[[0]],
        weights=[1.0],
        noise_mean=1.0,
        noise_std=0.0,
    )
    dist = create_joint_distribution(cfg, torch.device("cpu"))

    x, y = dist.sample(3, seed=0)
    assert x.shape == (3, *dist.input_shape)
    assert y.shape == (3, *dist.output_shape)
