import torch

from src.data.joint_distributions import create_joint_distribution
from src.data.joint_distributions.configs.hypercube import HypercubeConfig


def test_hypercube_sample_values():
    cfg = HypercubeConfig(input_shape=torch.Size([3]))
    dist = create_joint_distribution(cfg, torch.device("cpu"))

    x, y = dist.sample(100, seed=0)
    assert x.shape == (100, 3)
    assert y.shape == (100, 1)
    assert torch.all((x == -1) | (x == 1))
    assert torch.all(y == 0)
