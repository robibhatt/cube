import math
import torch

from src.data.joint_distributions import create_joint_distribution
from src.data.joint_distributions.configs.staircase import StaircaseConfig


def test_staircase_sample_values():
    cfg = StaircaseConfig(input_shape=torch.Size([4]), k=3)
    dist = create_joint_distribution(cfg, torch.device("cpu"))
    x, y = dist.sample(10, seed=0)
    assert x.shape == (10, 4)
    assert y.shape == (10, 1)
    # compute expected staircase values for k=3
    flat = x
    expected = flat[:, 0]
    expected = expected + flat[:, 0] * flat[:, 1]
    expected = expected + flat[:, 0] * flat[:, 1] * flat[:, 2]
    expected = expected / math.sqrt(3)
    assert torch.allclose(y.squeeze(1), expected)
