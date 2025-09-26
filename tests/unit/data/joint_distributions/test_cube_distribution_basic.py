import pytest
import torch
import torch.nn as nn

from src.data.cube_distribution import CubeDistribution
from src.data.cube_distribution_config import CubeDistributionConfig


def test_cube_distribution_construct_and_sample():
    cfg = CubeDistributionConfig(
        input_dim=2,
        indices_list=[[0]],
        weights=[1.0],
        noise_mean=1.0,
        noise_std=0.0,
    )
    dist = CubeDistribution(cfg, torch.device("cpu"))

    x, y = dist.sample(3, seed=0)
    assert x.shape == (3, *dist.input_shape)
    assert y.shape == (3, *dist.output_shape)


def test_cube_distribution_requires_scalar_target(monkeypatch):
    class MultiTarget(nn.Module):
        def __init__(self, cfg):  # pragma: no cover - validation occurs via exception
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - tested via exception
            batch = x.shape[0]
            return torch.zeros(batch, 2, device=x.device, dtype=x.dtype)

    monkeypatch.setattr(
        "src.data.cube_distribution.SumProdTarget",
        MultiTarget,
    )

    cfg = CubeDistributionConfig(
        input_dim=2,
        indices_list=[[0]],
        weights=[1.0],
        noise_mean=0.0,
        noise_std=1.0,
    )

    with pytest.raises(ValueError, match="single output dimension"):
        CubeDistribution(cfg, torch.device("cpu"))
