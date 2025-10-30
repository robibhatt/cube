import pytest
import types

import torch

from src.data.cube_distribution import CubeDistribution
from src.data.cube_distribution_config import CubeDistributionConfig
from src.data.noisy_data_provider import NoisyProvider


def _make_distribution():
    cfg = CubeDistributionConfig(
        input_dim=2,
        indices_list=[[0]],
        weights=[0.0],
        normalize=False,
        noise_std=0.0,
    )
    dist = CubeDistribution(cfg, torch.device("cpu"))

    def _constant_target(self: CubeDistribution, x: torch.Tensor) -> torch.Tensor:
        return torch.ones((x.size(0), 1), dtype=torch.float32, device=self.device)

    dist.target = types.MethodType(_constant_target, dist)  # type: ignore[assignment]
    return dist


def test_noisy_iterator_batches_apply_noise():
    dist = _make_distribution()
    iterator = NoisyProvider(dist, seed=0, dataset_size=4, batch_size=2)

    batches = list(iterator)

    assert len(batches) == 2
    for X, y in batches:
        assert X.shape == (2, *dist.input_shape)
        assert y.shape == (2, *dist.output_shape)
        assert torch.allclose(y, torch.full((2, 1), 1.0))


def test_noisy_iterator_deterministic():
    dist = _make_distribution()

    iterator1 = NoisyProvider(dist, seed=42, batch_size=2, dataset_size=4)
    first = list(iterator1)
    iterator2 = NoisyProvider(dist, seed=42, batch_size=2, dataset_size=4)
    second = list(iterator2)

    assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(first, second))


def test_noisy_iterator_requires_cube_distribution():
    dist = _make_distribution()
    dist.config.distribution_type = "NotCubeDistribution"
    with pytest.raises(AssertionError):
        NoisyProvider(dist, seed=0, batch_size=1, dataset_size=1)
