import pytest
import torch

from src.data.joint_distributions.cube_distribution import CubeDistribution
from src.data.joint_distributions.configs.cube_distribution import CubeDistributionConfig
from src.data.providers.noisy_provider import NoisyProvider
from src.data.providers import create_data_provider_from_distribution


def _make_distribution():
    cfg = CubeDistributionConfig(
        input_dim=2,
        indices_list=[[0]],
        weights=[0.0],
        normalize=False,
        noise_mean=1.0,
        noise_std=0.0,
    )
    return CubeDistribution(cfg, torch.device("cpu"))


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


def test_preferred_provider():
    dist = _make_distribution()
    provider = create_data_provider_from_distribution(
        dist,
        batch_size=2,
        dataset_size=4,
        seed=0,
    )
    assert isinstance(provider, NoisyProvider)
