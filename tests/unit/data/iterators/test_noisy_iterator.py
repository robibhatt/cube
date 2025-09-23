import torch
import pytest

from src.data.joint_distributions import create_joint_distribution
from src.data.joint_distributions.configs.noisy_distribution import NoisyDistributionConfig
from src.data.providers.noisy_provider import NoisyProvider
from src.data.providers import create_data_provider_from_distribution
from tests.unit.data.conftest import (
    DummyJointDistribution,
    AddOneNoiseDistributionConfig,
    dummy_distribution,
)


def _make_distribution():
    cfg = NoisyDistributionConfig(
        base_distribution_config=DummyJointDistribution._Config(),
        noise_distribution_config=AddOneNoiseDistributionConfig(),
    )
    return create_joint_distribution(cfg, torch.device("cpu"))


def test_noisy_iterator_batches_apply_noise(tmp_path):
    dist = _make_distribution()
    iterator = NoisyProvider(dist, tmp_path, seed=0, dataset_size=4, batch_size=2)

    batches = list(iterator)

    assert len(batches) == 2
    for X, y in batches:
        assert X.shape == (2, *dist.input_shape)
        assert y.shape == (2, *dist.output_shape)
        assert torch.allclose(y, torch.full((2, 1), 6.0))


def test_noisy_iterator_deterministic(tmp_path):
    dist = _make_distribution()

    iterator1 = NoisyProvider(dist, tmp_path, seed=42, batch_size=2, dataset_size=4)
    first = list(iterator1)
    iterator2 = NoisyProvider(dist, tmp_path, seed=42, batch_size=2, dataset_size=4)
    second = list(iterator2)

    assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(first, second))


def test_noisy_iterator_requires_noisy_distribution(tmp_path, dummy_distribution):
    with pytest.raises(AssertionError):
        NoisyProvider(dummy_distribution, tmp_path, seed=0, batch_size=1, dataset_size=1)


def test_preferred_provider(tmp_path):
    dist = _make_distribution()
    provider = create_data_provider_from_distribution(
        dist,
        tmp_path,
        batch_size=2,
        dataset_size=4,
        seed=0,
    )
    assert isinstance(provider, NoisyProvider)
