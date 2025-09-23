import torch
import pytest

from src.data.joint_distributions import create_joint_distribution
from src.data.joint_distributions.configs.mapped_joint_distribution import (
    MappedJointDistributionConfig,
)
from src.data.joint_distributions.configs.gaussian import GaussianConfig
from src.data.providers.tensor_data_provider import TensorDataProvider
from src.data.providers import create_data_provider_from_distribution
from tests.unit.data.conftest import gaussian_base, DummyJointDistribution
from conftest import LinearTargetFunctionConfig


def _make_distribution(base):
    cfg = MappedJointDistributionConfig(
        distribution_config=GaussianConfig(
            input_shape=base.config.input_shape, mean=0.0, std=1.0
        ),
        target_function_config=LinearTargetFunctionConfig(input_shape=base.input_shape),
    )
    return create_joint_distribution(cfg, torch.device("cpu"))


def test_tensor_iterator_batches_apply_target(tmp_path, gaussian_base):
    dist = _make_distribution(gaussian_base)
    iterator = TensorDataProvider(dist, tmp_path, seed=0, dataset_size=4, batch_size=2)

    batches = list(iterator)

    assert len(batches) == 2
    for X, y in batches:
        assert X.shape == (2, *dist.input_shape)
        assert y.shape == (2, *dist.output_shape)
        assert torch.allclose(y, dist.target_function(X))


def test_tensor_iterator_deterministic(tmp_path, gaussian_base):
    dist = _make_distribution(gaussian_base)

    iterator1 = TensorDataProvider(dist, tmp_path, seed=42, batch_size=2, dataset_size=4)
    first = list(iterator1)
    iterator2 = TensorDataProvider(dist, tmp_path, seed=42, batch_size=2, dataset_size=4)
    second = list(iterator2)

    assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(first, second))


def test_tensor_iterator_accepts_any_distribution(tmp_path, dummy_distribution):
    iterator = TensorDataProvider(dummy_distribution, tmp_path, seed=0, batch_size=1, dataset_size=1)
    assert next(iter(iterator))[0].shape[0] == 1


def test_preferred_provider(tmp_path, gaussian_base):
    dist = _make_distribution(gaussian_base)
    provider = create_data_provider_from_distribution(
        dist,
        tmp_path,
        batch_size=2,
        dataset_size=4,
        seed=0,
    )
    assert isinstance(provider, TensorDataProvider)

