import torch
import pytest

from src.data.joint_distributions import create_joint_distribution
from src.data.joint_distributions.configs.representor_distribution import (
    RepresentorDistributionConfig,
)
from src.data.providers.tensor_data_provider import TensorDataProvider
from tests.unit.data.conftest import trained_trainer


def _make_distribution(trainer):
    cfg = RepresentorDistributionConfig(
        base_distribution_config=trainer.config.joint_distribution_config,
        model_config=trainer.config.model_config,
        checkpoint_dir=trainer.checkpoint_dir,
        from_rep=0,
        to_rep=1,
    )
    return create_joint_distribution(cfg, torch.device("cpu"))


def test_tensor_iterator_batches_expected_shape(tmp_path, trained_trainer):
    dist = _make_distribution(trained_trainer)

    iterator = TensorDataProvider(
        dist,
        tmp_path,
        seed=0,
        dataset_size=4,
        batch_size=2,
    )

    batches = list(iterator)
    rep0 = dist.model_representor.representation_shape(dist.from_rep)
    rep1 = dist.model_representor.representation_shape(dist.to_rep)

    assert len(batches) == 2
    for X, y in batches:
        assert X.shape == (2, *rep0)
        assert y.shape == (2, *rep1)


def test_tensor_iterator_deterministic(tmp_path, trained_trainer):
    dist = _make_distribution(trained_trainer)

    iterator1 = TensorDataProvider(dist, tmp_path, seed=42, batch_size=2, dataset_size=4)
    first = list(iterator1)
    iterator2 = TensorDataProvider(dist, tmp_path, seed=42, batch_size=2, dataset_size=4)
    second = list(iterator2)

    assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(first, second))


