import torch
import pytest
from src.data.providers.tensor_data_provider import TensorDataProvider
from tests.unit.data.conftest import DummyJointDistribution



def test_make_loader_returns_expected_batches(tmp_path):
    cfg = DummyJointDistribution._Config()
    jd = DummyJointDistribution(cfg, torch.device("cpu"))
    seed = 0
    iterator = TensorDataProvider(
        jd,
        tmp_path,
        seed,
        dataset_size=4,
        batch_size=2,
    )

    loader = iterator.data_loader
    batches = list(loader)

    assert loader.batch_size == 2
    assert len(batches) == 2

    x_all = torch.cat([b[0] for b in batches])
    y_all = torch.cat([b[1] for b in batches])

    assert x_all.shape == (4, 2)
    assert y_all.shape == (4, 1)
    assert torch.allclose(y_all, torch.full((4, 1), 5.0))


def test_tensor_iterator_deterministic_across_calls(tmp_path):
    cfg = DummyJointDistribution._Config()
    jd = DummyJointDistribution(cfg, torch.device("cpu"))

    seed = 0
    iterator1 = TensorDataProvider(jd, tmp_path, seed, batch_size=2, dataset_size=4)
    first = list(iterator1.data_loader)

    iterator2 = TensorDataProvider(jd, tmp_path, seed, batch_size=2, dataset_size=4)
    second = list(iterator2.data_loader)

    assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(first, second))
