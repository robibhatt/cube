import pytest
import torch
from src.data.providers.tensor_data_provider import TensorDataProvider
from src.data.providers import create_data_provider_from_distribution
from tests.unit.data.conftest import DummyJointDistribution


class TensorPrefDist(DummyJointDistribution):
    def __init__(self):
        cfg = DummyJointDistribution._Config()
        super().__init__(cfg, torch.device("cpu"))

    def preferred_provider(self) -> str:
        return "TensorDataProvider"


def test_create_data_provider_from_distribution_tensor(tmp_path):
    dist = TensorPrefDist()
    seed = 0
    provider = create_data_provider_from_distribution(
        dist,
        tmp_path,
        batch_size=2,
        dataset_size=10,
        seed=seed,
    )
    assert isinstance(provider, TensorDataProvider)
    assert provider.dataset_dir == tmp_path
