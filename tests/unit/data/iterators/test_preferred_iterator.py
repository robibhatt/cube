from src.data.providers import create_data_provider_from_distribution
from src.data.providers.noisy_provider import NoisyProvider


def test_create_data_provider_from_distribution_prefers_noisy(constant_cube_distribution):
    provider = create_data_provider_from_distribution(
        constant_cube_distribution,
        batch_size=2,
        dataset_size=10,
        seed=0,
    )
    assert isinstance(provider, NoisyProvider)
