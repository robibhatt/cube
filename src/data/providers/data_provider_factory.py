from __future__ import annotations

from .provider_registry import DATA_PROVIDER_REGISTRY


def create_data_provider_from_distribution(
    dist,
    batch_size,
    dataset_size,
    seed,
):
    """Return the appropriate data provider for ``dist``."""
    assert batch_size is not None, "Batch size must be provided"

    provider_type = dist.preferred_provider()
    provider_cls = DATA_PROVIDER_REGISTRY[provider_type]
    return provider_cls(
        dist,
        seed,
        batch_size=batch_size,
        dataset_size=dataset_size,
    )
