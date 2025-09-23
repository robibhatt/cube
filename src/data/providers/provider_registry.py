from __future__ import annotations

from .data_provider import DataProvider


DATA_PROVIDER_REGISTRY: dict[str, type[DataProvider]] = {}


def register_data_provider(provider_type: str):
    """Class decorator to register a :class:`DataIterator` subclass."""

    def decorator(cls: type[DataProvider]):
        DATA_PROVIDER_REGISTRY[provider_type] = cls
        return cls

    return decorator
