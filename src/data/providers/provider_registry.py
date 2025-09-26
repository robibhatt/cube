from __future__ import annotations

DATA_PROVIDER_REGISTRY: dict[str, type] = {}


def register_data_provider(provider_type: str):
    """Class decorator to register a data provider implementation."""

    def decorator(cls: type):
        DATA_PROVIDER_REGISTRY[provider_type] = cls
        return cls

    return decorator
