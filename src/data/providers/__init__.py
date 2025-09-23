"""Data provider package bootstrap."""

from src.utils.plugin_loader import import_submodules

from .provider_registry import DATA_PROVIDER_REGISTRY, register_data_provider
from .data_provider_factory import create_data_provider_from_distribution

# Import all submodules so their registration decorators run and populate
# :data:`DATA_PROVIDER_REGISTRY`.
import_submodules(__name__)

__all__ = [
    "DATA_PROVIDER_REGISTRY",
    "register_data_provider",
    "create_data_provider_from_distribution",
]
