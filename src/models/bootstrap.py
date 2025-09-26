"""Ensure core model components are importable with a single import."""

# Importing these modules re-exports the key classes via ``src.models``.
from . import mlp  # noqa: F401
from . import mlp_config  # noqa: F401

