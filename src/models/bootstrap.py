"""Ensure core model components are importable with a single import."""

# Importing these modules re-exports the key classes via ``src.models``.
from src.models import mlp  # noqa: F401
from src.models import mlp_config  # noqa: F401

