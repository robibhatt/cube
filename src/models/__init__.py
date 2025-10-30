"""Public API for model components."""

from src.models.mlp import MLP
from src.models.mlp_config import MLPConfig

__all__ = [
    "MLP",
    "MLPConfig",
]
