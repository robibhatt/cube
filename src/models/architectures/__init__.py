"""Public API for model architectures."""

from .mlp import MLP
from .configs.mlp import MLPConfig

__all__ = ["MLP", "MLPConfig"]
