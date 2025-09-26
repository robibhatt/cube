"""Public API for joint distributions."""

from .cube_distribution import CubeDistribution
from .configs.cube_distribution import CubeDistributionConfig

__all__ = ["CubeDistribution", "CubeDistributionConfig"]
