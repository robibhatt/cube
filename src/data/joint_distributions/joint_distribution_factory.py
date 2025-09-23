from __future__ import annotations

import torch

from .configs.base import JointDistributionConfig
from .joint_distribution import JointDistribution
from .joint_distribution_registry import JOINT_DISTRIBUTION_REGISTRY


def create_joint_distribution(
    config: JointDistributionConfig, device: torch.device
) -> JointDistribution:
    """Instantiate a joint distribution based on ``config.distribution_type``."""

    dist_cls = JOINT_DISTRIBUTION_REGISTRY.get(config.distribution_type)
    if dist_cls is None:
        raise ValueError(
            f"Joint distribution type '{config.distribution_type}' is not registered."
        )

    return dist_cls(config, device)
