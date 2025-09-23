from __future__ import annotations

from .joint_distribution import JointDistribution


JOINT_DISTRIBUTION_REGISTRY: dict[str, type[JointDistribution]] = {}


def register_joint_distribution(distribution_type: str):
    """Class decorator to register a ``JointDistribution`` subclass."""

    def decorator(cls: type[JointDistribution]):
        JOINT_DISTRIBUTION_REGISTRY[distribution_type] = cls
        return cls

    return decorator
