"""Public API for joint distributions."""

from .joint_distribution_registry import (
    JOINT_DISTRIBUTION_REGISTRY,
    register_joint_distribution,
)
from .joint_distribution_factory import create_joint_distribution
from .configs.joint_distribution_config_registry import (
    JOINT_DISTRIBUTION_CONFIG_REGISTRY,
    register_joint_distribution_config,
    build_joint_distribution_config,
    build_joint_distribution_config_from_dict,
)

__all__ = [
    "JOINT_DISTRIBUTION_REGISTRY",
    "register_joint_distribution",
    "create_joint_distribution",
    "JOINT_DISTRIBUTION_CONFIG_REGISTRY",
    "register_joint_distribution_config",
    "build_joint_distribution_config",
    "build_joint_distribution_config_from_dict",
]
