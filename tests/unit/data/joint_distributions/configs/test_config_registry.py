import pytest

from src.data.joint_distributions.configs.joint_distribution_config_registry import (
    build_joint_distribution_config,
)


def test_unknown_joint_distribution_config():
    with pytest.raises(ValueError):
        build_joint_distribution_config("UnknownConfig")
