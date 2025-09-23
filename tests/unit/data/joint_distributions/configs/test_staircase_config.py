import torch
import pytest

from src.data.joint_distributions.configs.staircase import StaircaseConfig
from src.data.joint_distributions.configs.joint_distribution_config_registry import (
    JOINT_DISTRIBUTION_CONFIG_REGISTRY,
    build_joint_distribution_config,
    build_joint_distribution_config_from_dict,
)


def test_staircase_config_registered():
    assert "Staircase" in JOINT_DISTRIBUTION_CONFIG_REGISTRY
    assert JOINT_DISTRIBUTION_CONFIG_REGISTRY["Staircase"] is StaircaseConfig


def test_build_staircase_config():
    cfg = build_joint_distribution_config("Staircase", input_shape=torch.Size([5]), k=3)
    assert isinstance(cfg, StaircaseConfig)
    assert cfg.input_shape == torch.Size([5])
    assert cfg.k == 3
    assert cfg.output_shape == torch.Size([1])


def test_staircase_config_json_roundtrip():
    cfg = StaircaseConfig(input_shape=torch.Size([4]), k=2)
    json_str = cfg.to_json()
    restored = StaircaseConfig.from_json(json_str)
    assert restored == cfg


def test_staircase_config_from_dict_via_registry():
    data = {"distribution_type": "Staircase", "input_shape": [3], "k": 2, "dtype": "float32"}
    cfg = build_joint_distribution_config_from_dict(data)
    assert isinstance(cfg, StaircaseConfig)
    assert cfg.input_shape == torch.Size([3])
    assert cfg.k == 2


def test_staircase_config_invalid_k():
    with pytest.raises(ValueError):
        StaircaseConfig(input_shape=torch.Size([2]), k=3)
