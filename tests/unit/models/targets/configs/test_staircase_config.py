import torch
import torch

from src.models.targets.configs.staircase import StaircaseTargetConfig
from src.models.targets.configs.target_function_config_registry import (
    TARGET_FUNCTION_CONFIG_REGISTRY,
    build_target_function_config,
    build_target_function_config_from_dict,
)


def test_staircase_config_registered():
    assert "StaircaseTarget" in TARGET_FUNCTION_CONFIG_REGISTRY
    assert TARGET_FUNCTION_CONFIG_REGISTRY["StaircaseTarget"] is StaircaseTargetConfig


def test_build_staircase_config():
    cfg = build_target_function_config("StaircaseTarget", input_shape=torch.Size([4]), k=2)
    assert isinstance(cfg, StaircaseTargetConfig)
    assert cfg.input_shape == torch.Size([4])
    assert cfg.k == 2
    assert cfg.output_shape == torch.Size([1])


def test_staircase_config_json_roundtrip():
    original = StaircaseTargetConfig(input_shape=torch.Size([5]), k=3)
    json_str = original.to_json()
    restored = StaircaseTargetConfig.from_json(json_str)
    assert restored == original


def test_staircase_config_from_dict_via_registry():
    data = {"model_type": "StaircaseTarget", "input_shape": [6], "k": 2}
    cfg = build_target_function_config_from_dict(data)
    assert isinstance(cfg, StaircaseTargetConfig)
    assert cfg.input_shape == torch.Size([6])
    assert cfg.k == 2
