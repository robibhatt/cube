import torch

from src.models.targets.configs.prod_1234 import Prod1234Config
from src.models.targets.configs.target_function_config_registry import (
    TARGET_FUNCTION_CONFIG_REGISTRY,
    build_target_function_config,
    build_target_function_config_from_dict,
)


def test_prod_1234_config_registered():
    assert "1234_prod" in TARGET_FUNCTION_CONFIG_REGISTRY
    assert TARGET_FUNCTION_CONFIG_REGISTRY["1234_prod"] is Prod1234Config


def test_build_prod_1234_config():
    cfg = build_target_function_config("1234_prod", input_shape=torch.Size([4]))
    assert isinstance(cfg, Prod1234Config)
    assert cfg.input_shape == torch.Size([4])
    assert cfg.output_shape == torch.Size([1])
    assert cfg.model_type == "1234_prod"


def test_prod_1234_config_json_roundtrip():
    original = Prod1234Config(input_shape=torch.Size([5]))
    json_str = original.to_json()
    restored = Prod1234Config.from_json(json_str)
    assert restored == original


def test_prod_1234_config_from_dict_via_registry():
    data = {"model_type": "1234_prod", "input_shape": [6]}
    cfg = build_target_function_config_from_dict(data)
    assert isinstance(cfg, Prod1234Config)
    assert cfg.input_shape == torch.Size([6])
    assert cfg.model_type == "1234_prod"
