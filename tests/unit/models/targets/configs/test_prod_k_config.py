import torch

from src.models.targets.configs.prod_k import ProdKTargetConfig
from src.models.targets.configs.target_function_config_registry import (
    TARGET_FUNCTION_CONFIG_REGISTRY,
    build_target_function_config,
    build_target_function_config_from_dict,
)


def test_prod_k_config_registered():
    assert "ProdKTarget" in TARGET_FUNCTION_CONFIG_REGISTRY
    assert TARGET_FUNCTION_CONFIG_REGISTRY["ProdKTarget"] is ProdKTargetConfig


def test_build_prod_k_config():
    cfg = build_target_function_config(
        "ProdKTarget", input_shape=torch.Size([5]), indices=[0, 2, 4]
    )
    assert isinstance(cfg, ProdKTargetConfig)
    assert cfg.input_shape == torch.Size([5])
    assert cfg.indices == [0, 2, 4]
    assert cfg.output_shape == torch.Size([1])
    assert cfg.model_type == "ProdKTarget"


def test_prod_k_config_json_roundtrip():
    original = ProdKTargetConfig(input_shape=torch.Size([4]), indices=[0, 1])
    json_str = original.to_json()
    restored = ProdKTargetConfig.from_json(json_str)
    assert restored == original


def test_prod_k_config_from_dict_via_registry():
    data = {"model_type": "ProdKTarget", "input_shape": [3], "indices": [0, 2]}
    cfg = build_target_function_config_from_dict(data)
    assert isinstance(cfg, ProdKTargetConfig)
    assert cfg.input_shape == torch.Size([3])
    assert cfg.indices == [0, 2]
    assert cfg.model_type == "ProdKTarget"
