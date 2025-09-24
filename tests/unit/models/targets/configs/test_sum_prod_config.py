import torch

import torch

from src.models.targets.configs.sum_prod import SumProdTargetConfig
from src.models.targets.configs.target_function_config_registry import (
    TARGET_FUNCTION_CONFIG_REGISTRY,
    build_target_function_config,
    build_target_function_config_from_dict,
)


def test_sum_prod_config_registered():
    assert "SumProdTarget" in TARGET_FUNCTION_CONFIG_REGISTRY
    assert TARGET_FUNCTION_CONFIG_REGISTRY["SumProdTarget"] is SumProdTargetConfig


def test_build_sum_prod_config():
    cfg = build_target_function_config(
        "SumProdTarget",
        input_shape=torch.Size([5]),
        indices_list=[[0, 1], [2, 3, 4]],
        weights=[0.5, 1.5],
    )
    assert isinstance(cfg, SumProdTargetConfig)
    assert cfg.input_shape == torch.Size([5])
    assert cfg.indices_list == [[0, 1], [2, 3, 4]]
    assert cfg.weights == [0.5, 1.5]
    assert cfg.output_shape == torch.Size([1])
    assert cfg.model_type == "SumProdTarget"


def test_sum_prod_config_json_roundtrip():
    original = SumProdTargetConfig(
        input_shape=torch.Size([4]),
        indices_list=[[0, 1]],
        weights=[1.0],
    )
    json_str = original.to_json()
    restored = SumProdTargetConfig.from_json(json_str)
    assert restored == original


def test_sum_prod_config_from_dict_via_registry():
    data = {
        "model_type": "SumProdTarget",
        "input_shape": [3],
        "indices_list": [[0, 1], [2]],
        "weights": [1.0, 0.25],
    }
    cfg = build_target_function_config_from_dict(data)
    assert isinstance(cfg, SumProdTargetConfig)
    assert cfg.input_shape == torch.Size([3])
    assert cfg.indices_list == [[0, 1], [2]]
    assert cfg.weights == [1.0, 0.25]
    assert cfg.model_type == "SumProdTarget"
