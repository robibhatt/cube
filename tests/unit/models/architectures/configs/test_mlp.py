import pytest
import torch

import src.models.bootstrap  # noqa: F401
from src.models.architectures.configs.mlp import MLPConfig


@pytest.fixture
def example_args():
    return {
        "input_dim": 8,
        "output_dim": 3,
        "hidden_dims": [16, 32],
        "activation": "relu",
        "start_activation": True,
        "end_activation": False,
    }


def test_direct_instantiation(example_args):
    cfg = MLPConfig(**example_args)
    assert cfg.input_dim == example_args["input_dim"]
    assert cfg.output_dim == example_args["output_dim"]
    assert isinstance(cfg.input_shape, torch.Size)
    assert tuple(cfg.input_shape) == (example_args["input_dim"],)
    assert isinstance(cfg.output_shape, torch.Size)
    assert tuple(cfg.output_shape) == (example_args["output_dim"],)
    assert cfg.model_type == "MLP"


def test_roundtrip_via_json(example_args):
    original = MLPConfig(**example_args)
    json_str = original.to_json()
    restored = MLPConfig.from_json(json_str)
    assert restored == original


def test_invalid_frozen_layer_index_raises(example_args):
    with pytest.raises(ValueError):
        MLPConfig(**example_args, frozen_layers=[0])
    with pytest.raises(ValueError):
        MLPConfig(**example_args, frozen_layers=[len(example_args["hidden_dims"]) + 2])
