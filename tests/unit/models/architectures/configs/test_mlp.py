import pytest
import torch
import json

import json

import pytest
import torch

import src.models.bootstrap  # noqa: F401
from src.models.architectures.configs.mlp import MLPConfig
from src.models.architectures.configs.model_config_registry import (
    MODEL_CONFIG_REGISTRY,
    build_model_config,
    build_model_config_from_json_args,
    build_model_config_from_dict,
)


def test_mlp_registered():
    assert "MLP" in MODEL_CONFIG_REGISTRY
    assert MODEL_CONFIG_REGISTRY["MLP"] is MLPConfig


def test_build_mlp():
    cfg = build_model_config(
        "MLP",
        input_dim=3,
        output_dim=1,
        hidden_dims=[4, 2],
        activation="relu",
        start_activation=False,
        end_activation=False,
    )

    assert isinstance(cfg, MLPConfig)
    assert cfg.model_type == "MLP"
    assert cfg.input_dim == 3
    assert cfg.output_dim == 1
    assert cfg.hidden_dims == [4, 2]

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


def test_build_from_json_args(example_args):
    cfg = build_model_config_from_json_args("MLP", **example_args)
    assert isinstance(cfg, MLPConfig)
    assert cfg.input_dim == example_args["input_dim"]
    assert cfg.output_dim == example_args["output_dim"]
    assert cfg.model_type == "MLP"


def test_build_from_dict_roundtrip(example_args):
    # roundtrip via JSON string -> dict -> build
    original = MLPConfig(**example_args)
    json_str = original.to_json()
    data = json.loads(json_str)
    cfg = build_model_config_from_dict(data)
    assert isinstance(cfg, MLPConfig)
    assert cfg == original


def test_missing_model_type_raises():
    with pytest.raises(ValueError):
        build_model_config_from_dict({})


def test_unregistered_model_type_raises(example_args):
    with pytest.raises(ValueError):
        build_model_config_from_json_args("UnknownModel", **example_args)






