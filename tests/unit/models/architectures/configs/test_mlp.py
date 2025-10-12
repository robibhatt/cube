import pytest

import pytest

import src.models.bootstrap  # noqa: F401
from src.models.mlp_config import MLPConfig


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


def test_roundtrip_via_json(example_args):
    original = MLPConfig(**example_args)
    json_str = original.to_json()
    restored = MLPConfig.from_json(json_str)
    assert restored == original
