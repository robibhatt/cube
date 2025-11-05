import pytest

from src.models.mlp_config import MLPConfig


@pytest.fixture
def example_args():
    return {
        "input_dim": 8,
        "hidden_dims": [16, 32],
    }


def test_direct_instantiation(example_args):
    cfg = MLPConfig(**example_args)
    assert cfg.input_dim == example_args["input_dim"]
    assert cfg.output_dim == 1


def test_roundtrip_via_json(example_args):
    original = MLPConfig(**example_args)
    json_str = original.to_json()
    restored = MLPConfig.from_json(json_str)
    assert restored == original


def test_exact_base_shapes_defaults_to_false(example_args):
    cfg = MLPConfig(**example_args)

    assert cfg.exact_base_shapes is False

    as_dict = cfg.to_dict()
    assert "exact_base_shapes" not in as_dict

    restored = MLPConfig.from_dict(as_dict)
    assert restored.exact_base_shapes is False
