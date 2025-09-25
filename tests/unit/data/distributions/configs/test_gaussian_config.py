import torch
from src.data.joint_distributions.configs.gaussian import GaussianConfig
import json
from src.data.joint_distributions.configs.joint_distribution_config_registry import build_joint_distribution_config_from_dict

def test_gaussian_config_serialization_roundtrip():
    # Create instance
    cfg = GaussianConfig(
        input_dim=100,
        dtype=torch.bfloat16,
        mean=0.0,
        std=1.0,
    )

    # Serialize to JSON
    json_str = cfg.to_json()
    assert isinstance(json_str, str)

    # Deserialize
    cfg2 = GaussianConfig.from_json(json_str)

    # Check fields
    assert cfg2.input_dim == 100
    assert cfg2.input_shape == torch.Size([100])
    assert cfg2.dtype == torch.bfloat16
    assert cfg2.distribution_type == "Gaussian"

def test_gaussian_config_dict_output():
    cfg = GaussianConfig(
        input_dim=16,
        dtype=torch.float32,
        mean=0.0,
        std=1.0,
    )
    d = cfg.to_dict()

    assert d["input_dim"] == 16
    assert d["dtype"] == "float32"
    assert d["distribution_type"] == "Gaussian"

def test_gaussian_config_from_raw_json():
    json_str = json.dumps(
        {
            "input_dim": 64,
            "dtype": "float64",
            "mean": 0.0,
            "std": 1.0,
        }
    )

    cfg = GaussianConfig.from_json(json_str)

    assert cfg.input_dim == 64
    assert cfg.input_shape == torch.Size([64])
    assert cfg.dtype == torch.float64
    assert cfg.distribution_type == "Gaussian"

def test_gaussian_config_serialization_and_deserialization():
    # Create a GaussianConfig object
    original_config = GaussianConfig(
        input_dim=5,
        dtype=torch.float64,
        mean=0.0,
        std=1.0,
    )

    # Serialize to JSON
    json_str = original_config.to_json()

    # Deserialize from JSON
    restored_config = GaussianConfig.from_json(json_str)

    # Check equality of attributes
    assert restored_config.distribution_type == "Gaussian"
    assert restored_config.input_dim == 5
    assert restored_config.input_shape == torch.Size([5])
    assert restored_config.dtype == torch.float64

def test_gaussian_config_from_dict_via_registry():
    # Simulate a dictionary like from a parsed JSON blob
    cfg_dict = {
        "distribution_type": "Gaussian",
        "input_dim": 5,
        "dtype": "float64",
        "mean": 0.0,
        "std": 1.0,
    }

    # Build config from dict using registry
    config = build_joint_distribution_config_from_dict(cfg_dict)

    assert isinstance(config, GaussianConfig)
    assert config.input_dim == 5
    assert config.input_shape == torch.Size([5])
    assert config.dtype == torch.float64
    assert config.distribution_type == "Gaussian"
