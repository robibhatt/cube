import pytest
import torch

from src.data.joint_distributions.configs.cube_distribution import CubeDistributionConfig
from src.data.joint_distributions.configs.joint_distribution_config_registry import (
    JOINT_DISTRIBUTION_CONFIG_REGISTRY,
    build_joint_distribution_config,
    build_joint_distribution_config_from_dict,
)


def test_cube_config_registered():
    assert "CubeDistribution" in JOINT_DISTRIBUTION_CONFIG_REGISTRY
    assert (
        JOINT_DISTRIBUTION_CONFIG_REGISTRY["CubeDistribution"]
        is CubeDistributionConfig
    )


def test_build_cube_config():
    cfg = build_joint_distribution_config(
        "CubeDistribution",
        input_dim=2,
        indices_list=[[0]],
        weights=[0.0],
        normalize=False,
        noise_mean=1.0,
        noise_std=0.5,
    )
    assert isinstance(cfg, CubeDistributionConfig)
    assert cfg.distribution_type == "CubeDistribution"
    assert cfg.input_shape == torch.Size([2])
    assert cfg.output_shape == torch.Size([1])
    assert cfg.noise_mean == pytest.approx(1.0)
    assert cfg.noise_std == pytest.approx(0.5)
    assert cfg.target_function_config.indices_list == [[0]]


def test_cube_config_json_roundtrip():
    cfg = CubeDistributionConfig(
        input_dim=2,
        indices_list=[[0], [1]],
        weights=[1.0, 1.0],
        normalize=False,
        noise_mean=0.0,
        noise_std=1.0,
    )
    json_str = cfg.to_json()
    restored = CubeDistributionConfig.from_json(json_str)
    assert restored == cfg


def test_cube_config_from_dict_via_registry():
    data = {
        "distribution_type": "CubeDistribution",
        "input_dim": 2,
        "indices_list": [[0], [1]],
        "weights": [1.0, 1.0],
        "normalize": False,
        "noise_mean": 0.25,
        "noise_std": 0.75,
    }
    cfg = build_joint_distribution_config_from_dict(data)
    assert isinstance(cfg, CubeDistributionConfig)
    assert cfg.input_shape == torch.Size([2])
    assert cfg.output_shape == torch.Size([1])
    assert cfg.noise_mean == pytest.approx(0.25)
    assert cfg.noise_std == pytest.approx(0.75)
