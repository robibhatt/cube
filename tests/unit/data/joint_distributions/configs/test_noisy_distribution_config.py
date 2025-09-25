import pytest
import torch

from src.data.joint_distributions.configs.noisy_distribution import NoisyDistributionConfig
from src.data.joint_distributions.configs.joint_distribution_config_registry import (
    JOINT_DISTRIBUTION_CONFIG_REGISTRY,
    build_joint_distribution_config,
    build_joint_distribution_config_from_dict,
)


def test_noisy_config_registered():
    assert "NoisyDistribution" in JOINT_DISTRIBUTION_CONFIG_REGISTRY
    assert (
        JOINT_DISTRIBUTION_CONFIG_REGISTRY["NoisyDistribution"]
        is NoisyDistributionConfig
    )


def test_build_noisy_config():
    cfg = build_joint_distribution_config(
        "NoisyDistribution",
        input_dim=2,
        indices_list=[[0]],
        weights=[0.0],
        normalize=False,
        noise_mean=1.0,
        noise_std=0.5,
    )
    assert isinstance(cfg, NoisyDistributionConfig)
    assert cfg.distribution_type == "NoisyDistribution"
    assert cfg.input_shape == torch.Size([2])
    assert cfg.output_shape == torch.Size([1])
    assert cfg.noise_mean == pytest.approx(1.0)
    assert cfg.noise_std == pytest.approx(0.5)
    assert cfg.base_distribution_config.distribution_type == "Hypercube"
    assert cfg.target_function_config.indices_list == [[0]]


def test_noisy_config_json_roundtrip():
    cfg = NoisyDistributionConfig(
        input_dim=2,
        indices_list=[[0], [1]],
        weights=[1.0, 1.0],
        normalize=False,
        noise_mean=0.0,
        noise_std=1.0,
    )
    json_str = cfg.to_json()
    restored = NoisyDistributionConfig.from_json(json_str)
    assert restored == cfg


def test_noisy_config_from_dict_via_registry():
    data = {
        "distribution_type": "NoisyDistribution",
        "input_dim": 2,
        "indices_list": [[0], [1]],
        "weights": [1.0, 1.0],
        "normalize": False,
        "noise_mean": 0.25,
        "noise_std": 0.75,
    }
    cfg = build_joint_distribution_config_from_dict(data)
    assert isinstance(cfg, NoisyDistributionConfig)
    assert cfg.input_shape == torch.Size([2])
    assert cfg.output_shape == torch.Size([1])
    assert cfg.noise_mean == pytest.approx(0.25)
    assert cfg.noise_std == pytest.approx(0.75)
