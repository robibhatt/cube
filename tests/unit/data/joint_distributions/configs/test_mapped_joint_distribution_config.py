import torch

from src.data.joint_distributions.configs.mapped_joint_distribution import (
    MappedJointDistributionConfig,
)
from src.data.joint_distributions.configs.joint_distribution_config_registry import (
    JOINT_DISTRIBUTION_CONFIG_REGISTRY,
    build_joint_distribution_config,
    build_joint_distribution_config_from_dict,
)
from src.data.joint_distributions.configs.gaussian import GaussianConfig
from src.models.targets.configs.sum_prod import SumProdTargetConfig


def test_mapped_config_registered():
    assert "MappedJointDistribution" in JOINT_DISTRIBUTION_CONFIG_REGISTRY
    assert (
        JOINT_DISTRIBUTION_CONFIG_REGISTRY["MappedJointDistribution"]
        is MappedJointDistributionConfig
    )


def test_build_mapped_config():
    cfg = build_joint_distribution_config(
        "MappedJointDistribution",
        distribution_config=GaussianConfig(
            input_shape=torch.Size([2]), mean=0.0, std=1.0
        ),
        target_function_config=SumProdTargetConfig(
            input_shape=torch.Size([2]),
            indices_list=[[0], [1]],
            weights=[1.0, 1.0],
            normalize=False,
        ),
    )
    assert isinstance(cfg, MappedJointDistributionConfig)
    assert cfg.distribution_type == "MappedJointDistribution"
    assert cfg.input_shape == torch.Size([2])
    assert cfg.output_shape == torch.Size([1])


def test_mapped_config_json_roundtrip():
    cfg = MappedJointDistributionConfig(
        distribution_config=GaussianConfig(
            input_shape=torch.Size([2]), mean=0.0, std=1.0
        ),
        target_function_config=SumProdTargetConfig(
            input_shape=torch.Size([2]),
            indices_list=[[0], [1]],
            weights=[1.0, 1.0],
            normalize=False,
        ),
    )
    json_str = cfg.to_json()
    restored = MappedJointDistributionConfig.from_json(json_str)
    assert restored == cfg


def test_mapped_config_from_dict_via_registry():
    data = {
        "distribution_type": "MappedJointDistribution",
        "distribution_config": {
            "distribution_type": "Gaussian",
            "input_shape": [2],
            "dtype": "float32",
            "mean": 0.0,
            "std": 1.0,
        },
        "target_function_config": {
            "model_type": "SumProdTarget",
            "input_shape": [2],
            "indices_list": [[0], [1]],
            "weights": [1.0, 1.0],
            "normalize": False,
        },
    }
    cfg = build_joint_distribution_config_from_dict(data)
    assert isinstance(cfg, MappedJointDistributionConfig)
    assert cfg.input_shape == torch.Size([2])
    assert cfg.output_shape == torch.Size([1])
