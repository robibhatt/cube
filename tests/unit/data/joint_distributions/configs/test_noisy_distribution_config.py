import torch
from src.data.joint_distributions.configs.noisy_distribution import NoisyDistributionConfig
from src.data.joint_distributions.configs.joint_distribution_config_registry import (
    JOINT_DISTRIBUTION_CONFIG_REGISTRY,
    build_joint_distribution_config,
    build_joint_distribution_config_from_dict,
)
from src.data.joint_distributions.configs.mapped_joint_distribution import MappedJointDistributionConfig
from src.data.joint_distributions.configs.gaussian import GaussianConfig
from src.models.targets.configs.sum_prod import SumProdTargetConfig
from tests.unit.data.conftest import (
    DummyJointDistribution,
    add_one_noise_dist_cfg,
)


def test_noisy_config_registered():
    assert "NoisyDistribution" in JOINT_DISTRIBUTION_CONFIG_REGISTRY
    assert (
        JOINT_DISTRIBUTION_CONFIG_REGISTRY["NoisyDistribution"]
        is NoisyDistributionConfig
    )


def test_build_noisy_config(dummy_distribution, add_one_noise_dist_cfg):
    cfg = build_joint_distribution_config(
        "NoisyDistribution",
        base_distribution_config=DummyJointDistribution._Config(),
        noise_distribution_config=add_one_noise_dist_cfg,
    )
    assert isinstance(cfg, NoisyDistributionConfig)
    assert cfg.distribution_type == "NoisyDistribution"
    dummy_cfg = DummyJointDistribution._Config()
    assert cfg.input_shape == dummy_cfg.input_shape
    assert cfg.output_shape == dummy_cfg.output_shape


def test_noisy_config_json_roundtrip(add_one_noise_dist_cfg):
    base_cfg = MappedJointDistributionConfig(
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
    cfg = NoisyDistributionConfig(
        base_distribution_config=base_cfg,
        noise_distribution_config=add_one_noise_dist_cfg,
    )
    json_str = cfg.to_json()
    restored = NoisyDistributionConfig.from_json(json_str)
    assert restored == cfg


def test_noisy_config_from_dict_via_registry(add_one_noise_dist_cfg):
    data = {
        "distribution_type": "NoisyDistribution",
        "base_distribution_config": {
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
        },
        "noise_distribution_config": {
            "distribution_type": "AddOneNoiseDistribution",
            "input_shape": [1],
            "dtype": "float32",
        },
    }
    cfg = build_joint_distribution_config_from_dict(data)
    assert isinstance(cfg, NoisyDistributionConfig)
    assert cfg.input_shape == torch.Size([2])
    assert cfg.output_shape == torch.Size([1])
