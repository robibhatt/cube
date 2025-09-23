import torch
# Import provider to register with the configuration registry for tests.
import src.data.providers.tensor_data_provider
from src.data.joint_distributions.configs.representor_distribution import (
    RepresentorDistributionConfig,
)
from src.data.joint_distributions.configs.joint_distribution_config_registry import (
    JOINT_DISTRIBUTION_CONFIG_REGISTRY,
    build_joint_distribution_config,
    build_joint_distribution_config_from_dict,
)
from src.data.joint_distributions.configs.mapped_joint_distribution import MappedJointDistributionConfig
from src.data.joint_distributions.configs.gaussian import GaussianConfig
from conftest import LinearTargetFunctionConfig
from tests.unit.data.conftest import DummyJointDistribution
from tests.unit.data.conftest import trained_trainer


def test_representor_config_registered():
    assert "RepresentorDistribution" in JOINT_DISTRIBUTION_CONFIG_REGISTRY
    assert (
        JOINT_DISTRIBUTION_CONFIG_REGISTRY["RepresentorDistribution"]
        is RepresentorDistributionConfig
    )


def test_build_representor_config(trained_trainer):
    cfg = build_joint_distribution_config(
        "RepresentorDistribution",
        base_distribution_config=trained_trainer.config.joint_distribution_config,
        model_config=trained_trainer.config.model_config,
        checkpoint_dir=trained_trainer.checkpoint_dir,
        from_rep=0,
        to_rep=1,
    )
    assert isinstance(cfg, RepresentorDistributionConfig)
    assert cfg.distribution_type == "RepresentorDistribution"
    assert isinstance(cfg.input_shape, torch.Size)
    assert isinstance(cfg.output_shape, torch.Size)


def test_representor_config_json_roundtrip(trained_trainer):
    base_cfg = MappedJointDistributionConfig(
        distribution_config=GaussianConfig(
            input_shape=torch.Size([2]), mean=0.0, std=1.0
        ),
        target_function_config=LinearTargetFunctionConfig(input_shape=torch.Size([2])),
    )
    cfg = RepresentorDistributionConfig(
        base_distribution_config=trained_trainer.config.joint_distribution_config,
        model_config=trained_trainer.config.model_config,
        checkpoint_dir=trained_trainer.checkpoint_dir,
        from_rep=0,
        to_rep=1,
    )
    json_str = cfg.to_json()
    restored = RepresentorDistributionConfig.from_json(json_str)
    assert restored.model_config.to_dict() == cfg.model_config.to_dict()
    assert restored.base_distribution_config.to_dict() == cfg.base_distribution_config.to_dict()
    assert restored.checkpoint_dir == cfg.checkpoint_dir
    assert restored.from_rep == cfg.from_rep
    assert restored.to_rep == cfg.to_rep


def test_representor_config_from_dict_via_registry(trained_trainer):
    data = {
        "distribution_type": "RepresentorDistribution",
        "base_distribution_config": trained_trainer.config.joint_distribution_config.to_dict(),
        "model_config": trained_trainer.config.model_config.to_dict(),
        "checkpoint_dir": str(trained_trainer.checkpoint_dir),
        "from_rep": 0,
        "to_rep": 1,
    }
    cfg = build_joint_distribution_config_from_dict(data)
    assert isinstance(cfg, RepresentorDistributionConfig)
    assert isinstance(cfg.input_shape, torch.Size)
    assert isinstance(cfg.output_shape, torch.Size)
