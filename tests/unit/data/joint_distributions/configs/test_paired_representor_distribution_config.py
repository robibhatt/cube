import torch

from src.data.joint_distributions.configs.paired_representor_distribution import (
    PairedRepresentorDistributionConfig,
)
from src.data.joint_distributions.configs.joint_distribution_config_registry import (
    JOINT_DISTRIBUTION_CONFIG_REGISTRY,
    build_joint_distribution_config_from_dict,
)


def test_paired_representor_config_registry():
    assert "PairedRepresentorDistribution" in JOINT_DISTRIBUTION_CONFIG_REGISTRY
    assert (
        JOINT_DISTRIBUTION_CONFIG_REGISTRY["PairedRepresentorDistribution"]
        is PairedRepresentorDistributionConfig
    )


def test_paired_representor_config_json_roundtrip(trained_trainer):
    cfg = PairedRepresentorDistributionConfig(
        base_distribution_config=trained_trainer.config.joint_distribution_config,
        teacher_model_config=trained_trainer.config.model_config,
        teacher_checkpoint_dir=trained_trainer.checkpoint_dir,
        teacher_rep_id=1,
        teacher_from_rep_id=0,
        student_model_config=trained_trainer.config.model_config,
        student_checkpoint_dir=trained_trainer.checkpoint_dir,
        student_rep_id=0,
        student_from_rep_id=0,
    )
    json_str = cfg.to_json()
    restored = PairedRepresentorDistributionConfig.from_json(json_str)
    assert restored.teacher_model_config.to_dict() == cfg.teacher_model_config.to_dict()
    assert restored.student_model_config.to_dict() == cfg.student_model_config.to_dict()
    assert restored.base_distribution_config.to_dict() == cfg.base_distribution_config.to_dict()
    assert restored.teacher_checkpoint_dir == cfg.teacher_checkpoint_dir
    assert restored.student_checkpoint_dir == cfg.student_checkpoint_dir
    assert restored.teacher_rep_id == cfg.teacher_rep_id
    assert restored.student_rep_id == cfg.student_rep_id
    assert restored.teacher_from_rep_id == cfg.teacher_from_rep_id
    assert restored.student_from_rep_id == cfg.student_from_rep_id


def test_paired_representor_config_from_dict_via_registry(trained_trainer):
    cfg_dict = {
        "distribution_type": "PairedRepresentorDistribution",
        "base_distribution_config": trained_trainer.config.joint_distribution_config.to_dict(),
        "teacher_model_config": trained_trainer.config.model_config.to_dict(),
        "teacher_checkpoint_dir": str(trained_trainer.checkpoint_dir),
        "teacher_rep_id": 1,
        "teacher_from_rep_id": 0,
        "student_model_config": trained_trainer.config.model_config.to_dict(),
        "student_checkpoint_dir": str(trained_trainer.checkpoint_dir),
        "student_rep_id": 0,
        "student_from_rep_id": 0,
    }
    cfg = build_joint_distribution_config_from_dict(cfg_dict)
    assert isinstance(cfg, PairedRepresentorDistributionConfig)
    assert cfg.teacher_rep_id == 1
    assert cfg.student_rep_id == 0
    assert cfg.teacher_from_rep_id == 0
    assert cfg.student_from_rep_id == 0
