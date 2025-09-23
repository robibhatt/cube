from dataclasses import dataclass, field
from pathlib import Path
import torch
from dataclasses_json import dataclass_json, config

from .base import JointDistributionConfig
from .joint_distribution_config_registry import (
    register_joint_distribution_config,
    build_joint_distribution_config_from_dict,
)
from src.models.representors.representor_factory import create_model_representor
from src.models.architectures.configs.base import ModelConfig
from src.models.architectures.configs.model_config_registry import build_model_config_from_dict
from src.utils.serialization_utils import encode_path, decode_path


def _decode_joint_cfg(d):
    from src.data.joint_distributions.configs.base import JointDistributionConfig

    return (
        d
        if isinstance(d, JointDistributionConfig)
        else build_joint_distribution_config_from_dict(d)
    )


def _decode_model_cfg(d):
    return (
        d if isinstance(d, ModelConfig) else build_model_config_from_dict(d)
    )


@register_joint_distribution_config("PairedRepresentorDistribution")
@dataclass_json
@dataclass(kw_only=True)
class PairedRepresentorDistributionConfig(JointDistributionConfig):
    base_distribution_config: JointDistributionConfig = field(
        metadata=config(encoder=lambda c: c.to_dict(), decoder=_decode_joint_cfg)
    )
    teacher_model_config: ModelConfig = field(
        metadata=config(encoder=lambda m: m.to_dict(), decoder=_decode_model_cfg)
    )
    teacher_checkpoint_dir: Path = field(
        metadata=config(encoder=encode_path, decoder=decode_path)
    )
    teacher_rep_id: int
    teacher_from_rep_id: int = 0
    student_model_config: ModelConfig = field(
        metadata=config(encoder=lambda m: m.to_dict(), decoder=_decode_model_cfg)
    )
    student_checkpoint_dir: Path = field(
        metadata=config(encoder=encode_path, decoder=decode_path)
    )
    student_rep_id: int
    student_from_rep_id: int = 0

    def __post_init__(self) -> None:
        teacher_repr = create_model_representor(
            self.teacher_model_config,
            self.teacher_checkpoint_dir,
            device=torch.device("cpu"),
        )
        student_repr = create_model_representor(
            self.student_model_config,
            self.student_checkpoint_dir,
            device=torch.device("cpu"),
        )
        self.input_shape = student_repr.representation_shape(self.student_rep_id)
        self.output_shape = teacher_repr.representation_shape(self.teacher_rep_id)
        self.distribution_type = "PairedRepresentorDistribution"

