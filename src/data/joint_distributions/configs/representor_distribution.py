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


@register_joint_distribution_config("RepresentorDistribution")
@dataclass_json
@dataclass(kw_only=True)
class RepresentorDistributionConfig(JointDistributionConfig):
    base_distribution_config: JointDistributionConfig = field(
        metadata=config(encoder=lambda c: c.to_dict(), decoder=_decode_joint_cfg)
    )
    model_config: ModelConfig = field(
        metadata=config(encoder=lambda m: m.to_dict(), decoder=_decode_model_cfg)
    )
    checkpoint_dir: Path = field(
        metadata=config(encoder=encode_path, decoder=decode_path)
    )
    from_rep: int
    to_rep: int

    def __post_init__(self) -> None:
        representor = create_model_representor(
            self.model_config, self.checkpoint_dir, device=torch.device("cpu")
        )
        self.input_shape = representor.representation_shape(self.from_rep)
        self.output_shape = representor.representation_shape(self.to_rep)
        self.distribution_type = "RepresentorDistribution"

