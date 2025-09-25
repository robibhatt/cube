from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config
import torch

from .base import JointDistributionConfig
from .joint_distribution_config_registry import register_joint_distribution_config
from src.utils.serialization_utils import encode_dtype, decode_dtype


@register_joint_distribution_config("Hypercube")
@dataclass_json
@dataclass(kw_only=True)
class HypercubeConfig(JointDistributionConfig):
    input_dim: int
    dtype: torch.dtype = field(
        default=torch.float32,
        metadata=config(encoder=encode_dtype, decoder=decode_dtype)
    )

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        self.distribution_type = "Hypercube"
