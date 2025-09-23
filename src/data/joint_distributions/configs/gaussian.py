from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config
import torch

from .base import JointDistributionConfig
from .joint_distribution_config_registry import register_joint_distribution_config
from src.utils.serialization_utils import encode_shape, decode_shape, encode_dtype, decode_dtype

@register_joint_distribution_config("Gaussian")
@dataclass_json
@dataclass(kw_only=True)
class GaussianConfig(JointDistributionConfig):
    input_shape: torch.Size = field(
        metadata=config(encoder=encode_shape, decoder=decode_shape)
    )
    dtype: torch.dtype = field(
        default=torch.float32,
        metadata=config(encoder=encode_dtype, decoder=decode_dtype)
    )
    mean: float
    std: float

    def __post_init__(self) -> None:
        self.output_shape = torch.Size([1])
        self.distribution_type = "Gaussian"
