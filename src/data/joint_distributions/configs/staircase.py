from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config
import torch

from .base import JointDistributionConfig
from .joint_distribution_config_registry import register_joint_distribution_config
from src.utils.serialization_utils import encode_shape, decode_shape, encode_dtype, decode_dtype


@register_joint_distribution_config("Staircase")
@dataclass_json
@dataclass(kw_only=True)
class StaircaseConfig(JointDistributionConfig):
    input_shape: torch.Size = field(metadata=config(encoder=encode_shape, decoder=decode_shape))
    k: int
    dtype: torch.dtype = field(
        default=torch.float32,
        metadata=config(encoder=encode_dtype, decoder=decode_dtype),
    )

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.k > int(torch.tensor(self.input_shape).prod().item()):
            raise ValueError("k cannot exceed input dimension")
        self.output_shape = torch.Size([1])
        self.distribution_type = "Staircase"
