from dataclasses import dataclass, field
from abc import ABC
import torch
from dataclasses_json import dataclass_json, config

from src.utils.serialization_utils import encode_shape, decode_shape

@dataclass_json
@dataclass
class JointDistributionConfig(ABC):
    """Base configuration for joint distributions (X, y)."""

    distribution_type: str = field(init=False)

    input_shape: torch.Size = field(
        init=False,
        metadata=config(encoder=encode_shape, decoder=decode_shape),
    )

    output_shape: torch.Size = field(
        init=False,
        metadata=config(encoder=encode_shape, decoder=decode_shape),
    )
