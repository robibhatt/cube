from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from abc import ABC
import torch
from dataclasses_json import dataclass_json, config

from src.utils.serialization_utils import encode_shape, decode_shape

@dataclass_json
@dataclass
class ModelConfig(ABC):
    model_type: str = field(init=False)


    input_shape: Optional[torch.Size] = field(
        default=None,
        metadata=config(encoder=encode_shape, decoder=decode_shape)
    )

    output_shape: Optional[torch.Size] = field(
        default=None,
        metadata=config(encoder=encode_shape, decoder=decode_shape)
    )

