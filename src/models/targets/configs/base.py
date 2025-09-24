from dataclasses import dataclass, field
from abc import ABC

import torch
from dataclasses_json import dataclass_json, config

from src.utils.serialization_utils import encode_shape, decode_shape


@dataclass_json
@dataclass(kw_only=True)
class TargetFunctionConfig(ABC):
    """Base configuration for target functions."""

    model_type: str = field(init=False)
    input_shape: torch.Size = field(
        metadata=config(encoder=encode_shape, decoder=decode_shape)
    )
    output_shape: torch.Size = field(
        init=False,
        metadata=config(encoder=encode_shape, decoder=decode_shape),
    )

    def __post_init__(self) -> None:
        if not isinstance(self.input_shape, torch.Size):
            raise TypeError(
                f"input_shape must be torch.Size, got {type(self.input_shape)}"
            )
        if not isinstance(getattr(self, "model_type", None), str):
            raise TypeError(
                f"model_type must be a str, got {type(getattr(self, 'model_type', None))}"
            )

