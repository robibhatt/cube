from dataclasses import dataclass, field
import torch
from dataclasses_json import dataclass_json, config

from src.utils.serialization_utils import encode_shape, decode_shape
from .base import TargetFunctionConfig
from .target_function_config_registry import register_target_function_config


@register_target_function_config("1234_prod")
@dataclass_json
@dataclass(kw_only=True)
class Prod1234Config(TargetFunctionConfig):
    input_shape: torch.Size = field(metadata=config(encoder=encode_shape, decoder=decode_shape))

    def __post_init__(self) -> None:
        self.model_type = "1234_prod"
        self.output_shape = torch.Size([1])
        super().__post_init__()
