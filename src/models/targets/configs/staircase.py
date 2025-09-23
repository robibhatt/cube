from dataclasses import dataclass, field
import torch
from dataclasses_json import dataclass_json, config

from .base import TargetFunctionConfig
from .target_function_config_registry import register_target_function_config
from src.utils.serialization_utils import encode_shape, decode_shape


@register_target_function_config("StaircaseTarget")
@dataclass_json
@dataclass(kw_only=True)
class StaircaseTargetConfig(TargetFunctionConfig):
    input_shape: torch.Size = field(metadata=config(encoder=encode_shape, decoder=decode_shape))
    k: int

    def __post_init__(self) -> None:
        self.model_type = "StaircaseTarget"
        self.output_shape = torch.Size([1])
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.k > int(torch.tensor(self.input_shape).prod().item()):
            raise ValueError("k cannot exceed input dimension")
        super().__post_init__()
