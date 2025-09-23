from dataclasses import dataclass, field
from typing import List

import torch
from dataclasses_json import dataclass_json, config

from .base import TargetFunctionConfig
from .target_function_config_registry import register_target_function_config
from src.utils.serialization_utils import encode_shape, decode_shape


@register_target_function_config("ProdKTarget")
@dataclass_json
@dataclass(kw_only=True)
class ProdKTargetConfig(TargetFunctionConfig):
    """Configuration for :class:`ProdKTarget`.

    The :class:`ProdKTarget` computes the product of selected input coordinates
    specified by ``indices``.  Indices use zero-based numbering with respect to
    the flattened input tensor.
    """

    input_shape: torch.Size = field(
        metadata=config(encoder=encode_shape, decoder=decode_shape)
    )
    indices: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.model_type = "ProdKTarget"
        self.output_shape = torch.Size([1])
        if not self.indices:
            raise ValueError("indices must be a non-empty list")
        total_dim = int(torch.tensor(self.input_shape).prod().item())
        if any(i < 0 or i >= total_dim for i in self.indices):
            raise ValueError("indices cannot exceed input dimension or be negative")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("indices must be unique")
        super().__post_init__()
