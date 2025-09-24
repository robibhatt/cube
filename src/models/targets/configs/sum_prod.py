from dataclasses import dataclass, field
from typing import List

import torch
from dataclasses_json import dataclass_json, config

from .base import TargetFunctionConfig
from .target_function_config_registry import register_target_function_config
from src.utils.serialization_utils import encode_shape, decode_shape


@register_target_function_config("SumProdTarget")
@dataclass_json
@dataclass(kw_only=True)
class SumProdTargetConfig(TargetFunctionConfig):
    """Configuration for :class:`SumProdTarget`.

    ``indices_list`` is a list where each element is a list of indices for a
    product term. The resulting target function computes the weighted sum of
    the corresponding products of input coordinates using ``weights``.
    Set ``normalize`` to ``False`` to disable the variance normalization
    applied by :class:`SumProdTarget`.
    """

    input_shape: torch.Size = field(
        metadata=config(encoder=encode_shape, decoder=decode_shape)
    )
    indices_list: List[List[int]] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    normalize: bool = True

    def __post_init__(self) -> None:
        self.model_type = "SumProdTarget"
        self.output_shape = torch.Size([1])
        if not self.indices_list:
            raise ValueError("indices_list must be a non-empty list")
        if len(self.weights) != len(self.indices_list):
            raise ValueError("weights must have the same length as indices_list")
        total_dim = int(torch.tensor(self.input_shape).prod().item())
        for idx_group in self.indices_list:
            if not idx_group:
                raise ValueError("each indices group must be non-empty")
            if any(i < 0 or i >= total_dim for i in idx_group):
                raise ValueError("indices cannot exceed input dimension or be negative")
        super().__post_init__()
