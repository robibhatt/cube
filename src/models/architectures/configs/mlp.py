from dataclasses import dataclass, field
from typing import List
from dataclasses_json import dataclass_json

from .base import ModelConfig
from .model_config_registry import register_model_config
import torch

@register_model_config("MLP")
@dataclass_json
@dataclass(kw_only=True)
class MLPConfig(ModelConfig):
    input_dim: int
    output_dim: int
    hidden_dims: List[int]
    activation: str
    start_activation: bool
    end_activation: bool
    bias: bool = True
    mup: bool = False
    frozen_layers: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.model_type = "MLP"
        self.input_shape = torch.Size((self.input_dim,))
        self.output_shape = torch.Size((self.output_dim,))
        k = len(self.hidden_dims)
        for idx in self.frozen_layers:
            if idx < 1 or idx > k + 1:
                raise ValueError(
                    f"Invalid frozen layer index {idx}; valid range is [1, {k + 1}]"
                )

