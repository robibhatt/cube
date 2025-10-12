from dataclasses import dataclass, field
from typing import List
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(kw_only=True)
class MLPConfig:
    input_dim: int
    output_dim: int
    hidden_dims: List[int]
    activation: str
    start_activation: bool
    end_activation: bool
    bias: bool = True
    mup: bool = True
    frozen_layers: List[int] = field(default_factory=list)
    model_type: str = field(init=False, default="MLP")

    def __post_init__(self):
        self.model_type = "MLP"
        k = len(self.hidden_dims)
        for idx in self.frozen_layers:
            if idx < 1 or idx > k + 1:
                raise ValueError(
                    f"Invalid frozen layer index {idx}; valid range is [1, {k + 1}]"
                )

