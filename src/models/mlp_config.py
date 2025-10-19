from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(kw_only=True)
class MLPConfig:
    input_dim: int
    output_dim: int
    hidden_dims: List[int]
    start_activation: bool
    end_activation: bool

