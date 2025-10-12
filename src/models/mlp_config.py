from dataclasses import dataclass, field
from typing import List

from dataclasses_json import config, dataclass_json
from marshmallow import fields


@dataclass_json
@dataclass(kw_only=True)
class MLPConfig:
    input_dim: int
    output_dim: int
    hidden_dims: List[int]
    activation: str
    start_activation: bool
    end_activation: bool
    bias: bool = field(
        default=True,
        metadata=config(
            mm_field=fields.Boolean(load_default=True, dump_default=True)
        ),
    )

