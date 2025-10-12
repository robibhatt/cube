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
    mup: bool = field(
        default=True,
        metadata=config(
            mm_field=fields.Boolean(load_default=True, dump_default=True)
        ),
    )
    frozen_layers: List[int] = field(
        default_factory=list,
        metadata=config(
            mm_field=fields.List(
                fields.Integer(),
                load_default=list,
                dump_default=list,
            )
        ),
    )
    model_type: str = field(
        init=False,
        default="MLP",
        metadata=config(
            mm_field=fields.String(load_default="MLP", dump_default="MLP")
        ),
    )

    def __post_init__(self):
        self.model_type = "MLP"
        k = len(self.hidden_dims)
        for idx in self.frozen_layers:
            if idx < 1 or idx > k + 1:
                raise ValueError(
                    f"Invalid frozen layer index {idx}; valid range is [1, {k + 1}]"
                )

