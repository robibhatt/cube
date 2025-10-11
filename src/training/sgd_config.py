"""Configuration dataclass for the project SGD optimiser."""

from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(kw_only=True)
class SgdConfig:
    lr: float
    mup: bool = True
    weight_decay: float = 0.0
