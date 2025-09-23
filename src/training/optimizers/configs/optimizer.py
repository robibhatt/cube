from dataclasses import dataclass, field
from abc import ABC
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class OptimizerConfig(ABC):
    optimizer_type: str = field(init=False)
