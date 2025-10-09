from dataclasses import dataclass
from dataclasses_json import dataclass_json

from src.training.optimizers.configs.optimizer import OptimizerConfig
from src.training.optimizers.configs.optimizer_config_registry import register_optimizer_config


@register_optimizer_config("Adam")
@dataclass_json
@dataclass(kw_only=True)
class AdamConfig(OptimizerConfig):
    lr: float
    mup: bool = True
    weight_decay: float = 0.0

    def __post_init__(self):
        self.optimizer_type = "Adam"

