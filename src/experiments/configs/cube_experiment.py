from dataclasses import dataclass
from dataclasses_json import dataclass_json

from .experiment import ExperimentConfig
from .experiment_config_registry import register_experiment_config


@register_experiment_config("Cube")
@dataclass_json
@dataclass
class CubeExperimentConfig(ExperimentConfig):
    """Configuration for :class:`CubeExperiment`."""

    dimension: int
    k: int
    depth: int
    width: int
    learning_rate: float
    weight_decay: float
    weight_decay_l1: float
    mup: bool
    epochs: int
    early_stopping: float
    train_size: int
    batch_size: int

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.k < 0 or self.k > self.dimension:
            raise ValueError("k must satisfy 0 <= k <= dimension")
        if self.epochs <= 0:
            raise ValueError("epochs must be greater than 0")
        if self.weight_decay_l1 < 0:
            raise ValueError("weight_decay_l1 must be non-negative")
        self.experiment_type = "Cube"
