from dataclasses import dataclass
from dataclasses_json import dataclass_json

from .experiment import ExperimentConfig
from .experiment_config_registry import register_experiment_config
from src.training.trainer_config import TrainerConfig


@register_experiment_config("ClimbStairs")
@dataclass_json
@dataclass
class StaircaseExperimentConfig(ExperimentConfig):
    """Configuration for training an MLP on a staircase distribution."""

    trainer_config: TrainerConfig

    def __post_init__(self) -> None:
        self.experiment_type = "ClimbStairs"
