from dataclasses import dataclass
from dataclasses_json import dataclass_json

from src.experiments.configs.experiment import ExperimentConfig
from src.experiments.configs.experiment_config_registry import (
    register_experiment_config,
)
from src.training.trainer_config import TrainerConfig


@register_experiment_config("TrainMLP")
@dataclass_json
@dataclass
class TrainMLPExperimentConfig(ExperimentConfig):
    """Configuration for training a single MLP."""

    trainer_config: TrainerConfig
    edge_threshold: float = 0.0

    def __post_init__(self) -> None:
        self.experiment_type = "TrainMLP"
