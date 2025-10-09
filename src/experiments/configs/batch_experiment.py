from dataclasses import dataclass

from dataclasses_json import dataclass_json

from src.experiments.configs.experiment import ExperimentConfig
from src.experiments.configs.experiment_config_registry import (
    register_experiment_config,
)


@register_experiment_config("BatchExperiment")
@dataclass_json
@dataclass
class BatchExperimentConfig(ExperimentConfig):
    """Configuration for :class:`BatchExperiment`."""

    def __post_init__(self) -> None:
        self.experiment_type = "BatchExperiment"
