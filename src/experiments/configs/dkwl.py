from dataclasses import dataclass, field
from typing import List

from dataclasses_json import dataclass_json

from src.experiments.configs.experiment import ExperimentConfig
from src.experiments.configs.experiment_config_registry import (
    register_experiment_config,
)


@register_experiment_config("DKWL")
@dataclass_json
@dataclass
class DkwlExperimentConfig(ExperimentConfig):
    """Configuration for the DKWL batch experiment."""

    ds: List[int] = field(default_factory=list)
    ks: List[int] = field(default_factory=list)
    widths: List[int] = field(default_factory=list)
    layers: List[int] = field(default_factory=list)
    train_sizes: List[int] = field(default_factory=list)
    epochs: List[int] = field(default_factory=list)
    l1_decays: List[float] = field(default_factory=list)
    mse_threshold: float = 0.01
    mse_samples: int = 8192
    ancestor_threshold: int = 2
    learning_rates: List[float] = field(default_factory=list)
    batch_sizes: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.experiment_type = "DKWL"
