from dataclasses import dataclass

from dataclasses_json import dataclass_json

from .experiment import ExperimentConfig
from .experiment_config_registry import register_experiment_config
from src.training.trainer_config import TrainerConfig

@register_experiment_config("MLPRecovery")
@dataclass_json
@dataclass
class MLPRecoveryConfig(ExperimentConfig):
    """Configuration for MLP recovery experiments."""

    teacher_trainer_config: TrainerConfig
    student_trainer_config: TrainerConfig
    noise_variance: float = 0.0
    start_layer: int = 0

    def __post_init__(self) -> None:
        self.experiment_type = "MLPRecovery"
