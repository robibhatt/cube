from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import List

from .batch_experiment import BatchExperimentConfig
from .experiment_config_registry import register_experiment_config
from .mlp_recovery import MLPRecoveryConfig


@register_experiment_config("MassMLPRecovery")
@dataclass_json
@dataclass
class MassMLPRecoveryConfig(BatchExperimentConfig):
    """Configuration for :class:`MassMLPRecovery` experiments."""

    mlp_recovery_config: MLPRecoveryConfig
    train_sizes: List[int]
    learning_rates: List[float]
    weight_decays: List[float]
    rerun_plots: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:  # type: ignore[override]
        self.experiment_type = "MassMLPRecovery"
