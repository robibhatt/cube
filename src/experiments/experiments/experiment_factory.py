from __future__ import annotations

from typing import TYPE_CHECKING

from src.experiments.configs import ExperimentConfig
from .experiment_registry import EXPERIMENT_REGISTRY

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from .experiment import Experiment


def create_experiment(config: ExperimentConfig) -> "Experiment":
    """Instantiate an :class:`Experiment` using ``config``."""

    exp_cls = EXPERIMENT_REGISTRY.get(config.experiment_type)
    if exp_cls is None:
        raise ValueError(
            f"Experiment '{config.experiment_type}' is not registered."
        )

    return exp_cls(config)

