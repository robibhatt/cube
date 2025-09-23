from __future__ import annotations

  # pragma: no cover - only for type hints
from .experiment import ExperimentConfig

EXPERIMENT_CONFIG_REGISTRY: dict[str, ExperimentConfig] = {}


def register_experiment_config(name: str):
    """Class decorator to register an :class:`ExperimentConfig` subclass."""

    def decorator(cls: ExperimentConfig):
        EXPERIMENT_CONFIG_REGISTRY[name] = cls
        return cls

    return decorator


def build_experiment_config(name: str, **kwargs) -> ExperimentConfig:
    """Instantiate a registered ``ExperimentConfig`` by ``name``."""

    cfg_cls = EXPERIMENT_CONFIG_REGISTRY.get(name)
    if cfg_cls is None:
        raise ValueError(f"Experiment config '{name}' is not registered.")
    # ``fully_initialized`` used to be managed internally and may still
    # appear in old configuration files
    kwargs.pop("fully_initialized", None)
    # use dataclasses_json for encoding/decoding
    return cfg_cls.from_dict({"experiment_type": name, **kwargs})


def build_experiment_config_from_dict(data: dict) -> ExperimentConfig:
    """Construct an ``ExperimentConfig`` from a dictionary."""

    data = dict(data)
    # drop legacy ``fully_initialized`` key if present
    data.pop("fully_initialized", None)
    name = data.pop("experiment_type", None)
    if name is None:
        raise ValueError("Missing 'experiment_type' key in experiment config dictionary.")
    cfg = build_experiment_config(name, **data)
    return cfg
