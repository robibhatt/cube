"""Experiment config package bootstrap."""

from importlib import import_module

from src.utils.plugin_loader import import_submodules

from .experiment_config_registry import (
    EXPERIMENT_CONFIG_REGISTRY,
    register_experiment_config,
    build_experiment_config,
    build_experiment_config_from_dict,
)

# Import all submodules so their registration decorators run and populate
# :data:`EXPERIMENT_CONFIG_REGISTRY`.
import_submodules(__name__)

__all__ = [
    "EXPERIMENT_CONFIG_REGISTRY",
    "register_experiment_config",
    "build_experiment_config",
    "build_experiment_config_from_dict",
]


def __getattr__(name: str):
    if name == "ExperimentConfig":
        module = import_module(".experiment", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
