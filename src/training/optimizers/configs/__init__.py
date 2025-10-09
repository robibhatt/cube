"""Optimizer config package bootstrap."""

from src.utils.plugin_loader import import_submodules
from src.training.optimizers.configs.optimizer_config_registry import (
    OPTIMIZER_CONFIG_REGISTRY,
    register_optimizer_config,
    build_optimizer_config,
    build_optimizer_config_from_dict,
)

# Import all submodules so their registration decorators run and populate
# :data:`OPTIMIZER_CONFIG_REGISTRY`.
import_submodules(__name__)

__all__ = [
    "OPTIMIZER_CONFIG_REGISTRY",
    "register_optimizer_config",
    "build_optimizer_config",
    "build_optimizer_config_from_dict",
]
