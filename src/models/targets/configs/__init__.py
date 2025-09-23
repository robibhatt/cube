"""Target function config package bootstrap."""

from src.utils.plugin_loader import import_submodules
from .target_function_config_registry import (
    TARGET_FUNCTION_CONFIG_REGISTRY,
    register_target_function_config,
    build_target_function_config,
    build_target_function_config_from_dict,
)

# Import all submodules so their registration decorators run and populate
# :data:`TARGET_FUNCTION_CONFIG_REGISTRY`.
import_submodules(__name__)

__all__ = [
    "TARGET_FUNCTION_CONFIG_REGISTRY",
    "register_target_function_config",
    "build_target_function_config",
    "build_target_function_config_from_dict",
]