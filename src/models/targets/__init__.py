"""Target function package bootstrap."""

from src.utils.plugin_loader import import_submodules

from .target_function_registry import register_target_function, TARGET_FUNCTION_REGISTRY
from .target_function_factory import create_target_function

# Import all submodules so their registration decorators run and populate
# :data:`TARGET_FUNCTION_REGISTRY`.
import_submodules(__name__)

__all__ = [
    "register_target_function",
    "TARGET_FUNCTION_REGISTRY",
    "create_target_function",
]
