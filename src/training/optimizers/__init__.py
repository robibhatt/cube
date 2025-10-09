"""Optimizer package bootstrap.

This package exposes the optimizer registry and ensures that all optimizer
implementations register themselves via import side effects. Previously the
modules were imported manually, but ``import_submodules`` now loads them
automatically.
"""

from src.utils.plugin_loader import import_submodules

from src.training.optimizers.optimizer_registry import (
    OPTIMIZER_REGISTRY,
    register_optimizer,
)

# Import all submodules (e.g. ``adam`` and ``sgd``) so their registration
# decorators run and populate :data:`OPTIMIZER_REGISTRY`.
import_submodules(__name__)

__all__ = ["OPTIMIZER_REGISTRY", "register_optimizer"]

