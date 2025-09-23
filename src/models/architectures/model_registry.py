"""Registry for model architectures."""

from __future__ import annotations

from typing import Dict, Type

from .model import Model

MODEL_REGISTRY: Dict[str, Type[Model]] = {}


def register_model(model_type: str):
    """Class decorator to register a :class:`Model` subclass."""

    def decorator(cls: Type[Model]) -> Type[Model]:
        MODEL_REGISTRY[model_type] = cls
        return cls

    return decorator
