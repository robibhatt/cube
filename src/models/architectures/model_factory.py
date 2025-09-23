from __future__ import annotations

from typing import Any

from .model_registry import MODEL_REGISTRY
from .configs.base import ModelConfig
from .model import Model


def create_model(config: ModelConfig, **kwargs: Any) -> Model:
    """Factory method that takes a ModelConfig and returns the corresponding model instance."""
    model_cls = MODEL_REGISTRY.get(config.model_type)
    if model_cls is None:
        raise ValueError(f"Model type '{config.model_type}' is not registered.")

    return model_cls(config, **kwargs)
