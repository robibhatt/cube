from __future__ import annotations

import torch
from pathlib import Path

from .model_representor import ModelRepresentor
from .representor_registry import REPRESENTOR_REGISTRY
from src.models.architectures.configs.base import ModelConfig


def create_model_representor(
    model_config: ModelConfig, checkpoint_dir: Path, device: torch.device
) -> ModelRepresentor:
    """Instantiate a registered ``ModelRepresentor`` from ``model_config``."""

    model_type = model_config.model_type

    cls = REPRESENTOR_REGISTRY.get(model_type)
    if cls is None:
        raise ValueError(f"Model type '{model_type}' is not registered.")

    return cls(model_config, checkpoint_dir, device=device)

