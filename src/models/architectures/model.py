from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from .configs.base import ModelConfig
from torch import Tensor


class Model(nn.Module, ABC):
    """Base class for all models used in experiments."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    def initialize_weights(self) -> None:  # pragma: no cover - default no-op
        """Initialize model parameters.

        Subclasses may override this to set up their parameters.
        """
        pass

    def get_base_model(self) -> "Model | None":  # pragma: no cover - default no-op
        """Return a base-width reference model if available.

        Models that do not implement MuP-style width scaling simply return
        ``None``.  Subclasses may override this to provide a concrete model
        used for shape initialisation.
        """
        return None

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Compute the forward pass of the model."""
        pass


