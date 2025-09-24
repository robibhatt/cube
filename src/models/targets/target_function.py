from __future__ import annotations

import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from .configs.base import TargetFunctionConfig


class TargetFunction(nn.Module, ABC):
    def __init__(self, config: TargetFunctionConfig):
        super().__init__()
        self.config = config
        self.input_shape = config.input_shape
        self.output_shape = config.output_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        return self._forward(x)

    def initialize_weights(self) -> None:  # pragma: no cover - default no-op
        """Initialize model parameters.

        Subclasses may override this to set up their parameters.
        """
        pass

    def get_base_model(self) -> "TargetFunction | None":  # pragma: no cover - default no-op
        """Return a base-width reference model if available.

        Target functions that do not implement MuP-style width scaling simply
        return ``None``.  Subclasses may override this to provide a concrete
        model used for shape initialisation.
        """
        return None

    @abstractmethod
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


    def _validate_input(self, x: torch.Tensor) -> None:
        # Allow any number of leading batch dimensions. Only the trailing
        # dimensions must match ``input_shape``.
        expected_rank = len(self.input_shape)

        # 1) Require at least one batch dimension
        if x.ndim <= expected_rank:
            raise ValueError(
                f"Input must include at least one batch dimension. "
                f"Expected tensor of rank > {expected_rank} (batch + {self.input_shape}), "
                f"but got rank {x.ndim}.",
            )

        # 2) Ensure trailing dims match ``input_shape``
        if x.shape[-expected_rank:] != self.input_shape:
            raise ValueError(
                f"Input shape mismatch. Expected trailing dimensions {self.input_shape}, "
                f"but got {tuple(x.shape)}.",
            )
