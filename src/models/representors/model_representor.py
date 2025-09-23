from __future__ import annotations

from abc import ABC, abstractmethod
from torch import Tensor
import torch
from pathlib import Path
from typing import List, Any, Optional

import torch.nn as nn

from src.checkpoints.checkpoint import Checkpoint
from src.models.architectures.model import Model
from src.models.architectures.model_factory import create_model
from src.models.architectures.configs.base import ModelConfig

class ModelRepresentor(ABC):
    """Abstract base class for working with **internal representations**.

    The class defines two complementary capabilities:

    1. **Configuration builders** – methods that *construct* :class:`ModelConfig`
       objects describing the *sub‑network* bridging one internal representation
       to another (``forward_config``).

    2. **Tensor utilities** – methods that operate on *activations* at runtime.

    A concrete implementation **must** be initialised with a :class:`ModelConfig`
    that includes a non‑null ``checkpoint_path``; otherwise an ``AssertionError`` is
    raised.

    Notes
    -----
    *Representation identifiers* (``rep_id``) are assumed to be **integers** in
    the range ``0 ≤ rep_id < n_representations`` of the model.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __init__(
        self,
        model_config: ModelConfig,
        checkpoint_dir: Path,
        device: torch.device,
    ):
        """Load a trained model using ``model_config`` and ``checkpoint_dir``."""

        self.model_config: ModelConfig = model_config
        self.checkpoint_dir: Path = checkpoint_dir
        self.device: torch.device = device

        try:
            checkpoint = Checkpoint.from_dir(checkpoint_dir)
            model = create_model(model_config)
            checkpoint.load(model=model)
            model.to(self.device)
            model.eval()
            self.model = model
            self.modules = self.get_modules()
        except Exception:
            self.model = None
            self.modules = None
        

    
    def forward(self, input_tensor: Tensor, from_rep:int, to_rep:int, target: Optional[Model]):
        X = input_tensor.to(self.device)
        for i in range(from_rep):
            X = self.modules[i](X)
        y = X
        for j in range(from_rep, to_rep):
            y = self.modules[j](y)
        if target is not None:
            y = target(y)
        return X, y

    def get_module(self, from_rep: int, to_rep: int) -> nn.Module:
        """Return a module mapping ``from_rep`` to ``to_rep``.

        Parameters
        ----------
        from_rep:
            Index of the source representation (inclusive).
        to_rep:
            Index of the destination representation (exclusive).

        Returns
        -------
        nn.Module
            An ``nn.Sequential`` comprising the layers that map between the
            two representations.
        """

        if self.modules is None:
            raise RuntimeError("Model modules have not been initialised")
        assert 0 <= from_rep < to_rep <= len(self.modules), (
            "from_rep must be < to_rep and within the number of modules"
        )
        return nn.Sequential(*self.modules[from_rep:to_rep])
    
    @abstractmethod
    def forward_config(self, from_rep:int, to_rep:int) -> ModelConfig:
        pass
      
    @abstractmethod
    def get_modules(self) -> List[nn.Module]:
        """Return a list of modules."""
        pass

    @abstractmethod
    def to_representation_dict(self, rep_id: int) -> dict[str, Any]:
        """Return a dictionary of representation information."""
        pass

    @abstractmethod
    def from_representation_dict(self, rep_dict: dict[str, Any]) -> int:
        """Get the id from a representation information dictionary"""
        pass

    @abstractmethod
    def representation_shape(self, rep_id: int) -> torch.Size:
        """Return the tensor shape of representation ``rep_id``."""
        pass

