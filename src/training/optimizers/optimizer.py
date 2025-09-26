"""Base classes for project optimizers.

This module previously assumed that every optimiser would have at least one
parameter to update.  In some experiment settings, however, we may construct an
optimiser for a "teacher" network that is not meant to be trained and therefore
provides no parameters.  PyTorch raises a ``ValueError`` when an optimiser is
instantiated with an empty parameter list.  To keep the rest of the training
code unchanged we provide a tiny no-op stepper that mimics the required API but
performs no updates.  Optimiser implementations can fall back to this stepper
when no parameters are found.
"""

from src.training.optimizers.configs.optimizer import OptimizerConfig
from src.models.architectures.mlp import MLP
from abc import ABC
from typing import List


class NullStepper:
    """Minimal stand-in for a Torch optimiser with no parameters.

    The object exposes the methods and attributes that the training loop expects
    (`step`, `zero_grad`, `state_dict`, `load_state_dict`, and `state`).  Each of
    these is implemented as a no-op so that training continues smoothly when
    there is nothing to optimise.
    """

    def __init__(self) -> None:  # pragma: no cover - trivial initialisation
        self.state = {}

    def step(self, closure=None) -> None:  # pragma: no cover - no behaviour
        pass

    def zero_grad(self, set_to_none: bool = False) -> None:  # pragma: no cover - no behaviour
        pass

    def state_dict(self) -> dict:  # pragma: no cover - deterministic empty state
        return {}

    def load_state_dict(self, state_dict: dict) -> None:  # pragma: no cover - no behaviour
        pass

class Optimizer(ABC):
    def __init__(self, config: OptimizerConfig, model: MLP):
        self.config = config
        self.model = model

    def step(self):
        """Advance the underlying optimiser by one step."""
        self.stepper.step()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _parameter_groups(self) -> List[dict]:
        """Return parameter groups excluding biases from weight decay."""
        wd_params, no_wd_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            (no_wd_params if name.endswith("bias") else wd_params).append(param)

        groups: List[dict] = []
        if wd_params:
            groups.append({"params": wd_params})
        if no_wd_params:
            groups.append({"params": no_wd_params, "weight_decay": 0.0})
        return groups


