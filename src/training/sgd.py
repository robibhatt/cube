"""μP-aware stochastic gradient descent implementation."""

from __future__ import annotations

from typing import List

from mup import MuSGD, set_base_shapes

from src.models.mlp import MLP
from src.training.sgd_config import SgdConfig


class NullStepper:
    """Minimal stand-in for an optimiser with no parameters."""

    def __init__(self) -> None:
        self.state: dict = {}
        self.param_groups: list[dict] = []

    def step(self, closure=None) -> None:  # pragma: no cover - no behaviour
        pass

    def zero_grad(self, set_to_none: bool = False) -> None:  # pragma: no cover - no behaviour
        pass

    def state_dict(self) -> dict:  # pragma: no cover - deterministic empty state
        return {}

    def load_state_dict(self, state_dict: dict) -> None:  # pragma: no cover - no behaviour
        pass


class Sgd:
    """Project specific wrapper around :class:`mup.MuSGD`."""

    def __init__(self, config: SgdConfig, model: MLP) -> None:
        self.config = config
        self.model = model

        param_groups = self._parameter_groups()
        if not param_groups:
            self.stepper = NullStepper()
            return

        assert (
            self.config.mup
        ), "SGD optimizer now assumes μP training; set mup=True in the config."

        base_model = self.model.get_base_model()
        rescale = not getattr(self.model, "mup", False)
        set_base_shapes(self.model, base_model, rescale_params=rescale)
        self.stepper = MuSGD(
            param_groups,
            lr=self.config.lr,
            momentum=0.0,
            weight_decay=self.config.weight_decay,
        )

    def step(self) -> None:
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
