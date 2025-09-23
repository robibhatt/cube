import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple

from .joint_distribution import JointDistribution
from .configs.base import JointDistributionConfig


@dataclass
class _Config(JointDistributionConfig):
    """Lightweight config holding input and output shapes."""

    input_shape: torch.Size
    output_shape: torch.Size

    def __post_init__(self) -> None:  # type: ignore[override]
        self.distribution_type = "ModuleJointDistribution"


class ModuleJointDistribution(JointDistribution):
    """Joint distribution defined by arbitrary modules.

    This distribution wraps another :class:`JointDistribution` and applies two
    ``nn.Module`` transformations: ``x_module`` for the features and ``y_module``
    for the targets. ``y_module`` receives the base distribution's feature
    representation rather than its original targets. No configuration object is
    required; instead the shapes are inferred from the modules' outputs when
    applied to a dummy input matching the base distribution's input shape.
    """

    def __init__(
        self,
        base_distribution: JointDistribution,
        x_module: nn.Module,
        y_module: nn.Module,
    ) -> None:
        self.base_joint_distribution = base_distribution
        device = base_distribution.device
        self.x_module = x_module.to(device)
        self.y_module = y_module.to(device)

        # Infer shapes by sampling from the base distribution and passing the
        # resulting features through the provided modules.
        with torch.no_grad():
            X_base, _ = base_distribution.sample(1, seed=0)
            x_dummy = self.x_module(X_base.to(device))
            y_dummy = self.y_module(X_base.to(device))
        cfg = _Config(input_shape=x_dummy.shape[1:], output_shape=y_dummy.shape[1:])
        super().__init__(config=cfg, device=device)

    def __str__(self) -> str:
        return (
            f"ModuleJointDistribution(base_distribution={self.base_joint_distribution}, "
            f"x_module={self.x_module}, y_module={self.y_module})"
        )

    def base_sample(self, n_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.base_joint_distribution.base_sample(n_samples, seed)

    def forward_X(self, base_X: torch.Tensor) -> torch.Tensor:
        X_base = self.base_joint_distribution.forward_X(base_X)
        return self.x_module(X_base)

    def forward_Y(self, base_X: torch.Tensor) -> torch.Tensor:
        X_base = self.base_joint_distribution.forward_X(base_X)
        return self.y_module(X_base)

    def forward(self, base_X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X_base = self.base_joint_distribution.forward_X(base_X)
        return self.x_module(X_base), self.y_module(X_base)

    def sample(self, n_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        base_X, _ = self.base_sample(n_samples, seed)
        return self.forward(base_X)

    def preferred_provider(self) -> str:
        return self.base_joint_distribution.preferred_provider()
