import torch
from torch import Tensor
from typing import Tuple

from .joint_distribution import JointDistribution
from .joint_distribution_registry import register_joint_distribution
from .configs.staircase import StaircaseConfig
from .hypercube import Hypercube
from .configs.hypercube import HypercubeConfig
from src.models.targets.simple_functions import StaircaseTarget
from src.models.targets.configs.staircase import StaircaseTargetConfig


@register_joint_distribution("Staircase")
class Staircase(JointDistribution):
    r"""Hypercube inputs with staircase outputs.

    Samples ``X`` uniformly from the ``d``-dimensional hypercube ``{±1}^d`` and
    computes ``y`` via the staircase map
    ``(x₁ + x₁x₂ + … + x₁x₂⋯x_k) / \sqrt{k}`` using the first ``k`` coordinates of ``X``.
    """

    def __init__(self, config: StaircaseConfig, device: torch.device) -> None:
        super().__init__(config, device)
        self.k = config.k
        self.dtype = config.dtype
        hypercube_cfg = HypercubeConfig(input_shape=config.input_shape, dtype=config.dtype)
        self.hypercube = Hypercube(hypercube_cfg, device)
        target_cfg = StaircaseTargetConfig(input_shape=config.input_shape, k=config.k)
        self.target = StaircaseTarget(target_cfg).to(device)

    def __str__(self) -> str:
        return f"Staircase(d={self.input_shape[0]}, k={self.k})"

    def sample(self, n_samples: int, seed: int) -> Tuple[Tensor, Tensor]:
        x, _ = self.hypercube.sample(n_samples, seed)
        y = self.target(x)
        return x, y

    def base_sample(self, n_samples: int, seed: int) -> Tuple[Tensor, Tensor]:
        return self.hypercube.base_sample(n_samples, seed)

    def preferred_provider(self) -> str:
        return "TensorDataProvider"

    def forward(self, base_X: Tensor) -> Tuple[Tensor, Tensor]:
        y = self.target(base_X)
        return base_X, y

    def forward_X(self, base_X: Tensor) -> Tensor:
        return base_X
