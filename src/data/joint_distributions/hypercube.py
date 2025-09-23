import torch
from typing import Tuple

from .joint_distribution import JointDistribution
from .joint_distribution_registry import register_joint_distribution
from .configs.hypercube import HypercubeConfig


@register_joint_distribution("Hypercube")
class Hypercube(JointDistribution):
    """IID uniform distribution over {\u00b11}^d."""

    def __init__(self, config: HypercubeConfig, device: torch.device) -> None:
        super().__init__(config, device)
        self._shape = config.input_shape
        self.dtype = config.dtype

    def __str__(self) -> str:
        """Return a description of the uniform hypercube."""
        return f"{self.input_shape}-dimensional UniformHypercube"

    def _sample(self, n_samples: int, seed: int) -> torch.Tensor:
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        x = torch.randint(
            0,
            2,
            (n_samples, *self._shape),
            device=self.device,
            generator=g,
            dtype=torch.int64,
        )
        x = x * 2 - 1
        return x.to(self.dtype)

    def sample(self, n_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._sample(n_samples, seed)
        y = torch.zeros(n_samples, 1, dtype=self.dtype, device=self.device)
        return x, y

    def base_sample(self, n_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._sample(n_samples, seed)
        y = torch.zeros(n_samples, 1, dtype=self.dtype, device=self.device)
        return x, y

    def preferred_provider(self) -> str:
        return "TensorDataProvider"

    def forward(self, base_X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = torch.zeros(
            base_X.size(0), *self.output_shape, dtype=self.dtype, device=self.device
        )
        return base_X, y

    def forward_X(self, base_X: torch.Tensor) -> torch.Tensor:
        return base_X
