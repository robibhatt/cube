import torch
from typing import Tuple

from .joint_distribution import JointDistribution
from .joint_distribution_registry import register_joint_distribution
from .configs.gaussian import GaussianConfig


@register_joint_distribution("Gaussian")
class Gaussian(JointDistribution):
    """IID normal distribution with configurable mean and std."""

    def __init__(self, config: GaussianConfig, device: torch.device) -> None:
        super().__init__(config, device)
        self._shape = config.input_shape
        self.dtype = config.dtype
        self.mean = torch.full(
            self._shape, config.mean, dtype=self.dtype, device=device
        )
        self.std = config.std

    def __str__(self) -> str:
        """Return a readable description of the Gaussian distribution."""
        return (
            f"{self.input_shape}-dimensional Normal(mean={self.config.mean}, "
            f"std={self.config.std})"
        )

    def sample(self, n_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        x = torch.randn(
            (n_samples, *self._shape),
            dtype=self.dtype,
            device=self.device,
            generator=g,
        )
        x = x * self.std + self.mean
        y = torch.zeros(n_samples, 1, dtype=self.dtype, device=self.device)
        return x, y

    def base_sample(self, n_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        x = torch.randn(
            (n_samples, *self._shape),
            dtype=self.dtype,
            device=self.device,
            generator=g,
        )
        x = x * self.std + self.mean
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
