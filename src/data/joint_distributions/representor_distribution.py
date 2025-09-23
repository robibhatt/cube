from typing import TYPE_CHECKING, Tuple

import torch

from .joint_distribution import JointDistribution
from .configs.representor_distribution import RepresentorDistributionConfig
from .joint_distribution_registry import register_joint_distribution

if TYPE_CHECKING:  # For type hints
    from src.data.providers.data_provider import DataProvider

@register_joint_distribution("RepresentorDistribution")
class RepresentorDistribution(JointDistribution):
    def __init__(self, config: RepresentorDistributionConfig, device: torch.device) -> None:
        from .joint_distribution_factory import create_joint_distribution
        from src.models.representors.representor_factory import (
            create_model_representor,
        )

        self.base_joint_distribution = create_joint_distribution(
            config.base_distribution_config, device
        )
        self.model_representor = create_model_representor(
            config.model_config, config.checkpoint_dir, device=device
        )
        self.from_rep = config.from_rep
        self.to_rep = config.to_rep

        super().__init__(config=config, device=device)

    def __str__(self) -> str:
        return (
            f"RepresentorDistribution(base_distribution={self.base_joint_distribution}, "
            f"model_representor={self.model_representor}, from_rep={self.from_rep}, "
            f"to_rep={self.to_rep})"
        )

    def sample(self, n_samples: int, seed: int):
        X_base, _ = self.base_sample(n_samples, seed=seed)
        return self.forward(X_base)

    def base_sample(self, n_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.base_joint_distribution.base_sample(n_samples, seed)

    def forward(self, base_X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X_penultimate = self.base_joint_distribution.forward_X(base_X)
        return self.model_representor.forward(
            X_penultimate, self.from_rep, self.to_rep, None
        )

    def forward_X(self, base_X: torch.Tensor) -> torch.Tensor:
        X_final, _ = self.forward(base_X)
        return X_final

    def preferred_provider(self) -> str:
        """Return the default iterator type for this distribution."""
        return "TensorDataProvider"

