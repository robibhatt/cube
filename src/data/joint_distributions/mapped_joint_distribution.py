import torch
from torch import Tensor
from typing import Tuple, Optional

from .joint_distribution import JointDistribution
from .configs.mapped_joint_distribution import MappedJointDistributionConfig
from .joint_distribution_registry import register_joint_distribution


@register_joint_distribution("MappedJointDistribution")
class MappedJointDistribution(JointDistribution):
    """
    A joint distribution where:
      - X ~ any Distribution that returns Tensor samples
      - y = f(X), where f is a custom function mapping Tensor input to Tensor output

    Attributes:
        distribution: The input feature distribution.
        target_function: Callable mapping input Tensor to output Tensor.
    """

    def __init__(self, config: MappedJointDistributionConfig, device: torch.device) -> None:
        from .joint_distribution_factory import create_joint_distribution
        from src.models.targets.target_function_factory import create_target_function

        self.distribution = create_joint_distribution(
            config.distribution_config, device
        )
        self.target_function = create_target_function(
            config.target_function_config
        )

        super().__init__(config, device)

    def sample(self, n_samples: int, seed: int) -> Tuple[Tensor, Tensor]:
        """
        Sample n_samples from distribution and compute targets.

        Args:
            n_samples: Number of samples to generate

        Returns:
            A tuple (X, y):
                X: Tensor of shape (n_samples, *input_shape)
                y: Tensor of shape (n_samples, *output_shape)
        """
        x, _ = self.distribution.sample(n_samples, seed=seed)
        y = self.target_function(x)
        return x, y
    
    def base_sample(self, n_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.distribution.base_sample(n_samples, seed)
    
    def preferred_provider(self) -> str:
        return "TensorDataProvider"
    
    def forward_transform(
        self, base_X: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return base_X, self.target_function(base_X)

    def forward(self, base_X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X_final = self.distribution.forward_X(base_X)
        return X_final, self.target_function(X_final)

    def forward_X(self, base_X: torch.Tensor) -> torch.Tensor:
        return base_X

    def __str__(self) -> str:
        """
        Returns a human-readable description.
        """
        return (
            f"MappedJointDistribution: X ~ {self.distribution}, "
            f"input_shape={self.input_shape}, "
            f"output_shape={self.output_shape}, "
            f"target_function={self.target_function}"
        )
