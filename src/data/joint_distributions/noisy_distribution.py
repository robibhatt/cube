from typing import Tuple
import torch
import random

from .joint_distribution import JointDistribution
from .configs.noisy_distribution import NoisyDistributionConfig
from .joint_distribution_registry import register_joint_distribution

@register_joint_distribution("NoisyDistribution")
class NoisyDistribution(JointDistribution):
    def __init__(self, config: NoisyDistributionConfig, device: torch.device) -> None:
        from .joint_distribution_factory import create_joint_distribution

        self.base_joint_distribution = create_joint_distribution(
            config.base_distribution_config, device
        )
        self.noise_distribution = create_joint_distribution(
            config.noise_distribution_config, device
        )

        super().__init__(config=config, device=device)
        self.well_specified = False
        x, _ = self.base_joint_distribution.base_sample(1, seed=0)
        self.base_input_shape = x.shape[1:]
        self.base_input_size = torch.prod(torch.tensor(self.base_input_shape)).item()

    def __str__(self) -> str:
        base_str = str(self.base_joint_distribution)
        noise_str = str(self.noise_distribution)
        return f"NoisyDistribution with {base_str} and {noise_str}"
    
    def get_seeds(self, seed: int) -> Tuple[int, int]:
        splitter = random.Random(seed)
        base_seed = splitter.randint(0, 2**32 - 1)
        noise_seed = splitter.randint(0, 2**32 - 1)
        return base_seed, noise_seed

    def sample(
        self, n_samples: int, seed: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        base_seed, noise_seed = self.get_seeds(seed)
        x, y = self.base_joint_distribution.sample(n_samples, seed=base_seed)
        noise, _ = self.noise_distribution.sample(n_samples, seed=noise_seed)

        # Ensure all tensors reside on the same device.  Some joint
        # distributions used in tests (e.g. ``DummyJointDistribution``)
        # ignore the ``device`` argument and default to CPU, which causes
        # failures when the trainer runs on a GPU.
        x = x.to(self.device)
        y = y.to(self.device)
        noise = noise.to(self.device)

        y_noise = noise
        return x, y + y_noise
    
    def base_sample(self, n_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        base_seed, noise_seed = self.get_seeds(seed)
        x, y = self.base_joint_distribution.base_sample(n_samples, seed=base_seed)
        noise, _ = self.noise_distribution.sample(n_samples, seed=noise_seed)

        # ``DummyJointDistribution`` and similar fixtures return tensors on the
        # CPU regardless of the requested device.  When the trainer runs on a
        # GPU this leads to ``torch.cat`` receiving tensors on different
        # devices.  Explicitly move everything to the distribution's device to
        # keep them consistent across platforms.
        x = x.to(self.device)
        y = y.to(self.device)
        noise = noise.to(self.device)

        # Keep batch dimension (dim 0) and flatten the rest
        x_flat = x.reshape(n_samples, -1)
        noise_flat = noise.reshape(n_samples, -1)
        x_with_noise = torch.cat([x_flat, noise_flat], dim=1)
        return x_with_noise, y
    
    def preferred_provider(self) -> str:
        return "NoisyProvider"
    
    def forward(self, base_X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert False, "NoisyDistribution is not well-specified"

    def forward_X(self, base_X: torch.Tensor) -> torch.Tensor:
        batch_size = base_X.shape[0]
        X_base_flat = base_X[:, :self.base_input_size]
        X_base = X_base_flat.reshape(batch_size, *self.base_input_shape)
        return self.base_joint_distribution.forward_X(X_base)
