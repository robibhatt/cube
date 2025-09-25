from typing import Tuple
import math
import torch
import random

from .joint_distribution import JointDistribution
from .configs.cube_distribution import CubeDistributionConfig
from .joint_distribution_registry import register_joint_distribution
from src.models.targets.sum_prod import SumProdTarget

@register_joint_distribution("CubeDistribution")
class CubeDistribution(JointDistribution):
    def __init__(self, config: CubeDistributionConfig, device: torch.device) -> None:
        self.base_input_shape = config.input_shape
        self.base_input_size = math.prod(self.base_input_shape)
        self.base_dtype = torch.float32
        self._base_distribution_str = (
            f"{config.input_dim}-dimensional UniformHypercube"
        )
        self.base_distribution_description = self._base_distribution_str
        self.target_function = SumProdTarget(config.target_function_config).to(device)

        super().__init__(config=config, device=device)
        self.well_specified = False

        x = self._sample_base_inputs(1, seed=0).to(self.device)
        y_clean = self.target_function(x)
        if y_clean.dim() != 2 or y_clean.size(1) != 1:
            raise ValueError(
                "target_function must return 2D tensors with a single output dimension"
            )

        self.noise_shape = y_clean.shape[1:]
        noise_dtype = y_clean.dtype if torch.is_floating_point(y_clean) else torch.float32
        self.noise_dtype = noise_dtype
        self.noise_mean_tensor = torch.full(
            self.noise_shape, config.noise_mean, dtype=self.noise_dtype, device=self.device
        )
        self.noise_std = config.noise_std
        self.noise_distribution_description = (
            f"{self.output_dim}-dimensional Normal(mean={self.config.noise_mean}, "
            f"std={self.config.noise_std})"
        )

    def __str__(self) -> str:
        base_str = self._base_distribution_str
        noise_str = (
            f"{self.output_dim}-dimensional Normal(mean={self.config.noise_mean}, "
            f"std={self.config.noise_std})"
        )
        return f"CubeDistribution with {base_str} and {noise_str}"
    
    def get_seeds(self, seed: int) -> Tuple[int, int]:
        splitter = random.Random(seed)
        base_seed = splitter.randint(0, 2**32 - 1)
        noise_seed = splitter.randint(0, 2**32 - 1)
        return base_seed, noise_seed

    def sample(
        self, n_samples: int, seed: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        base_seed, noise_seed = self.get_seeds(seed)
        x = self._sample_base_inputs(n_samples, seed=base_seed)
        noise = self._sample_noise(n_samples, seed=noise_seed)

        # Ensure all tensors reside on the same device.  Some joint
        # distributions used in tests (e.g. ``DummyJointDistribution``)
        # ignore the ``device`` argument and default to CPU, which causes
        # failures when the trainer runs on a GPU.
        x = x.to(self.device)
        noise = noise.to(self.device)

        y_clean = self.target_function(x)
        y_noise = noise
        return x, y_clean + y_noise

    def base_sample(self, n_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        base_seed, noise_seed = self.get_seeds(seed)
        x = self._sample_base_inputs(n_samples, seed=base_seed)
        noise = self._sample_noise(n_samples, seed=noise_seed)

        # ``DummyJointDistribution`` and similar fixtures return tensors on the
        # CPU regardless of the requested device.  When the trainer runs on a
        # GPU this leads to ``torch.cat`` receiving tensors on different
        # devices.  Explicitly move everything to the distribution's device to
        # keep them consistent across platforms.
        x = x.to(self.device)
        noise = noise.to(self.device)

        y_clean = self.target_function(x)

        # Keep batch dimension (dim 0) and flatten the rest
        x_flat = x.reshape(n_samples, -1)
        noise_flat = noise.reshape(n_samples, -1)
        x_with_noise = torch.cat([x_flat, noise_flat], dim=1)
        return x_with_noise, y_clean
    
    def preferred_provider(self) -> str:
        return "NoisyProvider"

    def forward(self, base_X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert False, "CubeDistribution is not well-specified"

    def forward_X(self, base_X: torch.Tensor) -> torch.Tensor:
        if base_X.shape[1:] == self.base_input_shape:
            return base_X.to(self.device).to(self.base_dtype)

        batch_size = base_X.shape[0]
        X_base_flat = base_X.reshape(batch_size, -1)[:, : self.base_input_size]
        X_base = X_base_flat.reshape(batch_size, *self.base_input_shape)
        return X_base.to(self.device).to(self.base_dtype)

    def _sample_noise(self, n_samples: int, seed: int) -> torch.Tensor:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        noise = torch.randn(
            (n_samples, *self.noise_shape),
            dtype=self.noise_dtype,
            device=self.device,
            generator=generator,
        )
        noise = noise * self.noise_std + self.noise_mean_tensor
        return noise

    def _sample_base_inputs(self, n_samples: int, seed: int) -> torch.Tensor:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        x = torch.randint(
            0,
            2,
            (n_samples, *self.base_input_shape),
            device=self.device,
            generator=generator,
            dtype=torch.int64,
        )
        x = x * 2 - 1
        return x.to(self.base_dtype)
