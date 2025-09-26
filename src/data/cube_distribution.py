from typing import Optional, Tuple
import math
import torch
import random

from .cube_distribution_config import CubeDistributionConfig
from src.models.targets.sum_prod import SumProdTarget


class CubeDistribution:
    def __init__(self, config: CubeDistributionConfig, device: torch.device) -> None:
        self.config = config
        self.device = torch.device(device)
        self.input_dim = config.input_dim
        self.well_specified = False

        self.base_input_shape = config.input_shape
        self.base_input_size = math.prod(self.base_input_shape)
        self.base_dtype = torch.float32
        self._base_distribution_str = (
            f"{config.input_dim}-dimensional UniformHypercube"
        )
        self.base_distribution_description = self._base_distribution_str
        self.target_function = SumProdTarget(config.target_function_config).to(device)

        self._output_dim: Optional[int] = None

        x = self._sample_base_inputs(1, seed=0).to(self.device)
        y_clean = self.target_function(x)
        if y_clean.dim() != 2 or y_clean.size(1) != 1:
            raise ValueError(
                "target_function must return 2D tensors with a single output dimension"
            )

        self._output_dim = y_clean.size(1)
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

        y_noise = noise
        return x, self.target(x) + y_noise

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

        return x, noise

    def target(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_function(x)
    
    def preferred_provider(self) -> str:
        return "NoisyProvider"

    def forward(self, base_X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert False, "CubeDistribution is not well-specified"

    def forward_X(self, base_X: torch.Tensor) -> torch.Tensor:
        return base_X.to(self.device).to(self.base_dtype)

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

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.input_dim])

    @property
    def output_dim(self) -> int:
        if self._output_dim is None:
            raise RuntimeError("CubeDistribution output dimension has not been initialised")
        return self._output_dim

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.output_dim])

    def average_output_variance(
        self, n_samples: int = 1000, seed: int = 0
    ) -> float:
        _, y = self.sample(n_samples, seed)
        y_flat = y.reshape(n_samples, -1)
        var_per_coord = y_flat.var(dim=0, unbiased=False)
        return var_per_coord.mean().item()
