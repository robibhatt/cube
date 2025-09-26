import torch
from abc import abstractmethod, ABC
from typing import Tuple


from src.data.joint_distributions.configs.base import JointDistributionConfig


class JointDistribution(ABC):
    def __init__(self, config: JointDistributionConfig, device: torch.device) -> None:
        """Store shapes and the ``device`` from ``config``."""

        self.input_dim: int = config.input_dim
        self.config = config

        # Accept the provided device without additional canonicalization.
        self.device: torch.device = torch.device(device)

        self.well_specified: bool = True

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.input_dim])

    @property
    def output_dim(self) -> int:
        return 1

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.output_dim])
    
    @abstractmethod
    def sample(self, n_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate ``n_samples`` from the joint distribution.

        Parameters
        ----------
        n_samples:
            Number of samples to draw.
        seed:
            Random seed used to create a ``torch.Generator`` on this
            distribution's device.

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            Input and output tensors of shapes ``(n_samples, input_dim)`` and
            ``(n_samples, output_dim)`` respectively.
        """
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        """Return a textual description of the distribution."""
        pass

    @abstractmethod
    def base_sample(
        self, n_samples: int, seed: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from the distribution's base distribution.

        The returned ``(base_X, base_y)`` pair is typically passed to
        :meth:`forward` or :meth:`forward_X` for transformation into the final
        ``(X, y)`` samples.
        """
        pass

    @abstractmethod
    def forward(self, base_X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform ``base_X`` from :meth:`base_sample` into ``(X, y)``."""
        pass

    @abstractmethod
    def forward_X(self, base_X: torch.Tensor) -> torch.Tensor:
        """Transform only the ``X`` portion of ``base_X`` returning ``X``."""
        pass
    
    @abstractmethod
    def preferred_provider(self) -> str:
        pass

    def average_output_variance(
        self, n_samples: int = 1000, seed: int = 0
    ) -> float:
        """Estimate the average variance of the output dimensions.

        Draws ``n_samples`` from the distribution and computes the variance of
        each coordinate of ``y``. The returned value is the mean of these
        variances.

        Parameters
        ----------
        n_samples:
            Number of samples used to estimate the variance.
        seed:
            Seed for ``torch.Generator`` passed to :meth:`sample`.

        Returns
        -------
        float
            The mean of the variances of each output coordinate.
        """
        _, y = self.sample(n_samples, seed)
        y_flat = y.reshape(n_samples, -1)
        var_per_coord = y_flat.var(dim=0, unbiased=False)
        return var_per_coord.mean().item()
