from dataclasses import dataclass, field
from abc import ABC
import torch
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class JointDistributionConfig(ABC):
    """Base configuration for joint distributions (X, y)."""

    distribution_type: str = field(init=False)
    input_dim: int = field(init=False)

    @property
    def input_shape(self) -> torch.Size:
        """Return the canonical input shape corresponding to ``input_dim``."""

        return torch.Size([self.input_dim])

    @property
    def output_shape(self) -> torch.Size:
        """Return the canonical output shape for scalar targets."""

        return torch.Size([1])
