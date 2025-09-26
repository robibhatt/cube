from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Tuple

import torch

from src.data.joint_distributions.cube_distribution import CubeDistribution


@dataclass
class DataProvider(ABC):
    """Abstract interface for dataset generation and ``DataLoader`` creation."""

    joint_distribution: CubeDistribution
    seed: int
    dataset_size: int
    batch_size: int
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.dataset_size
    
    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Return an iterator over the dataset."""
        pass
