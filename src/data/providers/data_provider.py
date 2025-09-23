from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
import torch
from src.data.joint_distributions.joint_distribution import JointDistribution
from typing import Iterator, Tuple


@dataclass
class DataProvider(ABC):
    """Abstract interface for dataset generation and ``DataLoader`` creation."""

    joint_distribution: JointDistribution
    dataset_dir: Path
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
