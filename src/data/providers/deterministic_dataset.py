from __future__ import annotations

import torch
from torch.utils.data import Dataset

from src.data.joint_distributions.cube_distribution import CubeDistribution

class DeterministicDataset(Dataset):
    """Dataset that deterministically samples from a distribution per index."""

    def __init__(
        self,
        distribution: CubeDistribution,
        size: int,
        seed: int = 0,
    ) -> None:
        self.distribution = distribution
        self.size = size
        self.seed = seed

    def __len__(self) -> int:  # type: ignore[override]
        return self.size

    def __getitem__(self, idx: int):  # type: ignore[override]
        seed = self.seed + int(idx)
        X, y = self.distribution.base_sample(1, seed=seed)
        if y is None:
            return X.squeeze(0)
        return X.squeeze(0), y.squeeze(0)

