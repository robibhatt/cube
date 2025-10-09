from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Iterator, Tuple
import random

from src.data.cube_distribution import CubeDistribution
from src.data.seeded_noisy_dataset import SeededNoisyDataset

@dataclass
class NoisyProvider:
    joint_distribution: CubeDistribution
    seed: int
    dataset_size: int
    batch_size: int

    def __post_init__(self) -> None:
        assert (
            self.joint_distribution.config.distribution_type
            == "CubeDistribution"
        ), "NoisyIterator can only be used with CubeDistribution"
        self.data_loader = self.make_loader()

    def make_loader(self) -> DataLoader:
        splitter = random.Random(self.seed)
        dataset_seed = splitter.randint(0, 2**32 - 1)
        loader_seed = splitter.randint(0, 2**32 - 1)

        dataset = SeededNoisyDataset(
            self.joint_distribution,
            self.dataset_size,
            seed=dataset_seed,
        )

        g = torch.Generator()
        g.manual_seed(loader_seed)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=g,
        )

    def __len__(self) -> int:
        return self.dataset_size

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for batch in self.data_loader:
            X_base, y_noise = batch
            X_base = X_base.to(self.joint_distribution.device)
            y_noise = y_noise.to(self.joint_distribution.device)
            y_clean = self.joint_distribution.target(X_base)
            yield X_base, y_clean + y_noise
