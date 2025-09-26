from __future__ import annotations

from torch.utils.data import Dataset

from src.data.joint_distributions.joint_distribution import JointDistribution


class SeededNoisyDataset(Dataset):
    """Dataset that deterministically samples from a distribution per index."""

    def __init__(
        self,
        distribution: JointDistribution,
        size: int,
        seed: int = 0,
    ) -> None:
        assert (
            distribution.config.distribution_type == "CubeDistribution"
        ), "SeededNoisyDataset requires a CubeDistribution"
        self.distribution = distribution
        self.size = size
        self.seed = seed

    def __len__(self) -> int:  # type: ignore[override]
        return self.size

    def __getitem__(self, idx: int):  # type: ignore[override]
        X_base, noise = self.distribution.base_sample(1, self.seed + int(idx))

        # Remove the batch dimension for easier slicing
        X_base = X_base.squeeze(0)
        noise = noise.squeeze(0)

        return X_base, noise

