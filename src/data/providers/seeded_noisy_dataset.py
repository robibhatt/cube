from __future__ import annotations

import math

import torch
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
            distribution.config.distribution_type == "NoisyDistribution"
        ), "SeededNoisyDataset requires a NoisyDistribution"
        self.distribution = distribution
        self.size = size
        self.seed = seed

        # Cache shape metadata provided directly by ``NoisyDistribution`` so
        # that ``__getitem__`` does not need to recompute it for every sample.
        self._base_size = distribution.base_input_size
        self._base_shape = distribution.base_input_shape
        self._noise_shape = tuple(distribution.noise_shape)
        self._noise_size = math.prod(self._noise_shape)

    def __len__(self) -> int:  # type: ignore[override]
        return self.size

    def __getitem__(self, idx: int):  # type: ignore[override]
        # ``NoisyDistribution.base_sample`` concatenates the flattened base
        # features with a noise sample. Draw a single combined sample and then
        # split it back into the two components.

        X_with_noise, y_base = self.distribution.base_sample(
            1, self.seed + int(idx)
        )

        # Remove the batch dimension for easier slicing
        X_with_noise = X_with_noise.squeeze(0)
        y_base = y_base.squeeze(0)

        X_base_flat = X_with_noise[: self._base_size]
        X_noise_flat = X_with_noise[self._base_size : self._base_size + self._noise_size]

        X_base = X_base_flat.reshape(*self._base_shape)
        X_noise = X_noise_flat.reshape(*self._noise_shape)

        # Return the noise sample so that the provider can add it to ``y_base``.
        return X_base, y_base, X_noise

