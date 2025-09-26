from dataclasses import dataclass, field
from typing import List

import torch
from dataclasses_json import dataclass_json, config

from src.models.targets.configs.sum_prod import SumProdTargetConfig


@dataclass_json
@dataclass(kw_only=True)
class CubeDistributionConfig:
    """Configuration for :class:`CubeDistribution`."""

    input_dim: int
    indices_list: List[List[int]] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    normalize: bool = False
    noise_mean: float = 0.0
    noise_std: float = 1.0
    distribution_type: str = field(init=False, default="CubeDistribution")
    target_function_config: SumProdTargetConfig = field(
        init=False,
        metadata=config(
            encoder=lambda c: c.to_dict(),
            decoder=lambda d: d
            if isinstance(d, SumProdTargetConfig)
            else SumProdTargetConfig.from_dict(d),
        ),
    )

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")

        if not self.indices_list:
            raise ValueError("indices_list must be a non-empty list")

        if len(self.weights) != len(self.indices_list):
            raise ValueError("weights must have the same length as indices_list")

        for idx_group in self.indices_list:
            if not idx_group:
                raise ValueError("each indices group must be non-empty")
            for index in idx_group:
                if index < 0:
                    raise ValueError("indices cannot be negative")
                if index >= self.input_dim:
                    raise ValueError(
                        "indices_list contains an index that is >= input_dim"
                    )

        self.target_function_config = SumProdTargetConfig(
            input_shape=torch.Size([self.input_dim]),
            indices_list=self.indices_list,
            weights=self.weights,
            normalize=self.normalize,
        )

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.input_dim])

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([1])
