from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config

from .base import JointDistributionConfig
from .joint_distribution_config_registry import (
    register_joint_distribution_config,
    build_joint_distribution_config_from_dict,
)
from src.models.targets.configs.sum_prod import SumProdTargetConfig


@register_joint_distribution_config("NoisyDistribution")
@dataclass_json
@dataclass(kw_only=True)
class NoisyDistributionConfig(JointDistributionConfig):
    base_distribution_config: JointDistributionConfig = field(
        metadata=config(
            encoder=lambda c: c.to_dict(),
            decoder=lambda d: d
            if isinstance(d, JointDistributionConfig)
            else build_joint_distribution_config_from_dict(d),
        )
    )
    target_function_config: SumProdTargetConfig = field(
        metadata=config(
            encoder=lambda c: c.to_dict(),
            decoder=lambda d: d
            if isinstance(d, SumProdTargetConfig)
            else SumProdTargetConfig.from_dict(d),
        )
    )
    noise_mean: float = 0.0
    noise_std: float = 1.0

    def __post_init__(self) -> None:
        self.input_shape = self.base_distribution_config.input_shape
        self.output_shape = self.target_function_config.output_shape
        self.distribution_type = "NoisyDistribution"

