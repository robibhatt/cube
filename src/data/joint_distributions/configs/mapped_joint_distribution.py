from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config

from .base import JointDistributionConfig
from .joint_distribution_config_registry import (
    register_joint_distribution_config,
    build_joint_distribution_config_from_dict,
)
from src.models.targets.configs.base import TargetFunctionConfig
from src.models.targets.configs.target_function_config_registry import (
    build_target_function_config_from_dict,
)


@register_joint_distribution_config("MappedJointDistribution")
@dataclass_json
@dataclass(kw_only=True)
class MappedJointDistributionConfig(JointDistributionConfig):
    distribution_config: JointDistributionConfig = field(
        metadata=config(
            encoder=lambda c: c.to_dict(),
            decoder=lambda d: d if isinstance(d, JointDistributionConfig) else build_joint_distribution_config_from_dict(d),
        )
    )
    target_function_config: TargetFunctionConfig = field(
        metadata=config(
            encoder=lambda c: c.to_dict(),
            decoder=lambda d: d if isinstance(d, TargetFunctionConfig) else build_target_function_config_from_dict(d),
        )
    )

    def __post_init__(self) -> None:
        self.input_shape = self.distribution_config.input_shape
        self.output_shape = self.target_function_config.output_shape
        self.distribution_type = "MappedJointDistribution"
