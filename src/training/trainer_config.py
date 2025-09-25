from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config
from pathlib import Path
from typing import Optional
import copy
import torch

from src.models.architectures.configs.base import ModelConfig
from src.models.architectures.configs.model_config_registry import build_model_config_from_dict
from src.data.joint_distributions.configs.cube_distribution import (
    CubeDistributionConfig,
)
from src.training.loss.configs.loss import LossConfig
from src.utils.serialization_utils import encode_path, decode_path
from src.training.optimizers.configs.optimizer import OptimizerConfig
from src.training.optimizers.configs.optimizer_config_registry import (
    build_optimizer_config_from_dict,
)
from src.training.optimizers.configs.adam import AdamConfig


@dataclass_json
@dataclass(kw_only=True)
class TrainerConfig:
    model_config: Optional[ModelConfig] = field(
        default=None,
        metadata=config(
            encoder=lambda m: None if m is None else m.to_dict(),
            decoder=lambda d: None if d is None else build_model_config_from_dict(d),
        ),
    )
    cube_distribution_config: Optional[CubeDistributionConfig] = field(
        default=None,
        metadata=config(
            encoder=lambda jd: None if jd is None else jd.to_dict(),
            decoder=lambda d: None if d is None else CubeDistributionConfig.from_dict(d),
        ),
    )
    loss_config: Optional[LossConfig] = field(
        default=None,
        metadata=config(
            encoder=lambda l: None if l is None else l.to_dict(),
            decoder=lambda d: None if d is None else LossConfig.from_dict(d),
        ),
    )
    optimizer_config: Optional[OptimizerConfig] = field(
        default=None,
        metadata=config(
            encoder=lambda o: None if o is None else o.to_dict(),
            decoder=lambda d: None if d is None else build_optimizer_config_from_dict(d),
        ),
    )
    early_stopping: Optional[float] = None
    train_size: Optional[int] = None
    test_size: Optional[int] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    use_full_batch: bool = False
    weight_decay_l1: float = 0.0
    home_dir: Optional[Path] = field(
        default=None,
        metadata=config(encoder=encode_path, decoder=decode_path),
    )
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.optimizer_config is None:
            self.optimizer_config = AdamConfig(lr=0.001)

    def deep_copy(self) -> "TrainerConfig":
        return copy.deepcopy(self)

    def ready_for_trainer(self) -> None:
        assert self.model_config is not None, "model_config must be specified"
        assert (
            self.cube_distribution_config is not None
        ), "cube_distribution_config must be specified"
        assert self.loss_config is not None, "loss_config must be specified"
        assert hasattr(torch.nn, self.loss_config.name), (
            f"Loss '{self.loss_config.name}' must exist in torch.nn"
        )
        assert self.train_size is not None, "train_size must be specified"
        assert self.test_size is not None, "test_size must be specified"
        assert self.home_dir is not None, "home_dir must be specified"
        assert self.weight_decay_l1 >= 0.0, "weight_decay_l1 must be non-negative"

        assert self.model_config.input_shape == self.cube_distribution_config.input_shape, (
            f"Model input shape must match distribution input shape. "
            f"Model: {self.model_config.input_shape}, "
            f"Distribution: {self.cube_distribution_config.input_shape}"
        )
        assert (
            self.model_config.output_shape is not None
        ), "Model output shape must not be None"
        assert (
            self.cube_distribution_config.output_shape is not None
        ), "Cube distribution output shape must not be None"
        assert self.model_config.output_shape == self.cube_distribution_config.output_shape, (
            f"Model output shape must match distribution output shape. "
            f"Model: {self.model_config.output_shape}, "
            f"Distribution: {self.cube_distribution_config.output_shape}"
        )
        assert self.home_dir.exists(), f"Home directory {self.home_dir} does not exist"
        if self.batch_size is not None and self.epochs is not None:
            return
        # Allow incomplete configs for backward compatibility
