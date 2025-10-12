from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config
from pathlib import Path
from typing import Optional
import copy

from src.models.mlp_config import MLPConfig
from src.data.cube_distribution_config import (
    CubeDistributionConfig,
)
from src.utils.serialization_utils import encode_path, decode_path
from src.training.sgd_config import SgdConfig


@dataclass_json
@dataclass(kw_only=True)
class TrainerConfig:
    mlp_config: Optional[MLPConfig] = field(
        default=None,
        metadata=config(
            encoder=lambda m: None if m is None else m.to_dict(),
            decoder=lambda d: None if d is None else MLPConfig.from_dict(d),
        ),
    )
    cube_distribution_config: Optional[CubeDistributionConfig] = field(
        default=None,
        metadata=config(
            encoder=lambda jd: None if jd is None else jd.to_dict(),
            decoder=lambda d: None if d is None else CubeDistributionConfig.from_dict(d),
        ),
    )
    optimizer_config: Optional[SgdConfig] = field(
        default=None,
        metadata=config(
            encoder=lambda o: None if o is None else o.to_dict(),
            decoder=lambda d: None if d is None else SgdConfig.from_dict(d),
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
            self.optimizer_config = SgdConfig(lr=0.001)

    def deep_copy(self) -> "TrainerConfig":
        return copy.deepcopy(self)

    def ready_for_trainer(self) -> None:
        assert self.mlp_config is not None, "mlp_config must be specified"
        assert (
            self.cube_distribution_config is not None
        ), "cube_distribution_config must be specified"
        assert self.train_size is not None, "train_size must be specified"
        assert self.test_size is not None, "test_size must be specified"
        assert self.home_dir is not None, "home_dir must be specified"
        assert self.weight_decay_l1 >= 0.0, "weight_decay_l1 must be non-negative"

        assert (
            self.mlp_config.input_dim == self.cube_distribution_config.input_dim
        ), (
            "Model input_dim must match distribution input_dim. "
            f"Model: {self.mlp_config.input_dim}, "
            f"Distribution: {self.cube_distribution_config.input_dim}"
        )
        expected_output_dim = int(
            self.cube_distribution_config.output_shape.numel()
        )
        assert self.mlp_config.output_dim == expected_output_dim, (
            "Model output_dim must match distribution output dimension. "
            f"Model: {self.mlp_config.output_dim}, "
            f"Distribution: {expected_output_dim}"
        )
        assert self.home_dir.exists(), f"Home directory {self.home_dir} does not exist"
        if self.batch_size is not None and self.epochs is not None:
            return
        # Allow incomplete configs for backward compatibility
