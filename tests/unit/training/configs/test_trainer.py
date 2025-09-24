import pytest
from dataclasses import dataclass

import torch

from src.training.trainer_config import TrainerConfig
from src.training.loss.configs.loss import LossConfig
from src.data.joint_distributions.configs.mapped_joint_distribution import MappedJointDistributionConfig
from src.data.joint_distributions.configs.base import JointDistributionConfig
from src.data.joint_distributions.configs.gaussian import GaussianConfig
from src.models.architectures.configs.mlp import MLPConfig
from src.models.targets.configs.sum_prod import SumProdTargetConfig


def test_trainer_config_json_roundtrip(tmp_path):
    cfg = TrainerConfig(
        model_config=MLPConfig(
            input_dim=3,
            output_dim=1,
            hidden_dims=[4, 2],
            activation="relu",
            start_activation=False,
            end_activation=False,
        ),
        joint_distribution_config=MappedJointDistributionConfig(
            distribution_config=GaussianConfig(
                input_shape=torch.Size([3]), mean=0.0, std=1.0
            ),
            target_function_config=SumProdTargetConfig(
                input_shape=torch.Size([3]),
                indices_list=[[0]],
                weights=[1.0],
                normalize=False,
            ),
        ),
        train_size=10,
        test_size=5,
        batch_size=2,
        epochs=1,
        home_dir=tmp_path / "trainer_home",
        loss_config=LossConfig(name="MSELoss"),
        seed=42,
    )

    json_str = cfg.to_json()
    restored = TrainerConfig.from_json(json_str)
    assert restored == cfg


def test_trainer_config_requires_output_shape(tmp_path, mlp_config, adam_config):
    @dataclass
    class _Cfg(JointDistributionConfig):
        def __post_init__(self) -> None:  # type: ignore[override]
            self.input_shape = torch.Size([mlp_config.input_dim])
            self.output_shape = None
            self.distribution_type = "NoOutput"

    cfg = TrainerConfig(
        model_config=mlp_config,
        joint_distribution_config=_Cfg(),
        train_size=1,
        test_size=1,
        batch_size=1,
        epochs=1,
        home_dir=tmp_path / "h",
        loss_config=LossConfig(name="MSELoss"),
    )
    with pytest.raises(AssertionError):
        cfg.ready_for_trainer()


def test_ready_for_trainer_missing_fields(tmp_path):
    cfg = TrainerConfig(home_dir=tmp_path / "h")
    with pytest.raises(AssertionError):
        cfg.ready_for_trainer()




