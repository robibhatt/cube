import torch
import pytest

from src.training.trainer import Trainer
from src.training.trainer_config import TrainerConfig
from src.training.loss.configs.loss import LossConfig
from src.models.architectures.configs.mlp import MLPConfig
from src.training.optimizers.configs.adam import AdamConfig
from src.data.joint_distributions.configs.cube_distribution import (
    CubeDistributionConfig,
)


def test_trainer_uses_evaluator(tmp_path, monkeypatch):
    class ConstEval(torch.nn.Module):
        def forward(self, input, target):
            return torch.tensor(5.0)

    monkeypatch.setattr(LossConfig, "get_evaluator", lambda self: ConstEval())

    cfg = TrainerConfig(
        model_config=MLPConfig(
            input_dim=1,
            output_dim=1,
            hidden_dims=[],
            activation="relu",
            start_activation=False,
            end_activation=False,
            bias=False,
        ),
        optimizer_config=AdamConfig(lr=0.01),
        cube_distribution_config=CubeDistributionConfig(
            input_dim=1,
            indices_list=[[0]],
            weights=[1.0],
            noise_mean=0.0,
            noise_std=0.0,
        ),
        train_size=4,
        test_size=4,
        batch_size=2,
        epochs=0,
        home_dir=tmp_path,
        loss_config=LossConfig(name="MSELoss"),
    )

    trainer = Trainer(cfg)
    model, _ = trainer._initialize_model_and_optimizer()

    assert trainer.test_loss(model) == pytest.approx(5.0)

