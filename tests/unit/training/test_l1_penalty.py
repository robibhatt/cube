import torch
import pytest

from src.training.trainer import Trainer
from src.training.trainer_config import TrainerConfig
from src.training.optimizers.configs.adam import AdamConfig
from src.models.architectures.configs.mlp import MLPConfig
from src.models.architectures.mlp import MLP
from src.data.joint_distributions.configs.cube_distribution import (
    CubeDistributionConfig,
)


def _make_trainer(tmp_path, hidden_dims):
    cfg = TrainerConfig(
        mlp_config=MLPConfig(
            input_dim=1,
            output_dim=1,
            hidden_dims=hidden_dims,
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
    )
    trainer = Trainer(cfg)
    assert cfg.mlp_config is not None
    model = MLP(cfg.mlp_config)
    return trainer, model


def test_l1_penalty_averages_hidden_layers(tmp_path):
    trainer, model = _make_trainer(tmp_path, [1, 1, 1])
    with torch.no_grad():
        model.linear_layers[0].weight.fill_(1.0)
        model.linear_layers[1].weight.fill_(2.0)
        model.linear_layers[2].weight.fill_(3.0)
        model.linear_layers[3].weight.fill_(4.0)
    penalty = trainer._l1_penalty(model).item()
    assert penalty == pytest.approx(7.5)


def test_l1_penalty_single_hidden_layer(tmp_path):
    trainer, model = _make_trainer(tmp_path, [1])
    with torch.no_grad():
        model.linear_layers[0].weight.fill_(1.0)
        model.linear_layers[1].weight.fill_(4.0)
    penalty = trainer._l1_penalty(model).item()
    assert penalty == pytest.approx(5.0)
