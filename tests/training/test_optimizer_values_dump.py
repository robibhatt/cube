import json
import torch
import src.models.bootstrap  # noqa: F401

from src.training.trainer import Trainer
from src.training.trainer_config import TrainerConfig
from src.training.loss.configs.loss import LossConfig
from src.models.architectures.configs.mlp import MLPConfig
from src.training.optimizers.configs.adam import AdamConfig
from tests.helpers.stubs import StubJointDistribution


def test_optimizer_values_written(tmp_path):
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
        joint_distribution_config=StubJointDistribution._Config(
            X=torch.zeros(4, 1),
            y=torch.zeros(4, 1),
        ),
        train_size=4,
        test_size=4,
        batch_size=2,
        epochs=1,
        home_dir=tmp_path,
        loss_config=LossConfig(name="MSELoss"),
        seed=0,
    )
    trainer = Trainer(cfg)
    trainer.train()

    out_file = tmp_path / "optimizer_values.json"
    assert out_file.exists()

    data = json.loads(out_file.read_text())
    assert data["optimizer"] == "Adam"
    assert data["mup_used"] is False
    assert len(data["groups"]) == 1
    assert data["groups"][0]["n_params"] == 1
    assert len(data["params"]) == 1
    param = data["params"][0]
    assert param["name"] == "net.0.weight"
    assert param["shape"] == [1, 1]
    assert param["group_index"] == 0
