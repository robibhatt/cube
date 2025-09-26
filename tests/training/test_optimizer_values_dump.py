import json
import src.models.bootstrap  # noqa: F401

from src.training.trainer import Trainer
from src.training.trainer_config import TrainerConfig
from src.models.architectures.configs.mlp import MLPConfig
from src.training.optimizers.configs.adam import AdamConfig
from src.data.joint_distributions.configs.cube_distribution import (
    CubeDistributionConfig,
)


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
        epochs=1,
        home_dir=tmp_path,
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
