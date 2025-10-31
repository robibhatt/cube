import json
from src.training.trainer import Trainer
from src.training.trainer_config import TrainerConfig
from src.models.mlp_config import MLPConfig
from src.training.sgd_config import SgdConfig
from src.data.cube_distribution_config import (
    CubeDistributionConfig,
)


def test_optimizer_values_written(tmp_path):
    cfg = TrainerConfig(
        mlp_config=MLPConfig(
            input_dim=1,
            hidden_dims=[],
        ),
        optimizer_config=SgdConfig(lr=0.01),
        cube_distribution_config=CubeDistributionConfig(
            input_dim=1,
            indices_list=[[0]],
            weights=[1.0],
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
    assert data["optimizer"] == "SGD"
    assert data["mup_used"] is True
    assert len(data["groups"]) == 2
    assert data["groups"][0]["n_params"] == 1
    assert data["groups"][1]["n_params"] == 1
    assert data["groups"][1]["weight_decay"] == 0.0
    assert len(data["params"]) == 2

    params_by_name = {p["name"]: p for p in data["params"]}
    assert set(params_by_name) == {"net.0.weight", "net.0.bias"}

    weight = params_by_name["net.0.weight"]
    assert weight["shape"] == [1, 1]
    assert weight["group_index"] == 0

    bias = params_by_name["net.0.bias"]
    assert bias["shape"] == [1]
    assert bias["group_index"] == 1
    assert bias["weight_decay"] == 0.0
