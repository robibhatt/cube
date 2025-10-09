import pytest
from src.training.trainer_config import TrainerConfig
from src.models.mlp_config import MLPConfig
from src.data.cube_distribution_config import (
    CubeDistributionConfig,
)
import pytest


def test_trainer_config_json_roundtrip(tmp_path):
    home_dir = tmp_path / "trainer_home"
    home_dir.mkdir()
    cfg = TrainerConfig(
        mlp_config=MLPConfig(
            input_dim=3,
            output_dim=1,
            hidden_dims=[4, 2],
            activation="relu",
            start_activation=False,
            end_activation=False,
        ),
        cube_distribution_config=CubeDistributionConfig(
            input_dim=3,
            indices_list=[[0]],
            weights=[1.0],
            noise_mean=0.0,
            noise_std=0.0,
        ),
        train_size=10,
        test_size=5,
        batch_size=2,
        epochs=1,
        home_dir=home_dir,
        seed=42,
    )

    json_str = cfg.to_json()
    restored = TrainerConfig.from_json(json_str)
    assert restored == cfg


def test_trainer_config_requires_output_shape(tmp_path, mlp_config):
    home_dir = tmp_path / "h"
    home_dir.mkdir()
    cfg = TrainerConfig(
        mlp_config=mlp_config,
        cube_distribution_config=CubeDistributionConfig(
            input_dim=mlp_config.input_dim + 1,
            indices_list=[[0]],
            weights=[1.0],
        ),
        train_size=1,
        test_size=1,
        batch_size=1,
        epochs=1,
        home_dir=home_dir,
    )
    with pytest.raises(AssertionError):
        cfg.ready_for_trainer()


def test_ready_for_trainer_missing_fields(tmp_path):
    cfg = TrainerConfig(home_dir=tmp_path / "h")
    with pytest.raises(AssertionError):
        cfg.ready_for_trainer()




