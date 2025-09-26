import csv
import src.models.bootstrap  # noqa: F401

from src.training.trainer_config import TrainerConfig
from src.models.mlp_config import MLPConfig
from src.experiments.configs.train_mlp import TrainMLPExperimentConfig
from src.experiments.experiments.train_mlp_experiment import TrainMLPExperiment
from src.utils.seed_manager import SeedManager
from src.data.cube_distribution_config import (
    CubeDistributionConfig,
)


def _make_trainer_config() -> TrainerConfig:
    model_cfg = MLPConfig(
        input_dim=1,
        output_dim=1,
        hidden_dims=[],
        activation='relu',
        start_activation=False,
        end_activation=False,
    )
    dist_cfg = CubeDistributionConfig(
        input_dim=1,
        indices_list=[[0]],
        weights=[1.0],
        noise_mean=0.0,
        noise_std=0.0,
    )
    return TrainerConfig(
        mlp_config=model_cfg,
        cube_distribution_config=dist_cfg,
        train_size=1,
        test_size=1,
        batch_size=1,
        epochs=1,
    )


def test_trainer_seed_from_experiment_seed(tmp_path):
    trainer_cfg = _make_trainer_config()
    exp_seed = 42
    exp_cfg = TrainMLPExperimentConfig(
        trainer_config=trainer_cfg, home_directory=tmp_path, seed=exp_seed
    )
    experiment = TrainMLPExperiment(exp_cfg)
    trainer_seed = experiment.get_trainer_configs()[0][0].seed
    seed_mgr = SeedManager(exp_seed)
    expected_seed = seed_mgr.spawn_seed()
    assert trainer_seed == expected_seed
    trainer_seed_2 = experiment.get_trainer_configs()[0][0].seed
    assert trainer_seed_2 == trainer_seed


def test_train_and_consolidate(tmp_path):
    trainer_cfg = _make_trainer_config()
    exp_cfg = TrainMLPExperimentConfig(
        trainer_config=trainer_cfg, home_directory=tmp_path, seed=0
    )
    experiment = TrainMLPExperiment(exp_cfg)
    experiment.train()
    rows = experiment.consolidate_results()
    results_csv = tmp_path / 'results.csv'
    assert results_csv.exists()
    assert len(rows) == 1
    with open(results_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        data_row = next(reader)
    assert float(data_row['final_train_loss']) == rows[0]['final_train_loss']
