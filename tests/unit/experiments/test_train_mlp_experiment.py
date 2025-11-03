import csv

import pytest

from src.data.cube_distribution_config import CubeDistributionConfig
from src.experiments.configs.train_mlp import TrainMLPExperimentConfig
from src.experiments.experiments.train_mlp_experiment import TrainMLPExperiment
from src.models.mlp_config import MLPConfig
from src.training.trainer_config import TrainerConfig
from src.utils.seed_manager import SeedManager


def _make_trainer_config() -> TrainerConfig:
    model_cfg = MLPConfig(
        input_dim=1,
        hidden_dims=[1],
    )
    dist_cfg = CubeDistributionConfig(
        input_dim=1,
        indices_list=[[0]],
        weights=[1.0],
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
        trainer_config=trainer_cfg,
        home_directory=tmp_path,
        seed=exp_seed,
        mse_threshold=0.05,
        mse_samples=4,
    )
    experiment = TrainMLPExperiment(exp_cfg)
    trainer_seed = experiment.get_trainer_configs()[0].seed
    seed_mgr = SeedManager(exp_seed)
    expected_seed = seed_mgr.spawn_seed()
    assert trainer_seed == expected_seed
    trainer_seed_2 = experiment.get_trainer_configs()[0].seed
    assert trainer_seed_2 == trainer_seed


def test_train_and_consolidate(tmp_path):
    trainer_cfg = _make_trainer_config()
    edge_thresholds = [0.0, 0.5]
    exp_cfg = TrainMLPExperimentConfig(
        trainer_config=trainer_cfg,
        home_directory=tmp_path,
        seed=0,
        edge_thresholds=edge_thresholds,
        mse_threshold=0.05,
        mse_samples=4,
        ancestor_threshold=5,
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
    assert float(data_row['sparsify_threshold']) == rows[0]['sparsify_threshold']
    assert float(data_row['sparsified_mse']) == rows[0]['sparsified_mse']
    assert float(data_row['sparsified_mean_mse']) == rows[0]['sparsified_mean_mse']

    sparsified_dir = tmp_path / 'sparsified_mlp'
    assert sparsified_dir.exists()

    visualization_root = tmp_path / 'visualizations'
    original_png = visualization_root / 'original' / 'visualization.png'
    sparsified_png = visualization_root / 'sparsified' / 'visualization.png'
    assert original_png.exists()
    assert sparsified_png.exists()


@pytest.mark.parametrize("missing_strategy", ["delattr", "none"])
def test_defaults_applied_for_missing_mse_params(tmp_path, missing_strategy):
    trainer_cfg = _make_trainer_config()
    exp_cfg = TrainMLPExperimentConfig(
        trainer_config=trainer_cfg,
        home_directory=tmp_path,
        seed=7,
    )

    if missing_strategy == "delattr":
        delattr(exp_cfg, "mse_threshold")
        delattr(exp_cfg, "mse_samples")
        delattr(exp_cfg, "ancestor_threshold")
    else:
        exp_cfg.mse_threshold = None
        exp_cfg.mse_samples = None
        exp_cfg.ancestor_threshold = None

    experiment = TrainMLPExperiment(exp_cfg)

    assert experiment.config.mse_threshold == pytest.approx(0.01)
    assert experiment.config.mse_samples == 64
    assert experiment.config.ancestor_threshold == 4


def test_ancestor_flag_created_when_threshold_not_met(tmp_path):
    trainer_cfg = _make_trainer_config()
    exp_cfg = TrainMLPExperimentConfig(
        trainer_config=trainer_cfg,
        home_directory=tmp_path,
        seed=0,
        mse_threshold=0.05,
        mse_samples=4,
        ancestor_threshold=10,
    )
    experiment = TrainMLPExperiment(exp_cfg)
    experiment.train()
    experiment.consolidate_results()

    flag_path = tmp_path / "sparsified_mlp" / "ANCESTOR_FLAG"
    assert flag_path.exists()
    contents = flag_path.read_text(encoding="utf-8").strip().splitlines()
    assert contents
    assert any("threshold 10" in line for line in contents)
