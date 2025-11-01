from pathlib import Path

import pytest

from src.experiments.configs.dkwl import DkwlExperimentConfig
from src.experiments.experiments import create_experiment
from src.experiments.experiments.dkwl_experiment import DkwlExperiment
from src.experiments.configs.train_mlp import TrainMLPExperimentConfig
from src.utils.seed_manager import SeedManager


def test_get_experiment_configs_builds_expected_sub_configs(tmp_path: Path) -> None:
    cfg = DkwlExperimentConfig(
        ds=[3],
        ks=[2, 5],
        widths=[16],
        layers=[2],
        train_sizes=[256],
        epochs=[20],
        l1_decays=[0.001],
        mse_threshold=0.05,
        mse_samples=64,
        ancestor_threshold=3,
        learning_rates=[0.01],
        batch_sizes=[32],
        home_directory=tmp_path / "dkwl",
        seed=99,
    )

    experiment = create_experiment(cfg)
    assert isinstance(experiment, DkwlExperiment)

    sub_configs = experiment.get_experiment_configs()
    assert len(sub_configs) == 1

    sub_cfg = sub_configs[0]
    assert isinstance(sub_cfg, TrainMLPExperimentConfig)

    trainer_cfg = sub_cfg.trainer_config
    assert trainer_cfg.train_size == 256
    assert trainer_cfg.test_size == 256
    assert trainer_cfg.batch_size == 32
    assert trainer_cfg.epochs == 20
    assert trainer_cfg.weight_decay_l1 == 0.001
    assert trainer_cfg.optimizer_config is not None
    assert trainer_cfg.optimizer_config.lr == 0.01

    mlp_cfg = trainer_cfg.mlp_config
    assert mlp_cfg is not None
    assert mlp_cfg.input_dim == 3
    assert mlp_cfg.output_dim == 1
    assert mlp_cfg.hidden_dims == [16, 16]

    dist_cfg = trainer_cfg.cube_distribution_config
    assert dist_cfg is not None
    assert dist_cfg.input_dim == 3
    assert dist_cfg.indices_list == [[0, 1]]
    assert dist_cfg.weights == [1.0]
    assert dist_cfg.noise_std == 0.0

    assert sub_cfg.mse_threshold == 0.05
    assert sub_cfg.mse_samples == 64
    assert sub_cfg.ancestor_threshold == 3

    expected_dir = (
        tmp_path
        / "dkwl"
        / "d3"
        / "k2"
        / "width16"
        / "layers2"
        / "train256"
        / "epochs20"
        / "lr0p01"
        / "batch32"
        / "l10p001"
    )
    assert sub_cfg.home_directory == expected_dir

    seed_mgr = SeedManager(cfg.seed)
    assert sub_cfg.seed == seed_mgr.spawn_seed()


def test_get_config_params_reports_hyperparameters(tmp_path: Path) -> None:
    cfg = DkwlExperimentConfig(
        ds=[4],
        ks=[3],
        widths=[10],
        layers=[1],
        train_sizes=[512],
        epochs=[50],
        l1_decays=[0.0],
        learning_rates=[0.1],
        batch_sizes=[64],
        ancestor_threshold=4,
        home_directory=tmp_path / "exp",
        seed=7,
    )

    experiment = create_experiment(cfg)
    sub_cfg = experiment.get_experiment_configs()[0]

    params = experiment.get_config_params(sub_cfg)
    assert params == {
        "d": 4,
        "k": 3,
        "width": 10,
        "layers": 1,
        "train_size": 512,
        "epochs": 50,
        "l1_decay": 0.0,
        "learning_rate": 0.1,
        "batch_size": 64,
        "mse_threshold": cfg.mse_threshold,
        "mse_samples": cfg.mse_samples,
        "ancestor_threshold": cfg.ancestor_threshold,
        "seed": sub_cfg.seed,
    }


@pytest.mark.parametrize("missing_strategy", ["delattr", "none"])
def test_defaults_applied_for_missing_mse_params(tmp_path: Path, missing_strategy):
    cfg = DkwlExperimentConfig(
        ds=[2],
        ks=[1],
        widths=[8],
        layers=[1],
        train_sizes=[32],
        epochs=[5],
        l1_decays=[0.0],
        learning_rates=[0.1],
        batch_sizes=[16],
        home_directory=tmp_path / "exp",
        seed=11,
    )

    if missing_strategy == "delattr":
        delattr(cfg, "mse_threshold")
        delattr(cfg, "mse_samples")
        delattr(cfg, "ancestor_threshold")
    else:
        cfg.mse_threshold = None
        cfg.mse_samples = None
        cfg.ancestor_threshold = None

    experiment = create_experiment(cfg)
    assert experiment.config.mse_threshold == pytest.approx(0.01)
    assert experiment.config.mse_samples == 8192
    assert experiment.config.ancestor_threshold == 2

    sub_cfg = experiment.get_experiment_configs()[0]
    assert sub_cfg.mse_threshold == pytest.approx(0.01)
    assert sub_cfg.mse_samples == 8192
    assert sub_cfg.ancestor_threshold == 2
