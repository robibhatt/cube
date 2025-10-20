from pathlib import Path

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
        edge_thresholds=[0.1, 0.2],
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
    assert mlp_cfg.start_activation is False
    assert mlp_cfg.end_activation is False

    dist_cfg = trainer_cfg.cube_distribution_config
    assert dist_cfg is not None
    assert dist_cfg.input_dim == 3
    assert dist_cfg.indices_list == [[0, 1]]
    assert dist_cfg.weights == [1.0]
    assert dist_cfg.noise_std == 0.0
    assert dist_cfg.noise_mean == 0.0

    assert sub_cfg.edge_thresholds == [0.1, 0.2]

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
        edge_thresholds=[],
        learning_rates=[0.1],
        batch_sizes=[64],
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
        "edge_thresholds": [],
        "seed": sub_cfg.seed,
    }
