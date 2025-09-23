from pathlib import Path
import torch

from src.experiments.experiments import mass_mlp_recovery
from src.experiments.configs.mass_mlp_recovery import MassMLPRecoveryConfig
from src.experiments.configs.mlp_recovery import MLPRecoveryConfig
from src.training.trainer_config import TrainerConfig
from src.training.optimizers.configs.adam import AdamConfig
from src.models.architectures.configs.mlp import MLPConfig
from src.training.loss.configs.loss import LossConfig
from tests.helpers.stubs import StubJointDistribution
from src.utils.seed_manager import SeedManager


def _trainer_cfg(tmp_path: Path) -> TrainerConfig:
    return TrainerConfig(
        model_config=MLPConfig(
            input_dim=1,
            hidden_dims=[1],
            activation="relu",
            output_dim=1,
            start_activation=False,
            end_activation=False,
            bias=False,
        ),
        joint_distribution_config=StubJointDistribution._Config(
            X=torch.zeros(10, 1),
            y=torch.zeros(10, 1),
        ),
        optimizer_config=AdamConfig(lr=0.1),
        train_size=1,
        test_size=1,
        batch_size=1,
        epochs=1,
        loss_config=LossConfig(name="MSELoss"),
        home_dir=tmp_path,
    )


def test_get_experiment_configs(tmp_path, monkeypatch):
    teacher_cfg = _trainer_cfg(tmp_path / "teacher")
    student_cfg = _trainer_cfg(tmp_path / "student")

    base_cfg = MLPRecoveryConfig(
        home_directory=tmp_path / "base",
        seed=0,
        teacher_trainer_config=teacher_cfg,
        student_trainer_config=student_cfg,
    )

    cfg = MassMLPRecoveryConfig(
        home_directory=tmp_path / "mass",
        seed=0,
        mlp_recovery_config=base_cfg,
        train_sizes=[5, 6],
        learning_rates=[0.01],
        weight_decays=[0.1],
    )

    expected_mgr = SeedManager(cfg.seed)
    expected_seeds = [expected_mgr.spawn_seed() for _ in range(2)]

    exp = mass_mlp_recovery.MassMLPRecovery(cfg)
    sub_cfgs = exp.get_experiment_configs()

    assert len(sub_cfgs) == 2
    sub_cfg1, sub_cfg2 = sub_cfgs

    expected_dir1 = (
        cfg.home_directory / "train_size_5" / "lr_0.01" / "weight_decay_0.1"
    )
    expected_dir2 = (
        cfg.home_directory / "train_size_6" / "lr_0.01" / "weight_decay_0.1"
    )

    assert sub_cfg1.home_directory == expected_dir1
    assert sub_cfg1.home_directory.exists()
    assert sub_cfg2.home_directory == expected_dir2
    assert sub_cfg2.home_directory.exists()

    assert [sub_cfg1.seed, sub_cfg2.seed] == expected_seeds

    student1 = sub_cfg1.student_trainer_config
    assert student1.train_size == 5
    assert student1.optimizer_config.lr == 0.01
    assert student1.optimizer_config.weight_decay == 0.1

    params = exp.get_config_params(sub_cfg1)
    assert params == {
        "train_size": 5,
        "learning_rate": 0.01,
        "weight_decay": 0.1,
    }


def test_rerun_plots_deletes_results_json(tmp_path, monkeypatch):
    teacher_cfg = _trainer_cfg(tmp_path / "teacher")
    student_cfg = _trainer_cfg(tmp_path / "student")

    base_cfg = MLPRecoveryConfig(
        home_directory=tmp_path / "base",
        seed=0,
        teacher_trainer_config=teacher_cfg,
        student_trainer_config=student_cfg,
    )

    cfg = MassMLPRecoveryConfig(
        home_directory=tmp_path / "mass",
        seed=0,
        mlp_recovery_config=base_cfg,
        train_sizes=[1],
        learning_rates=[0.1],
        weight_decays=[0.0],
        rerun_plots=True,
    )

    exp = mass_mlp_recovery.MassMLPRecovery(cfg)
    sub_cfgs = exp.get_experiment_configs()
    for sub_cfg in sub_cfgs:
        res = sub_cfg.home_directory / "results.json"
        res.write_text("{}")
        assert res.exists()

    called = {}

    def fake_run(self):
        assert not any(self.config.home_directory.rglob("results.json"))
        called["called"] = True

    monkeypatch.setattr(mass_mlp_recovery.BatchExperiment, "run", fake_run)
    exp.run()
    assert called.get("called") is True


def test_script_rerun_plots_existing_experiment(tmp_path, monkeypatch):
    teacher_cfg = _trainer_cfg(tmp_path / "teacher")
    student_cfg = _trainer_cfg(tmp_path / "student")

    base_cfg = MLPRecoveryConfig(
        home_directory=tmp_path / "base",
        seed=0,
        teacher_trainer_config=teacher_cfg,
        student_trainer_config=student_cfg,
    )

    existing_cfg = MassMLPRecoveryConfig(
        home_directory=tmp_path / "mass",
        seed=0,
        mlp_recovery_config=base_cfg,
        train_sizes=[1],
        learning_rates=[0.1],
        weight_decays=[0.0],
        rerun_plots=False,
    )

    mass_mlp_recovery.MassMLPRecovery(existing_cfg)

    from scripts.mass_mlp_recovery import mass_mlp_recovery as script

    yaml_cfg = MassMLPRecoveryConfig(
        home_directory=tmp_path / "mass",
        seed=0,
        mlp_recovery_config=base_cfg,
        train_sizes=[1],
        learning_rates=[0.1],
        weight_decays=[0.0],
        rerun_plots=True,
    )

    def fake_build(_):
        return yaml_cfg

    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("{}")

    monkeypatch.setattr(script, "CONFIG_FILE", cfg_file)
    monkeypatch.setattr(script, "build_experiment_config_from_dict", fake_build)

    captured = {}

    def fake_run(self):
        captured["rerun_plots"] = self.config.rerun_plots

    monkeypatch.setattr(mass_mlp_recovery.BatchExperiment, "run", fake_run)
    script.main()

    assert captured["rerun_plots"] is True

