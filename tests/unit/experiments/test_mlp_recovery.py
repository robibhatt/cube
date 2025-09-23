import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import src.models.bootstrap  # noqa: F401
from src.models.architectures.model_factory import create_model
from src.models.architectures.configs.mlp import MLPConfig
from src.training.optimizers.configs.adam import AdamConfig
from src.training.loss.configs.loss import LossConfig
from tests.helpers.stubs import StubJointDistribution, StubOptimizer
from src.data.joint_distributions.configs.representor_distribution import (
    RepresentorDistributionConfig,
)
from src.data.joint_distributions.configs.noisy_distribution import (
    NoisyDistributionConfig,
)
from src.data.joint_distributions.configs.gaussian import GaussianConfig


def test_mlp_recovery_runs(tmp_path, monkeypatch):
    created = []
    trained = []
    saved = []

    from src.training.trainer_config import TrainerConfig

    class StubDistribution:
        def __init__(self, config, device):
            self.config = config

        def average_output_variance(self, n_samples: int = 1000, seed: int = 0):
            return 0.5

        def sample(self, n_samples: int, seed: int):
            return torch.zeros(n_samples, 1), torch.zeros(n_samples, 1)

        def base_sample(self, n_samples: int, seed: int):
            return self.sample(n_samples, seed)

        def forward_X(self, X: torch.Tensor):
            return X

    class StubTrainer:
        def __init__(self, config: TrainerConfig):
            self.config = config
            created.append(config.home_dir)
            config.home_dir.mkdir(parents=True, exist_ok=True)
            self.datasets_dir = config.home_dir / "datasets"
            self.datasets_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir = config.home_dir / "checkpoints"
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.joint_distribution = StubDistribution(None, None)
            cfg_path = config.home_dir / "trainer_config.json"
            cfg_path.write_text(config.to_json(indent=2))
            from src.checkpoints.checkpoint import Checkpoint
            if not (self.checkpoint_dir / "checkpoint.pkl").exists():
                checkpoint = Checkpoint(dir=self.checkpoint_dir)
                model, optimizer = self._load_model_and_optimizer()
                checkpoint.save(model=model, optimizer=optimizer.stepper)

        @classmethod
        def from_dir(cls, home_dir: Path):
            cfg = TrainerConfig(
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
                home_dir=home_dir,
                loss_config=LossConfig(name="MSELoss"),
            )
            return cls(cfg)

        def _load_model_and_optimizer(self):
            model = create_model(self.config.model_config)
            for p in model.parameters():
                nn.init.constant_(p, 0.0)
            return model, StubOptimizer(model)

        def _load_model(self):
            model, _ = self._load_model_and_optimizer()
            return model

        def get_iterator(self, split):
            return [(torch.zeros(1, 1), torch.zeros(1, 1))]

        def train(self):
            trained.append(self.config.home_dir)

        def get_results(self):
            return {
                "final_train_loss": 1.0,
                "final_test_loss": 0.0,
                "mean_output_loss": 3.0,
                "epochs_trained": 1,
            }

        def save_results(self):
            saved.append(self.config.home_dir)
            res_file = self.config.home_dir / "results.json"
            res_file.write_text(json.dumps(self.get_results()))

        @property
        def started_training(self):
            return True

        @property
        def finished_training(self):
            return True


    monkeypatch.setattr("src.training.trainer.Trainer", StubTrainer)
    monkeypatch.setattr(
        "src.experiments.experiments.experiment.create_trainer",
        lambda cfg: StubTrainer(cfg),
    )
    monkeypatch.setattr(
        "src.experiments.experiments.experiment.trainer_from_dir",
        lambda home_dir: StubTrainer.from_dir(home_dir),
    )
    monkeypatch.setattr(
        "src.experiments.experiments.mlp_recovery.trainer_from_dir",
        lambda home_dir: StubTrainer.from_dir(home_dir),
    )
    monkeypatch.setattr(
        "src.data.joint_distributions.joint_distribution_factory.create_joint_distribution",
        lambda cfg, device: StubDistribution(cfg, device),
    )

    from src.experiments.experiments.mlp_recovery import (
        MLPRecovery,
    )
    from src.experiments.configs.mlp_recovery import (
        MLPRecoveryConfig,
    )

    teacher_trainer_dir = tmp_path / "trainer"
    teacher_trainer_dir.mkdir()
    teacher_cfg = TrainerConfig(
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
            X=torch.randn(10, 1),
            y=torch.randn(10, 1),
        ),
        optimizer_config=AdamConfig(lr=0.1),
        train_size=1,
        test_size=1,
        batch_size=1,
        epochs=1,
        home_dir=teacher_trainer_dir,
        loss_config=LossConfig(name="MSELoss"),
    )

    student_cfg = teacher_cfg.deep_copy()
    student_cfg.joint_distribution_config = None

    cfg = MLPRecoveryConfig(
        teacher_trainer_config=teacher_cfg,
        student_trainer_config=student_cfg,
        home_directory=tmp_path / "exp",
        seed=0,
    )

    exp = MLPRecovery(cfg)

    teacher_dir = cfg.home_directory / "teacher"
    student_dir = cfg.home_directory / "student"

    assert created == []

    exp.train()

    assert teacher_dir in trained
    assert student_dir in trained
    assert teacher_dir in saved
    assert student_dir in saved

    results = exp.consolidate_results()
    res_file = cfg.home_directory / "results.csv"
    assert res_file.exists()
    import csv
    with open(res_file) as f:
        rows = list(csv.DictReader(f))

    assert rows == [
        {
            "train_size": "1",
            "trial_number": "0",
            "mean_output_loss": "3.0",
            "final_test_loss": "0.0",
            "final_train_loss": "1.0",
        }
    ]
    assert results == [
        {
            "train_size": 1,
            "trial_number": 0,
            "mean_output_loss": 3.0,
            "final_test_loss": 0.0,
            "final_train_loss": 1.0,
        }
    ]


def test_student_distribution_config(tmp_path):
    from src.training.trainer_config import TrainerConfig
    from src.experiments.experiments.mlp_recovery import MLPRecovery
    from src.experiments.configs.mlp_recovery import MLPRecoveryConfig

    teacher_cfg = TrainerConfig(
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
        home_dir=tmp_path / "teacher",
    )

    student_cfg = teacher_cfg.deep_copy()
    student_cfg.joint_distribution_config = None

    cfg = MLPRecoveryConfig(
        teacher_trainer_config=teacher_cfg,
        student_trainer_config=student_cfg,
        home_directory=tmp_path / "exp",
        seed=0,
        start_layer=1,
    )

    exp = MLPRecovery(cfg)
    _, student_cfgs = exp.get_trainer_configs()
    student_trainer = student_cfgs[0]

    jd_cfg = student_trainer.joint_distribution_config
    assert isinstance(jd_cfg, RepresentorDistributionConfig)
    assert jd_cfg.from_rep == 2


def test_student_distribution_with_noise(tmp_path):
    from src.training.trainer_config import TrainerConfig
    from src.experiments.experiments.mlp_recovery import MLPRecovery
    from src.experiments.configs.mlp_recovery import MLPRecoveryConfig

    teacher_cfg = TrainerConfig(
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
        home_dir=tmp_path / "teacher",
    )

    student_cfg = teacher_cfg.deep_copy()
    student_cfg.joint_distribution_config = None

    cfg = MLPRecoveryConfig(
        teacher_trainer_config=teacher_cfg,
        student_trainer_config=student_cfg,
        home_directory=tmp_path / "exp",
        seed=0,
        noise_variance=0.04,
    )

    exp = MLPRecovery(cfg)
    _, student_cfgs = exp.get_trainer_configs()
    student_trainer = student_cfgs[0]

    jd_cfg = student_trainer.joint_distribution_config
    assert isinstance(jd_cfg, NoisyDistributionConfig)
    assert isinstance(jd_cfg.base_distribution_config, RepresentorDistributionConfig)
    assert isinstance(jd_cfg.noise_distribution_config, GaussianConfig)

