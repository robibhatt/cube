from dataclasses import dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Optional

from dataclasses_json import dataclass_json

import src.experiments.experiments.experiment as experiment_module
from src.experiments.experiments import register_experiment
from src.experiments.configs import (
    register_experiment_config,
    build_experiment_config,
)
from src.experiments.experiments.experiment import Experiment
from src.experiments.configs import ExperimentConfig


@register_experiment_config("DummyExperiment")
@dataclass_json
@dataclass
class DummyExperimentConfig(ExperimentConfig):
    def __post_init__(self) -> None:
        self.experiment_type = "DummyExperiment"


@register_experiment("DummyExperiment")
class DummyExperiment(Experiment):
    def __init__(self, config: DummyExperimentConfig) -> None:
        super().__init__(config)
        self.called = getattr(self, "called", 0)

    def get_trainer_configs(self) -> list:
        self.called = getattr(self, "called", 0) + 1
        return []

    def _consolidate_results(self):
        out_file = self.config.home_directory / "results.csv"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text("")
        return None


def test_experiment_save_and_load(tmp_path: Path):
    home = tmp_path / "exp"
    cfg = build_experiment_config("DummyExperiment", home_directory=home, seed=1)
    exp = DummyExperiment(cfg)
    assert exp.called == 0

    exp.train()
    assert exp.called == 1

    cfg_path = home / "experiment_config.json"
    assert cfg_path.exists()

    # verify directory already exists when loading
    assert home.exists()

    loaded = Experiment.from_dir(home)
    assert isinstance(loaded, DummyExperiment)
    assert loaded.config.seed == exp.config.seed
    assert loaded.config.home_directory == exp.config.home_directory
    assert loaded.called == 0
    loaded.train()
    assert loaded.called == 1


def test_experiment_existing_directory_allowed(tmp_path: Path):
    home = tmp_path / "exp"
    home.mkdir()
    cfg = build_experiment_config("DummyExperiment", home_directory=home, seed=1)
    exp = DummyExperiment(cfg)
    assert exp.config.home_directory == home


def test_run_script_invokes_experiment(monkeypatch, tmp_path: Path):
    from scripts import run as run_script

    calls = []

    class Dummy:
        def run(self):
            calls.append("run")

    def fake_from_dir(path):
        assert path == tmp_path
        return Dummy()

    monkeypatch.setattr(run_script.Experiment, "from_dir", staticmethod(fake_from_dir))
    monkeypatch.setattr(run_script.sys, "argv", ["run.py", str(tmp_path)])

    run_script.main()

    assert calls == ["run"]


def test_build_experiment_config_ignores_unknown_fields(tmp_path: Path):
    cfg = build_experiment_config(
        "DummyExperiment",
        home_directory=tmp_path / "exp",
        seed=5,
        edge_thresholds=[0.1],
        extraneous_flag=True,
    )

    assert isinstance(cfg, DummyExperimentConfig)
    assert not hasattr(cfg, "edge_thresholds")
    assert not hasattr(cfg, "extraneous_flag")


def _build_dummy_experiment(tmp_path: Path) -> DummyExperiment:
    home = tmp_path / "exp"
    cfg = build_experiment_config("DummyExperiment", home_directory=home, seed=1)
    return DummyExperiment(cfg)


def test_prepare_for_execution_preserves_trainer_directory(tmp_path: Path):
    exp = _build_dummy_experiment(tmp_path)

    trainer = exp.config.home_directory / "trainer-0"
    trainer.mkdir()
    (trainer / "trainer_config.json").write_text("{}")
    (trainer / "checkpoint.pt").write_text("ckpt")

    stray_file = exp.config.home_directory / "old_results.csv"
    stray_file.write_text("old")
    stray_dir = exp.config.home_directory / "temp"
    stray_dir.mkdir()
    (stray_dir / "scratch.txt").write_text("data")

    exp.prepare_for_execution()

    assert trainer.exists()
    assert (trainer / "checkpoint.pt").exists()
    assert not stray_file.exists()
    assert not stray_dir.exists()
    assert (exp.config.home_directory / "experiment_config.json").exists()


def test_prepare_for_execution_keeps_completed_trainer(tmp_path: Path):
    exp = _build_dummy_experiment(tmp_path)

    trainer = exp.config.home_directory / "trainer-1"
    trainer.mkdir()
    (trainer / "trainer_config.json").write_text("{}")
    (trainer / "results.json").write_text("{}")
    (trainer / "checkpoint.pt").write_text("ckpt")

    exp.prepare_for_execution()

    assert trainer.exists()
    assert (trainer / "checkpoint.pt").exists()


class _TrainerStub:
    existing: dict[Path, "_TrainerStub"] = {}

    def __init__(self, config) -> None:
        self.config = config
        self._epochs = 0
        self._finished = False
        self._started = False
        self.train_calls = 0
        self.save_calls = 0
        self._epochs_after = 0
        _TrainerStub.existing[Path(self.config.home_dir)] = self

    @classmethod
    def reset(cls) -> None:
        cls.existing.clear()

    @classmethod
    def for_path(
        cls,
        home_dir: Path,
        *,
        epochs: int = 0,
        finished: bool = False,
        epochs_after: Optional[int] = None,
    ) -> "_TrainerStub":
        inst = cls.__new__(cls)
        inst.config = SimpleNamespace(home_dir=Path(home_dir))
        inst._epochs = epochs
        inst._finished = finished
        inst._started = epochs > 0
        inst.train_calls = 0
        inst.save_calls = 0
        inst._epochs_after = epochs if epochs_after is None else epochs_after
        cls.existing[Path(home_dir)] = inst
        return inst

    @classmethod
    def from_dir(cls, home_dir: Path) -> "_TrainerStub":
        return cls.existing[Path(home_dir)]

    def train(self) -> None:
        self.train_calls += 1
        self._started = True
        self._epochs = self._epochs_after
        self._finished = True

    def save_results(self) -> None:
        self.save_calls += 1

    @property
    def epochs_trained(self) -> int:
        return self._epochs

    @property
    def finished_training(self) -> bool:
        return self._finished

    @property
    def started_training(self) -> bool:
        return self._started


def test_train_logs_resumption(monkeypatch, tmp_path: Path, capsys):
    exp = _build_dummy_experiment(tmp_path)
    trainer_dir = exp.config.home_directory / "trainer-0"
    trainer_dir.mkdir(parents=True)

    _TrainerStub.reset()
    trainer = _TrainerStub.for_path(
        trainer_dir, epochs=5, finished=False, epochs_after=7
    )

    monkeypatch.setattr(experiment_module, "Trainer", _TrainerStub)

    def fake_get_configs(self):
        return [SimpleNamespace(home_dir=trainer_dir)]

    exp.get_trainer_configs = MethodType(fake_get_configs, exp)

    exp.train()

    captured = capsys.readouterr()
    assert "Resuming trainer at" in captured.out
    assert str(trainer_dir.resolve()) in captured.out
    assert "epoch 6" in captured.out
    assert f"epoch {trainer.epochs_trained}" in captured.out
    assert trainer.train_calls == 1
    assert trainer.save_calls == 1


def test_train_logs_when_trainer_already_complete(monkeypatch, tmp_path: Path, capsys):
    exp = _build_dummy_experiment(tmp_path)
    trainer_dir = exp.config.home_directory / "trainer-0"
    trainer_dir.mkdir(parents=True)

    _TrainerStub.reset()
    trainer = _TrainerStub.for_path(
        trainer_dir, epochs=12, finished=True, epochs_after=12
    )

    monkeypatch.setattr(experiment_module, "Trainer", _TrainerStub)

    def fake_get_configs(self):
        return [SimpleNamespace(home_dir=trainer_dir)]

    exp.get_trainer_configs = MethodType(fake_get_configs, exp)

    exp.train()

    captured = capsys.readouterr()
    assert "already completed 12 epochs" in captured.out
    assert "reusing results" in captured.out
    assert trainer.train_calls == 0
    assert trainer.save_calls == 1
