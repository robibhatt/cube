from dataclasses import dataclass
from pathlib import Path

from dataclasses_json import dataclass_json

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

    def consolidate_results(self):
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
        def train(self):
            calls.append("train")

        def consolidate_results(self):
            calls.append("consolidate")

    def fake_from_dir(path):
        assert path == tmp_path
        return Dummy()

    monkeypatch.setattr(run_script.Experiment, "from_dir", staticmethod(fake_from_dir))
    monkeypatch.setattr(run_script.sys, "argv", ["run.py", str(tmp_path)])

    run_script.main()

    assert calls == ["train", "consolidate"]
