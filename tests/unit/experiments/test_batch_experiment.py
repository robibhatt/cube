from dataclasses import dataclass
from pathlib import Path
import csv
import pytest

from dataclasses_json import dataclass_json

from src.experiments.experiments.batch_experiment import BatchExperiment
from src.experiments.configs.batch_experiment import BatchExperimentConfig
from src.experiments.experiments import register_experiment
from src.experiments.configs import register_experiment_config
from src.experiments.configs import build_experiment_config
from src.experiments.experiments.experiment import Experiment
from src.experiments.configs import ExperimentConfig


@register_experiment_config("DummyBatchSub")
@dataclass_json
@dataclass
class DummySubExperimentConfig(ExperimentConfig):
    param: int = 0

    def __post_init__(self) -> None:
        self.experiment_type = "DummyBatchSub"


@register_experiment("DummyBatchSub")
class DummySubExperiment(Experiment):
    called = 0

    def __init__(self, config: DummySubExperimentConfig) -> None:
        super().__init__(config)

    def get_trainer_configs(self) -> list[list]:
        DummySubExperiment.called += 1
        return [["a"], ["b"]]

    def consolidate_results(self):
        out_file = self.config.home_directory / "results.csv"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["a", "b"])
            writer.writeheader()
            writer.writerow({"a": 1, "b": 2})
        return [{"a": 1, "b": 2}]


class SimpleBatchExperiment(BatchExperiment):
    def __init__(self, config: BatchExperimentConfig, configs: list[ExperimentConfig]):
        super().__init__(config)
        self._configs = configs

    def get_experiment_configs(self) -> list[ExperimentConfig]:
        cfgs = self._configs
        self._configs = []
        return cfgs

    def get_config_params(self, config: ExperimentConfig) -> dict:
        return {"param": config.param}


def test_get_trainer_configs_returns_sub_exp(tmp_path: Path):
    sub_cfg1 = build_experiment_config(
        "DummyBatchSub", home_directory=tmp_path / "sub1", seed=0
    )
    sub_cfg2 = build_experiment_config(
        "DummyBatchSub", home_directory=tmp_path / "sub2", seed=0
    )

    batch_cfg = BatchExperimentConfig(home_directory=tmp_path / "batch", seed=0)
    batch_exp = SimpleBatchExperiment(batch_cfg, [sub_cfg1, sub_cfg2])

    cfgs = batch_exp.get_trainer_configs()
    assert cfgs == [["a"], ["b"], ["a"], ["b"]]
    assert DummySubExperiment.called == 2

    # No more experiments -> empty list
    cfgs_empty = batch_exp.get_trainer_configs()
    assert cfgs_empty == []


def test_train_consolidates_when_empty(tmp_path: Path, monkeypatch):
    batch_cfg = BatchExperimentConfig(home_directory=tmp_path / "batch", seed=0)
    batch_exp = SimpleBatchExperiment(batch_cfg, [])

    consolidated = {"done": False}

    def dummy_consolidate(self):
        consolidated["done"] = True

    monkeypatch.setattr(SimpleBatchExperiment, "consolidate_results", dummy_consolidate)

    batch_exp.train()
    assert consolidated["done"] is True


def test_consolidate_results(tmp_path: Path):
    sub_cfg1 = build_experiment_config(
        "DummyBatchSub", home_directory=tmp_path / "sub1", seed=0, param=1
    )
    sub_cfg2 = build_experiment_config(
        "DummyBatchSub", home_directory=tmp_path / "sub2", seed=0, param=2
    )

    batch_cfg = BatchExperimentConfig(home_directory=tmp_path / "batch", seed=0)

    # consolidation should fail if sub-experiment results are missing
    batch_exp_missing = SimpleBatchExperiment(batch_cfg, [sub_cfg1, sub_cfg2])
    with pytest.raises(FileNotFoundError):
        batch_exp_missing.consolidate_results()

    # create results for sub-experiments
    DummySubExperiment(sub_cfg1).consolidate_results()
    DummySubExperiment(sub_cfg2).consolidate_results()

    # consolidate once results exist
    batch_exp = SimpleBatchExperiment(batch_cfg, [sub_cfg1, sub_cfg2])
    rows = batch_exp.consolidate_results()

    assert rows == [
        {"param": 1, "a": "1", "b": "2"},
        {"param": 2, "a": "1", "b": "2"},
    ]

    out_file = batch_cfg.home_directory / "results.csv"
    assert out_file.exists()
    with open(out_file, newline="") as f:
        reader = csv.DictReader(f)
        assert list(reader) == [
            {"param": "1", "a": "1", "b": "2"},
            {"param": "2", "a": "1", "b": "2"},
        ]
