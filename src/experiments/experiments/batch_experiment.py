from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import csv

from src.experiments.experiments.experiment import Experiment
from src.training.trainer_config import TrainerConfig
from src.experiments.configs.batch_experiment import BatchExperimentConfig
from src.experiments.configs import ExperimentConfig


class BatchExperiment(Experiment, ABC):
    """Abstract experiment that sequentially yields sub experiments."""

    def __init__(self, config: BatchExperimentConfig) -> None:
        super().__init__(config)

    @abstractmethod
    def get_experiment_configs(self) -> List["ExperimentConfig"]:
        """Return a list of sub-experiment configs to run."""
        raise NotImplementedError

    @abstractmethod
    def get_config_params(self, config: "ExperimentConfig") -> Dict[str, Any]:
        """Return parameters that vary between sub-experiments for ``config``."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Run all sub-experiments sequentially and consolidate results."""

        from src.experiments.experiments.experiment_factory import create_experiment

        configs = self.get_experiment_configs()

        for cfg in configs:
            exp = create_experiment(cfg)
            exp.train()
            exp.consolidate_results()

        self.consolidate_results()

    def get_trainer_configs(self) -> List[TrainerConfig]:
        """Return trainer configs from all sub-experiments.

        The trainer configurations from each sub-experiment are concatenated,
        preserving their original order.
        """

        configs = self.get_experiment_configs()
        if not configs:
            return []

        from src.experiments.experiments.experiment_factory import create_experiment

        combined: List[TrainerConfig] = []
        for cfg in configs:
            exp = create_experiment(cfg)
            combined.extend(exp.get_trainer_configs())

        return combined

    def consolidate_results(self) -> List[Dict[str, Any]]:
        """Consolidate results from all sub-experiments into one CSV.

        Assumes that each sub-experiment has already saved its results to a
        ``results.csv`` file in its home directory. If any results file is
        missing, a ``FileNotFoundError`` is raised.
        """

        configs = self.get_experiment_configs()

        rows: List[Dict[str, Any]] = []
        for cfg in configs:
            results_file = cfg.home_directory / "results.csv"
            if not results_file.exists():
                raise FileNotFoundError(
                    f"Missing results.csv for sub-experiment at {cfg.home_directory}"
                )
            params = self.get_config_params(cfg)
            with open(results_file, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    combined = dict(params)
                    combined.update(row)
                    rows.append(combined)

        if rows:
            fieldnames = sorted({key for row in rows for key in row})
            out_file = self.config.home_directory / "results.csv"
            with open(out_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        return rows
