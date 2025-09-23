from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import csv
import shutil
import time

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
        """Run all sub-experiments.

        If ``self.config.run_parallel`` is ``True`` then experiments are
        submitted to the cluster scheduler and executed in parallel.  Otherwise
        they are executed sequentially on the local machine.  After all
        experiments finish, :meth:`consolidate_results` is invoked to aggregate
        the results across sub-experiments.
        """

        from .experiment_factory import create_experiment

        configs = self.get_experiment_configs()

        if self.config.run_parallel:
            self.run_parallel(configs)
            return

        for cfg in configs:
            exp = create_experiment(cfg)
            exp.train()
            exp.consolidate_results()

        self.consolidate_results()

    # ------------------------------ PARALLEL -----------------------------

    def _update_parallel_experiments(
        self, active_configs: List[Tuple[ExperimentConfig, str]]
    ) -> None:
        """Check running jobs and resubmit failed ones."""

        from .experiment_factory import create_experiment

        remaining: List[Tuple[ExperimentConfig, str]] = []

        for cfg, job_id in active_configs:
            results_file = cfg.home_directory / "results.csv"
            if results_file.exists():
                continue

            state, _ = self._get_job_status(job_id)
            if state and not state.startswith(("PENDING", "RUNNING", "CONFIGURING")):
                for item in cfg.home_directory.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()

                time.sleep(15)
                create_experiment(cfg)
                new_id = Experiment.server_run(cfg.home_directory)
                remaining.append((cfg, new_id))
            else:
                remaining.append((cfg, job_id))

        active_configs[:] = remaining

    def run_parallel(self, configs: List[ExperimentConfig]) -> None:
        """Run sub-experiments in parallel using the cluster scheduler."""

        from .experiment_factory import create_experiment

        if not configs:
            self.consolidate_results()
            return

        active: List[Tuple[ExperimentConfig, str]] = []

        for cfg in configs:
            self._update_parallel_experiments(active)

            while len(active) >= 10:
                time.sleep(10)
                self._update_parallel_experiments(active)

            if cfg.home_directory.exists():
                try:
                    self._verify_sub_exp(cfg.home_directory)
                except Exception:
                    for item in cfg.home_directory.iterdir():
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                    create_experiment(cfg)
            else:
                cfg.home_directory.mkdir(parents=True, exist_ok=True)
                create_experiment(cfg)

            results_file = cfg.home_directory / "results.csv"
            if not results_file.exists():
                job_id = Experiment.server_run(cfg.home_directory)
                time.sleep(15)
                active.append((cfg, job_id))

        while not all((cfg.home_directory / "results.csv").exists() for cfg in configs):
            self._update_parallel_experiments(active)
            time.sleep(15)

        self.consolidate_results()

    def get_trainer_configs(self) -> list[list[TrainerConfig]]:
        """Return trainer configs from all sub-experiments.

        The trainer configuration lists from each sub-experiment are simply
        concatenated, preserving their original order.
        """

        configs = self.get_experiment_configs()
        if not configs:
            return []

        from .experiment_factory import create_experiment

        combined: list[list[TrainerConfig]] = []
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
