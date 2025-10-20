from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import csv
import shutil
import time
import subprocess
import json

from src.experiments.experiments.experiment import Experiment
from src.training.trainer_config import TrainerConfig
from src.experiments.configs.experiment import ExperimentConfig
from src.experiments.configs.experiment_config_registry import build_experiment_config_from_dict
from src.experiments.configs import ExperimentConfig
from src.experiments.experiments.experiment_factory import create_experiment
from pathlib import Path


class BatchExperiment(Experiment, ABC):
    """Abstract experiment that sequentially yields sub experiments."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)

    @abstractmethod
    def get_experiment_configs(self) -> List[ExperimentConfig]:
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

    def _consolidate_results(self) -> List[Dict[str, Any]]:
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

        out_file = self.config.home_directory / "results.csv"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            fieldnames = sorted({key for row in rows for key in row})
            with open(out_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        else:
            out_file.write_text("")

        return rows

    # ------------------------------ PARALLEL -----------------------------

    def _get_job_status(self, job_id: str) -> tuple[str, str]:
        """Return (state, exit_code) from ``sacct`` for ``job_id``."""

        proc = subprocess.run(
            [
                "sacct",
                "-j",
                job_id,
                "-n",
                "-X",
                "-P",
                "--format",
                "State,ExitCode",
            ],
            capture_output=True,
            text=True,
        )

        for line in proc.stdout.splitlines():
            if line.strip():
                state, code = line.strip().split("|", 1)
                return state, code
        return "", ""

    def _update_parallel_experiments(
        self, active_configs: list[tuple[ExperimentConfig, str]]
    ) -> None:
        """Check running jobs and resubmit failed ones."""

        

        remaining: list[tuple[ExperimentConfig, str]] = []

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

    def _verify_sub_exp(self, sub_exp_dir: Path) -> None:
        """Load the sub-experiment configuration to ensure it's valid."""
        build_experiment_config_from_dict(
            json.loads((sub_exp_dir / "experiment_config.json").read_text())
        )

    def run_parallel(self, configs: List[ExperimentConfig]) -> None:
        """Run sub-experiments in parallel using the cluster scheduler."""

        if not configs:
            self.consolidate_results()
            return

        active: List[tuple[ExperimentConfig, str]] = []

        for cfg in configs:

            # gets rid of old jobs, and resubmits jobs that were just lame
            self._update_parallel_experiments(active)

            # continually wait and reactivate stuff if we are stuck
            while len(active) >= 10:
                time.sleep(10)
                self._update_parallel_experiments(active)

            # active is small so it is time to add some new jobs
            if cfg.home_directory.exists():
                try:
                    self._verify_sub_exp(cfg.home_directory)
                except Exception:
                    # the directory got made but somehow a config was never initiailized there so kill it.
                    for item in cfg.home_directory.iterdir():
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                    create_experiment(cfg)
            else:
                cfg.home_directory.mkdir(parents=True, exist_ok=True)
                create_experiment(cfg)

            # only add this experiment to the queue if it hasn't already been run
            results_file = cfg.home_directory / "results.csv"
            if not results_file.exists():
                job_id = Experiment.server_run(cfg.home_directory)
                time.sleep(15)
                active.append((cfg, job_id))

        # once we added everything, we sit and wait
        while not all((cfg.home_directory / "results.csv").exists() for cfg in configs):
            self._update_parallel_experiments(active)
            time.sleep(15)

        self.consolidate_results()