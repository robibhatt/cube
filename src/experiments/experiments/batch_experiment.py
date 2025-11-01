from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import csv
import shutil
import time
import subprocess

from src.experiments.experiments.experiment import Experiment
from src.training.trainer_config import TrainerConfig
from src.experiments.configs.experiment import ExperimentConfig
from src.experiments.configs import ExperimentConfig
from src.experiments.experiments.experiment_factory import create_experiment


class BatchExperiment(Experiment, ABC):
    """Abstract experiment that sequentially yields sub experiments."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        """Emit a timestamped log message with the experiment class name."""

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{now}] [{self.__class__.__name__}] {message}",
            flush=True,
        )

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

        self._log("Preparing for sequential execution")
        skip, _ = self.prepare_for_execution()
        if skip:
            self._log("Execution skipped because experiment is already complete")
            return

        configs = self.get_experiment_configs()

        self._log(f"Running {len(configs)} sub-experiments sequentially")

        for cfg in configs:
            if cfg.home_directory.exists():
                self._log(
                    f"Loading existing sub-experiment from {cfg.home_directory}"
                )
                exp = Experiment.from_dir(cfg.home_directory)
            else:
                self._log(
                    f"Creating new sub-experiment at {cfg.home_directory}"
                )
                exp = create_experiment(cfg)
            self._log(
                f"Starting sub-experiment in {cfg.home_directory.name}"
            )
            exp.run()

        self._log("Sequential execution finished; consolidating results")
        self.consolidate_results()

        self._log("Result consolidation complete")

    def get_trainer_configs(self) -> List[TrainerConfig]:
        """Return trainer configs from all sub-experiments.

        The trainer configurations from each sub-experiment are concatenated,
        preserving their original order.
        """

        configs = self.get_experiment_configs()
        if not configs:
            self._log("No sub-experiment configs found; consolidating results")
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

        if not active_configs:
            self._log("No active parallel jobs to update")
            return

        self._log(
            f"Updating status for {len(active_configs)} active parallel jobs"
        )
        remaining: list[tuple[ExperimentConfig, str]] = []

        for cfg, job_id in active_configs:
            results_file = cfg.home_directory / "results.csv"
            if results_file.exists():
                self._log(
                    f"Detected completed results for job {job_id} at {cfg.home_directory}"
                )
                continue

            state, _ = self._get_job_status(job_id)
            if state and not state.startswith(("PENDING", "RUNNING", "CONFIGURING")):
                self._log(
                    f"Job {job_id} for {cfg.home_directory} finished in state {state};"
                    " resubmitting"
                )
                for item in cfg.home_directory.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()

                time.sleep(15)
                create_experiment(cfg)
                new_id = Experiment.server_run(cfg.home_directory)
                self._log(
                    f"Resubmitted job {new_id} for {cfg.home_directory}"
                )
                remaining.append((cfg, new_id))
            else:
                if state:
                    self._log(
                        f"Job {job_id} for {cfg.home_directory} still in state {state}"
                    )
                remaining.append((cfg, job_id))

        active_configs[:] = remaining

    def run_parallel(self, configs: List[ExperimentConfig]) -> None:
        """Run sub-experiments in parallel using the cluster scheduler."""

        self._log(
            f"Preparing to launch {len(configs)} sub-experiments in parallel"
        )
        skip, _ = self.prepare_for_execution()
        if skip:
            self._log("Parallel execution skipped because experiment is complete")
            return

        if not configs:
            self._log("No configs provided for parallel execution; consolidating")
            self.consolidate_results()
            return

        active: List[tuple[ExperimentConfig, str]] = []

        for cfg in configs:

            # gets rid of old jobs, and resubmits jobs that were just lame
            self._update_parallel_experiments(active)

            # continually wait and reactivate stuff if we are stuck
            while len(active) >= 10:
                self._log(
                    "Scheduler queue full (>=10 active jobs); waiting before"
                    " submitting more"
                )
                time.sleep(10)
                self._update_parallel_experiments(active)

            # active is small so it is time to add some new jobs
            if cfg.home_directory.exists():
                try:
                    self._log(
                        f"Attempting to load existing sub-experiment at {cfg.home_directory}"
                    )
                    sub_experiment = Experiment.from_dir(cfg.home_directory)
                except Exception:
                    for item in cfg.home_directory.iterdir():
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                    sub_experiment = create_experiment(cfg)
            else:
                cfg.home_directory.mkdir(parents=True, exist_ok=True)
                self._log(
                    f"Creating sub-experiment directory at {cfg.home_directory}"
                )
                sub_experiment = create_experiment(cfg)

            skip_sub, _ = sub_experiment.prepare_for_execution()
            if skip_sub:
                self._log(
                    f"Skipping already completed sub-experiment at {cfg.home_directory}"
                )
                continue

            # only add this experiment to the queue if it hasn't already been run
            results_file = cfg.home_directory / "results.csv"
            if not results_file.exists():
                job_id = Experiment.server_run(cfg.home_directory)
                self._log(
                    f"Submitted job {job_id} for sub-experiment at {cfg.home_directory}"
                )
                time.sleep(15)
                active.append((cfg, job_id))

        # once we added everything, we sit and wait
        while not all((cfg.home_directory / "results.csv").exists() for cfg in configs):
            self._log(
                f"Waiting for {sum(not (cfg.home_directory / 'results.csv').exists() for cfg in configs)}"
                " remaining sub-experiments to finish"
            )
            self._update_parallel_experiments(active)
            time.sleep(15)

        self._log("All parallel sub-experiments completed; consolidating results")
        self.consolidate_results()
        self._log("Parallel result consolidation complete")
