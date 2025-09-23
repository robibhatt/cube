from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, List, Tuple
import time
import subprocess
import shutil
import torch
import os
import re

from src.training.trainer_factory import create_trainer, trainer_from_dir
from src.training.trainer_config import TrainerConfig

from src.experiments.configs import ExperimentConfig
from src.utils.seed_manager import SeedManager
from src.experiments.configs import build_experiment_config_from_dict
from src.experiments.experiments.experiment_factory import create_experiment
from src.training.trainer import Trainer

class Experiment(ABC):

    def __init__(self, config: ExperimentConfig) -> None:
        """Create a new experiment using ``config``."""

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.seed_mgr = SeedManager(config.seed)
        self.config = config
        self.config.home_directory.mkdir(parents=True, exist_ok=True)
        self.save()

    @abstractmethod
    def get_trainer_configs(self) -> List[List[TrainerConfig]]:
        """Return the list of lists of trainer configs for this experiment."""
        pass
    
    def save(self) -> None:
        cfg_file = self.config.home_directory / "experiment_config.json"
        cfg_file.write_text(self.config.to_json(indent=2))

    @classmethod
    def from_dir(cls, home_directory: Path) -> "Experiment":
        """Load an experiment stored in ``home_directory``."""

        cfg_file = Path(home_directory) / "experiment_config.json"
        if not cfg_file.exists():
            raise FileNotFoundError(f"No experiment_config.json at {cfg_file}")

        cfg = build_experiment_config_from_dict(
            json.loads(cfg_file.read_text())
        )
        cfg.home_directory = Path(home_directory)
        return create_experiment(cfg)

    @classmethod
    def server_run(cls, experiment_dir: Path) -> str:
        """Create an sbatch script in ``experiment_dir`` and submit it.

        Parameters
        ----------
        experiment_dir:
            Directory containing the experiment to run.

        Returns
        -------
        str
            The job id returned by ``sbatch``.
        """
        project_root = Path(__file__).resolve().parents[3]
        start_script = project_root / "scripts" / "start_experiment.sh"
        if not start_script.exists():
            raise FileNotFoundError(f"Reference script not found at {start_script}")

        lines = start_script.read_text().splitlines()
        job_name = f"{experiment_dir.parent.name}_{experiment_dir.name}"
        out_path = os.path.relpath(experiment_dir / "run.out", project_root)
        err_path = os.path.relpath(experiment_dir / "run.err", project_root)
        exp_rel = os.path.relpath(experiment_dir, project_root)

        new_lines = []
        for line in lines:
            if line.startswith("#SBATCH --job-name="):
                new_lines.append(f"#SBATCH --job-name={job_name}")
            elif line.startswith("#SBATCH --output="):
                new_lines.append(f"#SBATCH --output={out_path}")
            elif line.startswith("#SBATCH --error="):
                new_lines.append(f"#SBATCH --error={err_path}")
            elif line.strip().startswith("python"):
                new_lines.append(f"python -m scripts.run {exp_rel}")
            else:
                new_lines.append(line)

        script_path = experiment_dir / "run.sh"
        script_path.write_text("\n".join(new_lines) + "\n")
        script_path.chmod(0o755)
        script_rel = os.path.relpath(script_path, project_root)

        result = subprocess.run(
            ["sbatch", script_rel],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        match = re.search(r"(\d+)", result.stdout)
        if not match:
            raise RuntimeError(
                f"Could not parse job ID from sbatch output: {result.stdout!r}"
            )
        return match.group(1)

    def train(self) -> None:
        config_lists = self.get_trainer_configs()

        if not config_lists:
            self.consolidate_results()
            return

        if self.config.run_parallel:
            self._train_parallel(config_lists)
        else:
            self._train_serial(config_lists)

    @abstractmethod
    def consolidate_results(self) -> Any:
        """
        Aggregate and process results across all trainers
        Must be implemented by subclasses.
        """
        pass
    
    def _verify_sub_exp(self, sub_exp_dir: Path) -> None:
        """Load the sub-experiment configuration to ensure it's valid."""
        build_experiment_config_from_dict(
            json.loads((sub_exp_dir / "experiment_config.json").read_text())
        )

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _train_serial(self, config_lists: List[List[TrainerConfig]]) -> None:
        """Run all trainers sequentially on the local machine."""

        for idx, configs in enumerate(config_lists):
            for cfg in configs:
                if cfg.home_dir is None:
                    raise RuntimeError("TrainerConfig.home_dir must be set")
                if cfg.home_dir.exists():
                    try:
                        trainer = trainer_from_dir(cfg.home_dir)
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to load trainer from {cfg.home_dir}: {e}"
                        ) from e
                else:
                    cfg.home_dir.mkdir(parents=True, exist_ok=True)
                    trainer = create_trainer(cfg)


                trainer.train()
                trainer.save_results()

    # ------------------------------ PARALLEL -----------------------------

    def _get_job_status(self, job_id: str) -> Tuple[str, str]:
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

    def _update_parallel_jobs(self, active_configs: List[Tuple[TrainerConfig, str]]) -> None:
        """Check running jobs and resubmit failed ones."""

        remaining: List[Tuple[TrainerConfig, str]] = []

        for cfg, job_id in active_configs:
            results_file = cfg.home_dir / "results.json"
            if results_file.exists():
                continue

            state, _ = self._get_job_status(job_id)
            if state and not state.startswith(("PENDING", "RUNNING", "CONFIGURING")):
                # job finished but no results -> resubmit
                for item in cfg.home_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()

                time.sleep(15)
                create_trainer(cfg)
                new_id = Trainer.server_train(cfg.home_dir)
                remaining.append((cfg, new_id))
            else:
                remaining.append((cfg, job_id))

        active_configs[:] = remaining

    def _train_parallel(self, config_lists: List[List[TrainerConfig]]) -> None:
        """Run trainers using the cluster scheduler."""

        active: List[Tuple[TrainerConfig, str]] = []

        for idx, configs in enumerate(config_lists):
            if idx > 0:
                prev_configs = config_lists[idx - 1]
                while not all((cfg.home_dir / "results.json").exists() for cfg in prev_configs):
                    self._update_parallel_jobs(active)
                    time.sleep(1)

            for cfg in configs:
                if cfg.home_dir is None:
                    raise RuntimeError("TrainerConfig.home_dir must be set")

                self._update_parallel_jobs(active)

                while len(active) >= 10:
                    time.sleep(10)
                    self._update_parallel_jobs(active)

                if not cfg.home_dir.exists():
                    cfg.home_dir.mkdir(parents=True, exist_ok=True)
                    create_trainer(cfg)
                else:
                    try:
                        trainer_from_dir(cfg.home_dir)
                    except Exception:
                        create_trainer(cfg)

                results_file = cfg.home_dir / "results.json"
                if not results_file.exists():
                    job_id = Trainer.server_train(cfg.home_dir)
                    time.sleep(15)
                    active.append((cfg, job_id))

        all_configs = [cfg for configs in config_lists for cfg in configs]
        while not all((cfg.home_dir / "results.json").exists() for cfg in all_configs):
            self._update_parallel_jobs(active)
            time.sleep(15)
