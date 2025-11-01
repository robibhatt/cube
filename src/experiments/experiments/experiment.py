from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, List, Optional, Tuple
import torch
import os
import shutil
import subprocess
import re
import time

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

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _log_debug(message: str) -> None:
        """Emit a timestamped debug message for base experiment flow."""

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] [Experiment] {message}", flush=True)

    @abstractmethod
    def get_trainer_configs(self) -> List[TrainerConfig]:
        """Return the trainer configs to execute in order."""
        raise NotImplementedError
    
    def save(self) -> None:
        cfg_file = self.config.home_directory / "experiment_config.json"
        cfg_file.write_text(self.config.to_json(indent=2))

    @classmethod
    def from_dir(cls, home_directory: Path) -> "Experiment":
        """Load an experiment stored in ``home_directory``."""

        cfg_file = Path(home_directory) / "experiment_config.json"
        if not cfg_file.exists():
            raise FileNotFoundError(f"No experiment_config.json at {cfg_file}")

        cls._log_debug(
            f"Loading experiment configuration from {cfg_file.resolve()}"
        )
        cfg = build_experiment_config_from_dict(
            json.loads(cfg_file.read_text())
        )
        cfg.home_directory = Path(home_directory)
        cls._log_debug(
            f"Reconstructed experiment config for {cfg.__class__.__name__}"
        )
        return create_experiment(cfg)

    def train(self) -> None:
        configs = self.get_trainer_configs()

        if not configs:
            self._log_debug(
                f"No trainer configs for {self.config.home_directory}; consolidating"
            )
            self.consolidate_results()
            return

        for cfg in configs:
            if cfg.home_dir is None:
                raise RuntimeError("TrainerConfig.home_dir must be set")

            if cfg.home_dir.exists():
                self._log_debug(
                    f"Loading trainer state from {cfg.home_dir.resolve()}"
                )
                try:
                    trainer = Trainer.from_dir(cfg.home_dir)
                except Exception as exc:  # pragma: no cover - defensive branch
                    raise RuntimeError(
                        f"Failed to load trainer from {cfg.home_dir}: {exc}"
                    ) from exc
            else:
                self._log_debug(
                    f"Creating trainer directory at {cfg.home_dir.resolve()}"
                )
                cfg.home_dir.mkdir(parents=True, exist_ok=True)
                trainer = Trainer(cfg)

            self._log_debug(
                f"Starting training run for trainer at {cfg.home_dir.resolve()}"
            )
            trainer.train()
            trainer.save_results()
            self._log_debug(
                f"Completed training run for trainer at {cfg.home_dir.resolve()}"
            )

    def _done_file(self) -> Path:
        return self.config.home_directory / "done.txt"

    def _trainer_directories(self) -> List[Path]:
        home = self.config.home_directory
        if not home.exists():
            return []

        trainer_dirs: List[Path] = []
        for child in home.iterdir():
            if not child.is_dir():
                continue
            if (child / "results.json").exists() or (child / "trainer_config.json").exists():
                trainer_dirs.append(child)
        return trainer_dirs

    def _cleanup_non_trainer_items(self, trainer_dirs: List[Path]) -> None:
        keep_names = {
            "experiment_config.json",
            "steps.log",
            "run.err",
            "run.out",
            "run.sh",
        }

        trainer_set = {path.resolve() for path in trainer_dirs}
        home = self.config.home_directory
        for item in home.iterdir():
            if item.resolve() in trainer_set:
                continue
            if item.name in keep_names:
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    def _reset_experiment_directory(self) -> None:
        home = self.config.home_directory
        if home.exists():
            for item in home.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

        home.mkdir(parents=True, exist_ok=True)
        self.save()

    def prepare_for_execution(self) -> Tuple[bool, Optional[Any]]:
        self._log_debug(
            f"Preparing experiment execution in {self.config.home_directory}"
        )
        if self._done_file().exists():
            self._log_debug(
                "Found done.txt marker; skipping execution and returning cached results"
            )
            return True, None

        trainer_dirs = [d for d in self._trainer_directories() if d.exists()]
        if trainer_dirs and all((d / "results.json").exists() for d in trainer_dirs):
            self._log_debug(
                f"Detected {len(trainer_dirs)} completed trainer directories; consolidating"
            )
            self._cleanup_non_trainer_items(trainer_dirs)
            result = self.consolidate_results()
            return True, result

        if trainer_dirs:
            self._log_debug(
                f"Found {len(trainer_dirs)} incomplete trainer directories; resetting"
            )
            self._reset_experiment_directory()

        self._log_debug("Experiment directory ready for execution")
        return False, None

    def run(self) -> Any:
        """Execute the experiment end-to-end locally."""

        skip, result = self.prepare_for_execution()
        if skip:
            self._log_debug(
                f"Execution skipped for {self.config.home_directory}"
            )
            return result

        self._log_debug(
            f"Starting local execution for {self.config.home_directory}"
        )
        self.train()
        results = self.consolidate_results()
        self._log_debug(
            f"Local execution finished for {self.config.home_directory}"
        )
        return results

    def consolidate_results(self) -> Any:
        """Run subclass consolidation and ensure ``results.csv`` exists."""

        results = self._consolidate_results()

        home = self.config.home_directory
        results_file = home / "results.csv"
        if not results_file.exists():
            raise AssertionError(
                "Experiment consolidation must create results.csv at "
                f"{results_file}"
            )

        done_file = self._done_file()
        done_file.write_text("finished consolodating results in it.")

        return results

    @abstractmethod
    def _consolidate_results(self) -> Any:
        """Aggregate and process results across all trainers."""
        raise NotImplementedError

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

        cls._log_debug(
            "Submitting sbatch job for parallel sub-experiment with command "
            f"'sbatch {script_rel}' from {project_root}"
        )

        result = subprocess.run(
            ["sbatch", script_rel],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if stdout:
            cls._log_debug(f"sbatch stdout: {stdout}")
        if stderr:
            cls._log_debug(f"sbatch stderr: {stderr}")
        match = re.search(r"(\d+)", result.stdout)
        if not match:
            raise RuntimeError(
                f"Could not parse job ID from sbatch output: {result.stdout!r}"
            )
        cls._log_debug(
            f"Successfully submitted sbatch job {match.group(1)} for {experiment_dir}"
        )
        return match.group(1)
    
