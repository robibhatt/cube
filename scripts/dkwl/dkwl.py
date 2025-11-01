#!/usr/bin/env python3
"""Run a DKWL experiment from a YAML configuration."""
from __future__ import annotations

import sys
from pathlib import Path
import shutil
import time
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import modules to ensure all registry decorators are executed
import src.models.targets  # noqa: F401
import src.models.targets.configs  # noqa: F401
import src.data  # noqa: F401
import src.experiments.configs  # noqa: F401

# Import module to trigger registration of experiment and config classes
import src.experiments.experiments.dkwl_experiment  # noqa: F401

from src.experiments.configs import build_experiment_config_from_dict
from src.experiments.experiments import Experiment, create_experiment
from src.experiments.experiments.batch_experiment import BatchExperiment
from src.experiments.experiments.train_mlp_experiment import TrainMLPExperiment

CONFIG_FILE = SCRIPT_DIR / "dkwl.yaml"


def _log(message: str) -> None:
    """Log a message with a timestamp for easier debugging."""
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def main() -> None:
    start_time = time.perf_counter()
    _log("Starting DKWL launcher")
    _log(f"Loading configuration from {CONFIG_FILE}")
    with open(CONFIG_FILE, "r") as f:
        cfg_dict = yaml.safe_load(f)

    server_train = bool(cfg_dict.pop("server_train", False))
    _log("Building experiment configuration object")
    cfg = build_experiment_config_from_dict(cfg_dict)

    home = cfg.home_directory
    is_existing = home.exists()

    if is_existing:
        _log(f"Continuing experiment in {home}")
        resume_start = time.perf_counter()
        experiment = Experiment.from_dir(home)
        _log(
            "Loaded existing experiment from disk in "
            f"{time.perf_counter() - resume_start:.2f} seconds"
        )
    else:
        _log(f"Initializing new experiment in {home}")
        create_start = time.perf_counter()
        experiment = create_experiment(cfg)
        _log(
            "Created new experiment in "
            f"{time.perf_counter() - create_start:.2f} seconds"
        )

    if not isinstance(experiment, BatchExperiment):
        raise TypeError("DKWL experiment must inherit from BatchExperiment")

    _log("Copying configuration into experiment directory")
    shutil.copy(CONFIG_FILE, home / CONFIG_FILE.name)

    if server_train:
        _log("Retrieving experiment configs for parallel run")
        configs = experiment.get_experiment_configs()
        _log(f"Launching parallel run with {len(configs)} configs")
        experiment.run_parallel(configs)
        _log("Parallel run completed")
    else:
        _log("Launching single-machine run")
        experiment.run()
        _log("Single-machine run completed")

    _log(
        "DKWL launcher finished in "
        f"{time.perf_counter() - start_time:.2f} seconds"
    )


if __name__ == "__main__":
    main()
