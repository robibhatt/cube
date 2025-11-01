#!/usr/bin/env python3
"""Run a DKWL experiment from a YAML configuration."""
from __future__ import annotations

import sys
from pathlib import Path
import shutil
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


def main() -> None:
    with open(CONFIG_FILE, "r") as f:
        cfg_dict = yaml.safe_load(f)

    server_train = bool(cfg_dict.pop("server_train", False))
    cfg = build_experiment_config_from_dict(cfg_dict)

    home = cfg.home_directory
    is_existing = home.exists()

    if is_existing:
        print(f"Continuing experiment in {home}")
        experiment = Experiment.from_dir(home)
    else:
        print(f"Initializing new experiment in {home}")
        experiment = create_experiment(cfg)

    if not isinstance(experiment, BatchExperiment):
        raise TypeError("DKWL experiment must inherit from BatchExperiment")

    shutil.copy(CONFIG_FILE, home / CONFIG_FILE.name)

    if server_train:
        configs = experiment.get_experiment_configs()
        experiment.run_parallel(configs)
        print('running parallel')
    else:
        experiment.run()
        print('running solo')


if __name__ == "__main__":
    main()
