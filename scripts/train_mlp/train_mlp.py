#!/usr/bin/env python3
"""Run a TrainMLP experiment from a YAML configuration."""
from __future__ import annotations

import sys
from pathlib import Path
import shutil
import yaml
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import modules to ensure all registry decorators are executed
import src.models.bootstrap  # noqa: F401
import src.models.targets  # noqa: F401
import src.models.targets.configs  # noqa: F401
import src.data  # noqa: F401
import src.experiments.configs  # noqa: F401

# Import module to trigger registration of experiment and config classes
import src.experiments.experiments.train_mlp_experiment  # noqa: F401

from src.experiments.configs import build_experiment_config_from_dict
from src.experiments.experiments import Experiment, create_experiment

CONFIG_FILE = SCRIPT_DIR / "train_mlp.yaml"


def main() -> None:
    with open(CONFIG_FILE, "r") as f:
        cfg_dict = yaml.safe_load(f)

    cfg = build_experiment_config_from_dict(cfg_dict)
    home = cfg.home_directory
    is_existing = home.exists()

    if is_existing:
        print(f"Continuing experiment in {home}")
        experiment = Experiment.from_dir(home)
    else:
        print(f"Initializing new experiment in {home}")
        experiment = create_experiment(cfg)

    shutil.copy(CONFIG_FILE, home / CONFIG_FILE.name)

    experiment.train()
    experiment.consolidate_results()


if __name__ == "__main__":
    main()
