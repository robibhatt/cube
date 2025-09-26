#!/usr/bin/env python3
"""Run an experiment from an existing experiment directory."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path so `src` can be imported
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import modules to ensure all registry decorators are executed
import src.models.bootstrap  # noqa: F401
import src.models.targets  # noqa: F401
import src.models.targets.configs  # noqa: F401
import src.data.providers  # noqa: F401
import src.experiments.configs  # noqa: F401

from src.utils.plugin_loader import import_submodules

# Import all experiment implementations so they register themselves
import_submodules("src.experiments.experiments")

from src.experiments.experiments.experiment import Experiment


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {Path(__file__).name} EXPERIMENT_DIR")
        sys.exit(1)

    exp_dir = Path(sys.argv[1])
    experiment = Experiment.from_dir(exp_dir)
    experiment.train()
    experiment.consolidate_results()


if __name__ == "__main__":
    main()
