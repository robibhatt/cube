#!/usr/bin/env python3
"""Train a model from an existing trainer directory."""
from __future__ import annotations

import json
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

from src.training.trainer import Trainer
from src.training.trainer_config import TrainerConfig


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {Path(__file__).name} TRAINER_DIR")
        sys.exit(1)

    trainer_dir = Path(sys.argv[1])
    cfg_path = trainer_dir / "trainer_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No trainer_config.json at {cfg_path}")
    cfg_dict = json.loads(cfg_path.read_text())
    cfg = TrainerConfig.from_dict(cfg_dict)
    cfg.home_dir = trainer_dir
    trainer = Trainer(cfg)
    trainer.train()
    trainer.save_results()


if __name__ == "__main__":
    main()
