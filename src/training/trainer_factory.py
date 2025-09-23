"""Factory helpers for instantiating :class:`Trainer` objects."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import json

from src.training.trainer_config import TrainerConfig

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from src.training.trainer import Trainer


def create_trainer(config: TrainerConfig) -> "Trainer":
    """Build a :class:`Trainer` from ``config``."""

    from src.training.trainer import Trainer

    return Trainer(config)


def trainer_from_dir(home_dir: Path) -> "Trainer":
    """Reconstruct a :class:`Trainer` from ``home_dir``.

    The directory is expected to contain a ``trainer_config.json`` file created
    by :class:`Trainer.save_cfg` during initialisation.
    """

    cfg_json = home_dir / "trainer_config.json"
    if not cfg_json.exists():
        raise FileNotFoundError(f"No trainer_config.json at {cfg_json}")
    cfg_dict = json.loads(cfg_json.read_text())
    cfg = TrainerConfig.from_dict(cfg_dict)
    cfg.home_dir = home_dir
    return create_trainer(cfg)

