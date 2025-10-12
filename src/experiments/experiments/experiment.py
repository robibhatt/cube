from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, List
import torch

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

        cfg = build_experiment_config_from_dict(
            json.loads(cfg_file.read_text())
        )
        cfg.home_directory = Path(home_directory)
        return create_experiment(cfg)

    def train(self) -> None:
        configs = self.get_trainer_configs()

        if not configs:
            self.consolidate_results()
            return

        for cfg in configs:
            if cfg.home_dir is None:
                raise RuntimeError("TrainerConfig.home_dir must be set")

            if cfg.home_dir.exists():
                try:
                    trainer = Trainer.from_dir(cfg.home_dir)
                except Exception as exc:  # pragma: no cover - defensive branch
                    raise RuntimeError(
                        f"Failed to load trainer from {cfg.home_dir}: {exc}"
                    ) from exc
            else:
                cfg.home_dir.mkdir(parents=True, exist_ok=True)
                trainer = Trainer(cfg)

            trainer.train()
            trainer.save_results()

    @abstractmethod
    def consolidate_results(self) -> Any:
        """
        Aggregate and process results across all trainers
        Must be implemented by subclasses.
        """
        pass
    
