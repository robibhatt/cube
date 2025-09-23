from __future__ import annotations

import copy
from typing import Any, Dict, List

from src.experiments.experiments import register_experiment
from src.experiments.experiments.batch_experiment import BatchExperiment
from src.experiments.configs import ExperimentConfig
from src.experiments.configs.mass_mlp_recovery import MassMLPRecoveryConfig
from src.experiments.configs.mlp_recovery import MLPRecoveryConfig


@register_experiment("MassMLPRecovery")
class MassMLPRecovery(BatchExperiment):
    """Batch experiment running many MLP recovery experiments."""

    def __init__(self, config: MassMLPRecoveryConfig) -> None:
        super().__init__(config)

    def run(self) -> None:  # type: ignore[override]
        if self.config.rerun_plots:
            for results_file in self.config.home_directory.rglob("results.csv"):
                results_file.unlink()
            for json_file in self.config.home_directory.rglob("results.json"):
                json_file.unlink()
        super().run()

    def get_experiment_configs(self) -> List[ExperimentConfig]:
        cfgs: List[ExperimentConfig] = []
        base_cfg = self.config.mlp_recovery_config

        for train_size in self.config.train_sizes:
            train_dir = self.config.home_directory / f"train_size_{train_size}"
            for lr in self.config.learning_rates:
                lr_dir = train_dir / f"lr_{lr}"
                for wd in self.config.weight_decays:
                    wd_dir = lr_dir / f"weight_decay_{wd}"
                    wd_dir.mkdir(parents=True, exist_ok=True)

                    cfg: MLPRecoveryConfig = copy.deepcopy(base_cfg)
                    cfg.home_directory = wd_dir
                    cfg.seed = self.seed_mgr.spawn_seed()
                    student_cfg = cfg.student_trainer_config
                    student_cfg.train_size = train_size
                    if student_cfg.optimizer_config is None:
                        raise ValueError("student_trainer_config must have optimizer_config")
                    student_cfg.optimizer_config.lr = lr
                    student_cfg.optimizer_config.weight_decay = wd
                    cfgs.append(cfg)

        return cfgs

    def get_config_params(self, config: ExperimentConfig) -> Dict[str, Any]:
        cfg = config  # type: ignore[assignment]
        student_cfg = cfg.student_trainer_config
        opt_cfg = student_cfg.optimizer_config
        return {
            "train_size": student_cfg.train_size,
            "learning_rate": opt_cfg.lr if opt_cfg else None,
            "weight_decay": opt_cfg.weight_decay if opt_cfg else None,
        }
