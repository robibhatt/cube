import csv
import json
import math
from pathlib import Path
from typing import Dict, Any, List

import torch

from src.training.trainer_config import TrainerConfig
from src.experiments.experiments import register_experiment
from src.experiments.experiments.experiment import Experiment
from src.experiments.configs.mlp_recovery import MLPRecoveryConfig
from src.training.trainer_factory import trainer_from_dir
from src.models.representors.representor_factory import create_model_representor
from src.data.joint_distributions.configs.representor_distribution import (
    RepresentorDistributionConfig,
)
from src.data.joint_distributions.configs.noisy_distribution import (
    NoisyDistributionConfig,
)
from src.data.joint_distributions.configs.gaussian import GaussianConfig

@register_experiment("MLPRecovery")
class MLPRecovery(Experiment):
    """Experiment comparing student MLPs to a fixed teacher.

    The teacher and student networks may have different depths; all plots and
    comparisons will still be generated.
    """

    def __init__(self, config: MLPRecoveryConfig) -> None:
        """Create a new experiment using ``config``."""
        super().__init__(config)

    def get_trainer_configs(self) -> list[list[TrainerConfig]]:
        teacher_trainer_cfg = self.config.teacher_trainer_config.deep_copy()
        teacher_trainer_cfg.seed = self.seed_mgr.spawn_seed()
        teacher_trainer_cfg.home_dir = self.config.home_directory / "teacher"

        trainer_cfg = self.config.student_trainer_config.deep_copy()
        trainer_cfg.seed = self.seed_mgr.spawn_seed()
        trainer_cfg.home_dir = self.config.home_directory / "student"

        # Build representor distribution based on the teacher model.
        teacher_repr = create_model_representor(
            teacher_trainer_cfg.model_config,
            teacher_trainer_cfg.home_dir / "checkpoints",
            device=torch.device("cpu"),
        )

        if self.config.start_layer == 0:
            start_rep = 0
        else:
            rep_dict = {
                "layer_index": self.config.start_layer,
                "post_activation": True,
            }
            start_rep = teacher_repr.from_representation_dict(rep_dict)

        representor_cfg = RepresentorDistributionConfig(
            base_distribution_config=teacher_trainer_cfg.joint_distribution_config,
            model_config=teacher_trainer_cfg.model_config,
            checkpoint_dir=teacher_trainer_cfg.home_dir / "checkpoints",
            from_rep=start_rep,
            to_rep=teacher_repr.get_final_rep_id(),
        )

        if self.config.noise_variance > 0.0:
            noise_cfg = GaussianConfig(
                input_shape=representor_cfg.output_shape,
                mean=0.0,
                std=math.sqrt(self.config.noise_variance),
            )
            trainer_cfg.joint_distribution_config = NoisyDistributionConfig(
                base_distribution_config=representor_cfg,
                noise_distribution_config=noise_cfg,
            )
        else:
            trainer_cfg.joint_distribution_config = representor_cfg

        return [[teacher_trainer_cfg], [trainer_cfg]]

    def consolidate_results(self) -> List[Dict[str, Any]]:
        """Gather results from all trainers and save them to ``results.csv``."""
        # Gather trainer configurations
        teacher_configs, trainer_configs = self.get_trainer_configs()
        teacher_cfg = teacher_configs[0]

        # Verify that the teacher trainer exists and can provide a model. If this
        # fails, the experiment setup is broken and we should halt immediately.
        trainer_from_dir(teacher_cfg.home_dir)._load_model()

        # Only collect results for student trainers.
        rows: List[Dict[str, Any]] = []

        for cfg in trainer_configs:
            results_path = cfg.home_dir / "results.json"
            if not results_path.exists():
                raise FileNotFoundError(f"Missing metrics file: {results_path}")
            with open(results_path, "r") as f:
                metrics = json.load(f)

            row = {
                "train_size": cfg.train_size if cfg.train_size is not None else 0,
                "trial_number": 0,
                "mean_output_loss": metrics["mean_output_loss"],
                "final_test_loss": metrics["final_test_loss"],
                "final_train_loss": metrics["final_train_loss"],
            }
            rows.append(row)

        out_file = Path(self.config.home_directory) / "results.csv"
        with open(out_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "train_size",
                    "trial_number",
                    "mean_output_loss",
                    "final_test_loss",
                    "final_train_loss",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        return rows
