import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from src.training.trainer_config import TrainerConfig
from src.training.loss.configs.loss import LossConfig
from src.training.optimizers.configs.sgd import SgdConfig
from src.models.architectures.configs.mlp import MLPConfig
from src.data.joint_distributions.configs.hypercube import HypercubeConfig
from src.data.joint_distributions.configs.mapped_joint_distribution import (
    MappedJointDistributionConfig,
)
from src.models.targets.configs.sum_prod import SumProdTargetConfig
from src.experiments.experiments import register_experiment
from src.experiments.experiments.experiment import Experiment
from src.experiments.configs.cube_experiment import CubeExperimentConfig


@register_experiment("Cube")
class CubeExperiment(Experiment):
    """Experiment training an MLP on a hypercube with a sum-product target."""

    def __init__(self, config: CubeExperimentConfig) -> None:
        super().__init__(config)

        hidden_dims = [config.width] * config.depth
        model_cfg = MLPConfig(
            input_dim=config.dimension,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation="relu",
            start_activation=False,
            end_activation=False,
            mup=config.mup,
        )

        indices_list = [list(range(config.k))]
        weights = [1.0 for _ in indices_list]

        dist_cfg = MappedJointDistributionConfig(
            distribution_config=HypercubeConfig(
                input_shape=torch.Size([config.dimension])
            ),
            target_function_config=SumProdTargetConfig(
                input_shape=torch.Size([config.dimension]),
                indices_list=indices_list,
                weights=weights,
            ),
        )

        loss_cfg = LossConfig(name="MSELoss", eval_type="regression")
        opt_cfg = SgdConfig(
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            mup=config.mup,
        )

        trainer_cfg = TrainerConfig(
            model_config=model_cfg,
            joint_distribution_config=dist_cfg,
            loss_config=loss_cfg,
            optimizer_config=opt_cfg,
            early_stopping=config.early_stopping,
            train_size=config.train_size,
            test_size=4096,
            batch_size=config.batch_size,
            epochs=config.epochs,
            use_full_batch=False,
            weight_decay_l1=config.weight_decay_l1,
        )
        trainer_cfg.seed = self.seed_mgr.spawn_seed()
        trainer_cfg.home_dir = self.config.home_directory / "trainer"

        self._trainer_configs: List[List[TrainerConfig]] = [[trainer_cfg]]

    def get_trainer_configs(self) -> List[List[TrainerConfig]]:
        return self._trainer_configs

    def consolidate_results(self) -> List[Dict[str, Any]]:
        trainer_cfg = self.get_trainer_configs()[0][0]

        results_path = trainer_cfg.home_dir / "results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {results_path}")
        with open(results_path, "r") as f:
            metrics = json.load(f)

        row = {
            "final_train_loss": metrics["final_train_loss"],
            "final_test_loss": metrics["final_test_loss"],
        }

        out_file = Path(self.config.home_directory) / "results.csv"
        with open(out_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["final_train_loss", "final_test_loss"]
            )
            writer.writeheader()
            writer.writerow(row)

        return [row]
