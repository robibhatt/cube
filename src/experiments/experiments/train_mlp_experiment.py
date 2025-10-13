import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from src.experiments.experiments import register_experiment
from src.experiments.experiments.experiment import Experiment
from src.experiments.configs.train_mlp import TrainMLPExperimentConfig
from src.training.trainer_config import TrainerConfig
from src.checkpoints.checkpoint import Checkpoint
from src.mlp_graph.mlp_graph import MlpActivationGraph
from src.models.mlp import MLP
from src.models.targets.sum_prod import SumProdTarget


@register_experiment("TrainMLP")
class TrainMLPExperiment(Experiment):
    """Experiment training an MLP on an arbitrary joint distribution."""

    def __init__(self, config: TrainMLPExperimentConfig) -> None:
        super().__init__(config)

        trainer_cfg = self.config.trainer_config.deep_copy()
        trainer_cfg.seed = self.seed_mgr.spawn_seed()
        trainer_cfg.home_dir = self.config.home_directory / "trainer"

        # Store trainer configs so repeated calls use the same seed derived
        # from the experiment seed.
        self._trainer_configs: List[TrainerConfig] = [trainer_cfg]

    def get_trainer_configs(self) -> List[TrainerConfig]:
        """Return the trainer configuration seeded for this experiment run."""

        return self._trainer_configs

    def consolidate_results(self) -> List[Dict[str, Any]]:
        """Collect training metrics and trigger Fourier post-processing."""

        trainer_cfg = self.get_trainer_configs()[0]

        results_path = trainer_cfg.home_dir / "results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {results_path}")
        with open(results_path, "r") as f:
            metrics = json.load(f)

        row = {
            "train_size": trainer_cfg.train_size if trainer_cfg.train_size is not None else 0,
            "trial_number": 0,
            "mean_output_loss": metrics["mean_output_loss"],
            "final_test_loss": metrics["final_test_loss"],
            "final_train_loss": metrics["final_train_loss"],
        }

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
            writer.writerow(row)

        self._generate_mlp_graph(trainer_cfg)

        return [row]

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------
    def _generate_mlp_graph(self, trainer_cfg: TrainerConfig) -> None:
        """Create an activation graph for the trained MLP."""

        mlp = self._load_trained_mlp(trainer_cfg)
        mlp = self._apply_graph_scale(mlp, trainer_cfg)
        graph_root = Path(self.config.home_directory) / "mlp_graph"
        graph_root.mkdir(parents=True, exist_ok=True)
        MlpActivationGraph(
            mlp,
            eps=self.config.edge_threshold,
            output_dir=graph_root,
        )

    def _load_trained_mlp(self, trainer_cfg: TrainerConfig) -> MLP:
        """Return the trained MLP restored from the trainer checkpoint."""

        if trainer_cfg.mlp_config is None:
            raise ValueError("Trainer configuration is missing an MLP config")
        checkpoint_dir = trainer_cfg.home_dir / "checkpoints"
        checkpoint = Checkpoint.from_dir(checkpoint_dir)
        mlp = MLP(trainer_cfg.mlp_config)
        checkpoint.load(model=mlp)
        mlp.eval()
        return mlp

    def _apply_graph_scale(self, mlp: MLP, trainer_cfg: TrainerConfig) -> MLP:
        """Scale model parameters so the graph reflects the unnormalised target."""

        lambda_scale = self._compute_graph_scale(mlp, trainer_cfg)
        if lambda_scale == 1.0:
            return mlp

        with torch.no_grad():
            for param in mlp.parameters():
                param.mul_(lambda_scale)
        return mlp

    def _compute_graph_scale(self, mlp: MLP, trainer_cfg: TrainerConfig) -> float:
        """Return the multiplicative factor applied to each MLP parameter."""

        num_layers = len(mlp.linear_layers)
        if num_layers == 0:
            return 1.0

        normalization = self._get_normalization_factor(trainer_cfg)
        if normalization <= 0.0:
            return 1.0

        target_scale = 1.0 / normalization
        lambda_scale = target_scale ** (1.0 / num_layers)
        return lambda_scale

    def _get_normalization_factor(self, trainer_cfg: TrainerConfig) -> float:
        """Return the normalisation constant ``G`` used by the target function."""

        dist_cfg = trainer_cfg.cube_distribution_config
        if dist_cfg is None:
            return 1.0

        target_cfg = dist_cfg.target_function_config
        target = SumProdTarget(target_cfg)
        return float(target.scale)
