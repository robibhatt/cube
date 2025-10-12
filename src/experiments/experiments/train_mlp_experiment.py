import csv
import json
import shutil
from itertools import chain, combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from src.experiments.experiments import register_experiment
from src.experiments.experiments.experiment import Experiment
from src.experiments.configs.train_mlp import TrainMLPExperimentConfig
from src.training.trainer_config import TrainerConfig
from src.checkpoints.checkpoint import Checkpoint
from src.fourier.fourier_mlp import FourierMlp
from src.models.mlp import MLP


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

        self._generate_fourier_coefficients(trainer_cfg)

        return [row]

    # ------------------------------------------------------------------
    # Fourier analysis helpers
    # ------------------------------------------------------------------
    def _generate_fourier_coefficients(self, trainer_cfg: TrainerConfig) -> None:
        """Compute and persist Fourier coefficients for the trained MLP.

        The coefficients are computed for every neuron across all layers and for
        all index sets required by the sum-product structure of the trainer's
        target function.  Results are written to a dedicated subdirectory within
        the experiment's home directory.
        """

        mlp = self._load_trained_mlp(trainer_cfg)
        fourier_root = self._prepare_fourier_directory()
        fourier = FourierMlp(mlp, fourier_root)

        fourier_indices = self._collect_relevant_fourier_indices(trainer_cfg)
        total_layers = len(mlp.linear_layers)
        for indices in fourier_indices:
            for layer_idx in range(1, total_layers + 1):
                layer = mlp.linear_layers[layer_idx - 1]
                for neuron_idx in range(layer.out_features):
                    fourier.get_fourier_coefficient(indices, layer_idx, neuron_idx)

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

    def _prepare_fourier_directory(self) -> Path:
        """Create a clean directory under the experiment root for Fourier data."""

        base_dir = Path(self.config.home_directory) / "fourier_analysis"
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    def _collect_relevant_fourier_indices(
        self, trainer_cfg: TrainerConfig
    ) -> List[Tuple[int, ...]]:
        """Return all Fourier index sets derived from the target function.

        The target function is a sum of product terms.  Only subsets of the
        indices appearing in those product terms are required for the Fourier
        expansion.  The returned tuples are canonical (sorted, unique) to ensure
        deterministic filesystem layout.
        """

        distribution_cfg = trainer_cfg.cube_distribution_config
        if distribution_cfg is None:
            raise ValueError("Trainer configuration is missing a distribution config")

        index_groups = [tuple(sorted(set(group))) for group in distribution_cfg.indices_list]
        subsets = {tuple()}
        for group in index_groups:
            subsets.update(self._powerset(group))

        return sorted(subsets, key=lambda item: (len(item), item))

    @staticmethod
    def _powerset(indices: Sequence[int]) -> Iterable[Tuple[int, ...]]:
        """Yield all subsets of *indices* as sorted tuples."""

        return chain.from_iterable(combinations(indices, r) for r in range(len(indices) + 1))
