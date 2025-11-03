import csv
import json
from datetime import datetime, timezone
UTC = timezone.utc
from pathlib import Path
from typing import Any, Dict, List

from src.experiments.experiments import register_experiment
from src.experiments.experiments.experiment import Experiment
from src.experiments.configs.train_mlp import TrainMLPExperimentConfig
from src.experiments.config_defaults import (
    DEFAULT_ANCESTOR_THRESHOLD,
    DEFAULT_MSE_SAMPLES,
    DEFAULT_MSE_THRESHOLD,
    FINAL_MSE_AVERAGE_SAMPLES,
    ensure_config_value,
)
from src.models import (
    binary_search_sparsify_threshold,
    mse_average_diff,
    mse_diff,
    prune,
    sparsify_mlp,
    visualize_pruned_mlp,
)
from src.models.mlp_utils import visualize as visualize_mlp
from src.training.trainer import Trainer
from src.training.trainer_config import TrainerConfig


@register_experiment("TrainMLP")
class TrainMLPExperiment(Experiment):
    """Experiment training an MLP on an arbitrary joint distribution."""

    def __init__(self, config: TrainMLPExperimentConfig) -> None:
        super().__init__(config)

        self._mse_threshold = ensure_config_value(
            self.config, "mse_threshold", DEFAULT_MSE_THRESHOLD
        )
        # Always measure the sparsification MSE difference with a fixed
        # number of samples so old configuration files cannot override it.
        self.config.mse_samples = DEFAULT_MSE_SAMPLES
        self._mse_samples = DEFAULT_MSE_SAMPLES
        self._ancestor_threshold = ensure_config_value(
            self.config, "ancestor_threshold", DEFAULT_ANCESTOR_THRESHOLD
        )

        self._log_path = Path(self.config.home_directory) / "steps.log"
        self._log_path.write_text("", encoding="utf-8")
        self._log_step("Initialized TrainMLPExperiment")

        trainer_cfg = self.config.trainer_config.deep_copy()
        trainer_cfg.seed = self.seed_mgr.spawn_seed()
        trainer_cfg.home_dir = self.config.home_directory / "trainer"

        # Store trainer configs so repeated calls use the same seed derived
        # from the experiment seed.
        self._trainer_configs: List[TrainerConfig] = [trainer_cfg]

    def get_trainer_configs(self) -> List[TrainerConfig]:
        """Return the trainer configuration seeded for this experiment run."""

        return self._trainer_configs

    def train(self) -> None:
        self._log_step("Starting training run")
        super().train()
        self._log_step("Completed base training flow")
        self._log_step("Finished training")

    def _consolidate_results(self) -> List[Dict[str, Any]]:
        """Collect training metrics and generate auxiliary artefacts."""

        self._log_step("Starting consolidation of training artefacts")
        trainer_cfg = self.get_trainer_configs()[0]

        results_path = trainer_cfg.home_dir / "results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {results_path}")
        with open(results_path, "r") as f:
            metrics = json.load(f)

        self._log_step("Loaded training metrics")

        trainer = Trainer.from_dir(trainer_cfg.home_dir)
        trained_mlp = trainer._load_model()
        trained_mlp.eval()

        threshold = binary_search_sparsify_threshold(
            trained_mlp,
            self._mse_threshold,
            sample_count=self._mse_samples,
        )
        if threshold <= 0:
            threshold = 1e-12
        self._log_step(f"Computed sparsify threshold: {threshold:.6g}")
        sparsified_mlp = sparsify_mlp(trained_mlp, threshold)
        sparsified_mlp.eval()

        actual_mse = mse_diff(
            self._mse_samples,
            trained_mlp,
            sparsified_mlp,
        )
        self._log_step(f"Measured sparsified MSE difference: {actual_mse:.6g}")

        pruned_mlp, active_inputs = prune(sparsified_mlp)
        sparsified_dir = Path(self.config.home_directory) / "sparsified_mlp"
        visualize_pruned_mlp(pruned_mlp, active_inputs, sparsified_dir)

        visualization_root = Path(self.config.home_directory) / "visualizations"
        original_viz_dir = visualization_root / "original"
        sparsified_viz_dir = visualization_root / "sparsified"
        visualize_mlp(trained_mlp, original_viz_dir)
        visualize_mlp(pruned_mlp, sparsified_viz_dir)

        self._log_step(
            "Generated sparsified MLP artefacts and visualizations"
        )

        average_mse = mse_average_diff(
            FINAL_MSE_AVERAGE_SAMPLES,
            trained_mlp,
            pruned_mlp,
        )
        self._log_step(
            "Measured sparsified mean MSE difference over "
            f"{FINAL_MSE_AVERAGE_SAMPLES} samples: {average_mse:.6g}"
        )

        row = {
            "train_size": trainer_cfg.train_size if trainer_cfg.train_size is not None else 0,
            "trial_number": 0,
            "mean_output_loss": metrics["mean_output_loss"],
            "final_test_loss": metrics["final_test_loss"],
            "final_train_loss": metrics["final_train_loss"],
            "sparsify_threshold": threshold,
            "sparsified_mse": actual_mse,
            "sparsified_mean_mse": average_mse,
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
                    "sparsify_threshold",
                    "sparsified_mse",
                    "sparsified_mean_mse",
                ],
            )
            writer.writeheader()
            writer.writerow(row)

        self._log_step("Wrote results.csv")

        self._flag_insufficient_ancestors(sparsified_dir)

        self._log_step("Completed consolidation of training artefacts")

        return [row]

    def _flag_insufficient_ancestors(self, sparsified_dir: Path) -> None:
        if not sparsified_dir.exists():
            return

        ancestor_flag_path = sparsified_dir / "ANCESTOR_FLAG"
        if ancestor_flag_path.exists():
            ancestor_flag_path.unlink()

        flagged_neurons = []
        for layer_dir in sorted(sparsified_dir.glob("layer_*")):
            if not layer_dir.is_dir():
                continue

            for node_file in sorted(layer_dir.glob("*.json")):
                with open(node_file, "r", encoding="utf-8") as f:
                    node_data = json.load(f)

                ancestors = node_data.get("ancestors", [])
                if ancestors is None:
                    ancestors = []

                if len(ancestors) < self._ancestor_threshold:
                    layer_index = node_data.get("layer_index")
                    neuron_index = node_data.get("neuron_index")
                    flagged_neurons.append(
                        (
                            layer_index,
                            neuron_index,
                            len(ancestors),
                        )
                    )

        if not flagged_neurons:
            return

        lines = [
            (
                "layer {layer} neuron {neuron} has {count} ancestors, "
                "below threshold {threshold}"
            ).format(
                layer=layer_index,
                neuron=neuron_index,
                count=count,
                threshold=self._ancestor_threshold,
            )
            for layer_index, neuron_index, count in flagged_neurons
        ]

        ancestor_flag_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _log_step(self, message: str) -> None:
        timestamp = datetime.now(UTC).isoformat(timespec="seconds")
        line = f"[{timestamp} UTC] {message}\n"
        with self._log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(line)
        print(f"[{timestamp} UTC] [TrainMLPExperiment] {message}", flush=True)

