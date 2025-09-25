import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import itertools
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.training.trainer_config import TrainerConfig
from src.experiments.experiments import register_experiment
from src.experiments.experiments.experiment import Experiment
from src.experiments.configs.staircase_experiment import StaircaseExperimentConfig
from src.data.joint_distributions.configs.staircase import StaircaseConfig
from src.data.joint_distributions import create_joint_distribution
from src.data.joint_distributions.module_joint_distribution import ModuleJointDistribution
from src.models.representors.mlp_representor import MLPRepresentor
from src.models.targets.sum_prod import SumProdTarget
from src.models.targets.configs.sum_prod import SumProdTargetConfig


@register_experiment("ClimbStairs")
class StaircaseExperiment(Experiment):
    """Experiment training an MLP on a staircase distribution."""

    def __init__(self, config: StaircaseExperimentConfig) -> None:
        super().__init__(config)

        trainer_cfg = self.config.trainer_config.deep_copy()
        trainer_cfg.seed = self.seed_mgr.spawn_seed()
        trainer_cfg.home_dir = self.config.home_directory / "trainer"

        dist_cfg = trainer_cfg.joint_distribution_config
        if not isinstance(dist_cfg, StaircaseConfig):
            raise TypeError("trainer_config must use a Staircase distribution")

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

        # make the tables
        self.generate_tables()

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

        return [row]

    def _layer_module_distribution(
        self, layer_number: int, y_module: nn.Module
    ) -> ModuleJointDistribution:
        """Return a ``ModuleJointDistribution`` for a network layer and target module."""

        trainer_cfg = self.get_trainer_configs()[0][0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Base distribution matches the original training distribution
        base_dist = create_joint_distribution(
            trainer_cfg.joint_distribution_config, device
        )

        # Representor for the trained MLP
        representor = MLPRepresentor(
            trainer_cfg.model_config, trainer_cfg.home_dir / "checkpoints", device
        )

        n_hidden = len(trainer_cfg.model_config.hidden_dims)
        final_layer = n_hidden + 1
        if layer_number < 0 or layer_number > final_layer:
            raise ValueError("Invalid layer_number")

        if layer_number == 0:
            x_module = nn.Identity()
        else:
            rep_dict = {
                "layer_index": layer_number,
                # hidden layers use post-activation representations
                "post_activation": layer_number != final_layer,
            }
            rep_id = representor.from_representation_dict(rep_dict)
            x_module = representor.get_module(0, rep_id)

        return ModuleJointDistribution(base_dist, x_module, y_module)

    def layer_product_distribution(
        self, layer_number: int, indices: list[int]
    ) -> ModuleJointDistribution:
        """Return a ``ModuleJointDistribution`` for a network layer and input product.

        Parameters
        ----------
        layer_number:
            The layer of the trained network whose representation should be
            used.
        indices:
            A list of zero-based input indices whose product forms the
            ``SumProdTarget`` term.
        """

        trainer_cfg = self.get_trainer_configs()[0][0]
        indices_list = [list(indices)]
        weights = [1.0]
        prod_cfg = SumProdTargetConfig(
            input_shape=trainer_cfg.joint_distribution_config.input_shape,
            indices_list=indices_list,
            weights=weights,
            normalize=False,
        )
        y_module = SumProdTarget(prod_cfg)

        return self._layer_module_distribution(layer_number, y_module)

    def layer_staircase_distribution(
        self, layer_number: int, k: int
    ) -> ModuleJointDistribution:
        """Return a ``ModuleJointDistribution`` for a network layer and staircase target."""

        trainer_cfg = self.get_trainer_configs()[0][0]
        indices_list = [list(range(i + 1)) for i in range(k)]
        weight = 1.0 / math.sqrt(k)
        weights = [weight] * k
        stair_cfg = SumProdTargetConfig(
            input_shape=trainer_cfg.joint_distribution_config.input_shape,
            indices_list=indices_list,
            weights=weights,
            normalize=False,
        )
        y_module = SumProdTarget(stair_cfg)

        return self._layer_module_distribution(layer_number, y_module)

    def generate_tables(self) -> tuple[list[list[float]], list[list[float]]]:
        """Generate tables of test errors using unregularized linear models.

        For each layer of the trained network and for each module number ``k``
        of the underlying staircase distribution, this method fits a linear
        model with ridge parameter ``lambda=0`` on a fresh training set sampled
        from the corresponding :class:`ModuleJointDistribution`. The fitted
        model is then evaluated on a separate fresh test set.  The resulting
        mean squared errors are collected into two tables: one for all non-empty
        subsets of the first ``k`` coordinates using ``SumProdTarget`` instances
        configured with single product terms and one for staircase targets built
        from ``SumProdTarget`` weights representing orders ``1..k``.  If the
        network has ``L`` layers, the product table has size ``(2^k - 1)×L`` and
        the staircase table has size ``k×L``.

        Returns
        -------
        tuple[list[list[float]], list[list[float]]]
            ``(product_table, staircase_table)`` containing test errors.
        """

        trainer_cfg = self.get_trainer_configs()[0][0]
        dist_cfg = trainer_cfg.joint_distribution_config
        if not isinstance(dist_cfg, StaircaseConfig):
            raise TypeError("trainer_config must use a Staircase distribution")

        train_size = trainer_cfg.train_size
        test_size = trainer_cfg.test_size
        if train_size is None or test_size is None:
            raise ValueError("trainer_config must specify train_size and test_size")

        k = dist_cfg.k
        n_hidden = len(trainer_cfg.model_config.hidden_dims)
        final_layer = n_hidden + 1

        product_table: list[list[float]] = []
        staircase_table: list[list[float]] = []

        index_range = list(range(k))
        subset_list = [
            list(comb)
            for r in range(1, k + 1)
            for comb in itertools.combinations(index_range, r)
        ]

        # Compute product table for every non-empty subset of the first k indices
        for subset in subset_list:
            prod_row: list[float] = []
            for layer in range(final_layer + 1):
                prod_dist = self.layer_product_distribution(layer, subset)

                train_seed = self.seed_mgr.spawn_seed()
                prod_module, _ = prod_dist.linear_solve(
                    seed=train_seed,
                    n_samples=train_size,
                    lambda_=0.0,
                )
                test_seed = self.seed_mgr.spawn_seed()
                test_X, test_y = prod_dist.sample(test_size, test_seed)
                prod_err = torch.mean((prod_module(test_X) - test_y) ** 2).item()
                prod_row.append(prod_err)

            product_table.append(prod_row)

        # Compute staircase table for k = 1..k as before
        for m in range(1, k + 1):
            stair_row: list[float] = []
            for layer in range(final_layer + 1):
                stair_dist = self.layer_staircase_distribution(layer, m)

                train_seed = self.seed_mgr.spawn_seed()
                stair_module, _ = stair_dist.linear_solve(
                    seed=train_seed,
                    n_samples=train_size,
                    lambda_=0.0,
                )
                test_seed = self.seed_mgr.spawn_seed()
                test_X, test_y = stair_dist.sample(test_size, test_seed)
                stair_err = torch.mean((stair_module(test_X) - test_y) ** 2).item()
                stair_row.append(stair_err)

            staircase_table.append(stair_row)

        col_labels = [str(i) for i in range(final_layer + 1)]
        product_row_labels = [
            ",".join(str(i + 1) for i in subset) for subset in subset_list
        ]
        stair_row_labels = [str(i) for i in range(1, k + 1)]

        def _save_table(
            table: list[list[float]],
            row_labels: list[str],
            title: str,
            filename: str,
            row_axis_label: str,
        ) -> None:
            fig, ax = plt.subplots()
            ax.axis("off")
            ax.set_title(title)
            display_table = [[f"{val:.6f}" for val in row] for row in table]
            ax.table(
                cellText=display_table,
                rowLabels=row_labels,
                colLabels=col_labels,
                loc="center",
            )
            fig.text(0.5, 0.02, "Layer number", ha="center")
            fig.text(0.02, 0.5, row_axis_label, va="center", rotation="vertical")
            fig.tight_layout()
            fig.savefig(self.config.home_directory / filename, bbox_inches="tight")
            plt.close(fig)

        _save_table(
            product_table,
            product_row_labels,
            "Product test MSE",
            "product_table.png",
            "Subset",
        )
        _save_table(
            staircase_table,
            stair_row_labels,
            "Staircase test MSE",
            "staircase_table.png",
            "k",
        )

        return product_table, staircase_table
