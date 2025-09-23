from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.training.trainer_config import TrainerConfig
from src.models.representors.representor_factory import create_model_representor
from src.data.joint_distributions import create_joint_distribution
from src.data.joint_distributions.configs.paired_representor_distribution import (
    PairedRepresentorDistributionConfig,
)
from src.data.joint_distributions.paired_representor_distribution import (
    PairedRepresentorDistribution,
)
from src.data.providers import create_data_provider_from_distribution
from src.data.providers.data_provider import DataProvider
from src.utils.seed_manager import SeedManager


class MLPComparer:
    """Utility for comparing teacher and student MLPs.

    The comparer works even when the teacher and student networks have
    different numbers of hidden layers.  It can evaluate how well the
    representations of one network linearly map to those of another and report
    the resulting test losses in a table.
    """

    def __init__(
        self,
        teacher_dir: Path,
        student_dir: Path,
        seed: int,
        layer_matching: bool = True,
        start_layer: int = 0,
    ) -> None:
        self.teacher_dir = Path(teacher_dir)
        self.student_dir = Path(student_dir)

        self.teacher_cfg = self._load_trainer_config(self.teacher_dir)
        self.student_cfg = self._load_trainer_config(self.student_dir)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.seed_mgr = SeedManager(seed)
        self.layer_matching = layer_matching
        self.start_layer = start_layer

    def _load_trainer_config(self, directory: Path) -> TrainerConfig:
        cfg_path = Path(directory) / "trainer_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"trainer_config.json not found in {directory}")
        return TrainerConfig.from_json(cfg_path.read_text())

    def _layer_to_rep_id(self, representor, layer: int) -> int:
        n_hidden = len(representor.model_config.hidden_dims)
        if layer == 0:
            return 0
        if layer == n_hidden + 1:
            return representor.get_final_rep_id()
        rep_dict = {"layer_index": layer, "post_activation": True}
        return representor.from_representation_dict(rep_dict)

    def get_layers_distribution(
        self, student_layer: int, teacher_layer: int
    ) -> PairedRepresentorDistribution:
        """Return a ``PairedRepresentorDistribution`` for the given layers."""

        student_repr = create_model_representor(
            self.student_cfg.model_config,
            self.student_dir / "checkpoints",
            device=self.device,
        )
        teacher_repr = create_model_representor(
            self.teacher_cfg.model_config,
            self.teacher_dir / "checkpoints",
            device=self.device,
        )

        student_rep_id = self._layer_to_rep_id(student_repr, student_layer)
        teacher_rep_id = self._layer_to_rep_id(teacher_repr, teacher_layer)
        student_from_rep = self._layer_to_rep_id(student_repr, 0)
        teacher_from_rep = self._layer_to_rep_id(teacher_repr, self.start_layer)

        cfg = PairedRepresentorDistributionConfig(
            base_distribution_config=self.student_cfg.joint_distribution_config,
            teacher_model_config=self.teacher_cfg.model_config,
            teacher_checkpoint_dir=self.teacher_dir / "checkpoints",
            teacher_rep_id=teacher_rep_id,
            teacher_from_rep_id=teacher_from_rep,
            student_model_config=self.student_cfg.model_config,
            student_checkpoint_dir=self.student_dir / "checkpoints",
            student_rep_id=student_rep_id,
            student_from_rep_id=student_from_rep,
        )

        return PairedRepresentorDistribution(cfg, self.device)

    def get_test_provider(self, seed: int) -> DataProvider:
        """Return a data provider mirroring the student's test loader.

        Parameters
        ----------
        seed:
            Seed used to generate both the underlying dataset and the
            ``DataLoader`` shuffling. This mirrors the seeding strategy used in
            training so that repeated calls with the same seed yield identical
            batches.

        Returns
        -------
        DataProvider
            Provider configured like the student's test loader.
        """

        # Instantiate the joint distribution on the comparer's device
        joint_dist = create_joint_distribution(
            self.student_cfg.joint_distribution_config, self.device
        )

        dataset_dir = self.student_dir / "datasets" / "test"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        batch_size = self.student_cfg.batch_size or 1024
        if self.student_cfg.test_size is None:
            raise ValueError("student_cfg.test_size must be specified")

        provider = create_data_provider_from_distribution(
            joint_dist,
            dataset_dir,
            batch_size,
            self.student_cfg.test_size,
            seed=seed,
        )

        return provider

    def compute_test_loss(
        self,
        student_layer: int,
        teacher_layer: int,
        bias: bool = True,
        lambdas: list[float] | None = None,
        k: int = 5,
    ) -> float:
        """Return test loss when mapping a student layer to a teacher layer.

        A linear map is fitted from the student layer to the teacher layer using
        ``self.student_cfg.train_size`` samples.  The ridge parameter ``lambda``
        is selected via ``k``‑fold cross validation.  The learned map is then
        inserted between the student and teacher networks and the resulting
        model is evaluated on the student's test distribution.
        """

        if self.student_cfg.train_size is None:
            raise ValueError("student_cfg.train_size must be specified")

        n = self.student_cfg.train_size

        dist = self.get_layers_distribution(student_layer, teacher_layer)

        lin_module, _ = dist.k_fold_linear(
            seed=self.seed_mgr.spawn_seed(),
            n_samples=n,
            bias=bias,
            lambdas=lambdas,
            k=k,
        )

        student_repr = dist.student_representor
        teacher_repr = dist.teacher_representor

        if dist.student_rep_id == 0:
            student_module = nn.Identity().to(self.device)
        else:
            student_module = student_repr.get_module(0, dist.student_rep_id).to(
                self.device
            )

        final_rep = teacher_repr.get_final_rep_id()
        if dist.teacher_rep_id == final_rep:
            teacher_module = nn.Identity().to(self.device)
        else:
            teacher_module = teacher_repr.get_module(
                dist.teacher_rep_id, final_rep
            ).to(self.device)

        student_module.eval()
        teacher_module.eval()
        lin_module = lin_module.to(self.device)
        lin_module.eval()

        loss_fn = self.student_cfg.loss_config.build()
        provider = self.get_test_provider(seed=self.seed_mgr.spawn_seed())

        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for X, y in provider:
                X, y = X.to(self.device), y.to(self.device)
                s_rep = student_module(X)
                t_rep = lin_module(s_rep)
                y_pred = teacher_module(t_rep)
                batch_size = X.size(0)
                total_loss += loss_fn(y_pred, y).item() * batch_size
                total_samples += batch_size

        return total_loss / total_samples if total_samples > 0 else float("nan")

    def save_loss_table(
        self,
        filename: Path | str,
        bias: bool = True,
        lambdas: list[float] | None = None,
        start_teacher_layer: int = 0,
    ) -> None:
        """Compute losses for all layer pairs and save them as a table."""

        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        student_layers = len(self.student_cfg.model_config.hidden_dims) + 2
        total_teacher_layers = len(self.teacher_cfg.model_config.hidden_dims) + 2
        teacher_layers = list(range(start_teacher_layer, total_teacher_layers))

        # Collect loss values for each layer pair
        cell_data: list[list[str]] = []
        for s in range(student_layers):
            row: list[str] = []
            for t in teacher_layers:
                loss = self.compute_test_loss(s, t, bias=bias, lambdas=lambdas)
                row.append(f"{loss:.6g}")
            cell_data.append(row)

        col_labels = [str(t) for t in teacher_layers]
        row_labels = [str(s) for s in range(student_layers)]

        fig_width = max(4, len(teacher_layers))
        fig_height = max(4, student_layers)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")

        table = ax.table(
            cellText=cell_data,
            rowLabels=row_labels,
            colLabels=col_labels,
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        fig.tight_layout()
        fig.savefig(filename, dpi=300)
        plt.close(fig)
