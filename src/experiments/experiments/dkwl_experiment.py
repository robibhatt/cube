from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Dict, List

from src.data.cube_distribution_config import CubeDistributionConfig
from src.experiments.configs.dkwl import DkwlExperimentConfig
from src.experiments.configs.train_mlp import TrainMLPExperimentConfig
from src.experiments.experiments import BatchExperiment, register_experiment
from src.models.mlp_config import MLPConfig
from src.training.sgd_config import SgdConfig
from src.training.trainer_config import TrainerConfig


@register_experiment("DKWL")
class DkwlExperiment(BatchExperiment):
    """Batch experiment sweeping over MLP hyper-parameters."""

    config: DkwlExperimentConfig

    def __init__(self, config: DkwlExperimentConfig) -> None:
        super().__init__(config)

    def get_experiment_configs(self) -> List[TrainMLPExperimentConfig]:
        configs: List[TrainMLPExperimentConfig] = []

        for d in self.config.ds:
            for k in self.config.ks:
                if k <= 0 or k > d:
                    continue

                for (
                    width,
                    layer_count,
                    train_size,
                    epochs,
                    l1_decay,
                    learning_rate,
                    batch_size,
                ) in product(
                    self.config.widths,
                    self.config.layers,
                    self.config.train_sizes,
                    self.config.epochs,
                    self.config.l1_decays,
                    self.config.learning_rates,
                    self.config.batch_sizes,
                ):
                    trainer_cfg = TrainerConfig(
                        mlp_config=MLPConfig(
                            input_dim=d,
                            output_dim=1,
                            hidden_dims=[width] * layer_count,
                            start_activation=False,
                            end_activation=False,
                        ),
                        cube_distribution_config=CubeDistributionConfig(
                            input_dim=d,
                            indices_list=[list(range(k))],
                            weights=[1.0],
                            noise_mean=0.0,
                            noise_std=0.0,
                        ),
                        train_size=train_size,
                        test_size=train_size,
                        batch_size=batch_size,
                        epochs=epochs,
                        weight_decay_l1=l1_decay,
                        optimizer_config=SgdConfig(lr=learning_rate),
                    )

                    sub_cfg = TrainMLPExperimentConfig(
                        trainer_config=trainer_cfg,
                        edge_thresholds=list(self.config.edge_thresholds),
                        home_directory=self._subdirectory_for(
                            d=d,
                            k=k,
                            width=width,
                            layers=layer_count,
                            train_size=train_size,
                            epochs=epochs,
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                            l1_decay=l1_decay,
                        ),
                        seed=self.seed_mgr.spawn_seed(),
                    )
                    configs.append(sub_cfg)

        return configs

    def get_config_params(
        self, config: TrainMLPExperimentConfig
    ) -> Dict[str, Any]:
        trainer_cfg = config.trainer_config
        if trainer_cfg.mlp_config is None:
            raise ValueError("Trainer config is missing an MLP config")
        if trainer_cfg.cube_distribution_config is None:
            raise ValueError("Trainer config is missing a cube distribution config")

        mlp_cfg = trainer_cfg.mlp_config
        dist_cfg = trainer_cfg.cube_distribution_config

        hidden_dims = mlp_cfg.hidden_dims
        width = hidden_dims[0] if hidden_dims else 0
        optimizer_cfg = trainer_cfg.optimizer_config
        learning_rate = optimizer_cfg.lr if optimizer_cfg is not None else None

        return {
            "d": dist_cfg.input_dim,
            "k": len(dist_cfg.indices_list[0]) if dist_cfg.indices_list else 0,
            "width": width,
            "layers": len(hidden_dims),
            "train_size": trainer_cfg.train_size,
            "epochs": trainer_cfg.epochs,
            "l1_decay": trainer_cfg.weight_decay_l1,
            "learning_rate": learning_rate,
            "batch_size": trainer_cfg.batch_size,
            "edge_thresholds": list(config.edge_thresholds),
            "seed": config.seed,
        }

    def _subdirectory_for(
        self,
        *,
        d: int,
        k: int,
        width: int,
        layers: int,
        train_size: int,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        l1_decay: float,
    ) -> Path:
        parts = [
            f"d{d}",
            f"k{k}",
            f"width{width}",
            f"layers{layers}",
            f"train{train_size}",
            f"epochs{epochs}",
            f"lr{self._format_float(learning_rate)}",
            f"batch{batch_size}",
            f"l1{self._format_float(l1_decay)}",
        ]
        return self.config.home_directory.joinpath(*parts)

    @staticmethod
    def _format_float(value: float) -> str:
        text = format(value, "g")
        return text.replace(".", "p")
