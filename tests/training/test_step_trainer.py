import src.models.bootstrap  # noqa: F401

from src.training.trainer import Trainer
from src.training.trainer_config import TrainerConfig
from src.models.mlp_config import MLPConfig
from src.training.optimizers.configs.adam import AdamConfig
from src.data.cube_distribution_config import (
    CubeDistributionConfig,
)


def test_trainer_completes_epoch(tmp_path):
    """Ensure the :class:`Trainer` can run a minimal training loop."""

    cfg = TrainerConfig(
        mlp_config=MLPConfig(
            input_dim=1,
            output_dim=1,
            hidden_dims=[],
            activation="relu",
            start_activation=False,
            end_activation=False,
            bias=False,
        ),
        optimizer_config=AdamConfig(lr=0.01),
        cube_distribution_config=CubeDistributionConfig(
            input_dim=1,
            indices_list=[[0]],
            weights=[1.0],
            noise_mean=0.0,
            noise_std=0.0,
        ),
        train_size=4,
        test_size=4,
        batch_size=2,
        epochs=1,
        home_dir=tmp_path,
        seed=0,
    )

    trainer = Trainer(cfg)
    trainer.train()

    assert trainer.epochs_trained == 1
