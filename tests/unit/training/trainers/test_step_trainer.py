from src.training.trainer import Trainer
from src.training.trainer_config import TrainerConfig
from src.models.mlp_config import MLPConfig
from src.training.sgd_config import SgdConfig
from src.data.cube_distribution_config import (
    CubeDistributionConfig,
)


def test_trainer_completes_epoch(tmp_path):
    """Ensure the :class:`Trainer` can run a minimal training loop."""

    cfg = TrainerConfig(
        mlp_config=MLPConfig(
            input_dim=1,
            hidden_dims=[],
        ),
        optimizer_config=SgdConfig(lr=0.01),
        cube_distribution_config=CubeDistributionConfig(
            input_dim=1,
            indices_list=[[0]],
            weights=[1.0],
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


def test_trainer_uses_default_base_shapes_for_mlp(tmp_path):
    cfg = TrainerConfig(
        mlp_config=MLPConfig(
            input_dim=3,
            hidden_dims=[8, 5],
        ),
        optimizer_config=SgdConfig(lr=0.01),
        cube_distribution_config=CubeDistributionConfig(
            input_dim=3,
            indices_list=[[0]],
            weights=[1.0],
            noise_std=0.0,
        ),
        train_size=4,
        test_size=4,
        batch_size=2,
        epochs=0,
        home_dir=tmp_path,
        seed=0,
    )

    trainer = Trainer(cfg)
    model, _ = trainer._initialize_model_and_optimizer()

    assert model.config.exact_base_shapes is False
    base_model = model.get_base_model()
    assert base_model.config.hidden_dims == [64, 64]
