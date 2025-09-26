import pytest
import torch

import src.models.bootstrap  # noqa: F401
import src.data.providers.noisy_provider  # Register NoisyProvider
from src.data.joint_distributions.cube_distribution import CubeDistribution
from src.data.joint_distributions.configs.cube_distribution import CubeDistributionConfig
from src.models.architectures.configs.mlp import MLPConfig
from src.models.architectures.model_factory import create_model
from src.training.trainer import Trainer
from src.training.trainer_config import TrainerConfig


def _constant_cube_distribution(input_dim: int = 2, value: float = 5.0) -> CubeDistribution:
    config = CubeDistributionConfig(
        input_dim=input_dim,
        indices_list=[[0]],
        weights=[0.0],
        normalize=False,
        noise_mean=value,
        noise_std=0.0,
    )
    return CubeDistribution(config, torch.device("cpu"))


@pytest.fixture
def constant_cube_distribution() -> CubeDistribution:
    """Return a CubeDistribution that always produces a constant target."""

    return _constant_cube_distribution()


@pytest.fixture
def create_mlp_config(tmp_path) -> MLPConfig:
    """Return an ``MLPConfig`` with weights saved to ``tmp_path``."""

    config = MLPConfig(
        input_dim=8,
        hidden_dims=[3, 4, 5],
        activation="relu",
        output_dim=6,
        start_activation=False,
        end_activation=False,
    )

    model = create_model(config)
    weight_file = tmp_path / "checkpoint.pth"
    torch.save(model.state_dict(), weight_file)
    return config


@pytest.fixture
def trained_trainer(tmp_path, mlp_config, adam_config) -> Trainer:
    """Return a Trainer trained for one epoch."""

    home = tmp_path / "trainer_home"
    home.mkdir()
    cfg = TrainerConfig(
        mlp_config=mlp_config,
        optimizer_config=adam_config,
        cube_distribution_config=CubeDistributionConfig(
            input_dim=mlp_config.input_dim,
            indices_list=[[i] for i in range(mlp_config.input_dim)],
            weights=[1.0 for _ in range(mlp_config.input_dim)],
            noise_mean=0.0,
            noise_std=0.0,
        ),
        train_size=4,
        test_size=2,
        batch_size=2,
        epochs=1,
        home_dir=home,
        seed=0,
    )
    trainer = Trainer(cfg)
    trainer.train()
    return trainer


@pytest.fixture
def trained_noisy_trainer(tmp_path, adam_config) -> Trainer:
    """Return a Trainer using a CubeDistribution trained for one epoch."""

    home = tmp_path / "noisy_trainer_home"
    home.mkdir()
    model_cfg = MLPConfig(
        input_dim=2,
        hidden_dims=[],
        activation="relu",
        output_dim=1,
        start_activation=False,
        end_activation=False,
    )
    cfg = TrainerConfig(
        mlp_config=model_cfg,
        optimizer_config=adam_config,
        cube_distribution_config=CubeDistributionConfig(
            input_dim=2,
            indices_list=[[0]],
            weights=[0.0],
            normalize=False,
            noise_mean=1.0,
            noise_std=0.0,
        ),
        train_size=4,
        test_size=2,
        batch_size=2,
        epochs=1,
        home_dir=home,
        seed=0,
    )
    trainer = Trainer(cfg)
    trainer.train()
    # Overwrite model weights so the trained model outputs a constant value
    model, _ = trainer._load_model_and_optimizer()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        # The final layer may be a standard ``nn.Linear`` or a μP ``MuReadout``
        # depending on how ``MLP`` was constructed.
        if hasattr(model.layers[-1], "bias"):
            model.layers[-1].bias.fill_(5.0)
    from src.checkpoints.checkpoint import Checkpoint
    checkpoint = Checkpoint.from_dir(trainer.checkpoint_dir)
    checkpoint.save(model=model, optimizer=None)
    return trainer

