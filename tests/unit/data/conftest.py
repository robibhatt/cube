import pytest
import torch
from dataclasses import dataclass, field
from typing import Tuple

import src.models.bootstrap  # noqa: F401
from src.data.joint_distributions import create_joint_distribution
from src.data.joint_distributions.configs.gaussian import GaussianConfig
from src.data.joint_distributions.joint_distribution import JointDistribution
from src.data.joint_distributions.joint_distribution_registry import register_joint_distribution
from src.data.joint_distributions.configs.base import JointDistributionConfig
from src.data.joint_distributions.configs.joint_distribution_config_registry import register_joint_distribution_config
from src.models.architectures.configs.mlp import MLPConfig
from src.models.architectures.model_factory import create_model
from src.training.trainer import Trainer
from src.training.trainer_config import TrainerConfig
from src.training.loss.configs.loss import LossConfig
from tests.helpers.stubs import StubJointDistribution
import src.data.providers.noisy_provider  # Register NoisyIterator
from src.data.joint_distributions.configs.noisy_distribution import NoisyDistributionConfig


@pytest.fixture
def gaussian_base():
    cfg = GaussianConfig(input_shape=torch.Size([2]), mean=0.0, std=1.0)
    return create_joint_distribution(cfg, torch.device("cpu"))


@register_joint_distribution("DummyJointDistribution")
class DummyJointDistribution(JointDistribution):
    """Always returns x = arange and y = constant 5.0."""

    @dataclass
    @register_joint_distribution_config("DummyJointDistribution")
    class _Config(JointDistributionConfig):
        input_shape: torch.Size = field(default_factory=lambda: torch.Size([2]))
        output_shape: torch.Size = field(default_factory=lambda: torch.Size([1]))

        def __post_init__(self) -> None:  # type: ignore[override]
            self.distribution_type = "DummyJointDistribution"

    def __init__(self, config: _Config, device: torch.device):
        super().__init__(config, device)

    def sample(self, n_samples: int, seed: int):
        """Return deterministic samples on the distribution's device.

        The original test helper returned CPU tensors irrespective of the
        ``device`` argument, causing downstream operations to fail when the
        trainer runs on a GPU.  Creating the tensors directly on
        ``self.device`` keeps all tensors on the same device across platforms.
        """

        x = (
            torch.arange(n_samples * 2, dtype=torch.float32, device=self.device)
            .reshape(n_samples, 2)
        )
        y = torch.full(
            (n_samples, 1), 5.0, dtype=torch.float32, device=self.device
        )
        return x, y

    def __str__(self):
        return "DummyJointDistribution"

    def base_sample(self, n_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sample(n_samples, seed)

    def forward(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ``DummyJointDistribution`` is used in many unit tests where the
        # trainer may operate on GPU devices. Ensure that the returned tensors
        # live on the same device as the distribution to avoid
        # device-mismatch errors.
        X = X.to(self.device)
        y = torch.full(
            (X.size(0), 1), 5.0, dtype=torch.float32, device=self.device
        )
        return X, y

    def forward_X(
        self, X: torch.Tensor
    ) -> torch.Tensor:
        return X

    def preferred_provider(self) -> str:
        return "TensorDataProvider"


class BadTypeDistribution(JointDistribution):
    """sample() returns non-tensors."""

    @dataclass
    class _Config(JointDistributionConfig):
        input_shape: torch.Size = field(default_factory=lambda: torch.Size([1]))
        output_shape: torch.Size = field(default_factory=lambda: torch.Size([1]))

        def __post_init__(self) -> None:  # type: ignore[override]
            self.distribution_type = "BadTypeDistribution"

    def __init__(self, config: _Config, device: torch.device):
        super().__init__(config, device)

    def sample(self, n_samples: int, seed: int):
        return [1, 2, 3], "not a tensor"

    def __str__(self):
        return "BadTypeDistribution"

    def base_sample(self, n_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sample(n_samples, seed)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return X, torch.zeros(X.size(0), 1)

    def forward_X(self, X: torch.Tensor) -> torch.Tensor:
        return X

    def preferred_provider(self) -> str:
        return "TensorDataProvider"


class BadShapeDistribution(JointDistribution):
    """sample() returns tensors of wrong shape."""

    @dataclass
    class _Config(JointDistributionConfig):
        input_shape: torch.Size = field(default_factory=lambda: torch.Size([2]))
        output_shape: torch.Size = field(default_factory=lambda: torch.Size([1]))

        def __post_init__(self) -> None:  # type: ignore[override]
            self.distribution_type = "BadShapeDistribution"

    def __init__(self, config: _Config, device: torch.device):
        super().__init__(config, device)

    def sample(self, n_samples: int, seed: int):
        x = torch.zeros(n_samples, 3)  # should be (n_samples,2)
        y = torch.zeros(n_samples, 2)  # should be (n_samples,1)
        return x, y

    def __str__(self):
        return "BadShapeDistribution"

    def base_sample(self, n_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sample(n_samples, seed)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return X, torch.zeros(X.size(0), 1)

    def forward_X(self, X: torch.Tensor) -> torch.Tensor:
        return X

    def preferred_provider(self) -> str:
        return "TensorDataProvider"


@pytest.fixture
def dummy_distribution():
    """Return a simple joint distribution for testing."""
    cfg = DummyJointDistribution._Config()
    return DummyJointDistribution(cfg, torch.device("cpu"))


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
        model_config=mlp_config,
        optimizer_config=adam_config,
        joint_distribution_config=StubJointDistribution._Config(
            X=torch.zeros(4, mlp_config.input_dim),
            y=torch.zeros(4, mlp_config.output_dim),
        ),
        train_size=4,
        test_size=2,
        batch_size=2,
        epochs=1,
        home_dir=home,
        loss_config=LossConfig(name="MSELoss"),
        seed=0,
    )
    trainer = Trainer(cfg)
    trainer.train()
    return trainer


@pytest.fixture
def trained_noisy_trainer(tmp_path, adam_config) -> Trainer:
    """Return a Trainer using a NoisyDistribution trained for one epoch."""
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
        model_config=model_cfg,
        optimizer_config=adam_config,
        joint_distribution_config=NoisyDistributionConfig(
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
        loss_config=LossConfig(name="MSELoss"),
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
