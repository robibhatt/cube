import pytest
import torch
from typing import Tuple
from dataclasses import dataclass, field
from src.data.joint_distributions.configs.base import JointDistributionConfig
from src.data.joint_distributions import create_joint_distribution
from src.data.joint_distributions.configs.representor_distribution import (
    RepresentorDistributionConfig,
)
from src.data.joint_distributions.joint_distribution import JointDistribution
from src.data.joint_distributions.joint_distribution_registry import register_joint_distribution
from src.data.joint_distributions.configs.joint_distribution_config_registry import register_joint_distribution_config
from tests.helpers.stubs import StubJointDistribution
from src.models.representors.mlp_representor import MLPRepresentor
from src.training.trainer import Trainer
from src.training.trainer_config import TrainerConfig
from src.training.loss.configs.loss import LossConfig




def test_representor_distribution_initialization(trained_trainer, model_representor):
    # Setup
    from_rep = 0
    to_rep = 1
    # Create distribution
    cfg = RepresentorDistributionConfig(
        base_distribution_config=trained_trainer.config.joint_distribution_config,
        model_config=trained_trainer.config.model_config,
        checkpoint_dir=trained_trainer.checkpoint_dir,
        from_rep=from_rep,
        to_rep=to_rep,
    )
    dist = create_joint_distribution(cfg, torch.device("cpu"))

    # Assertions
    assert isinstance(dist.base_joint_distribution, StubJointDistribution)
    assert isinstance(dist.model_representor, MLPRepresentor)
    assert dist.model_representor.model_config == model_representor.model_config
    assert dist.from_rep == from_rep
    assert dist.to_rep == to_rep
    assert dist.input_shape == model_representor.representation_shape(from_rep)
    assert dist.output_shape == model_representor.representation_shape(to_rep)

def test_representor_distribution_str(trained_trainer, model_representor):
    # Setup
    from_rep = 0
    to_rep = 1
    # Create distribution
    cfg = RepresentorDistributionConfig(
        base_distribution_config=trained_trainer.config.joint_distribution_config,
        model_config=trained_trainer.config.model_config,
        checkpoint_dir=trained_trainer.checkpoint_dir,
        from_rep=from_rep,
        to_rep=to_rep,
    )
    dist = create_joint_distribution(cfg, torch.device("cpu"))

    # Test string representation
    dist_str = str(dist)
    assert "RepresentorDistribution" in dist_str
    assert "StubJointDistribution" in dist_str
    assert "MLPRepresentor" in dist_str
    assert f"from_rep={from_rep}" in dist_str
    assert f"to_rep={to_rep}" in dist_str


def test_representor_distribution_omits_noise(trained_noisy_trainer):
    """Noise used during training should not appear when sampling."""

    cfg = RepresentorDistributionConfig(
        base_distribution_config=trained_noisy_trainer.config.joint_distribution_config,
        model_config=trained_noisy_trainer.config.model_config,
        checkpoint_dir=trained_noisy_trainer.checkpoint_dir,
        from_rep=0,
        to_rep=1,
    )
    dist = create_joint_distribution(cfg, torch.device("cpu"))

    _, y = dist.sample(3, seed=0)

    assert torch.equal(y, torch.full((3, 1), 5.0))

def test_representor_distribution_sample(trained_trainer, model_representor):
    # Setup
    from_rep = 0
    to_rep = 1
    n_samples = 4

    # Create distribution
    cfg = RepresentorDistributionConfig(
        base_distribution_config=trained_trainer.config.joint_distribution_config,
        model_config=trained_trainer.config.model_config,
        checkpoint_dir=trained_trainer.checkpoint_dir,
        from_rep=from_rep,
        to_rep=to_rep,
    )
    dist = create_joint_distribution(cfg, torch.device("cpu"))

    # Sample from distribution
    X, y = dist.sample(n_samples, seed=0)

    # Assertions
    assert X.shape == (n_samples, *model_representor.representation_shape(from_rep))
    assert y.shape == (n_samples, *model_representor.representation_shape(to_rep))


def test_representor_distribution_sample_with_joint_base(tmp_path, mlp_config, adam_config):
    """Sampling should also work when the base is a ``JointDistribution``."""

    @register_joint_distribution("JointBase")
    class JointBase(JointDistribution):
        @dataclass
        @register_joint_distribution_config("JointBase")
        class _Cfg(JointDistributionConfig):
            input_shape: torch.Size = field(default_factory=lambda: torch.Size([3]))
            output_shape: torch.Size = field(default_factory=lambda: torch.Size([1]))

            def __post_init__(self) -> None:  # type: ignore[override]
                self.distribution_type = "JointBase"

        def __init__(self, config: "JointBase._Cfg", device: torch.device) -> None:
            super().__init__(config, device)

        def sample(self, n_samples: int, seed: int):
            g = torch.Generator()
            g.manual_seed(seed)
            x = torch.randn(n_samples, 3, generator=g)
            y = torch.zeros(n_samples, 1)
            return x, y

        def __str__(self) -> str:
            return "JointBase"

        def base_sample(self, n_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.sample(n_samples, seed)

        def forward(
            self, X: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            return X, torch.zeros(X.size(0), 1)

        def forward_X(
            self, X: torch.Tensor
        ) -> torch.Tensor:
            return X

        def preferred_provider(self) -> str:
            return "TensorDataProvider"

    from_rep = 0
    to_rep = 1
    n_samples = 4

    trainer_dir = tmp_path / "trainer"
    trainer_dir.mkdir()
    tr_cfg = TrainerConfig(
        model_config=mlp_config,
        optimizer_config=adam_config,
        joint_distribution_config=JointBase._Cfg(),
        train_size=4,
        test_size=2,
        batch_size=2,
        epochs=1,
        home_dir=trainer_dir,
        loss_config=LossConfig(name="MSELoss"),
        seed=0,
    )
    trainer = Trainer(tr_cfg)
    trainer.train()
    representor = MLPRepresentor(trainer.config.model_config, trainer.checkpoint_dir, device=torch.device("cpu"))
    cfg = RepresentorDistributionConfig(
        base_distribution_config=tr_cfg.joint_distribution_config,
        model_config=tr_cfg.model_config,
        checkpoint_dir=trainer.checkpoint_dir,
        from_rep=from_rep,
        to_rep=to_rep,
    )
    dist = create_joint_distribution(cfg, torch.device("cpu"))
    X, y = dist.sample(n_samples, seed=0)

    assert X.shape == (n_samples, *representor.representation_shape(from_rep))
    assert y.shape == (n_samples, *representor.representation_shape(to_rep))

def test_representor_distribution_field_parsing(trained_trainer, model_representor):
    """Test that we can correctly parse and access all fields of a RepresentorDistribution object."""
    # Setup with target config
    from_rep = 0
    to_rep = 1
    # Create distribution
    cfg = RepresentorDistributionConfig(
        base_distribution_config=trained_trainer.config.joint_distribution_config,
        model_config=trained_trainer.config.model_config,
        checkpoint_dir=trained_trainer.checkpoint_dir,
        from_rep=from_rep,
        to_rep=to_rep,
    )
    dist = create_joint_distribution(cfg, torch.device("cpu"))
    
    # Test base distribution fields
    assert isinstance(dist.base_joint_distribution, StubJointDistribution)
    assert dist.base_joint_distribution.input_shape == torch.Size([3])
    assert dist.base_joint_distribution.output_shape == torch.Size([1])
    
    # Test model representor fields
    assert isinstance(dist.model_representor, MLPRepresentor)
    assert len(dist.model_representor.modules) > 0
    assert dist.model_representor.model_config.input_dim == 3
    assert dist.model_representor.model_config.output_dim == 1
    assert dist.model_representor.model_config.hidden_dims == [4, 2]
    assert dist.model_representor.model_config.activation == "relu"
    
    # Test representation indices
    assert dist.from_rep == from_rep
    assert dist.to_rep == to_rep
    assert dist.from_rep < dist.to_rep  # Ensure valid representation order
    
    # Test shape fields
    assert dist.input_shape == model_representor.representation_shape(from_rep)
    assert dist.output_shape == model_representor.representation_shape(to_rep)
    
    # Test string representation contains all fields
    dist_str = str(dist)
    assert "RepresentorDistribution" in dist_str
    assert "StubJointDistribution" in dist_str
    assert "MLPRepresentor" in dist_str
    assert f"from_rep={from_rep}" in dist_str
    assert f"to_rep={to_rep}" in dist_str
