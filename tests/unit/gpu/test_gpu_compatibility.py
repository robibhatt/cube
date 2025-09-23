import pytest
import torch
import torch.nn.functional as F
from typing import Tuple

import pytest
import torch

import src.models.bootstrap  # noqa: F401
from src.data.joint_distributions import create_joint_distribution
from src.data.joint_distributions.configs.gaussian import GaussianConfig
from src.data.joint_distributions.joint_distribution import JointDistribution
from src.data.joint_distributions.configs.base import JointDistributionConfig
from src.data.joint_distributions.joint_distribution_registry import (
    register_joint_distribution,
)
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from src.data.joint_distributions.configs.joint_distribution_config_registry import (
    register_joint_distribution_config,
)
from src.data.providers.tensor_data_provider import TensorDataProvider
from src.training.trainer import Trainer
from src.training.trainer_config import TrainerConfig
from src.training.loss.configs.loss import LossConfig


def available_gpu():
    """Return the first available GPU device."""
    if torch.cuda.is_available():
        # Explicitly specify index 0 so equality checks work across devices
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mpu") and torch.backends.mpu.is_available():
        return torch.device("mpu:0")
    return None


@register_joint_distribution("GPUJointDistribution")
class GPUJointDistribution(JointDistribution):
    @register_joint_distribution_config("GPUJointDistribution")
    @dataclass_json
    @dataclass(kw_only=True)
    class _Config(JointDistributionConfig):
        input_shape: torch.Size = field(default_factory=lambda: torch.Size([3]))
        output_shape: torch.Size = field(default_factory=lambda: torch.Size([1]))

        def __post_init__(self) -> None:  # type: ignore[override]
            self.distribution_type = "GPUJointDistribution"

    def __init__(self, config: _Config, device: torch.device):
        super().__init__(config, device)

    def sample(self, n_samples: int, seed: int):
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        x = (
            torch.arange(n_samples * 3, dtype=torch.float32, device=self.device)
            .reshape(n_samples, 3)
        )
        y = torch.full((n_samples, 1), 5.0, dtype=torch.float32, device=self.device)
        return x, y

    def __str__(self) -> str:
        return "GPUJointDistribution"

    def base_sample(self, n_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sample(n_samples, seed)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return X, torch.zeros(X.size(0), 1, device=self.device)

    def forward_X(self, X: torch.Tensor) -> torch.Tensor:
        return X

    def preferred_provider(self) -> str:
        return "TensorDataProvider"


@pytest.mark.gpu
def test_gaussian_sample_device():
    device = available_gpu()
    if device is None:
        pytest.skip("GPU not available")
    cfg = GaussianConfig(input_shape=torch.Size([2]), mean=0.0, std=1.0)
    g = create_joint_distribution(cfg, device)
    samples, _ = g.sample(3, seed=0)
    assert samples.device == device


@pytest.mark.gpu
def test_tensor_provider_yields_on_device(tmp_path):
    device = available_gpu()
    if device is None:
        pytest.skip("GPU not available")
    cfg = GPUJointDistribution._Config()
    dist = GPUJointDistribution(cfg, device)
    seed = 0
    provider = TensorDataProvider(dist, tmp_path, seed, batch_size=2, dataset_size=4)
    loader = provider.data_loader
    xb, yb = next(iter(loader))
    assert xb.device == device
    assert yb.device == device


@pytest.mark.gpu
def test_trainer_runs_on_gpu(tmp_path, mlp_config, adam_config):
    device = available_gpu()
    if device is None:
        pytest.skip("GPU not available")
    home = tmp_path / "trainer_gpu"
    home.mkdir()
    dist_cfg = GPUJointDistribution._Config()
    cfg = TrainerConfig(
        model_config=mlp_config,
        optimizer_config=adam_config,
        joint_distribution_config=dist_cfg,
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
    model, _ = trainer._load_model_and_optimizer()
    param_device = next(model.parameters()).device
    loader = trainer.get_iterator("train")
    xb, yb = next(iter(loader))
    assert param_device == device
    assert xb.device == device
    assert yb.device == device

