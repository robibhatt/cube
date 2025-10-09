import pytest
import torch

import src.models.bootstrap  # noqa: F401

from src.data.cube_distribution import CubeDistribution
from src.data.cube_distribution_config import (
    CubeDistributionConfig,
)
from src.data.noisy_data_provider import NoisyProvider
from src.training.trainer import Trainer
from src.training.trainer_config import TrainerConfig


def available_gpu():
    """Return the first available GPU device."""

    if torch.cuda.is_available():
        # Explicitly specify index 0 so equality checks work across devices
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mpu") and torch.backends.mpu.is_available():
        return torch.device("mpu:0")
    return None


def _cube_config(input_dim: int = 3) -> CubeDistributionConfig:
    return CubeDistributionConfig(
        input_dim=input_dim,
        indices_list=[[i] for i in range(input_dim)],
        weights=[1.0 for _ in range(input_dim)],
        noise_mean=0.0,
        noise_std=0.0,
    )


@pytest.mark.gpu
def test_cube_sample_device():
    device = available_gpu()
    if device is None:
        pytest.skip("GPU not available")

    dist = CubeDistribution(_cube_config(2), device)
    samples, _ = dist.sample(3, seed=0)
    assert samples.device == device


@pytest.mark.gpu
def test_noisy_provider_yields_on_device():
    device = available_gpu()
    if device is None:
        pytest.skip("GPU not available")

    dist = CubeDistribution(_cube_config(3), device)
    provider = NoisyProvider(
        dist,
        seed=0,
        dataset_size=4,
        batch_size=2,
    )
    xb, yb = next(iter(provider))
    assert xb.device == device
    assert yb.device == device


@pytest.mark.gpu
def test_trainer_runs_on_gpu(tmp_path, mlp_config, sgd_config):
    device = available_gpu()
    if device is None:
        pytest.skip("GPU not available")

    home = tmp_path / "trainer_gpu"
    home.mkdir()

    cfg = TrainerConfig(
        mlp_config=mlp_config,
        optimizer_config=sgd_config,
        cube_distribution_config=_cube_config(mlp_config.input_dim),
        train_size=4,
        test_size=2,
        batch_size=2,
        epochs=1,
        home_dir=home,
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
