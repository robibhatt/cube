import json
import pytest
import torch

import src.models.bootstrap  # noqa: F401
from tests.unit.gpu.test_gpu_compatibility import available_gpu, _cube_config
from src.training.trainer import Trainer
from src.training.trainer_config import TrainerConfig
from src.training.loss.configs.loss import LossConfig


def _assert_on_cpu(model, optimizer):
    cpu = torch.device("cpu")
    for p in model.parameters():
        assert p.device == cpu
    for state in optimizer.stepper.state.values():
        for v in state.values():
            if isinstance(v, torch.Tensor):
                assert v.device == cpu


@pytest.mark.gpu
def test_cross_device_loading(tmp_path, mlp_config, adam_config):
    device = available_gpu()
    if device is None:
        pytest.skip("GPU not available")

    home = tmp_path / "trainer_gpu"
    home.mkdir()

    cfg = TrainerConfig(
        model_config=mlp_config,
        optimizer_config=adam_config,
        cube_distribution_config=_cube_config(mlp_config.input_dim),
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

    # Simulate loading the trained model on a CPU-only machine
    cfg_dict = json.loads((home / "trainer_config.json").read_text())
    reloaded_cfg = TrainerConfig.from_dict(cfg_dict)
    reloaded_cfg.home_dir = home

    # Force the reloaded trainer to operate on the CPU
    torch_device_available = torch.cuda.is_available
    try:
        torch.cuda.is_available = lambda: False  # type: ignore[assignment]
        cpu_trainer = Trainer(reloaded_cfg)
    finally:
        torch.cuda.is_available = torch_device_available

    model, optimizer = cpu_trainer._load_model_and_optimizer()
    _assert_on_cpu(model, optimizer)

    cpu_trainer.test_loss()
