import pytest
import torch
from src.data.joint_distributions import create_joint_distribution
from src.data.joint_distributions.configs.gaussian import GaussianConfig


def available_devices() -> list[torch.device]:
    """Return CPU and all GPU-like devices present."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    return devices


def test_shape_and_dim():
    cfg = GaussianConfig(input_dim=4, mean=0.0, std=1.0)
    g = create_joint_distribution(cfg, torch.device("cpu"))
    assert g.input_dim == 4
    assert g.input_shape == torch.Size([4])


def test_str_and_repr():
    cfg = GaussianConfig(input_dim=3, mean=0.0, std=1.0)
    g = create_joint_distribution(cfg, torch.device("cpu"))
    s = str(g)

    assert f"{g.input_dim}-dimensional Normal(mean=0.0, std=1.0)" == s


def test_sample_shape_and_dtype():
    cfg = GaussianConfig(input_dim=6, dtype=torch.float64, mean=0.0, std=1.0)
    g = create_joint_distribution(cfg, torch.device("cpu"))
    samples, y = g.sample(5, seed=0)
    assert isinstance(samples, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert samples.shape == (5, cfg.input_dim)
    assert y.shape == (5, 1)
    assert torch.allclose(y, torch.zeros(5, 1, dtype=g.dtype))
    assert samples.dtype == torch.float64




@pytest.mark.parametrize("device", available_devices())
def test_sample_on_requested_device(device: torch.device):
    cfg = GaussianConfig(input_dim=2, dtype=torch.float32, mean=0.0, std=1.0)
    g = create_joint_distribution(cfg, device)
    samples, y = g.sample(4, seed=0)
    assert samples.device == device
    assert y.device == device


