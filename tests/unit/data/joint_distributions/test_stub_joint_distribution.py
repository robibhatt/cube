import torch
from tests.helpers.stubs import StubJointDistribution


def test_forward_repeats_output():
    cfg = StubJointDistribution._Config(
        X=torch.tensor([[1.0, 2.0]]),
        y=torch.tensor([[3.0]]),
    )
    dist = StubJointDistribution(cfg, device=torch.device("cpu"))
    batch_size = 5
    X = torch.zeros(batch_size, 2)
    _, y = dist.forward(X)

    expected_y = cfg.y.repeat(batch_size, 1)
    assert y.shape[0] == batch_size
    assert torch.allclose(y, expected_y)
