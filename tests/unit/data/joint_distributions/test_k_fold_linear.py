import torch

from tests.helpers.stubs import StubJointDistribution


def test_k_fold_linear_recovers_parameters() -> None:
    """k_fold_linear should recover the generating linear map."""

    A_true = torch.tensor([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]])
    b_true = torch.tensor([0.1, -0.2, 0.3])

    n = 30
    g = torch.Generator()
    g.manual_seed(0)
    X = torch.randn(n, 2, generator=g)
    y = X @ A_true.T + b_true

    cfg = StubJointDistribution._Config(X=X, y=y)
    dist = StubJointDistribution(cfg, torch.device("cpu"))

    module, lam = dist.k_fold_linear(seed=0, lambdas=[0.0, 0.1], k=5)

    assert isinstance(module, torch.nn.Module)
    assert module.weight.shape == (3, 2)
    assert module.bias.shape == (3,)
    assert torch.allclose(module.weight, A_true, atol=1e-3)
    assert torch.allclose(module.bias, b_true, atol=1e-3)
    assert torch.allclose(module(X), y, atol=1e-3)
    assert lam in [0.0, 0.1]
