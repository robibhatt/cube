import torch

from tests.helpers.stubs import StubJointDistribution


def test_linear_solve_recovers_parameters() -> None:
    """``linear_solve`` should recover the generating linear map."""

    A_true = torch.tensor([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]])
    b_true = torch.tensor([0.1, -0.2, 0.3])

    n = 20
    g = torch.Generator()
    g.manual_seed(0)
    X = torch.randn(n, 2, generator=g)
    y = X @ A_true.T + b_true

    cfg = StubJointDistribution._Config(X=X, y=y)
    dist = StubJointDistribution(cfg, torch.device("cpu"))

    module, lam = dist.linear_solve(seed=0, lambda_=0.0)

    assert isinstance(module, torch.nn.Module)
    assert module.weight.shape == (3, 2)
    assert module.bias.shape == (3,)
    assert torch.allclose(module.weight, A_true, atol=1e-3)
    assert torch.allclose(module.bias, b_true, atol=1e-3)
    assert torch.allclose(module(X), y, atol=1e-3)
    assert lam >= 0
    # Ensure batch dimensions are preserved
    X_batch = X.unsqueeze(0)
    y_batch = module(X_batch)
    assert y_batch.shape == (1, n, 3)
    assert torch.allclose(y_batch.squeeze(0), y, atol=1e-6)


def test_linear_solve_no_bias() -> None:
    """When ``bias=False`` the solve should not learn an intercept."""

    A_true = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    n = 20
    g = torch.Generator()
    g.manual_seed(0)
    X = torch.randn(n, 2, generator=g)
    y = X @ A_true.T

    cfg = StubJointDistribution._Config(X=X, y=y)
    dist = StubJointDistribution(cfg, torch.device("cpu"))

    module, lam = dist.linear_solve(seed=0, bias=False, lambda_=0.0)

    assert isinstance(module, torch.nn.Module)
    assert module.weight.shape == (2, 2)
    assert module.bias is None
    assert torch.allclose(module.weight, A_true, atol=1e-3)
    assert torch.allclose(module(X), y, atol=1e-3)
    assert lam >= 0

