import torch
from torch import nn

from tests.helpers.stubs import StubJointDistribution
from src.data.joint_distributions.module_joint_distribution import ModuleJointDistribution


def build_base_distribution():
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = torch.zeros(3, 1)
    cfg = StubJointDistribution._Config(X=X, y=y)
    return StubJointDistribution(cfg, torch.device("cpu"))


def test_shapes_and_forward():
    base_dist = build_base_distribution()
    x_module = nn.Linear(2, 3, bias=False)
    x_module.weight.data = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y_module = nn.Linear(2, 1, bias=False)
    y_module.weight.data = torch.tensor([[2.0, -1.0]])

    dist = ModuleJointDistribution(base_dist, x_module, y_module)

    assert dist.input_shape == torch.Size([3])
    assert dist.output_shape == torch.Size([1])

    base_X, _ = dist.base_sample(3, seed=0)
    base_features = base_dist.forward_X(base_X)
    expected_X = x_module(base_features)
    expected_y = y_module(base_features)

    X, y = dist.forward(base_X)
    assert torch.allclose(X, expected_X)
    assert torch.allclose(y, expected_y)

    X2 = dist.forward_X(base_X)
    y2 = dist.forward_Y(base_X)
    assert torch.allclose(X2, expected_X)
    assert torch.allclose(y2, expected_y)


def test_sample():
    base_dist = build_base_distribution()
    x_module = nn.Linear(2, 2, bias=False)
    x_module.weight.data = torch.eye(2)
    y_module = nn.Linear(2, 1, bias=False)
    y_module.weight.data = torch.tensor([[1.0, 1.0]])

    dist = ModuleJointDistribution(base_dist, x_module, y_module)

    X, y = dist.sample(2, seed=0)

    base_X, _ = base_dist.base_sample(2, seed=0)
    base_features = base_dist.forward_X(base_X)
    assert torch.allclose(X, x_module(base_features))
    assert torch.allclose(y, y_module(base_features))


def test_respects_base_forward():
    base_dist = build_base_distribution()

    def forward_X(base_X):
        return base_X + 1

    def forward(base_X):
        X = forward_X(base_X)
        y = X.sum(dim=1, keepdim=True)
        return X, y

    base_dist.forward_X = forward_X  # type: ignore[assignment]
    base_dist.forward = forward  # type: ignore[assignment]

    x_module = nn.Identity()
    y_module = nn.Identity()
    dist = ModuleJointDistribution(base_dist, x_module, y_module)

    base_X, _ = base_dist.base_sample(2, seed=0)
    expected = forward_X(base_X)

    X, y = dist.forward(base_X)
    assert torch.allclose(X, expected)
    assert torch.allclose(y, expected)
    assert not torch.allclose(y, base_dist.forward(base_X)[1])

    X2 = dist.forward_X(base_X)
    y2 = dist.forward_Y(base_X)
    assert torch.allclose(X2, expected)
    assert torch.allclose(y2, expected)
