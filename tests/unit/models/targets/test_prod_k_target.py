import torch
import pytest

from src.models.targets.simple_functions import ProdKTarget
from src.models.targets.configs.prod_k import ProdKTargetConfig


def test_prod_k_target_computes_expected():
    X = torch.tensor([[1.0, 2.0, 3.0], [0.5, 4.0, -2.0]])
    func = ProdKTarget(ProdKTargetConfig(input_shape=torch.Size([3]), indices=[0, 1, 2]))
    result = func(X)
    expected = torch.tensor([[1.0 * 2.0 * 3.0], [0.5 * 4.0 * -2.0]])
    assert torch.allclose(result, expected)


def test_prod_k_target_dim_error():
    X = torch.tensor([[1.0, 2.0]])
    func = ProdKTarget(ProdKTargetConfig(input_shape=torch.Size([2]), indices=[0, 1]))
    with pytest.raises(ValueError):
        func(X[:, :1])


def test_prod_k_target_handles_multiple_batch_dims():
    X = torch.tensor([
        [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
        [[0.5, 4.0, -2.0], [1.0, 2.0, 3.0]],
    ])
    func = ProdKTarget(ProdKTargetConfig(input_shape=torch.Size([3]), indices=[0, 1, 2]))
    result = func(X)
    expected = torch.tensor([
        [[1.0 * 2.0 * 3.0], [2.0 * 3.0 * 4.0]],
        [[0.5 * 4.0 * -2.0], [1.0 * 2.0 * 3.0]],
    ])
    assert torch.allclose(result, expected)
