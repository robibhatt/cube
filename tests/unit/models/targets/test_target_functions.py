import torch
import pytest
from src.models.targets.simple_functions import Prod1234
from src.models.targets.configs.prod_1234 import Prod1234Config


def test_prod1234_computes_expected():
    X = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 1.5, 2.0, 0.5]])
    func = Prod1234(Prod1234Config(input_shape=torch.Size([4])))
    result = func(X)
    expected = torch.tensor([[1.0 * 2.0 + 3.0 * 4.0], [0.0 * 1.5 + 2.0 * 0.5]])
    assert torch.allclose(result, expected)


def test_prod1234_dim_error():
    X = torch.tensor([[1.0, 2.0, 3.0]])
    func = Prod1234(Prod1234Config(input_shape=torch.Size([3])))
    with pytest.raises(ValueError):
        func(X)
