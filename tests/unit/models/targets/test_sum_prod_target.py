import torch
import pytest

from src.models.targets.sum_prod import SumProdTarget
from src.models.targets.configs.sum_prod import SumProdTargetConfig


def test_sum_prod_target_computes_expected():
    X = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, 4.0, -2.0, 1.0]])
    func = SumProdTarget(
        SumProdTargetConfig(
            input_shape=torch.Size([4]),
            indices_list=[[0, 1], [2, 3]],
            weights=[0.5, 1.5],
        )
    )
    result = func(X)
    unscaled = torch.tensor(
        [
            [0.5 * (1.0 * 2.0) + 1.5 * (3.0 * 4.0)],
            [0.5 * (0.5 * 4.0) + 1.5 * (-2.0 * 1.0)],
        ]
    )
    expected = unscaled * func.scale
    assert torch.allclose(result, expected)


def test_sum_prod_target_dim_error():
    X = torch.tensor([[1.0, 2.0, 3.0]])
    func = SumProdTarget(
        SumProdTargetConfig(
            input_shape=torch.Size([4]),
            indices_list=[[0, 1], [2, 3]],
            weights=[0.5, 1.5],
        )
    )
    with pytest.raises(ValueError):
        func(X)


def test_sum_prod_target_handles_multiple_batch_dims():
    X = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
            [[0.5, 4.0, -2.0, 1.0], [1.0, 2.0, 3.0, 4.0]],
        ]
    )
    func = SumProdTarget(
        SumProdTargetConfig(
            input_shape=torch.Size([4]),
            indices_list=[[0, 1], [2, 3]],
            weights=[0.5, 1.5],
        )
    )
    result = func(X)
    unscaled = torch.tensor(
        [
            [
                [0.5 * (1.0 * 2.0) + 1.5 * (3.0 * 4.0)],
                [0.5 * (2.0 * 3.0) + 1.5 * (4.0 * 5.0)],
            ],
            [
                [0.5 * (0.5 * 4.0) + 1.5 * (-2.0 * 1.0)],
                [0.5 * (1.0 * 2.0) + 1.5 * (3.0 * 4.0)],
            ],
        ]
    )
    expected = unscaled * func.scale
    assert torch.allclose(result, expected)
