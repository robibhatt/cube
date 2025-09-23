import math
import torch
import pytest

from src.models.targets.simple_functions import StaircaseTarget
from src.models.targets.configs.staircase import StaircaseTargetConfig


def test_staircase_target_computes_expected():
    X = torch.tensor([[1.0, 2.0, -1.0], [1.0, -1.0, -1.0]])
    func = StaircaseTarget(StaircaseTargetConfig(input_shape=torch.Size([3]), k=3))
    result = func(X)
    expected = torch.tensor([
        [1.0 + 1.0 * 2.0 + 1.0 * 2.0 * -1.0],
        [1.0 + 1.0 * -1.0 + 1.0 * -1.0 * -1.0],
    ]) / math.sqrt(3)
    assert torch.allclose(result, expected)


def test_staircase_target_dim_error():
    X = torch.tensor([[1.0, 2.0]])
    func = StaircaseTarget(StaircaseTargetConfig(input_shape=torch.Size([3]), k=3))
    with pytest.raises(ValueError):
        func(X)


def test_staircase_target_handles_multiple_batch_dims():
    X = torch.tensor([
        [[1.0, 2.0, -1.0], [1.0, -1.0, 2.0]],
        [[1.0, -1.0, -1.0], [1.0, 2.0, 3.0]],
    ])
    func = StaircaseTarget(StaircaseTargetConfig(input_shape=torch.Size([3]), k=3))
    result = func(X)
    expected = torch.tensor([
        [[1.0 + 1.0 * 2.0 + 1.0 * 2.0 * -1.0], [1.0 + 1.0 * -1.0 + 1.0 * -1.0 * 2.0]],
        [[1.0 + 1.0 * -1.0 + 1.0 * -1.0 * -1.0], [1.0 + 1.0 * 2.0 + 1.0 * 2.0 * 3.0]],
    ]) / math.sqrt(3)
    assert torch.allclose(result, expected)
