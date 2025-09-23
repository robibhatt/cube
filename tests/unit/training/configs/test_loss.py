import torch
import pytest

from src.training.loss.configs.loss import LossConfig


def test_regression_evaluator_matches_loss():
    cfg = LossConfig(name="MSELoss", eval_type="regression")
    loss_fn = cfg.build()
    evaluator = cfg.get_evaluator()

    pred = torch.tensor([[0.0], [1.0]])
    target = torch.tensor([[1.0], [1.0]])

    assert torch.isclose(loss_fn(pred, target), evaluator(pred, target))


def test_binary_pm1_evaluator():
    cfg = LossConfig(name="MSELoss", eval_type="binary_pm1")
    evaluator = cfg.get_evaluator()

    pred = torch.tensor([1.0, -0.5, 0.2])
    target = torch.tensor([1.0, -1.0, 1.0])
    assert evaluator(pred, target).item() == pytest.approx(0.0)

    pred2 = torch.tensor([1.0, -0.5, -0.2])
    target2 = torch.tensor([1.0, -1.0, 1.0])
    assert evaluator(pred2, target2).item() == pytest.approx(1 / 3)


def test_binary_01_evaluator():
    cfg = LossConfig(name="MSELoss", eval_type="binary_01")
    evaluator = cfg.get_evaluator()

    pred = torch.tensor([0.2, -0.1, 1.2])
    target = torch.tensor([0.0, 1.0, 1.0])
    assert evaluator(pred, target).item() == pytest.approx(2 / 3)

