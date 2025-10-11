import pytest

from src.training.sgd_config import SgdConfig


def test_to_dict_without_optional_fields():
    cfg = SgdConfig(lr=0.05)
    assert cfg.to_dict() == {"lr": 0.05, "mup": True, "weight_decay": 0.0}


def test_from_dict_partial():
    data = {"lr": 0.5}
    cfg = SgdConfig.from_dict(data)
    assert cfg.lr == pytest.approx(0.5)
    assert cfg.mup is True
    assert cfg.weight_decay == pytest.approx(0.0)
