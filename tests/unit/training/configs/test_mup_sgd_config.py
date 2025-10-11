import pytest

from src.training.sgd_config import SgdConfig


def test_defaults():
    cfg = SgdConfig(lr=0.01)
    assert cfg.mup is True
    assert cfg.weight_decay == pytest.approx(0.0)


def test_json_roundtrip():
    original = SgdConfig(lr=0.5, mup=True, weight_decay=0.2)
    json_str = original.to_json()
    restored = SgdConfig.from_json(json_str)
    assert restored == original


def test_dict_roundtrip():
    data = {"lr": 1.23, "mup": False, "weight_decay": 0.4}
    cfg = SgdConfig.from_dict(data)
    assert cfg.lr == pytest.approx(1.23)
    assert cfg.mup is False
    assert cfg.weight_decay == pytest.approx(0.4)
    assert cfg.to_dict() == data
