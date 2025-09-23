import pytest

from src.training.optimizers.configs.sgd import SgdConfig
from src.training.optimizers.configs.optimizer_config_registry import (
    OPTIMIZER_CONFIG_REGISTRY,
    build_optimizer_config,
    build_optimizer_config_from_dict,
)


def test_sgd_registered():
    assert "sgd" in OPTIMIZER_CONFIG_REGISTRY
    assert OPTIMIZER_CONFIG_REGISTRY["sgd"] is SgdConfig


def test_build_sgd_config():
    cfg = build_optimizer_config("sgd", lr=0.01, weight_decay=0.1)
    assert isinstance(cfg, SgdConfig)
    assert cfg.lr == pytest.approx(0.01)
    assert cfg.weight_decay == pytest.approx(0.1)
    assert cfg.optimizer_type == "sgd"


def test_json_roundtrip():
    original = SgdConfig(lr=0.5)
    json_str = original.to_json()
    restored = SgdConfig.from_json(json_str)
    assert restored == original


def test_build_from_dict():
    data = {"optimizer_type": "sgd", "lr": 1.23, "weight_decay": 0.2}
    cfg = build_optimizer_config_from_dict(data)
    assert isinstance(cfg, SgdConfig)
    assert cfg.lr == pytest.approx(1.23)
    assert cfg.weight_decay == pytest.approx(0.2)
