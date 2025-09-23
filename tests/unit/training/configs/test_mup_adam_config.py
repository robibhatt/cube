import pytest

from src.training.optimizers.configs.adam import AdamConfig
from src.training.optimizers.configs.optimizer_config_registry import (
    OPTIMIZER_CONFIG_REGISTRY,
    build_optimizer_config,
    build_optimizer_config_from_dict,
)


def test_adam_registered():
    assert "Adam" in OPTIMIZER_CONFIG_REGISTRY
    assert OPTIMIZER_CONFIG_REGISTRY["Adam"] is AdamConfig


def test_build_mup_adam_config():
    cfg = build_optimizer_config("Adam", lr=0.01, mup=True, weight_decay=0.1)
    assert isinstance(cfg, AdamConfig)
    assert cfg.lr == pytest.approx(0.01)
    assert cfg.mup is True
    assert cfg.weight_decay == pytest.approx(0.1)
    assert cfg.optimizer_type == "Adam"


def test_json_roundtrip():
    original = AdamConfig(lr=0.5, mup=True)
    json_str = original.to_json()
    restored = AdamConfig.from_json(json_str)
    assert restored == original


def test_build_from_dict():
    data = {"optimizer_type": "Adam", "lr": 1.23, "mup": True, "weight_decay": 0.2}
    cfg = build_optimizer_config_from_dict(data)
    assert isinstance(cfg, AdamConfig)
    assert cfg.mup is True
    assert cfg.lr == pytest.approx(1.23)
    assert cfg.weight_decay == pytest.approx(0.2)
