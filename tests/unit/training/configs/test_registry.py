import pytest

from src.training.optimizers.configs.optimizer_config_registry import build_optimizer_config


def test_unknown_optimizer_config():
    with pytest.raises(ValueError):
        build_optimizer_config("Unknown")

