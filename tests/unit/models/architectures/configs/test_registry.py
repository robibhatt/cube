import pytest
from src.models.architectures.configs.model_config_registry import build_model_config


def test_unknown_model_config():
    with pytest.raises(ValueError):
        build_model_config("UnknownConfig")
