import pytest
import torch

from src.data.cube_distribution_config import CubeDistributionConfig


def test_cube_config_validates_and_sets_defaults():
    cfg = CubeDistributionConfig(
        input_dim=2,
        indices_list=[[0], [1]],
        weights=[1.0, 2.0],
        normalize=False,
        noise_mean=1.0,
        noise_std=0.5,
    )
    assert cfg.distribution_type == "CubeDistribution"
    assert cfg.input_shape == torch.Size([2])
    assert cfg.output_shape == torch.Size([1])
    assert cfg.noise_mean == pytest.approx(1.0)
    assert cfg.noise_std == pytest.approx(0.5)
    assert cfg.target_function_config.indices_list == [[0], [1]]


def test_cube_config_json_roundtrip():
    cfg = CubeDistributionConfig(
        input_dim=3,
        indices_list=[[0], [1, 2]],
        weights=[0.5, 1.5],
        normalize=True,
        noise_mean=0.0,
        noise_std=1.0,
    )
    json_str = cfg.to_json()
    restored = CubeDistributionConfig.from_json(json_str)
    assert restored == cfg


def test_cube_config_validation_errors():
    with pytest.raises(ValueError, match="positive integer"):
        CubeDistributionConfig(
            input_dim=0,
            indices_list=[[0]],
            weights=[1.0],
        )

    with pytest.raises(ValueError, match="non-empty list"):
        CubeDistributionConfig(
            input_dim=2,
            indices_list=[],
            weights=[],
        )

    with pytest.raises(ValueError, match="same length"):
        CubeDistributionConfig(
            input_dim=2,
            indices_list=[[0]],
            weights=[1.0, 2.0],
        )

    with pytest.raises(ValueError, match="non-empty"):
        CubeDistributionConfig(
            input_dim=2,
            indices_list=[[]],
            weights=[1.0],
        )

    with pytest.raises(ValueError, match=">= input_dim"):
        CubeDistributionConfig(
            input_dim=2,
            indices_list=[[2]],
            weights=[1.0],
        )
