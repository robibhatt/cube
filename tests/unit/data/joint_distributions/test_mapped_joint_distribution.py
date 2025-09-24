import pytest
import torch

from src.data.joint_distributions import create_joint_distribution
from src.data.joint_distributions.configs.mapped_joint_distribution import (
    MappedJointDistributionConfig,
)
from tests.unit.data.conftest import gaussian_base
from src.data.joint_distributions.configs.gaussian import GaussianConfig
from src.models.targets.configs.sum_prod import SumProdTargetConfig
from src.models.targets.sum_prod import SumProdTarget

@pytest.fixture
def input_shape(gaussian_base):
    # Grab the input_shape from the base distribution
    return gaussian_base.input_shape

@pytest.mark.parametrize(
    "indices_list",
    [
        [[0], [1]],
        [[0, 0], [1, 1]],
    ],
)
def test_initialization(gaussian_base, indices_list):
    cfg_tf = SumProdTargetConfig(
        input_shape=gaussian_base.input_shape,
        indices_list=indices_list,
        weights=[1.0] * len(indices_list),
        normalize=False,
    )
    cfg = MappedJointDistributionConfig(
        distribution_config=GaussianConfig(
            input_shape=gaussian_base.config.input_shape, mean=0.0, std=1.0
        ),
        target_function_config=cfg_tf,
    )
    dist = create_joint_distribution(cfg, torch.device("cpu"))

    # input_shape should mirror the base distribution
    assert dist.input_shape == gaussian_base.input_shape
    # output_shape should come from the target function
    assert dist.output_shape == cfg_tf.output_shape
    assert isinstance(dist.distribution.config, GaussianConfig)
    assert isinstance(dist.target_function, SumProdTarget)

@pytest.mark.parametrize(
    "indices_list",
    [
        [[0], [1]],
        [[0, 0], [1, 1]],
    ],
)
def test_sample_shape(gaussian_base, indices_list):
    cfg_tf = SumProdTargetConfig(
        input_shape=gaussian_base.input_shape,
        indices_list=indices_list,
        weights=[1.0] * len(indices_list),
        normalize=False,
    )
    cfg = MappedJointDistributionConfig(
        distribution_config=GaussianConfig(
            input_shape=gaussian_base.config.input_shape, mean=0.0, std=1.0
        ),
        target_function_config=cfg_tf,
    )
    dist = create_joint_distribution(cfg, torch.device("cpu"))
    n_samples = 100
    X, y = dist.sample(n_samples, seed=0)

    assert X.shape == (n_samples, *dist.input_shape)
    assert y.shape == (n_samples, *dist.output_shape)
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)

@pytest.mark.parametrize(
    "indices_list, expected",
    [
        ([[0], [1]], torch.tensor([[3.0], [7.0]])),
        ([[0, 0], [1, 1]], torch.tensor([[5.0], [25.0]])),
    ],
)
def test_target_function_output(gaussian_base, indices_list, expected):
    cfg_tf = SumProdTargetConfig(
        input_shape=gaussian_base.input_shape,
        indices_list=indices_list,
        weights=[1.0] * len(indices_list),
        normalize=False,
    )
    cfg = MappedJointDistributionConfig(
        distribution_config=GaussianConfig(
            input_shape=gaussian_base.config.input_shape, mean=0.0, std=1.0
        ),
        target_function_config=cfg_tf,
    )
    dist = create_joint_distribution(cfg, torch.device("cpu"))
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = dist.target_function(X)

    assert y.shape == (2, 1)
    assert torch.allclose(y, expected)
