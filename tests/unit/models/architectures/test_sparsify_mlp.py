import torch
import pytest

from src.models.mlp import MLP
from src.models.mlp_config import MLPConfig
from src.models.sparsify_mlp import mse_diff, sparsify_mlp


@pytest.fixture()
def small_mlp_config() -> MLPConfig:
    return MLPConfig(input_dim=2, hidden_dims=[3])


@pytest.fixture()
def populated_mlp(small_mlp_config: MLPConfig) -> MLP:
    model = MLP(small_mlp_config)
    with torch.no_grad():
        first = model.linear_layers[0]
        first.weight.data.copy_(
            torch.tensor(
                [
                    [0.05, -0.25],
                    [0.4, -0.8],
                    [-0.02, 0.12],
                ],
                dtype=first.weight.dtype,
            )
        )
        last = model.linear_layers[-1]
        last.weight.data.copy_(
            torch.tensor([[0.18, -0.35, 0.9]], dtype=last.weight.dtype)
        )
    return model


def test_sparsify_mlp_zeroes_weights_below_threshold(populated_mlp: MLP) -> None:
    threshold = 0.2

    result = sparsify_mlp(populated_mlp, threshold)

    expected_first = torch.tensor(
        [
            [0.0, -0.25],
            [0.4, -0.8],
            [0.0, 0.0],
        ],
        dtype=populated_mlp.linear_layers[0].weight.dtype,
    )
    expected_last = torch.tensor(
        [[0.0, -0.35, 0.9]], dtype=populated_mlp.linear_layers[-1].weight.dtype
    )

    first_weight = result.linear_layers[0].weight.detach()
    last_weight = result.linear_layers[-1].weight.detach()

    assert torch.equal(first_weight, expected_first)
    assert torch.equal(last_weight, expected_last)


def test_sparsify_mlp_does_not_modify_original(populated_mlp: MLP) -> None:
    original_first = populated_mlp.linear_layers[0].weight.clone()

    sparsify_mlp(populated_mlp, threshold=0.3)

    assert torch.equal(populated_mlp.linear_layers[0].weight, original_first)


def test_sparsify_mlp_requires_positive_threshold(populated_mlp: MLP) -> None:
    with pytest.raises(ValueError):
        sparsify_mlp(populated_mlp, threshold=0.0)

    with pytest.raises(ValueError):
        sparsify_mlp(populated_mlp, threshold=-0.1)


def test_mse_diff_returns_zero_for_identical_models(small_mlp_config: MLPConfig) -> None:
    mlp_a = MLP(small_mlp_config)
    mlp_b = MLP(small_mlp_config)
    mlp_b.load_state_dict(mlp_a.state_dict())

    assert mse_diff(10, mlp_a, mlp_b) == pytest.approx(0.0)


def test_mse_diff_detects_bias_shift() -> None:
    config = MLPConfig(input_dim=2, hidden_dims=[])
    mlp_a = MLP(config)
    mlp_b = MLP(config)

    with torch.no_grad():
        last_a = mlp_a.linear_layers[-1]
        last_b = mlp_b.linear_layers[-1]
        last_a.weight.zero_()
        last_b.weight.zero_()
        if last_a.bias is not None and last_b.bias is not None:
            last_a.bias.fill_(0.0)
            last_b.bias.fill_(1.0)

    assert mse_diff(32, mlp_a, mlp_b) == pytest.approx(1.0, rel=1e-6, abs=1e-6)


def test_mse_diff_validates_arguments(small_mlp_config: MLPConfig) -> None:
    mlp = MLP(small_mlp_config)
    other_config = MLPConfig(input_dim=3, hidden_dims=[2])
    other_mlp = MLP(other_config)

    with pytest.raises(ValueError):
        mse_diff(0, mlp, mlp)

    with pytest.raises(TypeError):
        mse_diff(1, mlp, object())  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        mse_diff(1, mlp, other_mlp)
