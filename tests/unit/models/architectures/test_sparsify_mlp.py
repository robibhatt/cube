import torch
import pytest

from src.models.mlp import MLP
from src.models.mlp_config import MLPConfig
from src.models.sparsify_mlp import sparsify_mlp


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
