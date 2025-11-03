from unittest import mock

import pytest
import torch
from torch.testing import assert_close

from src.models.mlp import MLP
from src.models.mlp_config import MLPConfig
from src.models.pruned_mlp import prune
from src.models.sparsify import (
    binary_search_sparsify_threshold,
    mse_diff,
    sparsify_mlp,
)
import src.models.sparsify as sparsify_module


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


def test_sparsify_mlp_thresholds_by_absolute_value(populated_mlp: MLP) -> None:
    with torch.no_grad():
        first = populated_mlp.linear_layers[0]
        first.weight.data[0, 0] = -0.19
        first.weight.data[0, 1] = -0.21

    threshold = 0.2

    result = sparsify_mlp(populated_mlp, threshold)
    first_weight = result.linear_layers[0].weight.detach()

    assert first_weight[0, 0].item() == pytest.approx(0.0)
    assert first_weight[0, 1].item() == pytest.approx(-0.21)


def test_sparsify_mlp_does_not_modify_original(populated_mlp: MLP) -> None:
    original_first = populated_mlp.linear_layers[0].weight.clone()

    sparsify_mlp(populated_mlp, threshold=0.3)

    assert torch.equal(populated_mlp.linear_layers[0].weight, original_first)


def test_sparsify_mlp_requires_positive_threshold(populated_mlp: MLP) -> None:
    with pytest.raises(ValueError):
        sparsify_mlp(populated_mlp, threshold=0.0)

    with pytest.raises(ValueError):
        sparsify_mlp(populated_mlp, threshold=-0.1)


def test_prune_returns_connected_inputs(populated_mlp: MLP) -> None:
    with torch.no_grad():
        first = populated_mlp.linear_layers[0]
        first.weight.zero_()
        first.weight[1, 1] = 0.5

        last = populated_mlp.linear_layers[-1]
        last.weight.zero_()
        last.weight[0, 1] = -0.75

    pruned, connected = prune(populated_mlp)

    assert pruned.config.hidden_dims == [1]
    assert connected == {1}


def test_prune_handles_models_without_hidden_layers() -> None:
    config = MLPConfig(input_dim=3, hidden_dims=[])
    model = MLP(config)

    with torch.no_grad():
        out = model.linear_layers[0]
        out.weight.zero_()
        out.weight[0, 1] = 0.3
        out.weight[0, 2] = -0.2

    pruned, connected = prune(model)

    assert pruned.config.hidden_dims == []
    assert connected == {1, 2}


def test_prune_preserves_outputs_after_sparsification() -> None:
    torch.manual_seed(42)
    config = MLPConfig(input_dim=4, hidden_dims=[8, 6, 5])
    model = MLP(config)

    with torch.no_grad():
        for param in model.parameters():
            param.uniform_(-0.5, 0.5)

    sparsified = sparsify_mlp(model, threshold=0.25)
    pruned, _ = prune(sparsified)

    inputs = torch.randint(0, 2, (128, config.input_dim), dtype=torch.float32)
    inputs.mul_(2.0).sub_(1.0)

    with torch.no_grad():
        expected = sparsified(inputs)
        actual = pruned(inputs)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_prune_preserves_config_attributes_and_outputs() -> None:
    torch.manual_seed(7)
    config = MLPConfig(input_dim=5, hidden_dims=[9, 7, 4])
    config.base_width = 32
    config.annotation = {"keep": True}

    model = MLP(config)

    with torch.no_grad():
        for param in model.parameters():
            param.normal_(mean=0.0, std=0.2)

    sparsified = sparsify_mlp(model, threshold=0.15)
    pruned, _ = prune(sparsified)

    assert hasattr(pruned.config, "base_width")
    assert pruned.config.base_width == config.base_width
    assert getattr(pruned.config, "annotation", None) == config.annotation

    inputs = torch.randn(256, config.input_dim)

    with torch.no_grad():
        expected = sparsified(inputs)
        actual = pruned(inputs)

    assert_close(actual, expected, rtol=0.0, atol=1e-7)


def test_prune_reduces_structure_but_matches_outputs() -> None:
    config = MLPConfig(input_dim=3, hidden_dims=[4, 5])
    model = MLP(config)

    with torch.no_grad():
        first = model.linear_layers[0]
        second = model.linear_layers[1]
        out = model.linear_layers[-1]

        first.weight.zero_()
        first.bias.zero_()
        first.weight[2, 1] = 0.75

        second.weight.zero_()
        second.bias.zero_()
        second.weight[4, 2] = -0.5

        out.weight.zero_()
        if out.bias is not None:
            out.bias.zero_()
        out.weight[0, 4] = 1.2

    sparsified = sparsify_mlp(model, threshold=0.05)
    pruned, connected = prune(sparsified)

    assert sparsified.config.hidden_dims == [4, 5]
    assert pruned.config.hidden_dims == [1, 1]
    assert connected == {1}

    first_sparse = sparsified.linear_layers[0].weight
    first_pruned = pruned.linear_layers[0].weight
    second_sparse = sparsified.linear_layers[1].weight
    second_pruned = pruned.linear_layers[1].weight

    assert first_sparse.shape == (4, 3)
    assert first_pruned.shape == (1, 3)
    assert second_sparse.shape == (5, 4)
    assert second_pruned.shape == (1, 1)

    inputs = torch.randn(32, config.input_dim)

    with torch.no_grad():
        expected = sparsified(inputs)
        actual = pruned(inputs)

    assert_close(actual, expected, rtol=0.0, atol=1e-8)


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


def test_mse_diff_accepts_explicit_inputs() -> None:
    config = MLPConfig(input_dim=2, hidden_dims=[])
    mlp_a = MLP(config)
    mlp_b = MLP(config)

    with torch.no_grad():
        out_a = mlp_a.linear_layers[0]
        out_a.weight.zero_()
        out_a.bias.zero_()

        out_b = mlp_b.linear_layers[0]
        out_b.weight.zero_()
        out_b.bias.fill_(1.0)

    provided_inputs = torch.zeros((3, config.input_dim))
    mse = mse_diff(3, mlp_a, mlp_b, inputs=provided_inputs)

    assert mse == pytest.approx(1.0)


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

    with pytest.raises(TypeError):
        mse_diff(1, mlp, mlp, inputs=[1, 2])  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        mse_diff(1, mlp, mlp, inputs=torch.zeros(2))

    with pytest.raises(ValueError):
        mse_diff(2, mlp, mlp, inputs=torch.zeros((1, mlp.config.input_dim)))

    with pytest.raises(ValueError):
        mse_diff(1, mlp, mlp, inputs=torch.zeros((1, mlp.config.input_dim + 1)))


def test_binary_search_sparsify_threshold_respects_budget(populated_mlp: MLP) -> None:
    torch.manual_seed(0)
    mse_budget = 0.05

    threshold = binary_search_sparsify_threshold(
        populated_mlp,
        mse_budget,
        sample_count=512,
        relative_tolerance=0.02,
    )

    sparsified = sparsify_mlp(populated_mlp, threshold)
    torch.manual_seed(0)
    measured_mse = mse_diff(2000, populated_mlp, sparsified)

    assert measured_mse <= mse_budget * 1.2

    torch.manual_seed(0)
    slightly_higher = sparsify_mlp(populated_mlp, threshold * 1.1 + 1e-6)
    torch.manual_seed(0)
    higher_mse = mse_diff(2000, populated_mlp, slightly_higher)

    assert higher_mse >= measured_mse


def test_binary_search_sparsify_threshold_validates_inputs(populated_mlp: MLP) -> None:
    with pytest.raises(ValueError):
        binary_search_sparsify_threshold(populated_mlp, 0.0, sample_count=32)

    with pytest.raises(ValueError):
        binary_search_sparsify_threshold(populated_mlp, 0.1, sample_count=0)

    with pytest.raises(ValueError):
        binary_search_sparsify_threshold(
            populated_mlp,
            0.1,
            sample_count=32,
            max_iterations=0,
        )

    with pytest.raises(ValueError):
        binary_search_sparsify_threshold(
            populated_mlp,
            0.1,
            sample_count=32,
            relative_tolerance=0.0,
        )

    with pytest.raises(TypeError):
        binary_search_sparsify_threshold(object(), 0.1, sample_count=32)  # type: ignore[arg-type]


def test_binary_search_returns_zero_for_zero_weights(small_mlp_config: MLPConfig) -> None:
    model = MLP(small_mlp_config)
    with torch.no_grad():
        for layer in model.linear_layers:
            weight = getattr(layer, "weight", None)
            if weight is not None:
                weight.zero_()

    threshold = binary_search_sparsify_threshold(model, 0.1, sample_count=16)

    assert threshold == 0.0


def test_binary_search_reuses_single_sample(populated_mlp: MLP, monkeypatch: pytest.MonkeyPatch) -> None:
    call_counter = mock.Mock()

    def fake_sample(*args, **kwargs):
        call_counter()
        sample_count = args[0]
        input_dim = args[1]
        return torch.ones((sample_count, input_dim), dtype=torch.float32)

    monkeypatch.setattr(
        sparsify_module,
        "_generate_binary_hypercube_samples",
        fake_sample,
    )

    binary_search_sparsify_threshold(populated_mlp, 0.05, sample_count=8)

    assert call_counter.call_count == 1
