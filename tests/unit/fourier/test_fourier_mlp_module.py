import pytest
import torch

from src.fourier import FourierMlpModule
from src.models.mlp import MLP
from src.models.mlp_config import MLPConfig


def _compute_fourier_products(x: torch.Tensor, indices: list[list[int]]) -> torch.Tensor:
    products = []
    for idxs in indices:
        if len(idxs) == 0:
            products.append(torch.ones(x.size(0), dtype=x.dtype, device=x.device))
        else:
            products.append(x[:, idxs].prod(dim=1))
    return torch.stack(products, dim=1)


def _build_mlp(
    *,
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    activation: str = "relu",
) -> MLP:
    config = MLPConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        start_activation=False,
        end_activation=False,
        bias=True,
    )
    return MLP(config)


def test_fourier_mlp_computes_expected_values() -> None:
    torch.manual_seed(123)
    input_dim = 3
    mlp = _build_mlp(input_dim=input_dim, hidden_dims=[4], output_dim=2)

    fourier_indices = [[0], [1, 2], []]
    model = FourierMlpModule(
        input_dim=input_dim,
        fourier_indices=fourier_indices,
        mlp=mlp,
        neuron_start_index=1,
        neuron_end_index=3,
    )

    batch = torch.randn(5, input_dim)
    outputs = model(batch)

    assert len(outputs) == 2

    with torch.no_grad():
        hidden = mlp.linear_layers[0](batch)
        hidden_act = mlp.net[1](hidden)
        final_out = mlp.linear_layers[1](hidden_act)

    fourier_products = _compute_fourier_products(batch, fourier_indices)

    # Hidden layer slice (neurons 1 and 2)
    hidden_slice = hidden_act[:, 1:3]
    expected_hidden = (
        hidden_slice.unsqueeze(2) * fourier_products.unsqueeze(1)
    ).mean(dim=0)

    assert outputs[0] is not None
    assert outputs[0].shape == expected_hidden.shape
    torch.testing.assert_close(outputs[0], expected_hidden)

    # Output layer slice (neuron 1 only)
    output_slice = final_out[:, 1:2]
    expected_output = (
        output_slice.unsqueeze(2) * fourier_products.unsqueeze(1)
    ).mean(dim=0)

    assert outputs[1] is not None
    assert outputs[1].shape == expected_output.shape
    torch.testing.assert_close(outputs[1], expected_output)


def test_fourier_mlp_returns_none_for_empty_ranges() -> None:
    input_dim = 2
    mlp = _build_mlp(input_dim=input_dim, hidden_dims=[3], output_dim=1)
    model = FourierMlpModule(
        input_dim=input_dim,
        fourier_indices=[[0]],
        mlp=mlp,
        neuron_start_index=5,
        neuron_end_index=7,
    )

    batch = torch.randn(4, input_dim)
    outputs = model(batch)

    assert outputs == (None, None)


def test_fourier_mlp_validates_fourier_indices() -> None:
    input_dim = 3
    mlp = _build_mlp(input_dim=input_dim, hidden_dims=[], output_dim=2)

    with pytest.raises(ValueError):
        FourierMlpModule(
            input_dim=input_dim,
            fourier_indices=[[0, 3]],
            mlp=mlp,
            neuron_start_index=0,
            neuron_end_index=2,
        )


def test_fourier_mlp_validates_input_dim_match() -> None:
    mlp = _build_mlp(input_dim=3, hidden_dims=[4], output_dim=2)

    with pytest.raises(ValueError):
        FourierMlpModule(
            input_dim=4,
            fourier_indices=[[0]],
            mlp=mlp,
            neuron_start_index=0,
            neuron_end_index=1,
        )


def test_fourier_mlp_handles_mup_linear_layers() -> None:
    try:
        import mup  # noqa: F401
    except ImportError:
        pytest.skip("mup not installed")

    input_dim = 2
    mlp = _build_mlp(input_dim=input_dim, hidden_dims=[3], output_dim=1)

    model = FourierMlpModule(
        input_dim=input_dim,
        fourier_indices=[[0], [1]],
        mlp=mlp,
        neuron_start_index=0,
        neuron_end_index=3,
    )

    batch = torch.randn(6, input_dim)
    outputs = model(batch)

    assert len(outputs) == 2
    assert outputs[0] is not None
    assert outputs[1] is not None

    with torch.no_grad():
        hidden = mlp.linear_layers[0](batch)
        hidden_act = mlp.net[1](hidden)
        final_out = mlp.linear_layers[1](hidden_act)

    fourier_products = _compute_fourier_products(batch, [[0], [1]])

    expected_hidden = (hidden_act.unsqueeze(2) * fourier_products.unsqueeze(1)).mean(dim=0)
    expected_output = (final_out.unsqueeze(2) * fourier_products.unsqueeze(1)).mean(dim=0)

    torch.testing.assert_close(outputs[0], expected_hidden)
    torch.testing.assert_close(outputs[1], expected_output)
