import json
from pathlib import Path

import pytest
import torch

from src.fourier import FourierMlp
from src.models.mlp import MLP
from src.models.mlp_config import MLPConfig


def _build_simple_mlp() -> MLP:
    config = MLPConfig(
        input_dim=2,
        output_dim=1,
        hidden_dims=[2],
        activation="relu",
        start_activation=False,
        end_activation=False,
        bias=False,
    )
    mlp = MLP(config)
    with torch.no_grad():
        mlp.linear_layers[0].weight.copy_(torch.tensor([[1.0, -1.0], [0.5, 2.0]]))
        mlp.linear_layers[1].weight.copy_(torch.tensor([[1.5, -0.75]]))
    return mlp


def test_initialization_creates_expected_files(tmp_path: Path) -> None:
    mlp = _build_simple_mlp()
    fourier = FourierMlp(mlp, tmp_path, sample_size=4, batch_size=2)

    expected_dir = tmp_path / FourierMlp.FOURIER_SUBDIR_NAME
    assert expected_dir.exists()

    config_path = expected_dir / FourierMlp.CONFIG_FILENAME
    metadata_path = expected_dir / FourierMlp.METADATA_FILENAME
    checkpoint_dir = expected_dir / FourierMlp.CHECKPOINT_SUBDIR
    second_moment_path = expected_dir / FourierMlp.SECOND_MOMENTS_FILENAME

    assert config_path.exists()
    assert metadata_path.exists()
    assert (checkpoint_dir / "checkpoint.pth").exists()
    assert second_moment_path.exists()

    with open(metadata_path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    assert metadata["sample_size"] == 4
    assert metadata["batch_size"] == 2

    payload = torch.load(second_moment_path, map_location="cpu")
    assert payload["sample_size"] == 4
    assert set(payload["layer_second_moments"].keys()) == {"1", "2"}

    with pytest.raises(FileExistsError):
        FourierMlp(mlp, tmp_path)


def test_from_dir_reuses_cached_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mlp = _build_simple_mlp()
    fourier = FourierMlp(mlp, tmp_path, sample_size=4, batch_size=2)

    # Pre-compute and cache a coefficient
    samples = torch.tensor(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
        dtype=torch.float32,
    )

    batches = [samples[:2], samples[2:]]
    call_count = {"value": 0}

    def _draw(batch_size: int) -> torch.Tensor:
        batch = batches[call_count["value"]]
        call_count["value"] += 1
        assert batch.shape[0] == batch_size
        return batch.clone()

    fourier._draw_boolean_batch = _draw  # type: ignore[method-assign]
    value = fourier.get_fourier_coefficient([0, 1], layer_index=1, neuron_index=0)
    assert isinstance(value, float)

    restored = FourierMlp.from_dir(tmp_path)

    def _fail(*args, **kwargs):  # pragma: no cover - ensured by assertion
        raise AssertionError("cache should have been used")

    monkeypatch.setattr(restored, "_compute_and_store_coefficients", _fail)
    cached = restored.get_fourier_coefficient([1, 0], layer_index=1, neuron_index=0)
    assert pytest.approx(cached, rel=1e-6) == value

    original_moment = fourier.get_neuron_second_moment(layer_index=1, neuron_index=0)
    restored_moment = restored.get_neuron_second_moment(layer_index=1, neuron_index=0)
    assert pytest.approx(restored_moment, rel=1e-6) == original_moment


def test_fourier_coefficient_matches_manual_computation(tmp_path: Path) -> None:
    mlp = _build_simple_mlp()
    fourier = FourierMlp(mlp, tmp_path, sample_size=4, batch_size=2)

    samples = torch.tensor(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
        dtype=torch.float32,
    )
    batches = [samples[:2], samples[2:]]
    call_state = {"idx": 0}

    def _draw(batch_size: int) -> torch.Tensor:
        batch = batches[call_state["idx"]]
        call_state["idx"] += 1
        return batch.clone()

    fourier._draw_boolean_batch = _draw  # type: ignore[method-assign]

    coeff = fourier.get_fourier_coefficient([0, 1], layer_index=1, neuron_index=0)

    with torch.no_grad():
        hidden_linear = mlp.linear_layers[0](samples)
        hidden_activation = torch.relu(hidden_linear)
    fourier_term = (samples[:, 0] * samples[:, 1]).to(hidden_activation.dtype)
    expected = (hidden_activation[:, 0] * fourier_term).mean().item()

    assert pytest.approx(coeff, rel=1e-6) == expected


def test_second_moment_matches_manual_computation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mlp = _build_simple_mlp()

    samples = torch.tensor(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
        dtype=torch.float32,
    )
    batches = [samples[:2], samples[2:]]
    call_state = {"idx": 0}

    def _draw(self: FourierMlp, batch_size: int) -> torch.Tensor:
        batch = batches[call_state["idx"]]
        call_state["idx"] += 1
        assert batch.shape[0] == batch_size
        return batch.clone()

    monkeypatch.setattr(FourierMlp, "_draw_boolean_batch", _draw, raising=False)

    fourier = FourierMlp(mlp, tmp_path, sample_size=4, batch_size=2)

    with torch.no_grad():
        hidden_linear = mlp.linear_layers[0](samples)
        hidden_activation = torch.relu(hidden_linear)
        output_linear = mlp.linear_layers[1](hidden_activation)

    expected_hidden = hidden_activation.pow(2).mean(dim=0)
    expected_output = output_linear.pow(2).mean(dim=0)

    for neuron_idx, expected in enumerate(expected_hidden):
        value = fourier.get_neuron_second_moment(layer_index=1, neuron_index=neuron_idx)
        assert pytest.approx(value, rel=1e-6) == expected.item()

    value = fourier.get_neuron_second_moment(layer_index=2, neuron_index=0)
    assert pytest.approx(value, rel=1e-6) == expected_output.item()

def test_invalid_index_raises(tmp_path: Path) -> None:
    mlp = _build_simple_mlp()
    fourier = FourierMlp(mlp, tmp_path, sample_size=4, batch_size=2)

    with pytest.raises(ValueError):
        fourier.get_fourier_coefficient([5], layer_index=1, neuron_index=0)

