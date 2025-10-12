import itertools
import json

import pytest
import torch
import torch.nn.functional as F

from src.fourier import detect_mlp_fourier_components
from src.models import MLP, MLPConfig


def _build_test_mlp() -> MLP:
    config = MLPConfig(
        input_dim=2,
        hidden_dims=[2],
        activation="relu",
        output_dim=1,
        start_activation=False,
        end_activation=False,
    )

    model = MLP(config)
    with torch.no_grad():
        hidden = model.linear_layers[0]
        hidden.weight.copy_(torch.tensor([[1.0, -0.5], [0.5, 1.5]]))
        hidden.bias.copy_(torch.tensor([0.25, -0.75]))

        output = model.linear_layers[1]
        output.weight.copy_(torch.tensor([[1.2, -0.8]]))
        output.bias.copy_(torch.tensor([0.1]))

    return model


def _sample_hypercube(dim: int, num_samples: int, seed: int) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    samples = torch.randint(0, 2, (num_samples, dim), generator=generator, dtype=torch.float32)
    return samples.mul_(2).sub_(1)


def _manual_fourier_components(
    model: MLP, max_degree: int, epsilon: float, *, num_samples: int, seed: int
):
    inputs = _sample_hypercube(model.config.input_dim, num_samples, seed)
    hidden_layer, output_layer = model.linear_layers

    with torch.no_grad():
        hidden_pre = F.linear(inputs, hidden_layer.weight, hidden_layer.bias)
        hidden_post = torch.relu(hidden_pre)
        output_pre = F.linear(hidden_post, output_layer.weight, output_layer.bias)

    activations = [hidden_post, output_pre]

    subsets = [()]  # empty subset
    for degree in range(1, min(model.config.input_dim, max_degree) + 1):
        subsets.extend(itertools.combinations(range(model.config.input_dim), degree))

    features = {}
    for subset in subsets:
        if not subset:
            features[subset] = torch.ones(inputs.shape[0])
        else:
            features[subset] = inputs[:, subset].prod(dim=1)

    expected = []
    for layer_idx, activation in enumerate(activations):
        for neuron_idx in range(activation.shape[1]):
            comps = []
            values = activation[:, neuron_idx]
            for subset in subsets:
                dot_value = float((features[subset] * values).mean().item())
                if abs(dot_value) >= epsilon:
                    comps.append({"coordinates": list(subset), "value": dot_value})
            comps.sort(key=lambda item: abs(item["value"]), reverse=True)
            if comps:
                expected.append({"layer": layer_idx, "index": neuron_idx, "components": comps})
    return expected


def test_fourier_detector_matches_manual_computation(tmp_path):
    model = _build_test_mlp()
    seed = 123
    num_samples = 64
    output_path = detect_mlp_fourier_components(
        model,
        max_degree=2,
        epsilon=0.0,
        output_directory=tmp_path,
        num_samples=num_samples,
        random_seed=seed,
    )

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["max_degree"] == 2
    assert data["epsilon"] == 0.0
    assert data["num_samples"] == num_samples
    assert data["random_seed"] == seed

    expected = _manual_fourier_components(
        model, max_degree=2, epsilon=0.0, num_samples=num_samples, seed=seed
    )

    assert data["results"]
    assert len(data["results"]) == len(expected)

    for actual, manual in zip(data["results"], expected):
        assert actual["layer"] == manual["layer"]
        assert actual["index"] == manual["index"]
        assert len(actual["components"]) == len(manual["components"])
        for actual_comp, manual_comp in zip(actual["components"], manual["components"]):
            assert actual_comp["coordinates"] == manual_comp["coordinates"]
            assert actual_comp["value"] == pytest.approx(manual_comp["value"])


def test_fourier_detector_applies_epsilon_threshold(tmp_path):
    model = _build_test_mlp()
    seed = 999
    num_samples = 128

    output_path = detect_mlp_fourier_components(
        model,
        max_degree=2,
        epsilon=0.5,
        output_directory=tmp_path,
        num_samples=num_samples,
        random_seed=seed,
    )
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["num_samples"] == num_samples
    assert data["random_seed"] == seed

    expected = _manual_fourier_components(
        model, max_degree=2, epsilon=0.5, num_samples=num_samples, seed=seed
    )

    assert data["results"] == expected
    for entry in data["results"]:
        for component in entry["components"]:
            assert abs(component["value"]) >= 0.5
