"""Utilities for analysing Fourier components of MLP neuron activations."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from mup import Linear as MuLinear, MuReadout

from src.models.mlp import MLP


LinearLike = (nn.Linear, MuLinear, MuReadout)


def _sample_hypercube_points(
    dim: int, num_samples: int, device: torch.device, *, generator: torch.Generator
) -> torch.Tensor:
    """Sample ``num_samples`` inputs from the ``{±1}^dim`` hypercube."""

    if dim <= 0:
        raise ValueError("Input dimension must be positive to sample hypercube points")
    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")

    samples = torch.randint(
        0,
        2,
        (num_samples, dim),
        generator=generator,
        dtype=torch.float32,
    )
    samples = samples.mul_(2).sub_(1)
    return samples.to(device)


def _fourier_subsets(dim: int, max_degree: int) -> List[Tuple[int, ...]]:
    """Enumerate coordinate subsets representing the Fourier basis up to ``max_degree``."""

    if max_degree < 0:
        raise ValueError("max_degree must be non-negative")

    capped_degree = min(dim, max_degree)
    subsets: List[Tuple[int, ...]] = [()]  # Empty set represents the constant basis element.
    for degree in range(1, capped_degree + 1):
        subsets.extend(itertools.combinations(range(dim), degree))
    return subsets


def _compute_basis_features(
    inputs: torch.Tensor, subsets: Sequence[Tuple[int, ...]]
) -> Dict[Tuple[int, ...], torch.Tensor]:
    """Compute values of each Fourier basis element on ``inputs``."""

    features: Dict[Tuple[int, ...], torch.Tensor] = {}
    num_samples = inputs.shape[0]
    device = inputs.device
    ones = torch.ones(num_samples, device=device)
    for subset in subsets:
        if not subset:
            features[subset] = ones
        else:
            subset_tensor = inputs[:, subset]
            features[subset] = subset_tensor.prod(dim=1)
    return features


def _collect_neuron_activations(model: MLP, inputs: torch.Tensor) -> List[torch.Tensor]:
    """Return activations for each linear layer in order of appearance."""

    activations: List[torch.Tensor] = []
    pending_value: torch.Tensor | None = None
    x = inputs

    for module in model.net:  # ``MLP`` always exposes ``net`` as ``nn.Sequential``
        x = module(x)
        if isinstance(module, LinearLike):
            if isinstance(module, MuReadout):
                # ``MuReadout`` rescales its input internally to preserve μP
                # behaviour.  For the purposes of the Fourier analysis we need
                # activations that correspond to a standard linear readout so we
                # undo the scaling before recording the activation.
                scale = float(module.width_mult())
                output_mult = float(getattr(module, "output_mult", 1.0) or 1.0)
                factor = scale / output_mult
                if module.bias is not None:
                    bias = module.bias.detach()
                    pending_value = ((x - bias) * factor + bias).detach()
                else:
                    pending_value = (x * factor).detach()
            else:
                pending_value = x.detach()
        else:
            if pending_value is not None:
                activations.append(x.detach())
                pending_value = None

    if pending_value is not None:
        activations.append(pending_value)

    return activations


def detect_mlp_fourier_components(
    mlp: MLP,
    *,
    max_degree: int,
    epsilon: float,
    output_directory: Path | str,
    num_samples: int,
    random_seed: int,
) -> Path:
    """Analyse Fourier components of each neuron's activation in an MLP.

    Parameters
    ----------
    mlp:
        The multi-layer perceptron whose neuron activations should be analysed.
        The network is expected to expose a ``net`` attribute that is an
        ``nn.Sequential`` object, as provided by :class:`src.models.mlp.MLP`.
    max_degree:
        Maximum degree of Fourier basis subsets to consider.
    epsilon:
        Threshold below which dot products are discarded from the report.
    output_directory:
        Directory where ``fourier.json`` will be written.
    num_samples:
        Number of Monte Carlo samples to draw from the input hypercube.
    random_seed:
        Seed controlling the sampling procedure for reproducible estimates.

    Notes
    -----
    The computation estimates the Fourier correlations via Monte Carlo
    averaging over ``num_samples`` draws from the uniform distribution on the
    ``{±1}^d`` hypercube (where ``d`` is the input dimension). Basis functions
    are defined as products of input coordinates.

    Returns
    -------
    Path
        Path to the generated JSON report.
    """

    if not isinstance(mlp, MLP):
        raise TypeError("detect_mlp_fourier_components expects an instance of src.models.mlp.MLP")

    if not hasattr(mlp, "config") or not hasattr(mlp.config, "input_dim"):
        raise ValueError("The provided model does not expose an input_dim via config")

    input_dim = int(mlp.config.input_dim)  # type: ignore[attr-defined]
    try:
        first_param = next(mlp.parameters())
    except StopIteration:  # pragma: no cover - rare case of parameterless model
        device = torch.device("cpu")
    else:
        device = first_param.device

    generator = torch.Generator()
    generator.manual_seed(int(random_seed))

    points = _sample_hypercube_points(input_dim, num_samples, device, generator=generator)
    subsets = _fourier_subsets(input_dim, max_degree)
    features = _compute_basis_features(points, subsets)

    was_training = mlp.training
    mlp.eval()
    with torch.no_grad():
        activations = _collect_neuron_activations(mlp, points)
    if was_training:
        mlp.train()

    results: List[Dict[str, object]] = []
    for layer_idx, activation in enumerate(activations):
        if activation.ndim != 2:
            raise ValueError(
                "Expected activation tensors to be two-dimensional (samples x neurons)"
            )
        num_neurons = activation.shape[1]
        for neuron_idx in range(num_neurons):
            values = activation[:, neuron_idx]
            components: List[Dict[str, object]] = []
            for subset in subsets:
                dot_product = float((features[subset] * values).mean().item())
                if abs(dot_product) >= epsilon:
                    components.append(
                        {
                            "coordinates": list(subset),
                            "value": dot_product,
                        }
                    )

            components.sort(key=lambda item: abs(item["value"]), reverse=True)

            if components:
                results.append(
                    {
                        "layer": layer_idx,
                        "index": neuron_idx,
                        "components": components,
                    }
                )

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "fourier.json"
    payload: Dict[str, object] = {
        "max_degree": max_degree,
        "epsilon": epsilon,
        "num_samples": num_samples,
        "random_seed": random_seed,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


__all__ = ["detect_mlp_fourier_components"]
