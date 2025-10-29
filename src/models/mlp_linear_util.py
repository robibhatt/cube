"""Utilities for probing MLP representations with linear models."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from mup import Linear as MuLinear

from src.data.cube_distribution import CubeDistribution
from src.data.cube_distribution_config import CubeDistributionConfig
from src.models.mlp import MLP


@dataclass
class LinearProbeResults:
    """Container for the results of a linear probe."""

    best_lambda: float
    train_samples: int
    test_samples: int
    test_mse: float

    def to_row(self) -> dict[str, float | int]:
        return {
            "best_lambda": float(self.best_lambda),
            "train_samples": int(self.train_samples),
            "test_samples": int(self.test_samples),
            "test_mse": float(self.test_mse),
        }


def run_first_layer_linear_regression(
    mlp: MLP,
    cube_config: CubeDistributionConfig,
    output_dir: Path,
    *,
    sample_count: int,
    lambda_value: float = 1e-3,
    seed: int = 0,
    batch_size: int = 4096,
) -> LinearProbeResults:
    """Fit a ridge regressor on the first hidden layer activations of *mlp*.

    The function samples data from the cube distribution described by
    ``cube_config`` and learns a linear model mapping the activations of the
    first hidden layer to the distribution outputs.  The number of samples used
    both for training the probe and evaluating its test loss is controlled by
    ``sample_count``.  Ridge regularisation is applied with the fixed strength
    ``lambda_value``.

    Parameters
    ----------
    mlp:
        The trained :class:`~src.models.mlp.MLP` to probe.
    cube_config:
        Configuration of the :class:`~src.data.cube_distribution.CubeDistribution`
        used during training.
    output_dir:
        Directory where ``linear_results.csv`` will be created.
    sample_count:
        Number of samples drawn from the distribution for both the training and
        evaluation datasets.
    lambda_value:
        Regularisation strength applied to the ridge regression weights.
    seed:
        Random seed controlling data shuffling and sampling from the
        distribution.
    batch_size:
        Batch size used when computing hidden activations.
    """

    if not mlp.config.hidden_dims:
        raise ValueError("MLP must have at least one hidden layer to run the linear probe")

    device = next(mlp.parameters()).device
    mlp = mlp.eval()

    if sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")

    train_samples = sample_count
    test_samples = sample_count

    distribution = CubeDistribution(cube_config, device)

    train_seed = seed
    test_seed = seed + 1

    train_inputs, train_targets = distribution.sample(train_samples, train_seed)
    test_inputs, test_targets = distribution.sample(test_samples, test_seed)

    train_features = _first_hidden_layer_activations(mlp, train_inputs, batch_size)
    test_features = _first_hidden_layer_activations(mlp, test_inputs, batch_size)

    train_targets = train_targets.to(device)
    test_targets = test_targets.to(device)

    weights, bias = _fit_ridge_with_gradient_descent(
        train_features,
        train_targets,
        lambda_value,
        seed=seed,
    )
    predictions = _predict_ridge(test_features, weights, bias)
    test_mse = torch.mean((predictions - test_targets) ** 2).item()

    results = LinearProbeResults(
        best_lambda=lambda_value,
        train_samples=train_samples,
        test_samples=test_samples,
        test_mse=test_mse,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "linear_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["best_lambda", "train_samples", "test_samples", "test_mse"],
        )
        writer.writeheader()
        writer.writerow(results.to_row())

    return results


def _first_hidden_layer_activations(
    mlp: MLP, inputs: torch.Tensor, batch_size: int
) -> torch.Tensor:
    """Return the activations of the first hidden layer for ``inputs``."""

    device = next(mlp.parameters()).device
    activations: list[torch.Tensor] = []

    mlp = mlp.eval()
    with torch.no_grad():
        for batch in torch.split(inputs, batch_size):
            hidden = batch.to(device)
            seen_hidden_linear = False
            for module in mlp.net:
                hidden = module(hidden)
                if isinstance(module, MuLinear) and not seen_hidden_linear:
                    seen_hidden_linear = True
                elif seen_hidden_linear and isinstance(module, nn.ReLU):
                    activations.append(hidden.detach())
                    break
            else:  # pragma: no cover - defensive branch
                raise RuntimeError("Failed to locate the first hidden layer activation")

    return torch.cat(activations, dim=0)


def _fit_ridge_with_gradient_descent(
    features: torch.Tensor,
    targets: torch.Tensor,
    lambda_value: float,
    *,
    seed: int,
    initial_subset_size: int = 5000,
    gd_batch_size: int = 8192,
    learning_rate: float = 1e-3,
    max_epochs: int = 200,
    convergence_tol: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a ridge regressor using a two-stage optimisation procedure."""

    n_samples, _ = features.shape
    if n_samples == 0:
        raise ValueError("Cannot fit ridge regression without data")

    generator = (
        torch.Generator(device=features.device) if features.is_cuda else torch.Generator()
    )
    generator.manual_seed(seed)

    subset_size = min(initial_subset_size, n_samples)
    subset_perm = torch.randperm(n_samples, generator=generator, device=features.device)
    subset_idx = subset_perm[:subset_size]

    init_weights, init_bias = _solve_ridge_closed_form(
        features.index_select(0, subset_idx),
        targets.index_select(0, subset_idx),
        lambda_value,
    )

    weights = init_weights.clone().detach().requires_grad_(True)
    bias = init_bias.clone().detach().requires_grad_(True)

    prev_loss = float("inf")
    for _ in range(max_epochs):
        perm = torch.randperm(n_samples, generator=generator, device=features.device)
        for start in range(0, n_samples, gd_batch_size):
            end = min(start + gd_batch_size, n_samples)
            batch_idx = perm[start:end]
            batch_features = features.index_select(0, batch_idx)
            batch_targets = targets.index_select(0, batch_idx)

            predictions = batch_features @ weights + bias
            mse = torch.mean((predictions - batch_targets) ** 2)
            reg = lambda_value * torch.sum(weights**2)
            loss = 0.5 * mse + 0.5 * reg

            loss.backward()
            with torch.no_grad():
                weights -= learning_rate * weights.grad
                bias -= learning_rate * bias.grad
            weights.grad.zero_()
            bias.grad.zero_()

        with torch.no_grad():
            full_predictions = features @ weights + bias
            full_mse = torch.mean((full_predictions - targets) ** 2)
            reg_term = lambda_value * torch.sum(weights**2)
            total_loss = 0.5 * full_mse + 0.5 * reg_term

        improvement = prev_loss - float(total_loss)
        if improvement >= 0.0 and improvement < convergence_tol * max(1.0, prev_loss):
            break
        prev_loss = float(total_loss)

    return weights.detach(), bias.detach()


def _solve_ridge_closed_form(
    features: torch.Tensor, targets: torch.Tensor, lambda_value: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ridge solution using the closed-form normal equations."""

    n_samples, n_features = features.shape
    if n_samples == 0:
        raise ValueError("Cannot fit ridge regression without data")

    ones = torch.ones(n_samples, 1, device=features.device, dtype=features.dtype)
    design = torch.cat([features, ones], dim=1)

    gram = design.T @ design / n_samples
    reg = torch.eye(n_features + 1, device=features.device, dtype=features.dtype)
    reg[-1, -1] = 0.0  # Do not regularise the bias term
    reg = reg * lambda_value

    rhs = design.T @ targets / n_samples
    solution = torch.linalg.solve(gram + reg, rhs)

    weights = solution[:-1]
    bias = solution[-1]
    return weights, bias


def _predict_ridge(
    features: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return features @ weights + bias

