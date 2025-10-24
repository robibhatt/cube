"""Utilities for probing MLP representations with linear models."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

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
    k_folds: int = 5,
    lambda_values: Sequence[float] | None = None,
    seed: int = 0,
    batch_size: int = 4096,
) -> LinearProbeResults:
    """Fit a ridge regressor on the first hidden layer activations of *mlp*.

    The function samples data from the cube distribution described by
    ``cube_config`` and learns a linear model mapping the activations of the
    first hidden layer to the distribution outputs.  The amount of training
    data used scales with the width ``w`` of the first hidden layer; by default
    ``20 * w`` samples are used for training to emphasise representability over
    optimisation challenges.  The regularisation strength for the ridge model is
    selected via ``k``-fold cross validation (default ``k=5``).

    Parameters
    ----------
    mlp:
        The trained :class:`~src.models.mlp.MLP` to probe.
    cube_config:
        Configuration of the :class:`~src.data.cube_distribution.CubeDistribution`
        used during training.
    output_dir:
        Directory where ``linear_results.csv`` will be created.
    k_folds:
        Number of folds used during cross validation.  The value is clamped to
        the number of available training samples.
    lambda_values:
        Optional iterable of candidate regularisation strengths.  When not
        provided a log-spaced grid spanning ``1e-6`` to ``1e2`` is used.
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

    width = mlp.config.hidden_dims[0]
    train_samples = max(20 * width, width)
    test_samples = max(5 * width, width)

    distribution = CubeDistribution(cube_config, device)

    train_seed = seed
    test_seed = seed + 1

    train_inputs, train_targets = distribution.sample(train_samples, train_seed)
    test_inputs, test_targets = distribution.sample(test_samples, test_seed)

    train_features = _first_hidden_layer_activations(mlp, train_inputs, batch_size)
    test_features = _first_hidden_layer_activations(mlp, test_inputs, batch_size)

    train_targets = train_targets.to(device)
    test_targets = test_targets.to(device)

    if lambda_values is None:
        lambda_values = torch.logspace(-6, 2, steps=9, base=10.0, device=device).tolist()

    best_lambda = _select_ridge_lambda(
        train_features,
        train_targets,
        lambda_values,
        k_folds=k_folds,
        seed=seed,
    )

    weights, bias = _fit_ridge(train_features, train_targets, best_lambda)
    predictions = _predict_ridge(test_features, weights, bias)
    test_mse = torch.mean((predictions - test_targets) ** 2).item()

    results = LinearProbeResults(
        best_lambda=best_lambda,
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


def _select_ridge_lambda(
    features: torch.Tensor,
    targets: torch.Tensor,
    lambda_values: Sequence[float],
    *,
    k_folds: int,
    seed: int,
) -> float:
    """Return the regularisation strength with the best cross-validation score."""

    n_samples = features.shape[0]
    if k_folds <= 1:
        raise ValueError("k_folds must be greater than 1 for cross validation")

    k_folds = min(k_folds, n_samples)

    generator = (
        torch.Generator(device=features.device) if features.is_cuda else torch.Generator()
    )
    generator.manual_seed(seed)
    perm = torch.randperm(n_samples, generator=generator, device=features.device)

    fold_sizes = _compute_fold_sizes(n_samples, k_folds)
    folds: list[torch.Tensor] = []
    start = 0
    for size in fold_sizes:
        end = start + size
        folds.append(perm[start:end])
        start = end

    best_lambda = None
    best_loss = float("inf")

    for lambda_value in lambda_values:
        lambda_loss = 0.0
        for fold_idx in range(k_folds):
            val_idx = folds[fold_idx]
            train_idx = torch.cat(
                [folds[i] for i in range(k_folds) if i != fold_idx], dim=0
            )

            weights, bias = _fit_ridge(
                features.index_select(0, train_idx),
                targets.index_select(0, train_idx),
                float(lambda_value),
            )
            preds = _predict_ridge(features.index_select(0, val_idx), weights, bias)
            fold_loss = torch.mean((preds - targets.index_select(0, val_idx)) ** 2)
            lambda_loss += fold_loss.item()

        avg_loss = lambda_loss / k_folds
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_lambda = float(lambda_value)

    assert best_lambda is not None, "No lambda value evaluated"
    return best_lambda


def _compute_fold_sizes(n_samples: int, k_folds: int) -> Iterable[int]:
    base_size = n_samples // k_folds
    remainder = n_samples % k_folds
    for fold in range(k_folds):
        yield base_size + (1 if fold < remainder else 0)


def _fit_ridge(
    features: torch.Tensor, targets: torch.Tensor, lambda_value: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the ridge regression normal equations."""

    n_samples, n_features = features.shape
    ones = torch.ones(n_samples, 1, device=features.device, dtype=features.dtype)
    design = torch.cat([features, ones], dim=1)

    gram = design.T @ design
    reg = torch.eye(n_features + 1, device=features.device, dtype=features.dtype)
    reg[-1, -1] = 0.0  # Do not regularise the bias term
    reg = reg * lambda_value

    rhs = design.T @ targets
    solution = torch.linalg.solve(gram + reg, rhs)

    weights = solution[:-1]
    bias = solution[-1]
    return weights, bias


def _predict_ridge(
    features: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return features @ weights + bias

