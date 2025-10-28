from __future__ import annotations

import csv
from pathlib import Path

import torch

from src.data.cube_distribution_config import CubeDistributionConfig
from src.models.mlp import MLP
from src.models.mlp_config import MLPConfig
from src.models.mlp_linear_util import run_first_layer_linear_regression


def _build_cube_config(input_dim: int) -> CubeDistributionConfig:
    return CubeDistributionConfig(
        input_dim=input_dim,
        indices_list=[[0], [1]],
        weights=[1.0, -0.5],
        noise_std=0.0,
    )


def _build_mlp(input_dim: int, hidden_width: int) -> MLP:
    config = MLPConfig(
        input_dim=input_dim,
        output_dim=1,
        hidden_dims=[hidden_width],
        start_activation=False,
        end_activation=False,
    )
    mlp = MLP(config)
    mlp.eval()
    return mlp


def test_run_first_layer_linear_regression_creates_csv(tmp_path: Path) -> None:
    hidden_width = 2
    mlp = _build_mlp(input_dim=2, hidden_width=hidden_width)
    cube_config = _build_cube_config(input_dim=2)

    results = run_first_layer_linear_regression(
        mlp,
        cube_config,
        tmp_path,
        sample_count=32,
        seed=123,
        batch_size=8,
    )

    csv_path = tmp_path / "linear_results.csv"
    assert csv_path.exists(), "linear regression should write its metrics to CSV"

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]

    assert results.train_samples == 32
    assert results.test_samples == 32
    assert results.best_lambda == 1e-3
    assert torch.isfinite(torch.tensor(results.test_mse)), "test MSE must be finite"

    # Values in the CSV should match the returned dataclass.
    assert float(row["best_lambda"]) == results.best_lambda
    assert int(row["train_samples"]) == results.train_samples
    assert int(row["test_samples"]) == results.test_samples
    assert float(row["test_mse"]) == results.test_mse
