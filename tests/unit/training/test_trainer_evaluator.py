import torch
import torch.nn.functional as F
import pytest

from src.training.trainer import Trainer
from src.training.trainer_config import TrainerConfig
from src.models.mlp_config import MLPConfig
from src.training.optimizers.configs.sgd import SgdConfig
from src.data.cube_distribution_config import (
    CubeDistributionConfig,
)


def test_trainer_reports_mse_loss(tmp_path):
    cfg = TrainerConfig(
        mlp_config=MLPConfig(
            input_dim=1,
            output_dim=1,
            hidden_dims=[],
            activation="relu",
            start_activation=False,
            end_activation=False,
            bias=False,
        ),
        optimizer_config=SgdConfig(lr=0.01),
        cube_distribution_config=CubeDistributionConfig(
            input_dim=1,
            indices_list=[[0]],
            weights=[1.0],
            noise_mean=0.0,
            noise_std=0.0,
        ),
        train_size=4,
        test_size=4,
        batch_size=2,
        epochs=0,
        home_dir=tmp_path,
    )

    trainer = Trainer(cfg)
    model, _ = trainer._initialize_model_and_optimizer()

    test_loader = trainer.get_iterator("test")
    total_loss = 0.0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(trainer.device), y.to(trainer.device)
            batch_size = X.size(0)
            preds = model(X)
            batch_loss = F.mse_loss(preds, y).item()
            total_loss += batch_loss * batch_size
            total_samples += batch_size

    expected_loss = total_loss / total_samples
    assert trainer.test_loss(model) == pytest.approx(expected_loss)

