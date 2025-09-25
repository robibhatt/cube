import torch
from src.training.trainer_config import TrainerConfig
from src.training.loss.configs.loss import LossConfig
from src.data.joint_distributions.configs.mapped_joint_distribution import MappedJointDistributionConfig
from src.data.joint_distributions.configs.gaussian import GaussianConfig
from src.models.architectures.configs.mlp import MLPConfig
from src.training.optimizers.configs.adam import AdamConfig
from src.models.targets.configs.sum_prod import SumProdTargetConfig


def test_default_optimizer(tmp_path):
    cfg = TrainerConfig(
        model_config=MLPConfig(
            input_dim=3,
            output_dim=1,
            hidden_dims=[4, 2],
            activation="relu",
            start_activation=False,
            end_activation=False,
        ),
        joint_distribution_config=MappedJointDistributionConfig(
            distribution_config=GaussianConfig(
                input_dim=3, mean=0.0, std=1.0
            ),
            target_function_config=SumProdTargetConfig(
                input_shape=torch.Size([3]),
                indices_list=[[0]],
                weights=[1.0],
                normalize=False,
            ),
        ),
        train_size=10,
        test_size=5,
        batch_size=2,
        epochs=1,
        home_dir=tmp_path / "trainer_home",
        loss_config=LossConfig(name="MSELoss"),
    )
    assert isinstance(cfg.optimizer_config, AdamConfig)
    assert cfg.optimizer_config.lr == 0.001
