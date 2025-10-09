from src.training.trainer_config import TrainerConfig
from src.models.mlp_config import MLPConfig
from src.training.optimizers.configs.sgd import SgdConfig
from src.data.cube_distribution_config import (
    CubeDistributionConfig,
)


def test_default_optimizer(tmp_path):
    home_dir = tmp_path / "trainer_home"
    home_dir.mkdir()
    cfg = TrainerConfig(
        mlp_config=MLPConfig(
            input_dim=3,
            output_dim=1,
            hidden_dims=[4, 2],
            activation="relu",
            start_activation=False,
            end_activation=False,
        ),
        cube_distribution_config=CubeDistributionConfig(
            input_dim=3,
            indices_list=[[0]],
            weights=[1.0],
        ),
        train_size=10,
        test_size=5,
        batch_size=2,
        epochs=1,
        home_dir=home_dir,
    )
    assert isinstance(cfg.optimizer_config, SgdConfig)
    assert cfg.optimizer_config.lr == 0.001
