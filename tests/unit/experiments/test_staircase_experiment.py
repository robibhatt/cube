import math
import torch
import matplotlib

matplotlib.use("Agg")
import src.models.bootstrap  # noqa: F401

from src.training.trainer import Trainer
from src.training.trainer_config import TrainerConfig
from src.models.architectures.configs.mlp import MLPConfig
from src.data.joint_distributions.configs.staircase import StaircaseConfig
from src.training.loss.configs.loss import LossConfig
from src.experiments.configs.staircase_experiment import StaircaseExperimentConfig
from src.experiments.experiments.staircase_experiment import StaircaseExperiment
from src.models.representors.mlp_representor import MLPRepresentor
from src.utils.seed_manager import SeedManager


def _make_trainer_config() -> TrainerConfig:
    model_cfg = MLPConfig(
        input_dim=1,
        output_dim=1,
        hidden_dims=[],
        activation="relu",
        start_activation=False,
        end_activation=False,
    )
    dist_cfg = StaircaseConfig(input_shape=torch.Size([1]), k=1)
    loss_cfg = LossConfig(name="MSELoss", eval_type="regression")

    return TrainerConfig(
        model_config=model_cfg,
        joint_distribution_config=dist_cfg,
        loss_config=loss_cfg,
        train_size=1,
        test_size=1,
        batch_size=1,
        epochs=1,
    )


def test_trainer_seed_from_experiment_seed(tmp_path):
    trainer_cfg = _make_trainer_config()
    exp_seed = 42
    exp_cfg = StaircaseExperimentConfig(
        trainer_config=trainer_cfg,
        home_directory=tmp_path,
        seed=exp_seed,
    )
    experiment = StaircaseExperiment(exp_cfg)

    trainer_seed = experiment.get_trainer_configs()[0][0].seed

    expected_seed = SeedManager(exp_seed).spawn_seed()
    assert trainer_seed == expected_seed

    # Repeated calls should return the same trainer configuration/seed
    trainer_seed_2 = experiment.get_trainer_configs()[0][0].seed
    assert trainer_seed_2 == trainer_seed


def test_layer_product_distribution(tmp_path):
    trainer_cfg = TrainerConfig(
        model_config=MLPConfig(
            input_dim=3,
            output_dim=1,
            hidden_dims=[2],
            activation="relu",
            start_activation=False,
            end_activation=False,
        ),
        joint_distribution_config=StaircaseConfig(
            input_shape=torch.Size([3]), k=2
        ),
        loss_config=LossConfig(name="MSELoss"),
        train_size=4,
        test_size=2,
        batch_size=2,
        epochs=1,
    )

    exp_cfg = StaircaseExperimentConfig(
        trainer_config=trainer_cfg,
        home_directory=tmp_path,
        seed=0,
    )
    experiment = StaircaseExperiment(exp_cfg)

    # Train the underlying model so checkpoints exist
    trainer_cfg = experiment.get_trainer_configs()[0][0]
    trainer_cfg.home_dir.mkdir(exist_ok=True)
    trainer = Trainer(trainer_cfg)
    trainer.train()

    dist_in = experiment.layer_product_distribution(layer_number=0, indices=[0, 1])
    dist_hidden = experiment.layer_product_distribution(layer_number=1, indices=[0, 1])
    dist_out = experiment.layer_product_distribution(layer_number=2, indices=[0, 1])

    n = 3
    seed = 0
    base_X, _ = dist_in.base_joint_distribution.base_sample(n, seed)
    expected_y = base_X[:, [0, 1]].prod(dim=1, keepdim=True)

    # Input layer should return the base inputs
    X_in, y_in = dist_in.sample(n, seed)
    assert torch.allclose(X_in, base_X)
    assert torch.allclose(y_in, expected_y)

    representor = MLPRepresentor(
        trainer_cfg.model_config, trainer.checkpoint_dir, device=torch.device("cpu")
    )

    # Hidden layer representation uses post-activation rep id
    rep_hidden = representor.from_representation_dict(
        {"layer_index": 1, "post_activation": True}
    )
    X_hidden, y_hidden = dist_hidden.sample(n, seed)
    expected_hidden = representor.get_module(0, rep_hidden)(base_X)
    assert torch.allclose(X_hidden, expected_hidden)
    assert torch.allclose(y_hidden, expected_y)

    # Output layer representation uses pre-activation rep id
    rep_out = representor.from_representation_dict(
        {"layer_index": 2, "post_activation": False}
    )
    X_out, y_out = dist_out.sample(n, seed)
    expected_out = representor.get_module(0, rep_out)(base_X)
    assert torch.allclose(X_out, expected_out)
    assert torch.allclose(y_out, expected_y)


def test_generate_tables(tmp_path):
    trainer_cfg = TrainerConfig(
        model_config=MLPConfig(
            input_dim=3,
            output_dim=1,
            hidden_dims=[2],
            activation="relu",
            start_activation=False,
            end_activation=False,
        ),
        joint_distribution_config=StaircaseConfig(
            input_shape=torch.Size([3]), k=2
        ),
        loss_config=LossConfig(name="MSELoss"),
        train_size=10,
        test_size=4,
        batch_size=5,
        epochs=1,
    )

    exp_cfg = StaircaseExperimentConfig(
        trainer_config=trainer_cfg,
        home_directory=tmp_path,
        seed=0,
    )
    experiment = StaircaseExperiment(exp_cfg)

    # Train underlying model so checkpoints exist
    trainer_cfg = experiment.get_trainer_configs()[0][0]
    trainer_cfg.home_dir.mkdir(exist_ok=True)
    trainer = Trainer(trainer_cfg)
    trainer.train()

    product_table, staircase_table = experiment.generate_tables()

    product_img = tmp_path / "product_table.png"
    staircase_img = tmp_path / "staircase_table.png"
    assert product_img.exists()
    assert staircase_img.exists()

    k = trainer_cfg.joint_distribution_config.k
    assert len(product_table) == 2 ** k - 1
    assert len(staircase_table) == k

    n_layers = len(trainer_cfg.model_config.hidden_dims) + 2
    for row in product_table:
        assert len(row) == n_layers
    for row in staircase_table:
        assert len(row) == n_layers


def test_generate_tables_axis_labels(tmp_path, monkeypatch):
    trainer_cfg = TrainerConfig(
        model_config=MLPConfig(
            input_dim=3,
            output_dim=1,
            hidden_dims=[2],
            activation="relu",
            start_activation=False,
            end_activation=False,
        ),
        joint_distribution_config=StaircaseConfig(
            input_shape=torch.Size([3]), k=2
        ),
        loss_config=LossConfig(name="MSELoss"),
        train_size=10,
        test_size=4,
        batch_size=5,
        epochs=1,
    )

    exp_cfg = StaircaseExperimentConfig(
        trainer_config=trainer_cfg,
        home_directory=tmp_path,
        seed=0,
    )
    experiment = StaircaseExperiment(exp_cfg)

    trainer_cfg = experiment.get_trainer_configs()[0][0]
    trainer_cfg.home_dir.mkdir(exist_ok=True)
    trainer = Trainer(trainer_cfg)
    trainer.train()

    import src.experiments.experiments.staircase_experiment as se

    figs: list[matplotlib.figure.Figure] = []
    original_subplots = se.plt.subplots

    def patched_subplots(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        figs.append(fig)
        return fig, ax

    monkeypatch.setattr(se.plt, "subplots", patched_subplots)

    experiment.generate_tables()

    assert len(figs) == 2
    product_fig, staircase_fig = figs
    prod_texts = {t.get_text() for t in product_fig.texts}
    stair_texts = {t.get_text() for t in staircase_fig.texts}
    assert "Layer number" in prod_texts
    assert "Subset" in prod_texts
    assert "Layer number" in stair_texts
    assert "k" in stair_texts


def test_layer_staircase_distribution(tmp_path):
    trainer_cfg = TrainerConfig(
        model_config=MLPConfig(
            input_dim=3,
            output_dim=1,
            hidden_dims=[2],
            activation="relu",
            start_activation=False,
            end_activation=False,
        ),
        joint_distribution_config=StaircaseConfig(
            input_shape=torch.Size([3]), k=3
        ),
        loss_config=LossConfig(name="MSELoss"),
        train_size=4,
        test_size=2,
        batch_size=2,
        epochs=1,
    )

    exp_cfg = StaircaseExperimentConfig(
        trainer_config=trainer_cfg,
        home_directory=tmp_path,
        seed=0,
    )
    experiment = StaircaseExperiment(exp_cfg)

    # Train the underlying model so checkpoints exist
    trainer_cfg = experiment.get_trainer_configs()[0][0]
    trainer_cfg.home_dir.mkdir(exist_ok=True)
    trainer = Trainer(trainer_cfg)
    trainer.train()

    dist_in = experiment.layer_staircase_distribution(layer_number=0, k=3)
    dist_hidden = experiment.layer_staircase_distribution(layer_number=1, k=3)
    dist_out = experiment.layer_staircase_distribution(layer_number=2, k=3)

    n = 3
    seed = 0
    base_X, _ = dist_in.base_joint_distribution.base_sample(n, seed)
    prod = base_X[:, 0]
    expected_y = prod.clone()
    for i in range(1, 3):
        prod = prod * base_X[:, i]
        expected_y = expected_y + prod
    expected_y = expected_y / math.sqrt(3)
    expected_y = expected_y.unsqueeze(1)

    # Input layer should return the base inputs
    X_in, y_in = dist_in.sample(n, seed)
    assert torch.allclose(X_in, base_X)
    assert torch.allclose(y_in, expected_y)

    representor = MLPRepresentor(
        trainer_cfg.model_config, trainer.checkpoint_dir, device=torch.device("cpu")
    )

    # Hidden layer representation uses post-activation rep id
    rep_hidden = representor.from_representation_dict(
        {"layer_index": 1, "post_activation": True}
    )
    X_hidden, y_hidden = dist_hidden.sample(n, seed)
    expected_hidden = representor.get_module(0, rep_hidden)(base_X)
    assert torch.allclose(X_hidden, expected_hidden)
    assert torch.allclose(y_hidden, expected_y)

    # Output layer representation uses pre-activation rep id
    rep_out = representor.from_representation_dict(
        {"layer_index": 2, "post_activation": False}
    )
    X_out, y_out = dist_out.sample(n, seed)
    expected_out = representor.get_module(0, rep_out)(base_X)
    assert torch.allclose(X_out, expected_out)
    assert torch.allclose(y_out, expected_y)

