import pytest
import torch
import torch.nn as nn
from mup import Linear as MuLinear, MuReadout

import src.models.bootstrap  # noqa: F401
from src.models.representors.mlp_representor import MLPRepresentor
from src.models.architectures.configs.mlp import MLPConfig
from src.training.trainer import Trainer
from src.training.trainer_config import TrainerConfig
from src.training.loss.configs.loss import LossConfig
from src.models.architectures.model_factory import create_model
from src.training.optimizers.optimizer_factory import create_optimizer
from tests.helpers.stubs import StubJointDistribution

# ---------------------------------------------------------------------------
# PyTest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=[False, True])
def representor(tmp_path_factory, mlp_config, adam_config, request):
    """Instantiate an ``MLPRepresentor`` for different ``MLP`` layer types."""

    mlp_config.mup = request.param
    cfg_model = mlp_config

    tmp_path = tmp_path_factory.mktemp("trainer")
    cfg = TrainerConfig(
        model_config=cfg_model,
        optimizer_config=adam_config,
        joint_distribution_config=StubJointDistribution._Config(
            X=torch.zeros(4, cfg_model.input_dim),
            y=torch.zeros(4, cfg_model.output_dim),
        ),
        train_size=4,
        test_size=2,
        batch_size=2,
        epochs=1,
        home_dir=tmp_path,
        loss_config=LossConfig(name="MSELoss"),
        seed=0,
    )
    trainer = Trainer(cfg)
    model = create_model(cfg.model_config)
    model.to(trainer.device)
    optimizer = create_optimizer(cfg.optimizer_config, model)
    trainer._save_checkpoint(
        model=model,
        optimizer=optimizer,
        training_loss=0.0,
        training_loss_with_l1=0.0,
        epoch=cfg.epochs,
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    return MLPRepresentor(
        trainer.config.model_config,
        trainer.checkpoint_dir,
        device=device,
    )


@pytest.fixture
def sample_input(representor):
    """Random batch of five input vectors matching the model's input dim."""
    return torch.randn(5, representor.model_config.input_dim)

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_metadata_consistency(representor):
    """``representor`` should expose modules and representation shapes."""
    modules = representor.modules
    hidden = representor.model_config.hidden_dims

    expected_num_modules = 2 * len(hidden) + 1  # input/output/pre‑+post‑acts
    if representor.model_config.start_activation:
        expected_num_modules += 1
    if representor.model_config.end_activation:
        expected_num_modules += 1

    assert len(modules) == expected_num_modules

    # First and last representation shapes must match config IO dims
    assert representor.representation_shape(0) == torch.Size([representor.model_config.input_dim])
    assert representor.representation_shape(len(modules)) == torch.Size([representor.model_config.output_dim])


def test_modules_match_config(representor):
    """Modules should match the layer type implied by the config."""
    first_layer = 0 if not representor.model_config.start_activation else 1
    first = representor.modules[first_layer]
    last = representor.modules[-1]
    if representor.model_config.mup:
        assert isinstance(first, MuLinear)
        assert isinstance(last, MuReadout)
    else:
        assert isinstance(first, nn.Linear)
        assert isinstance(last, nn.Linear)


def test_forward_config_roundtrip(representor):
    """``forward_config`` should return a valid ``MLPConfig``."""
    src_idx = 0
    dst_idx = len(representor.modules)  # full model slice
    subcfg = representor.forward_config(src_idx, dst_idx)

    print(f"\nSubconfig:\n{subcfg}")
    print(f"\nOriginal config:\n{representor.model_config}")
    print("\nRepresentations:")
    for i in range(len(representor.modules) + 1):
        rep_dict = representor.to_representation_dict(i)
        dim = representor.representation_shape(i)[0]
        print(f"Rep {i}: layer={rep_dict['layer_index']}, dim={dim}, post_act={rep_dict['post_activation']}")

    assert isinstance(subcfg, MLPConfig)
    assert subcfg.input_dim == representor.representation_shape(src_idx)[0]
    assert subcfg.output_dim == representor.representation_shape(dst_idx)[0]
    assert subcfg.mup == representor.model_config.mup
    assert subcfg.bias == representor.model_config.bias

    # Hidden dim sanity — ensure we captured at least one hidden layer
    assert len(subcfg.hidden_dims) == len(representor.model_config.hidden_dims)


def test_forward_outputs(representor, sample_input):
    """``ModelRepresentor.forward`` returns expected tensors."""
    with torch.no_grad():
        x = sample_input.to(representor.device)
        # from_rep = 0, to_rep = 1 -> single linear layer
        expected_y = representor.modules[0](x)
        X, y = representor.forward(sample_input, 0, 1, None)

        assert torch.allclose(X, x)
        assert torch.allclose(y, expected_y)

        # from_rep = 0, to_rep = 3 -> traverse multiple modules
        expected_y = representor.modules[2](representor.modules[1](representor.modules[0](x)))
        X, y = representor.forward(sample_input, 0, 3, None)

        assert torch.allclose(X, x)
        assert torch.allclose(y, expected_y)

        # from_rep = 1, to_rep = 4 -> start from representation 1
        rep1 = representor.modules[0](x)
        expected_y = rep1
        for m in representor.modules[1:4]:
            expected_y = m(expected_y)
        X, y = representor.forward(sample_input, 1, 4, None)

        assert torch.allclose(X, rep1)
        assert torch.allclose(y, expected_y)

        # from_rep = 1, to_rep = 2 -> activation on representation 1
        rep = representor.modules[0](x)
        expected_y = representor.modules[1](rep)
        X, y = representor.forward(sample_input, 1, 2, None)

        assert torch.allclose(X, rep)
        assert torch.allclose(y, expected_y)


def test_get_module(representor, sample_input):
    """``ModelRepresentor.get_module`` assembles submodules correctly."""
    with torch.no_grad():
        x = sample_input.to(representor.device)

        # Multi-layer slice
        module = representor.get_module(0, 3)
        expected = representor.modules[2](
            representor.modules[1](representor.modules[0](x))
        )
        assert isinstance(module, nn.Sequential)
        assert torch.allclose(module(x), expected)

        # Single layer slice
        single = representor.get_module(1, 2)
        rep = representor.modules[0](x)
        expected_single = representor.modules[1](rep)
        assert torch.allclose(single(rep), expected_single)


def test_get_base_rep_ids(representor):
    rep_ids = representor.get_base_rep_ids()
    n_hidden = len(representor.model_config.hidden_dims)
    expected = [0] + [2 * (i + 1) for i in range(n_hidden)] + [2 * n_hidden + 1]
    assert rep_ids == expected
