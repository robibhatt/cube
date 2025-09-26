import pytest
import torch
import torch.nn as nn
from mup import MuSGD

import src.models.bootstrap  # noqa: F401
from src.models.architectures.mlp import MLP
from src.models.architectures.configs.mlp import MLPConfig


def _basic_cfg(mup: bool = False, frozen_layers=None):
    if frozen_layers is None:
        frozen_layers = []
    return MLPConfig(
        input_dim=3,
        hidden_dims=[4, 2],
        activation="relu",
        output_dim=1,
        start_activation=False,
        end_activation=False,
        mup=mup,
        frozen_layers=frozen_layers,
    )


def test_frozen_layer_indices_validation():
    base_kwargs = dict(
        input_dim=3,
        hidden_dims=[4, 2],
        activation="relu",
        output_dim=1,
        start_activation=False,
        end_activation=False,
    )
    with pytest.raises(ValueError):
        MLPConfig(**base_kwargs, frozen_layers=[0])
    with pytest.raises(ValueError):
        MLPConfig(**base_kwargs, frozen_layers=[4])


def test_layer_indexing_order():
    model = MLP(_basic_cfg())
    assert len(model.linear_layers) == 3
    l1, l2, l3 = model.linear_layers
    assert isinstance(l1, nn.Linear) and l1.out_features == 4
    assert isinstance(l2, nn.Linear) and l2.out_features == 2
    assert isinstance(l3, nn.Linear) and l3.out_features == 1


def test_freezing_copies_and_freezes_non_mup():
    torch.manual_seed(0)
    donor = MLP(_basic_cfg())
    donor_w = donor.linear_layers[1].weight.clone()

    torch.manual_seed(1)
    model = MLP(_basic_cfg(frozen_layers=[2]))
    model.copy_weights_from_donor(donor, [2])

    assert torch.equal(model.linear_layers[1].weight, donor_w)
    assert model.linear_layers[1].weight.requires_grad is False

    other_before = model.linear_layers[0].weight.clone()

    opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
    x = torch.randn(4, 3)
    y = torch.randn(4, 1)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()
    opt.step()

    assert torch.equal(model.linear_layers[1].weight, donor_w)
    assert not torch.equal(model.linear_layers[0].weight, other_before)


def test_freezing_with_mup():
    torch.manual_seed(0)
    donor = MLP(_basic_cfg(mup=True))
    donor_w = donor.linear_layers[2].weight.clone()

    torch.manual_seed(1)
    model = MLP(_basic_cfg(mup=True, frozen_layers=[3]))
    model.copy_weights_from_donor(donor, [3])

    assert torch.equal(model.linear_layers[2].weight, donor_w)
    assert model.linear_layers[2].weight.requires_grad is False

    other_before = model.linear_layers[0].weight.clone()

    opt = MuSGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
    x = torch.randn(4, 3)
    y = torch.randn(4, 1)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()
    opt.step()

    assert torch.equal(model.linear_layers[2].weight, donor_w)
    assert not torch.equal(model.linear_layers[0].weight, other_before)
