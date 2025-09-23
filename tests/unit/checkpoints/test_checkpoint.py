import os
import pickle
import shutil
import pytest
import torch
import torch.nn as nn
from pathlib import Path

from src.checkpoints.checkpoint import Checkpoint  # Replace 'your_module' with the actual module name


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    """
    Provide a fresh temporary directory path for each test.
    """
    return tmp_path / "ckpt_dir"


def test_post_init_creates_directory_and_writes_pickle(tmp_checkpoint_dir):
    # Directory does not exist initially
    assert not tmp_checkpoint_dir.exists()

    # Construct a Checkpoint and expect creation of the directory and a pickle file.
    ckpt = Checkpoint(dir=tmp_checkpoint_dir)

    # Directory should now exist
    assert tmp_checkpoint_dir.exists()
    # checkpoint.pkl file should exist
    pkl_path = tmp_checkpoint_dir / "checkpoint.pkl"
    assert pkl_path.exists()

    # The pickle file should contain a Checkpoint instance equal to ckpt
    with open(pkl_path, "rb") as fh:
        loaded = pickle.load(fh)
    assert isinstance(loaded, Checkpoint)
    assert loaded == ckpt


def test_post_init_raises_if_directory_not_empty(tmp_path):
    # Create directory and place a non-dot file inside
    existing_dir = tmp_path / "non_empty_dir"
    existing_dir.mkdir()
    (existing_dir / "some_file.txt").write_text("not empty")

    with pytest.raises(ValueError) as excinfo:
        _ = Checkpoint(dir=existing_dir)
    assert "must be empty on construction" in str(excinfo.value)


def test_from_dir_success_and_type_check(tmp_checkpoint_dir):
    # Create and pickle a Checkpoint via __post_init__
    ckpt_original = Checkpoint(dir=tmp_checkpoint_dir)

    # from_dir should load the same object
    ckpt_loaded = Checkpoint.from_dir(tmp_checkpoint_dir)
    assert isinstance(ckpt_loaded, Checkpoint)
    assert ckpt_loaded == ckpt_original

    # Manually overwrite the pickle with a different type (e.g., an integer)
    with open(tmp_checkpoint_dir / "checkpoint.pkl", "wb") as fh:
        pickle.dump(123, fh)

    with pytest.raises(TypeError) as excinfo:
        _ = Checkpoint.from_dir(tmp_checkpoint_dir)
    assert "Expected Checkpoint, got" in str(excinfo.value)


def test_from_dir_no_pickle_raises(tmp_path):
    non_existent_dir = tmp_path / "no_pickle_dir"
    non_existent_dir.mkdir()
    # Ensure no checkpoint.pkl inside
    assert not (non_existent_dir / "checkpoint.pkl").exists()

    with pytest.raises(FileNotFoundError) as excinfo:
        _ = Checkpoint.from_dir(non_existent_dir)
    assert "No checkpoint pickle" in str(excinfo.value)


def test_save_without_optimizer(tmp_checkpoint_dir):
    ckpt = Checkpoint(dir=tmp_checkpoint_dir)

    # Create a simple model and record its initial parameters
    model = SimpleModel()
    before_params = {k: v.clone() for k, v in model.state_dict().items()}

    # Modify model parameters so that load can be tested later
    for param in model.parameters():
        param.data.add_(1.0)

    # Saving with optimizer=None should succeed
    ckpt.save(model=model, optimizer=None)

    # Verify that checkpoint.pth exists
    pth_path = tmp_checkpoint_dir / "checkpoint.pth"
    assert pth_path.exists()

    # Load the raw saved dictionary and check that model_state_dict matches
    saved = torch.load(pth_path)
    assert "model_state_dict" in saved
    # Since optimizer is None, optimizer_state_dict should be saved as None
    assert saved["optimizer_state_dict"] is None

    # Ensure that the saved model_state_dict corresponds to the modified model
    for k, v in saved["model_state_dict"].items():
        assert torch.allclose(v, model.state_dict()[k])


def test_load_without_optimizer(tmp_checkpoint_dir):
    ckpt = Checkpoint(dir=tmp_checkpoint_dir)
    model = SimpleModel()

    # Modify model, then save
    for param in model.parameters():
        param.data.mul_(2.0)
    ckpt.save(model=model, optimizer=None)

    # Create a fresh model and load from checkpoint
    model_new = SimpleModel()
    # Ensure parameters differ before loading
    for (k_old, v_old), (k_new, v_new) in zip(model.state_dict().items(), model_new.state_dict().items()):
        assert not torch.allclose(v_old, v_new)

    # Loading with optimizer=None should succeed
    ckpt.load(model=model_new, optimizer=None)

    # After loading, parameters should match the saved ones
    for (k_saved, v_saved), (k_new, v_new) in zip(model.state_dict().items(), model_new.state_dict().items()):
        assert torch.allclose(v_saved, v_new)


def test_load_no_checkpoint_file_raises(tmp_checkpoint_dir):
    ckpt = Checkpoint(dir=tmp_checkpoint_dir)
    model = SimpleModel()
    # Ensure no checkpoint.pth exists
    if (tmp_checkpoint_dir / "checkpoint.pth").exists():
        os.remove(tmp_checkpoint_dir / "checkpoint.pth")

    with pytest.raises(FileNotFoundError) as excinfo:
        ckpt.load(model=model, optimizer=None)
    assert "Checkpoint file not found" in str(excinfo.value)


def test_save_with_optimizer_state_included(tmp_checkpoint_dir):
    """When an optimizer is supplied the checkpoint must contain its state."""
    ckpt = Checkpoint(dir=tmp_checkpoint_dir)

    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # Do a single step to ensure the optimizer accumulates state (momentum buffer)
    dummy_x = torch.randn(8, 10)
    loss = model(dummy_x).sum()
    loss.backward()
    optimizer.step()

    ckpt.save(model=model, optimizer=optimizer)

    # Verify .pth exists and has a populated optimizer_state_dict
    pth_path = tmp_checkpoint_dir / "checkpoint.pth"
    saved = torch.load(pth_path)
    assert saved["optimizer_state_dict"] is not None
    # The saved optimizer state should contain momentum buffers (non-empty "state")
    assert len(saved["optimizer_state_dict"]["state"]) > 0


def test_load_restores_optimizer_state(tmp_checkpoint_dir):
    """
    Loading should faithfully restore both the model parameters and the
    optimizer’s internal state (e.g. momentum buffers).
    """
    ckpt = Checkpoint(dir=tmp_checkpoint_dir)

    # ----- create original model/optimizer and save -----
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # Populate optimizer state via a dummy update
    dummy_x = torch.randn(4, 10)
    loss = model(dummy_x).sum()
    loss.backward()
    optimizer.step()

    # Keep copies for later comparison
    original_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    original_optimizer_state = optimizer.state_dict()

    ckpt.save(model=model, optimizer=optimizer)

    # ----- load into fresh model/optimizer -----
    new_model = SimpleModel()
    new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1, momentum=0.9)

    # Optimizer state should be empty before loading
    assert len(new_optimizer.state_dict()["state"]) == 0

    ckpt.load(model=new_model, optimizer=new_optimizer)

    # Model parameters must now match the originals
    for k, v in new_model.state_dict().items():
        assert torch.allclose(v, original_model_state[k])

    # Optimizer state should have been populated
    new_optimizer_state = new_optimizer.state_dict()
    assert len(new_optimizer_state["state"]) > 0

    # Compare each scalar/tensor entry in the optimizer states
    for old_p, new_p in zip(original_optimizer_state["state"].values(),
                            new_optimizer_state["state"].values()):
        assert old_p.keys() == new_p.keys()
        for key in old_p:
            old_val, new_val = old_p[key], new_p[key]
            if isinstance(old_val, torch.Tensor):
                assert torch.allclose(old_val, new_val)
            else:
                assert old_val == new_val






