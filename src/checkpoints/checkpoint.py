from dataclasses import dataclass
from pathlib import Path
import torch
import pickle
import torch.nn as nn
from typing import Optional
from torch.optim import Optimizer as TorchOptimizer

@dataclass(frozen=True)
class Checkpoint:
    dir: Path

    def __post_init__(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        if any(not f.name.startswith('.') for f in self.dir.iterdir()):
            raise ValueError(f"Checkpoint directory {self.dir} must be empty on construction.")
        with open(self.dir / "checkpoint.pkl", "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def from_dir(cls, checkpoint_dir: Path) -> "Checkpoint":
        pkl_path = checkpoint_dir / "checkpoint.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"No checkpoint pickle at {pkl_path}")
        with open(pkl_path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj)}")
        return obj
    
    def save(self, model: nn.Module, optimizer: Optional[TorchOptimizer] = None) -> None:
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        }, self.dir / "checkpoint.pth")

    def load(self, model: nn.Module, optimizer: Optional[TorchOptimizer] = None) -> None:
        ckpt_file = self.dir / "checkpoint.pth"
        if not ckpt_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found at {ckpt_file}")

        data = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(data["model_state_dict"])
        opt_state = data.get("optimizer_state_dict")
        if optimizer is not None and opt_state is not None:
            optimizer.load_state_dict(opt_state)


