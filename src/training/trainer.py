"""Concrete training logic.

This module exposes a :class:`Trainer` class which combines the behaviour of
the previous ``StepTrainer`` implementation with the utility helpers that lived
on the abstract ``Trainer`` base class.  The result is a single concrete class
with no registry or inheritance machinery – exactly the functionality required
by the rest of the codebase.

The class provides utilities for preparing datasets, running a training loop
with checkpointing and timing information, and computing evaluation metrics.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import re
import subprocess
import time
import torch

from copy import deepcopy
from src.utils.seed_manager import SeedManager
from torch.optim import Optimizer as TorchOptimizer

from src.training.trainer_config import TrainerConfig
from src.data.joint_distributions import create_joint_distribution
from src.data.providers import create_data_provider_from_distribution
from src.data.providers.data_provider import DataProvider
from src.models.architectures.model import Model
from src.models.architectures.model_factory import create_model
from src.models.architectures.mlp import MLP
from src.training.optimizers.optimizer import Optimizer
from src.training.optimizers.optimizer_factory import create_optimizer
from src.checkpoints.checkpoint import Checkpoint


def detect_mup(optimizer: TorchOptimizer) -> bool:
    """Return ``True`` if *optimizer* comes from the ``mup`` package."""

    try:
        from mup.optim import MuSGD, MuAdam  # type: ignore

        return isinstance(optimizer, (MuSGD, MuAdam))
    except Exception:
        cls = optimizer.__class__.__name__
        mod = optimizer.__class__.__module__
        return cls.startswith("Mu") or "mup" in mod


def dump_optimizer_values(
    optimizer: TorchOptimizer,
    model: Model,
    out_dir: Path,
    mup_used: bool = False,
) -> None:
    """Write a JSON summary of *optimizer* parameter groups and parameters.

    The destination directory *out_dir* is required so that callers must
    explicitly choose where the file is written.
    """

    id_to_name = {id(p): name for name, p in model.named_parameters()}

    groups: list[dict] = []
    params: list[dict] = []
    for idx, group in enumerate(optimizer.param_groups):
        lr = float(group.get("lr", 0.0))
        wd = float(group.get("weight_decay", 0.0))
        n_params = sum(p.numel() for p in group.get("params", []))
        groups.append(
            {
                "group_index": idx,
                "lr": lr,
                "weight_decay": wd,
                "n_params": n_params,
            }
        )

        for p in group.get("params", []):
            params.append(
                {
                    "name": id_to_name.get(id(p), ""),
                    "shape": list(p.shape),
                    "group_index": idx,
                    "lr": lr,
                    "weight_decay": wd,
                }
            )

    payload = {
        "optimizer": optimizer.__class__.__name__,
        "mup_used": mup_used,
        "groups": groups,
        "params": params,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "optimizer_values.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


class Trainer:
    """Concrete trainer used throughout the project."""

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def server_train(cls, trainer_dir: Path) -> str:
        """Create an sbatch script in ``trainer_dir`` and submit it."""

        project_root = Path(__file__).resolve().parents[3]
        start_script = project_root / "scripts" / "start_experiment.sh"
        if not start_script.exists():
            raise FileNotFoundError(
                f"Reference script not found at {start_script}"
            )

        lines = start_script.read_text().splitlines()
        job_name = f"{trainer_dir.parent.name}_{trainer_dir.name}"
        out_path = os.path.relpath(trainer_dir / "train.out", project_root)
        err_path = os.path.relpath(trainer_dir / "train.err", project_root)
        trainer_rel = os.path.relpath(trainer_dir, project_root)

        new_lines = []
        for line in lines:
            if line.startswith("#SBATCH --job-name="):
                new_lines.append(f"#SBATCH --job-name={job_name}")
            elif line.startswith("#SBATCH --output="):
                new_lines.append(f"#SBATCH --output={out_path}")
            elif line.startswith("#SBATCH --error="):
                new_lines.append(f"#SBATCH --error={err_path}")
            elif line.strip().startswith("python"):
                new_lines.append(f"python -m scripts.train {trainer_rel}")
            else:
                new_lines.append(line)

        script_path = trainer_dir / "train.sh"
        script_path.write_text("\n".join(new_lines) + "\n")
        script_path.chmod(0o755)
        script_rel = os.path.relpath(script_path, project_root)

        result = subprocess.run(
            ["sbatch", script_rel],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        match = re.search(r"(\d+)", result.stdout)
        if not match:
            raise RuntimeError(
                f"Could not parse job ID from sbatch output: {result.stdout!r}"
            )
        return match.group(1)

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.config.ready_for_trainer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Derived state from config
        self.seed_mgr = SeedManager(self.config.seed if self.config.seed is not None else 0)
        self.train_seed = self.seed_mgr.spawn_seed()
        self.test_seed = self.seed_mgr.spawn_seed()
        self.model_seed = self.seed_mgr.spawn_seed()
        self.optimizer_seed = self.seed_mgr.spawn_seed()

        # ``loss_fn`` is used for training; ``evaluator`` for reporting metrics
        self.loss_fn = self.config.loss_config.build()
        self.evaluator = self.config.loss_config.get_evaluator()

        # Instantiate the joint distribution on the given device
        self.joint_distribution = create_joint_distribution(
            self.config.joint_distribution_config, self.device
        )

        self._create_directories()
        self.timing_profile_file = self.config.home_dir / "timing_profile.json"
        if not (self.checkpoint_dir / "checkpoint.pkl").exists():
            Checkpoint(dir=self.checkpoint_dir)
        self._save_cfg()

        # Optimizer configuration is copied so that we can mutate it freely.
        self.optimizer_config = deepcopy(config.optimizer_config)

    def _create_directories(self) -> None:
        self.datasets_dir = self.config.home_dir / "datasets"
        self.datasets_dir.mkdir(exist_ok=True)

        self.checkpoint_dir = self.config.home_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.logs_dir = self.config.home_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        self.log_file = self.logs_dir / "training.log"
        if not self.log_file.exists():
            with open(self.log_file, "w") as f:
                f.write("epochs_trained,training_loss\n")

        self.train_l1_log_file = self.logs_dir / "train_loss_with_l1.log"
        if not self.train_l1_log_file.exists():
            with open(self.train_l1_log_file, "w") as f:
                f.write("epochs_trained,train_loss_with_l1\n")

        self.test_log_file = self.logs_dir / "test.log"
        if not self.test_log_file.exists():
            with open(self.test_log_file, "w") as f:
                f.write("epoch,test_loss\n")

    def _save_cfg(self) -> None:
        cfg_json = self.config.home_dir / "trainer_config.json"
        cfg_json.write_text(self.config.to_json(indent=2))

    def _load_timing_profile(self) -> list[dict]:
        if self.timing_profile_file.exists():
            with open(self.timing_profile_file) as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return []
        return []

    def _save_timing_profile(self, data: list[dict]) -> None:
        with open(self.timing_profile_file, "w") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # Core training helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        model: Model,
        optimizer: Optimizer,
        training_loss: float,
        training_loss_with_l1: float,
        epoch: int,
    ) -> None:
        checkpoint = Checkpoint.from_dir(self.checkpoint_dir)
        checkpoint.save(model=model, optimizer=optimizer.stepper)

        with open(self.log_file, "a") as f:
            f.write(f"{epoch},{training_loss}\n")

        with open(self.train_l1_log_file, "a") as f:
            f.write(f"{epoch},{training_loss_with_l1}\n")

        if epoch % 10 == 0:
            test_loader = self.get_iterator("test")
            test_loss = self._loss(model, test_loader)
            with open(self.test_log_file, "a") as f:
                f.write(f"{epoch},{test_loss}\n")

    def get_iterator(self, split: str) -> DataProvider:
        size = self.config.train_size if split == "train" else self.config.test_size
        seed = self.train_seed if split == "train" else self.test_seed
        batch_size = self.config.batch_size or 1024
        iterator = create_data_provider_from_distribution(
            self.joint_distribution,
            self.datasets_dir / split,
            batch_size,
            size,
            seed=seed,
        )
        return iterator

    def get_fresh_iterator(self, size: int | None = None) -> DataProvider:
        """Return a data provider with a new, unique seed.

        Parameters
        ----------
        size:
            Optional dataset size.  Defaults to ``self.config.test_size``.
        """

        seed = self.seed_mgr.spawn_seed()
        size = size or self.config.test_size
        batch_size = self.config.batch_size or 1024
        iterator = create_data_provider_from_distribution(
            self.joint_distribution,
            self.datasets_dir / f"fresh_{seed}",
            batch_size,
            size,
            seed=seed,
        )
        return iterator

    def _initialize_model_and_optimizer(self) -> tuple[Model, Optimizer]:
        assert not self.started_training, "Training already started"
        torch.manual_seed(self.model_seed)
        model = create_model(self.config.model_config)
        model.to(self.device)

        # Save an independent copy of the initial model before any training
        init_model_path = self.checkpoint_dir / "initial_model.pth"
        if not init_model_path.exists():
            torch.save(model.state_dict(), init_model_path)

        # Also persist the initial weights of linear layers for drift computation
        if isinstance(model, MLP):
            init_weights_path = self.checkpoint_dir / "initial_linear_weights.pth"
            if not init_weights_path.exists():
                torch.save(
                    [
                        layer.weight.detach().cpu().clone()
                        for layer in model.linear_layers
                    ],
                    init_weights_path,
                )

        torch.manual_seed(self.optimizer_seed)
        optimizer = create_optimizer(self.optimizer_config, model)
        base_loss, l1_loss = self._train_losses(model)
        self._save_checkpoint(
            model=model,
            optimizer=optimizer,
            training_loss=base_loss,
            training_loss_with_l1=l1_loss,
            epoch=0,
        )
        return model, optimizer

    def _load_model_and_optimizer(self) -> tuple[Model, Optimizer]:
        checkpoint = Checkpoint.from_dir(self.checkpoint_dir)
        torch.manual_seed(self.model_seed)
        model = create_model(self.config.model_config)
        torch.manual_seed(self.optimizer_seed)
        optimizer = create_optimizer(self.optimizer_config, model)
        checkpoint.load(model=model, optimizer=optimizer.stepper)

        model.to(self.device)
        self._move_optimizer_state_to_device(optimizer.stepper)

        return model, optimizer

    def _load_model(self) -> Model:
        model, _ = self._load_model_and_optimizer()
        return model

    def _move_optimizer_state_to_device(self, stepper: TorchOptimizer) -> None:
        for state in stepper.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

    def _step_train(self) -> None:
        if not self.started_training:
            model, optimizer = self._initialize_model_and_optimizer()
        else:
            model, optimizer = self._load_model_and_optimizer()

        model.to(self.device)
        self._move_optimizer_state_to_device(optimizer.stepper)

        mup_flag = detect_mup(optimizer.stepper)
        dump_optimizer_values(
            optimizer.stepper,
            model,
            self.config.home_dir,
            mup_flag,
        )

        train_loader = self.get_iterator("train")
        timing_data = self._load_timing_profile()

        if self.config.early_stopping is not None:
            # Early stopping should be based on the training loss without any
            # L1 regularization component.  We therefore recompute the pure
            # training loss here instead of relying on previously logged
            # values.
            base_loss = self._train_loss(model)
            if base_loss < self.config.early_stopping:
                return

        start = self.epochs_trained + 1
        for epoch in range(start, self.config.epochs + 1):
            model.train()
            load_time, step_time = 0.0, 0.0
            batch_count = 0

            if self.config.use_full_batch:
                optimizer.stepper.zero_grad()

            for Xb, yb in train_loader:
                t0 = time.perf_counter()

                Xb, yb = Xb.to(self.device), yb.to(self.device)

                t1 = time.perf_counter()

                if not self.config.use_full_batch:
                    optimizer.stepper.zero_grad()

                loss = self.loss_fn(model(Xb), yb)
                if self.config.weight_decay_l1 != 0.0 and isinstance(model, MLP):
                    loss = loss + self.config.weight_decay_l1 * self._l1_penalty(model)
                loss.backward()

                if not self.config.use_full_batch:
                    optimizer.step()

                t2 = time.perf_counter()
                load_time += t1 - t0
                step_time += t2 - t1
                batch_count += 1

            if self.config.use_full_batch:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(batch_count)
                optimizer.step()

            base_loss, l1_loss = self._train_losses(model)
            self._save_checkpoint(
                model=model,
                optimizer=optimizer,
                training_loss=base_loss,
                training_loss_with_l1=l1_loss,
                epoch=epoch,
            )
            if self.config.early_stopping is not None and base_loss < self.config.early_stopping:
                break
            if batch_count > 0:
                timing_entry = {
                    "epoch": epoch,
                    "avg_batch_load_time": load_time / batch_count,
                    "avg_batch_step_time": step_time / batch_count,
                }
                timing_data.append(timing_entry)
                self._save_timing_profile(timing_data)

        # After training, compute relative Frobenius drift per layer for MLPs
        if isinstance(model, MLP):
            init_weights_path = self.checkpoint_dir / "initial_linear_weights.pth"
            if init_weights_path.exists():
                initial_weights = torch.load(init_weights_path, map_location="cpu")
                drifts: dict[str, float] = {}
                for idx, layer in enumerate(model.linear_layers, start=1):
                    w0 = initial_weights[idx - 1].to("cpu")
                    wt = layer.weight.detach().cpu()
                    denom = torch.norm(w0, p="fro").item()
                    num = torch.norm(wt - w0, p="fro").item()
                    drift = (num / denom) if denom != 0 else float("inf")
                    drifts[f"layer_{idx}"] = drift
                out_path = self.config.home_dir / "frobenius_drifts.json"
                with open(out_path, "w") as f:
                    json.dump(drifts, f, indent=2)

            # ------------------------------------------------------------------
            # gradient statistics and visualization
            # ------------------------------------------------------------------
            data_provider = self.get_fresh_iterator()
            grads_path = self.config.home_dir / "neuron_input_gradients.csv"
            model.export_neuron_input_gradients(data_provider, grads_path)
            model.visualize(self.config.home_dir)

    def train(self) -> None:
        if not self.started_training or not self.finished_training:
            self._step_train()

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _loss(self, model: Model, iterator: DataProvider) -> float:
        total_loss = 0.0
        total_samples = 0

        model.eval()
        with torch.no_grad():
            for X, y in iterator:
                X, y = X.to(self.device), y.to(self.device)
                batch_size = X.size(0)
                y_pred = model(X)
                batch_mean_loss = self.evaluator(y_pred, y).item()
                total_loss += batch_mean_loss * batch_size
                total_samples += batch_size

        return total_loss / total_samples

    def _train_loss(self, model: Model) -> float:
        train_loader = self.get_iterator("train")
        total_loss = 0.0
        total_samples = 0

        model.eval()
        with torch.no_grad():
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                batch_size = X.size(0)
                y_pred = model(X)
                batch_mean_loss = self.loss_fn(y_pred, y).item()
                total_loss += batch_mean_loss * batch_size
                total_samples += batch_size

        return total_loss / total_samples

    def _l1_penalty(self, model: Model) -> torch.Tensor:
        if not isinstance(model, MLP):
            return torch.tensor(0.0, device=self.device)

        layers = model.linear_layers
        if not layers:
            return torch.tensor(0.0, device=self.device)

        if len(layers) == 1:
            return layers[0].weight.abs().sum()

        penalty = layers[0].weight.abs().sum() + layers[-1].weight.abs().sum()

        if len(layers) > 2:
            intermediate = torch.tensor(0.0, device=self.device)
            for layer in layers[1:-1]:
                intermediate = intermediate + layer.weight.abs().sum()
            penalty = penalty + intermediate / (len(layers) - 2)

        return penalty

    def _train_losses(self, model: Model) -> tuple[float, float]:
        base_loss = self._train_loss(model)
        if self.config.weight_decay_l1 != 0.0 and isinstance(model, MLP):
            with torch.no_grad():
                l1 = self._l1_penalty(model).item()
            return base_loss, base_loss + self.config.weight_decay_l1 * l1
        return base_loss, base_loss

    def test_loss(self, model: Model | None = None) -> float:
        if model is None:
            model = self._load_model()
        test_loader = self.get_iterator("test")
        return self._loss(model, test_loader)

    def train_loss(self) -> float:
        with open(self.log_file, "r") as f:
            lines = f.readlines()
            if len(lines) <= 1:
                raise RuntimeError("No training data found in log file")
            last_line = lines[-1]
            _, train_loss = last_line.strip().split(",")
            return float(train_loss)

    def train_loss_with_l1(self) -> float:
        with open(self.train_l1_log_file, "r") as f:
            lines = f.readlines()
            if len(lines) <= 1:
                raise RuntimeError("No training data found in L1 log file")
            last_line = lines[-1]
            _, train_loss = last_line.strip().split(",")
            return float(train_loss)

    def mean_output_loss(self) -> float:
        loader = self.get_iterator("test")
        total_y = None
        count = 0
        with torch.no_grad():
            for _, y in loader:
                y = y.to(self.device)
                if total_y is None:
                    total_y = torch.zeros_like(y[0])
                total_y += y.sum(dim=0)
                count += y.size(0)
        mean_y = total_y / count

        loader = self.get_iterator("test")
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for _, y in loader:
                y = y.to(self.device)
                batch_size = y.size(0)
                y_pred = mean_y.expand_as(y)
                batch_mean_loss = self.evaluator(y_pred, y).item()
                total_loss += batch_mean_loss * batch_size
                total_samples += batch_size

        return total_loss / total_samples

    def get_results(self) -> dict:
        model = self._load_model()
        train_loader = self.get_iterator("train")
        test_loader = self.get_iterator("test")
        return {
            "final_train_loss": self._loss(model, train_loader),
            "final_test_loss": self._loss(model, test_loader),
            "mean_output_loss": self.mean_output_loss(),
            "epochs_trained": self.epochs_trained,
        }

    # ------------------------------------------------------------------
    # Result handling
    # ------------------------------------------------------------------

    def save_results(self) -> None:
        results = self.get_results()
        if self.config.epochs is not None and self.epochs_trained > self.config.epochs:
            raise RuntimeError(
                f"Number of epochs trained ({self.epochs_trained}) is greater than "
                f"number of epochs in config ({self.config.epochs})"
            )
        elif self.finished_training:
            results_file = self.config.home_dir / "results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

    # ------------------------------------------------------------------
    # Derived state helpers
    # ------------------------------------------------------------------

    @property
    def started_training(self) -> bool:
        ckpt_file = self.checkpoint_dir / "checkpoint.pth"
        return ckpt_file.exists() and self.log_file.exists() and self.epochs_trained >= 0

    @property
    def epochs_trained(self) -> int:
        if not self.log_file.exists():
            return 0
        with open(self.log_file) as fh:
            lines = fh.readlines()
        if len(lines) <= 1:
            return 0
        last_line = lines[-1].strip()
        if not last_line:
            return 0
        epoch_str, _ = last_line.split(",")
        return int(epoch_str)

    @property
    def finished_training(self) -> bool:
        if self.config.early_stopping is not None:
            try:
                if self.train_loss() < self.config.early_stopping:
                    return True
            except RuntimeError:
                return False
        return self.epochs_trained >= self.config.epochs

