import random
from pathlib import Path
from copy import deepcopy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from typing import Tuple
from mup import Linear as MuLinear, MuReadout, set_base_shapes

from src.models.architectures.model import Model

LINEAR_LAYERS = (nn.Linear, MuLinear, MuReadout)
from src.training.trainer_factory import trainer_from_dir


class NeuronComparator:
    """Analyse how well a student MLP mimics a teacher MLP.

    The comparator normalises the weight vectors of the *first* hidden layer in
    both networks so that each has unit norm and rescales the outgoing weights
    from those neurons accordingly.  After normalisation it computes, for every
    student neuron, the angle to the closest teacher neuron.  For two-layer
    networks (one hidden layer with scalar output) the rescaled outgoing weights
    and bias terms are also recorded, enabling additional diagnostic plots.  For
    deeper networks only the angle statistics and loss computations are
    available.

    The comparator is constructed from trainer directories for both the teacher
    and the student.  Models are loaded from those directories and a fresh test
    dataset is generated from the student's joint distribution using a random
    seed.
    """

    def __init__(self, teacher_dir: Path, student_dir: Path) -> None:
        teacher_trainer = trainer_from_dir(teacher_dir)
        student_trainer = trainer_from_dir(student_dir)

        teacher = teacher_trainer._load_model()
        student = student_trainer._load_model()

        if teacher.config.model_type != "MLP" or student.config.model_type != "MLP":
            raise TypeError("teacher and student must be instances of architectures.MLP")

        if teacher.config.input_dim != student.config.input_dim:
            raise ValueError("teacher and student must have the same input dimension")

        if teacher.config.output_dim != student.config.output_dim:
            raise ValueError("teacher and student must have matching output dimensions")

        self.teacher = teacher
        self.student = student

        from src.data.providers.data_provider_factory import (
            create_data_provider_from_distribution,
        )

        seed = random.randint(0, 2**32 - 1)
        batch_size = student_trainer.config.batch_size or 1024
        dataset_size = student_trainer.config.test_size
        self.test_loader = create_data_provider_from_distribution(
            student_trainer.joint_distribution,
            batch_size,
            dataset_size,
            seed,
        )

        # Track whether we can generate the full suite of plots (single hidden layer)
        self._supports_full_plots = (
            len(self.teacher.config.hidden_dims) == 1
            and self.teacher.config.output_dim == 1
        )

        # Normalise networks and compute pairwise angles
        t_w, _, _ = self._normalise_network(self.teacher)
        s_w, s_a, s_b = self._normalise_network(self.student)

        # cosines → angles
        cos = np.clip(t_w @ s_w.T, -1.0, 1.0)
        angles = np.arccos(cos)
        self.theta: np.ndarray = angles.min(axis=0)
        self.a: Optional[np.ndarray] = s_a if self._supports_full_plots else None
        self.b: Optional[np.ndarray] = s_b if self._supports_full_plots else None

        # Measure how well the student mimics the teacher on the supplied data
        self.mse: float = self._compute_mse()

    @property
    def supports_full_plots(self) -> bool:
        """Return ``True`` when the full set of plots is available."""
        return self._supports_full_plots

    # ------------------------------------------------------------------
    def _normalise_network(self, mlp: Model) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return unit first-layer weights, rescaled outgoing weights and biases."""
        hidden_w, out_w, hidden_b = self._extract_weights(mlp)
        norms = np.linalg.norm(hidden_w, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        hidden_unit = hidden_w / norms
        out_scaled = out_w * norms.squeeze()
        bias_scaled = hidden_b / norms.squeeze()
        return hidden_unit, out_scaled, bias_scaled

    def _extract_weights(self, mlp: Model) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract first-layer weights, next-layer weights and first-layer biases.

        Networks without bias parameters are treated as having zero bias."""
        linears = [
            m for m in mlp.modules() if isinstance(m, LINEAR_LAYERS) or hasattr(m, "weight")
        ]
        if len(linears) < 2:
            raise ValueError("MLP must contain at least two linear layers")

        hidden_layer = linears[0]
        next_layer = linears[1]

        hidden = hidden_layer.weight.detach().cpu().numpy()
        hidden_bias = hidden_layer.bias
        if hidden_bias is None:
            hidden_bias_arr = np.zeros(hidden.shape[0], dtype=hidden.dtype)
        else:
            hidden_bias_arr = hidden_bias.detach().cpu().numpy().reshape(-1)

        out = next_layer.weight.detach().cpu().numpy()
        if out.shape[1] != hidden.shape[0]:
            raise ValueError("First two layers must have compatible widths")
        if out.shape[0] == 1:
            out = out.reshape(-1)

        # ``MuReadout``/``MuLinear`` store unscaled parameters; apply width
        # multipliers where necessary to obtain effective weights.
        width_mult = getattr(next_layer, "width_mult", None)
        if callable(width_mult):
            out = out * getattr(next_layer, "output_mult", 1.0) / width_mult()

        return hidden, out, hidden_bias_arr

    def _compute_mse_for(self, student: Model) -> float:
        """Compute mean squared error between ``student`` and the teacher."""
        self.teacher.eval()
        student.eval()
        losses = []
        with torch.no_grad():
            for x, _ in self.test_loader:
                t_out = self.teacher(x)
                s_out = student(x)
                losses.append(torch.mean((t_out - s_out) ** 2).item())
        return float(np.mean(losses))

    def _compute_mse(self) -> float:
        """Compute mean squared error between teacher and stored student."""
        return self._compute_mse_for(self.student)

    # ------------------------------------------------------------------
    def _theta_threshold_losses(self, thetas: np.ndarray) -> list[float]:
        """Return test MSE for a sequence of ``theta`` thresholds."""

        losses: list[float] = []

        # Work on a copy so that original weights remain unchanged
        student_clone = deepcopy(self.student)

        # ``deepcopy`` strips μP shape metadata; restore if necessary
        base_model = student_clone.get_base_model()
        if base_model is not None:
            # The original student already has scaled parameters; avoid
            # rescaling when restoring μP metadata on the clone.
            set_base_shapes(student_clone, base_model, rescale_params=False)

        linears = [m for m in student_clone.modules() if isinstance(m, LINEAR_LAYERS)]
        out_layer = linears[1]
        base_weight = out_layer.weight.detach().clone()
        theta_tensor = torch.tensor(
            self.theta, dtype=base_weight.dtype, device=base_weight.device
        )

        for th in thetas:
            mask = (theta_tensor <= th).float()
            with torch.no_grad():
                out_layer.weight.copy_(base_weight * mask)
            losses.append(self._compute_mse_for(student_clone))

        return losses

    # ------------------------------------------------------------------
    def _select_thetas(self, total: int = 40, focus: int = 20) -> np.ndarray:
        """Return a subset of angles emphasising small ``theta`` values.

        Parameters
        ----------
        total : int, optional (default=40)
            Maximum number of angles to return.
        focus : int, optional (default=20)
            Number of smallest angles to include explicitly.  The remaining
            angles are selected evenly from the rest of the range.
        """

        sorted_theta = np.sort(self.theta)
        if sorted_theta.size <= total:
            return sorted_theta

        small = sorted_theta[:focus]
        remaining = sorted_theta[focus:]
        idx = np.linspace(0, remaining.size - 1, total - focus, dtype=int)
        return np.concatenate([small, remaining[idx]])

    # ------------------------------------------------------------------
    def plot_angle_histogram(self, bins: int = 30) -> None:
        """Plot a histogram of the angles ``theta`` between neurons.

        Parameters
        ----------
        bins : int, optional (default=30)
            Number of histogram bins.
        """
        plt.hist(self.theta, bins=bins)
        plt.xlabel("Angle to nearest teacher (rad)")
        plt.ylabel("Count")
        plt.title("Distribution of student-to-teacher angles")

    def plot_angle_weight_scatter(self) -> None:
        """Scatter ``theta`` against the rescaled outgoing weights ``a``."""
        if not self._supports_full_plots or self.a is None:
            return

        plt.scatter(self.theta, self.a)
        plt.xlabel("Angle to nearest teacher (rad)")
        plt.ylabel("Rescaled output weight")
        plt.title("Angle vs output weight")

    def plot_weight_bias_scatter(self) -> None:
        """Scatter rescaled output weights against rescaled bias terms.

        Points are coloured by ``theta`` and a colour bar indicates the
        mapping between colour and angle.
        """
        if not self._supports_full_plots or self.a is None or self.b is None:
            return

        sc = plt.scatter(self.a, self.b, c=self.theta, cmap="viridis")
        plt.colorbar(sc, label="Angle to nearest teacher (rad)")
        plt.xlabel("Rescaled output weight")
        plt.ylabel("Rescaled bias term")
        plt.title("Output weight vs bias (coloured by angle)")

    def plot_theta_test_loss(self) -> None:
        r"""Plot the test MSE when using neurons within angle ``theta``.

        The plot evaluates up to forty angle thresholds to speed up the
        computation.  The twenty smallest angles are always included while the
        remaining thresholds are sampled evenly from the rest of the range.  For
        each selected threshold :math:`\theta_i` the student is evaluated using
        only those neurons whose angle does not exceed :math:`\theta_i`.  The
        resulting test losses are connected with a line and each evaluated
        threshold is indicated with a tick mark.
        """

        thetas = self._select_thetas()
        losses = self._theta_threshold_losses(thetas)

        plt.plot(thetas, losses, marker="|")
        plt.xlabel("Theta threshold (rad)")
        plt.ylabel("Test MSE")
        plt.title("Test loss vs theta threshold")

    # ------------------------------------------------------------------
    def plot_gaussian_theta_test_loss(self) -> bool:
        """Plot empirical and Gaussian-theory test loss vs ``theta`` threshold.

        This assumes both teacher and student are two-layer ReLU networks with
        *no bias terms* and scalar outputs.  Up to forty angle thresholds are
        evaluated, comprising the twenty smallest angles and twenty more sampled
        evenly from the remaining range.  For each threshold the empirical test
        loss (using only neurons with angle at most :math:`\theta_i`) and the
        corresponding Gaussian-theory prediction are computed and plotted with
        tick marks.

        Returns
        -------
        bool
            ``True`` if the plot was generated, ``False`` if the network
            configuration does not satisfy the required assumptions.
        """

        if not self._supports_full_plots:
            return False

        if getattr(self.teacher.config, "bias", True) or getattr(self.student.config, "bias", True):
            return False

        if (
            self.teacher.config.activation.lower() != "relu"
            or self.student.config.activation.lower() != "relu"
        ):
            return False

        thetas = self._select_thetas()

        # Empirical losses using the existing routine
        empirical = self._theta_threshold_losses(thetas)

        # Extract weight matrices
        t_w1, t_w2, _ = self._extract_weights(self.teacher)
        s_w1, s_w2, _ = self._extract_weights(self.student)

        # Kernel computation helper using the ReLU arccosine kernel
        def _relu_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            a_norm = np.linalg.norm(a, axis=1, keepdims=True)
            b_norm = np.linalg.norm(b, axis=1, keepdims=True)
            a_safe = np.where(a_norm == 0.0, 1.0, a_norm)
            b_safe = np.where(b_norm == 0.0, 1.0, b_norm)
            cos = np.clip((a @ b.T) / (a_safe * b_safe.T), -1.0, 1.0)
            theta = np.arccos(cos)
            return (a_norm * b_norm.T) / (2 * np.pi) * (
                np.sin(theta) + (np.pi - theta) * cos
            )

        K_f = _relu_kernel(t_w1, t_w1)
        K_g = _relu_kernel(s_w1, s_w1)
        K_fg = _relu_kernel(t_w1, s_w1)

        def _quad_form(w: np.ndarray, K: np.ndarray) -> float:
            w = w.reshape(-1)
            return float(w @ K @ w)

        teacher_term = _quad_form(t_w2, K_f)

        theta_arr = self.theta
        theory_losses: list[float] = []
        for th in thetas:
            mask = (theta_arr <= th).astype(float)
            w2_mask = s_w2 * mask
            term_g = _quad_form(w2_mask, K_g)
            term_fg = float(t_w2 @ K_fg @ w2_mask)
            theory_losses.append(teacher_term + term_g - 2.0 * term_fg)

        plt.plot(thetas, empirical, marker="|", label="Empirical")
        plt.plot(thetas, theory_losses, marker="|", label="Gaussian theory")
        plt.xlabel("Theta threshold (rad)")
        plt.ylabel("Test MSE")
        plt.title("Gaussian vs empirical test loss")
        plt.legend()

        return True
