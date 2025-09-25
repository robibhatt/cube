import torch
import torch.nn as nn
from abc import abstractmethod, ABC
from typing import Tuple


from src.data.joint_distributions.configs.base import JointDistributionConfig


class _LinearModule(nn.Module):
    """A simple affine module ``x ↦ Ax + b`` used by ``linear_solve``."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        input_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias, requires_grad=False)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.shape[-1] != self.input_dim:
            raise RuntimeError(
                f"Expected input_dim={self.input_dim}, got tensor with trailing dim {x.shape[-1]}"
            )
        x_flat = x.reshape(-1, self.input_dim)
        y_flat = x_flat @ self.weight.T
        if self.bias is not None:
            y_flat = y_flat + self.bias
        return y_flat.reshape(*x.shape[:-1], self.output_dim)


class JointDistribution(ABC):
    def __init__(self, config: JointDistributionConfig, device: torch.device) -> None:
        """Store shapes and the ``device`` from ``config``."""

        self.input_dim: int = config.input_dim
        self.config = config

        # Accept the provided device without additional canonicalization.
        self.device: torch.device = torch.device(device)

        self.well_specified: bool = True

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.input_dim])

    @property
    def output_dim(self) -> int:
        return 1

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.output_dim])
    
    @abstractmethod
    def sample(self, n_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate ``n_samples`` from the joint distribution.

        Parameters
        ----------
        n_samples:
            Number of samples to draw.
        seed:
            Random seed used to create a ``torch.Generator`` on this
            distribution's device.

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            Input and output tensors of shapes ``(n_samples, input_dim)`` and
            ``(n_samples, output_dim)`` respectively.
        """
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        """Return a textual description of the distribution."""
        pass

    @abstractmethod
    def base_sample(
        self, n_samples: int, seed: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from the distribution's base distribution.

        The returned ``(base_X, base_y)`` pair is typically passed to
        :meth:`forward` or :meth:`forward_X` for transformation into the final
        ``(X, y)`` samples.
        """
        pass

    @abstractmethod
    def forward(self, base_X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform ``base_X`` from :meth:`base_sample` into ``(X, y)``."""
        pass

    @abstractmethod
    def forward_X(self, base_X: torch.Tensor) -> torch.Tensor:
        """Transform only the ``X`` portion of ``base_X`` returning ``X``."""
        pass
    
    @abstractmethod
    def preferred_provider(self) -> str:
        pass

    def average_output_variance(
        self, n_samples: int = 1000, seed: int = 0
    ) -> float:
        """Estimate the average variance of the output dimensions.

        Draws ``n_samples`` from the distribution and computes the variance of
        each coordinate of ``y``. The returned value is the mean of these
        variances.

        Parameters
        ----------
        n_samples:
            Number of samples used to estimate the variance.
        seed:
            Seed for ``torch.Generator`` passed to :meth:`sample`.

        Returns
        -------
        float
            The mean of the variances of each output coordinate.
        """
        _, y = self.sample(n_samples, seed)
        y_flat = y.reshape(n_samples, -1)
        var_per_coord = y_flat.var(dim=0, unbiased=False)
        return var_per_coord.mean().item()

    def _linear_regression_from_data(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        bias: bool,
        lambda_: float,
        device: torch.device,
    ) -> nn.Module:
        """Internal helper to fit a ridge regressor on provided data.

        Parameters
        ----------
        X, y:
            Tensors containing the training inputs and targets.  The first
            dimension is interpreted as the sample dimension and must match for
            ``X`` and ``y``.
        bias:
            Whether to fit an intercept term.  When ``True`` the data are
            centered before solving and the intercept is recovered in closed
            form.  When ``False`` the solve is performed without centering and
            the returned module has no bias parameter.
        lambda_:
            Ridge penalty ``λ``.  ``lambda_=0`` corresponds to an unregularized
            least squares fit.
        device:
            The device on which to return the fitted module.  The computations
            are always performed on CPU for numerical stability.
        """

        n_samples = X.shape[0]

        # Perform the solve on CPU
        X = X.to("cpu")
        y = y.to("cpu")

        if not torch.is_floating_point(X) or not torch.is_floating_point(y):
            raise TypeError(
                f"linear_solve expects floating-point tensors; got X.dtype={X.dtype}, y.dtype={y.dtype}"
            )
        if y.shape[0] != n_samples:
            raise RuntimeError(
                f"Expected X and y to have same first dim; got X{tuple(X.shape)}, y{tuple(y.shape)}."
            )

        # Flatten to (n, d_x) and (n, d_y)
        X_flat = X.reshape(n_samples, -1).contiguous()
        y_flat = y.reshape(n_samples, -1).contiguous()

        if torch.isnan(X_flat).any() or torch.isinf(X_flat).any():
            raise ValueError("X contains NaN or Inf values; cannot solve linear system.")
        if torch.isnan(y_flat).any() or torch.isinf(y_flat).any():
            raise ValueError("y contains NaN or Inf values; cannot solve linear system.")

        if y_flat.dtype != X_flat.dtype:
            y_flat = y_flat.to(X_flat.dtype)

        if bias:
            x_mean = X_flat.mean(dim=0, keepdim=True)
            y_mean = y_flat.mean(dim=0, keepdim=True)
        else:
            x_mean = torch.zeros(1, X_flat.shape[1], dtype=X_flat.dtype)
            y_mean = torch.zeros(1, y_flat.shape[1], dtype=y_flat.dtype)
        x_std = X_flat.std(dim=0, keepdim=True, unbiased=False)
        y_std = y_flat.std(dim=0, keepdim=True, unbiased=False)

        x_std_safe = x_std.clone()
        x_std_safe[x_std_safe == 0] = 1.0
        y_std_safe = y_std.clone()
        y_std_safe[y_std_safe == 0] = 1.0

        if bias:
            Xn = (X_flat - x_mean) / x_std_safe
            Yn = (y_flat - y_mean) / y_std_safe
        else:
            Xn = X_flat / x_std_safe
            Yn = y_flat / y_std_safe

        n, d = Xn.shape
        d_y = Yn.shape[1]

        def _ridge(A: torch.Tensor, B: torch.Tensor, lam: float) -> torch.Tensor:
            if lam <= 0.0:
                # Use least squares for the unregularized case to avoid
                # inverting potentially singular matrices.
                return torch.linalg.lstsq(A, B).solution
            sqrt_lam = lam ** 0.5
            A_aug = torch.cat([
                A,
                sqrt_lam * torch.eye(A.shape[1], dtype=A.dtype, device=A.device),
            ])
            B_aug = torch.cat(
                [
                    B,
                    torch.zeros(A.shape[1], B.shape[1], dtype=B.dtype, device=B.device),
                ]
            )
            return torch.linalg.lstsq(A_aug, B_aug).solution

        try:
            W_norm = _ridge(Xn, Yn, lambda_)
        except RuntimeError as e:
            raise RuntimeError(
                "ridge/least-squares solve failed. Context: "
                f"n_samples={n_samples}, d_x={d}, d_y={d_y}, lambda_={lambda_}, "
                f"dtype={X_flat.dtype}, device={X_flat.device}"
            ) from e

        M = (W_norm * y_std_safe) / x_std_safe.T
        if bias:
            b_flat = (y_mean - x_mean @ M).squeeze(0)
        else:
            b_flat = None
        A_flat = M.T

        d_y_expected = self.output_dim
        if A_flat.shape != (d_y_expected, d):
            raise RuntimeError(
                f"Computed A has shape {tuple(A_flat.shape)}; expected {(d_y_expected, d)}."
            )
        if bias and b_flat.shape != (d_y_expected,):
            raise RuntimeError(
                f"Computed b has shape {tuple(b_flat.shape)}; expected {(d_y_expected,)}."
            )

        if bias:
            module = _LinearModule(A_flat, b_flat, self.input_dim, self.output_dim).to(
                device
            )
        else:
            module = _LinearModule(A_flat, None, self.input_dim, self.output_dim).to(
                device
            )

        return module

    def linear_solve(
        self,
        seed: int,
        n_samples: int | None = None,
        bias: bool = True,
        lambda_: float = 0.0,
    ) -> tuple[nn.Module, float]:
        """Solve a ridge regression ``f(x)=Ax+b`` from samples.

        The regression is performed on feature‑wise normalized data and the
        resulting module is "unnormalized" so that it operates on the original
        input and output scales.  The ridge parameter ``λ`` is supplied by the
        caller via ``lambda_`` rather than being selected automatically.

        Bias terms are never regularized; when ``bias=True`` we solve on
        centered data and recover the intercept in closed form.  When
        ``bias=False`` the solve is performed without mean centering and the
        returned module contains no bias parameter.
        """
        # Number of features in x after flattening
        d_x = self.input_dim
        if n_samples is None:
            n_samples = max(10, d_x + 1)

        X, y = self.sample(n_samples, seed)

        if X.shape[0] != n_samples or y.shape[0] != n_samples:
            raise RuntimeError(
                f"Expected first dim == n_samples ({n_samples}); got X{tuple(X.shape)}, y{tuple(y.shape)}."
            )

        module = self._linear_regression_from_data(X, y, bias, lambda_, self.device)

        return module, float(lambda_)

    def k_fold_linear(
        self,
        seed: int,
        n_samples: int | None = None,
        bias: bool = True,
        lambdas: list[float] | None = None,
        k: int = 5,
    ) -> tuple[nn.Module, float]:
        """Fit a linear model selecting ``lambda`` via k-fold cross validation.

        This method samples ``n_samples`` from the joint distribution and
        performs ``k``-fold cross validation over the provided list of ridge
        penalties.  The ``lambda`` producing the lowest average validation MSE
        is selected, and a final model is trained on all of the data using that
        ``lambda``.

        Parameters
        ----------
        seed:
            Random seed used for sampling data and shuffling folds.
        n_samples:
            Number of samples to draw from the distribution.  If ``None``, a
            default of ``max(10, d_x + 1)`` is used, where ``d_x`` is the number
            of flattened input features.
        bias:
            Whether to include an intercept term in the regression.
        lambdas:
            List of ridge penalties to evaluate.  If ``None``, a small default
            set is used.
        k:
            Number of cross-validation folds.  Values larger than the number of
            available samples are clipped.

        Returns
        -------
        (nn.Module, float)
            The fitted linear module and the selected ``lambda`` value.
        """

        d_x = self.input_dim
        if n_samples is None:
            n_samples = max(10, d_x + 1)

        X, y = self.sample(n_samples, seed)

        if lambdas is None or len(lambdas) == 0:
            lambdas = [0.0, 0.1, 1.0]

        # Move data to CPU for solving
        X = X.to("cpu")
        y = y.to("cpu")

        n_total = X.shape[0]
        k = max(2, min(k, n_total))

        g = torch.Generator()
        g.manual_seed(seed)
        indices = torch.randperm(n_total, generator=g)
        folds = indices.chunk(k)

        best_lambda = lambdas[0]
        best_err = float("inf")

        for lam in lambdas:
            cv_err = 0.0
            for i in range(k):
                val_idx = folds[i]
                train_idx = torch.cat([folds[j] for j in range(k) if j != i])

                train_X, train_y = X[train_idx], y[train_idx]
                val_X, val_y = X[val_idx], y[val_idx]

                module = self._linear_regression_from_data(
                    train_X, train_y, bias, lam, torch.device("cpu")
                )
                pred = module(val_X)
                cv_err += torch.mean((pred - val_y) ** 2).item()

            avg_err = cv_err / k
            if avg_err < best_err:
                best_err = avg_err
                best_lambda = lam

        final_module = self._linear_regression_from_data(
            X, y, bias, best_lambda, self.device
        )

        return final_module, float(best_lambda)
