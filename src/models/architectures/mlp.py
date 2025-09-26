import math
import torch
import torch.nn as nn
import torch.nn.init as init
from dataclasses import replace
from typing import TYPE_CHECKING

from mup import Linear as MuLinear, MuReadout, set_base_shapes

from .activations import ACTIVATION_MAP

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .configs.mlp import MLPConfig


class MLP(nn.Module):
    """A configurable multi-layer perceptron.

    When ``config.mup`` is ``False`` (default) the network uses standard
    variance-preserving :class:`torch.nn.Linear` layers.  When ``config.mup`` is
    ``True`` the network mirrors the behaviour of :class:`~mup.MuLinear` based
    architectures used for μP scaling.
    """

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def __init__(self, config: "MLPConfig") -> None:
        super().__init__()
        self.config = config
        self.mup = config.mup

        # Honour the ``bias`` flag in the config (defaulting to ``True``)
        bias_flag = getattr(self.config, "bias", True)

        act_cls = self._get_activation()
        self.layers, self.linear_layers = self._build_layers(act_cls, bias_flag)
        self.net = nn.Sequential(*self.layers)

        # MuP base-shape setup before any freezing
        if self.mup and not getattr(self.config, "_is_base", False):
            base = self.get_base_model()
            set_base_shapes(self, base)

            # --- Re-initialize with μP-aware init now that infshape is attached ---
            try:
                import mup
                for m in self.modules():
                    if isinstance(m, (MuLinear, MuReadout)):
                        if self.config.activation == "quadratic":
                            mup.init.kaiming_uniform_(
                                m.weight, a=0.0, mode="fan_in", nonlinearity="linear"
                            )
                            with torch.no_grad():
                                m.weight.mul_(math.sqrt(0.5))
                        else:
                            mup.init.kaiming_uniform_(m.weight, a=0.0)
                        if getattr(m, "bias", None) is not None and m.bias is not None:
                            nn.init.constant_(m.bias, 0.01)
            except ImportError:
                # Fallback: call reset_parameters if available
                for m in self.modules():
                    if isinstance(m, (MuLinear, MuReadout)) and hasattr(m, "reset_parameters"):
                        m.reset_parameters()
                        if getattr(m, "bias", None) is not None and m.bias is not None:
                            nn.init.constant_(m.bias, 0.01)

        # Freeze any specified layers
        if self.config.frozen_layers:
            for idx in self.config.frozen_layers:
                layer = self.linear_layers[idx - 1]
                layer.weight.requires_grad_(False)
                if layer.bias is not None:
                    layer.bias.requires_grad_(False)

        if self.mup and not getattr(self.config, "_is_base", False):
            self._mup_post_init_sanity_check()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_layers(
        self, act_cls: type[nn.Module], bias_flag: bool
    ) -> tuple[list[nn.Module], list[nn.Module]]:
        """Construct network layers based on the current configuration."""
        layers: list[nn.Module] = []
        linear_layers: list[nn.Module] = []

        if self.config.start_activation:
            layers.append(act_cls())

        in_dim = self.config.input_dim

        if self.mup and self.config.end_activation:
            raise ValueError("end_activation is not supported when mup=True")

        for h in self.config.hidden_dims:
            lin = self._create_linear_layer(in_dim, h, bias_flag)
            layers.extend([lin, act_cls()])
            linear_layers.append(lin)
            in_dim = h

        out_lin = self._create_linear_layer(
            in_dim, self.config.output_dim, bias_flag, is_output=True
        )
        layers.append(out_lin)
        linear_layers.append(out_lin)

        if self.config.end_activation and not self.mup:
            layers.append(act_cls())

        return layers, linear_layers

    def _get_activation(self) -> type[nn.Module]:
        act = ACTIVATION_MAP.get(self.config.activation)
        if act is None:
            raise ValueError(f"Unsupported activation: {self.config.activation}")
        return act

    def _mup_post_init_sanity_check(
        self,
        *,
        max_report: int = 6,
        min_cos_numel: int = 256,
        bias_tol: float = 1e-6,
    ) -> None:
        """
        Post-__init__ μP sanity checks (only when mup=True and not on the base).

        Checks:
        1) Every trainable param has .infshape (means set_base_shapes() succeeded)
        2) No parameter storage is shared with a fresh base model
        3) Optional independence check on large weight matrices (skip biases)
        4) Within-model scale plausibility for hidden weights (exclude MuReadout)
        5) Biases are (approximately) the configured μP bias constant (default 0.01)
        """
        if not self.mup or getattr(self.config, "_is_base", False):
            return

        import math
        problems: list[str] = []

        # You can expose this in config; default here to 0.1 per your preference
        expected_bias = getattr(self.config, "mup_bias_init", 0.01)

        # ---- (1) Every param must carry μP infshape ----
        for n, p in self.named_parameters():
            if not hasattr(p, "infshape"):
                problems.append(f"Missing .infshape on param '{n}' (shape {tuple(p.shape)})")

        # Fresh base model (shape provider only; not μP-reinit'd)
        base = self.get_base_model()

        # ---- (2) No shared storage with base ----
        with torch.no_grad():
            for (n, p), (_, bp) in zip(self.named_parameters(), base.named_parameters()):
                if p.data_ptr() == bp.data_ptr():
                    problems.append(f"Shared storage between model and base for '{n}'")

        # ---- (5) Verify μP bias constant ~ expected_bias (and don't cosine-check biases) ----
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, (nn.Linear, MuLinear, MuReadout)) and (m.bias is not None):
                    b = m.bias.detach().float()
                    if not torch.allclose(b, torch.full_like(b, expected_bias), atol=bias_tol, rtol=0.0):
                        problems.append(
                            f"Bias != expected {expected_bias:g} in {m.__class__.__name__} "
                            f"(mean {b.mean().item():.6g}, max |Δ| {(b - expected_bias).abs().max().item():.3g})"
                        )

        # ---- (3) Independence check only on sizeable, equal-shape WEIGHTS (skip biases) ----
        with torch.no_grad():
            for (n, p), (_, bp) in zip(self.named_parameters(), base.named_parameters()):
                if p.ndim == 2 and p.shape == bp.shape and p.numel() >= min_cos_numel:
                    fp, fbp = p.view(-1).float(), bp.view(-1).float()
                    denom = (fp.norm() * fbp.norm()).clamp_min(1e-12)
                    cos = torch.dot(fp, fbp) / denom
                    if cos.abs().item() > 0.1:
                        problems.append(f"Unexpectedly high cosine ({cos.item():.3f}) model↔base for '{n}'")

        # ---- (4) Within-model scale plausibility for hidden weights (std * sqrt(fan_in)) ----
        with torch.no_grad():
            scales = []
            labels = []
            for mod in self.modules():
                is_linear = isinstance(mod, (nn.Linear, MuLinear))
                if is_linear and not isinstance(mod, MuReadout):
                    w = mod.weight.detach().float()
                    if w.ndim == 2 and w.numel() > 0:
                        fan_in = w.shape[1]
                        s = w.std().item() * math.sqrt(max(1, fan_in))
                        scales.append(s)
                        labels.append(mod.__class__.__name__)
            if scales:
                import torch as _t
                med = float(_t.tensor(scales).median().item())
                lo, hi = 0.3 * med, 3.0 * med  # very generous
                for s, lab in zip(scales, labels):
                    if not (lo <= s <= hi) or not math.isfinite(s):
                        problems.append(
                            f"Hidden weight scale outlier in {lab}: std*sqrt(fan_in)={s:.4g} (median {med:.4g})"
                        )

        if problems:
            msg = "\n  - " + "\n  - ".join(problems[:max_report])
            more = "" if len(problems) <= max_report else f"\n  (+{len(problems)-max_report} more)"
            raise RuntimeError(
                "μP post-init sanity check failed:" + msg + more +
                "\nNotes:\n"
                "  • Biases are intentionally initialized to a constant; this check enforces that.\n"
                "  • We avoid comparing scales to the base model (it is not μP-reinit'd).\n"
                "  • Cosine checks skip biases and tiny tensors; they only run where meaningful.\n"
            )

    def _init_nonlin_name_for_init(self) -> str:
        """Map activation names to what torch.init expects."""
        nonlin = self.config.activation.lower()
        alias = {"gelu": "relu", "silu": "relu"}  # Kaiming is okay for these
        return alias.get(nonlin, nonlin)

    def _init_quadratic(self, weight: torch.Tensor) -> None:
        """Variance-preserving init for quadratic activations.

        Uses a weight standard deviation of ``sqrt(0.5 / fan_in)`` which keeps the
        variance of ``x^2`` activations approximately constant across layers.
        """
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        std = math.sqrt(0.5 / fan_in)
        bound = math.sqrt(3.0) * std
        init.uniform_(weight, -bound, bound)

    def _create_linear_layer(
        self, in_features: int, out_features: int, bias: bool, is_output: bool = False
    ) -> nn.Module:
        """Return the appropriate linear layer for the current μP setting."""
        if self.mup:
            if is_output:
                return MuReadout(in_features, out_features, bias=bias)
            return MuLinear(in_features, out_features, bias=bias)
        return self._make_linear(in_features, out_features, bias=bias)

    def _make_linear(
        self, in_features: int, out_features: int, bias: bool = True
    ) -> nn.Linear:
        """Create a standard Linear layer initialised to preserve activation variance."""
        layer = nn.Linear(in_features, out_features, bias=bias)

        nonlin = self._init_nonlin_name_for_init()
        if nonlin == "quadratic":
            self._init_quadratic(layer.weight)
        elif nonlin in {"linear", "tanh", "sigmoid"}:
            # Xavier for saturated/linear activations
            init.xavier_uniform_(layer.weight)
        else:
            # ReLU family → Kaiming/He
            init.kaiming_uniform_(layer.weight, a=0.0, mode="fan_in", nonlinearity=nonlin)

        if layer.bias is not None:
            init.zeros_(layer.bias)
        return layer

    def copy_weights_from_donor(self, donor: "MLP", layers: list[int]) -> None:
        """Copy weights and biases from ``donor`` for the given ``layers``.

        Parameters
        ----------
        donor:
            An :class:`MLP` instance whose weights will be copied.
        layers:
            A list of one-indexed layer numbers indicating which linear layers
            should have their parameters overwritten.
        """
        self._validate_donor(donor)
        for idx in layers:
            tgt = self.linear_layers[idx - 1]
            src = donor.linear_layers[idx - 1]
            if tgt.weight.shape != src.weight.shape:
                raise ValueError(
                    f"Shape mismatch for layer {idx}: {tgt.weight.shape} vs {src.weight.shape}"
                )
            tgt.weight.data.copy_(src.weight.data)

            if tgt.bias is not None and src.bias is not None:
                if tgt.bias.shape != src.bias.shape:
                    raise ValueError(
                        f"Bias shape mismatch for layer {idx}: {tgt.bias.shape} vs {src.bias.shape}"
                    )
                tgt.bias.data.copy_(src.bias.data)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x):
        return self.net(x)

    # ------------------------------------------------------------------
    # MuP utilities
    # ------------------------------------------------------------------
    def get_base_model(self):  # type: ignore[override]
        """Return a clean base-width model for μP shape registration.

        - Strips any freezing (frozen_layers → []) so set_base_shapes() sees a fully
        trainable parameter tree.
        - Forces end_activation=False (μP doesn't support it).
        - Mirrors input/output dims, activation, bias, and start_activation.
        - Uses a stable small width for all hidden layers (default 64), or a user-
        specified `base_width` attribute on the config if present.
        """
        if not self.mup:
            return None

        # Allow an optional config.base_width override; default to 64.
        base_w = getattr(self.config, "base_width", 64)
        base_hidden = [base_w] * len(self.config.hidden_dims)

        # Build a fresh config with μP on, no freezing, and no end activation.
        base_cfg = replace(
            self.config,
            hidden_dims=base_hidden,
            mup=True,
            frozen_layers=[],           # never freeze the base
            end_activation=False,       # μP doesn't support an end activation
        )

        # Preserve bias and start_activation exactly as-is; replace() already did.
        # Mark as base so __init__ skips set_base_shapes() recursion.
        setattr(base_cfg, "_is_base", True)

        # IMPORTANT: do not mutate self or reuse modules; make a fresh model.
        base_model = MLP(base_cfg)

        return base_model

    def _validate_donor(self, donor: "MLP") -> None:
        """Ensure donor model matches architecture of this model."""
        if not isinstance(donor, MLP):
            raise TypeError("donor must be an instance of MLP")
        if len(donor.linear_layers) != len(self.linear_layers):
            raise ValueError("Donor and target have different number of layers")
        for i, (tgt, src) in enumerate(
            zip(self.linear_layers, donor.linear_layers), start=1
        ):
            if tgt.weight.shape != src.weight.shape:
                raise ValueError(
                    f"Shape mismatch for layer {i}: {tgt.weight.shape} vs {src.weight.shape}"
                )
            if (tgt.bias is None) != (src.bias is None):
                # different bias presence is allowed, but we cannot copy
                continue
