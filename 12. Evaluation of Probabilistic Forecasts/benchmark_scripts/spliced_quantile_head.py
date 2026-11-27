# ============================================================
# Spline + GPD-tail probabilistic head for NeuralForecast models
# ============================================================
#
# Lets NeuralForecast backbones (TiDE, TimeXer, ...) use the same
# distributional head as our model (ANGEL) instead of MQLoss:
#
#   backbone -> raw params (B,H,P)
#            -> I-spline quantile body  (ISplineQuantile)
#            -> spliced GPD tails       (SplicedGPDQuantile)
#            -> Q(u) on a fixed u-grid
#
# trained with the same CRPS / twCRPS / combined objective.
#
# The head classes and the twCRPS chaining function are imported from
# the supplement repo ("[SUPPLEMENT] Pytorch lightning forecasting repo"),
# so there is a single source of truth for the head.
#
# How it plugs into NeuralForecast
# --------------------------------
# The loss declares `is_distribution_output = True`, which makes NF call
#   domain_map(raw) -> scale_decouple(params, loc, scale) -> loss(y, distr_args)
# with `y` on the ORIGINAL scale (not scaler-normalised). The head builds
# Q_norm(u) in the scaler-normalised space and we map it back with
#   Q(u) = loc + scale * Q_norm(u)
# which is exact: the I-spline body is location/scale parametrised and the
# GPD tails are slope-matched to the body, so the spliced quantile function
# is affine-equivariant. Losses (incl. twCRPS thresholds in bps) are thus
# computed on the original target scale, as in the supplement repo.
#
# At prediction time NF calls `sample()`, which here returns the exact
# quantiles on `output_quantiles` (no Monte Carlo) plus the grid mean.
# ============================================================

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ------------------------------------------------------------
# Import the head from the supplement repo
# ------------------------------------------------------------

_DEFAULT_SUPPLEMENT_DIR = (
    Path(__file__).resolve().parents[2] / "[SUPPLEMENT] Pytorch lightning forecasting repo"
)
SUPPLEMENT_DIR = Path(os.environ.get("SUPPLEMENT_REPO_DIR", _DEFAULT_SUPPLEMENT_DIR))

if not SUPPLEMENT_DIR.exists():
    raise FileNotFoundError(
        f"Supplement repo not found at {SUPPLEMENT_DIR}. "
        "Set SUPPLEMENT_REPO_DIR to its location."
    )
if str(SUPPLEMENT_DIR) not in sys.path:
    sys.path.insert(0, str(SUPPLEMENT_DIR))

from models.helper_classes import (  # noqa: E402
    ChebyshevQuantile,
    ISplineQuantile,
    SplicedGPDQuantile,
)
from models.losses import ChainingFunction  # noqa: E402
from models.utils import power_tails_u  # noqa: E402


class SafeSplicedGPDQuantile(SplicedGPDQuantile):
    """
    SplicedGPDQuantile with NaN-safe gradients at xi -> 0.

    The parent switches to the exponential tail with torch.where when
    |xi| < 1e-6, but still evaluates the GPD branch beta/xi * (r^-xi - 1).
    With xi_min = 0 the sigmoid underflows (xi == 0 exactly once raw_xi is very
    negative), the unused branch is inf, and its gradient is inf * 0 = NaN,
    which poisons the weights. Forward values are identical to the parent; we
    only substitute xi = 1 inside the unused branch ("double where" trick).
    """

    def _upper_tail_quantile(self, u_tail, xU, betaU, xiU):
        r = ((1.0 - u_tail) / (1.0 - self.u_high)).view(1, 1, -1).clamp_min(self.eps)
        xU = xU.unsqueeze(-1)
        betaU = betaU.unsqueeze(-1).clamp_min(self.eps)
        xiU = xiU.unsqueeze(-1)

        near_zero = xiU.abs() < 1e-6
        xi_safe = torch.where(near_zero, torch.ones_like(xiU), xiU)

        Q_exp = xU - betaU * torch.log(r)
        Q_gpd = xU + (betaU / xi_safe) * (torch.pow(r, -xi_safe) - 1.0)
        Q_up = torch.where(near_zero, Q_exp, Q_gpd)

        q_exp = betaU / ((1.0 - self.u_high) * r)
        q_gpd = (betaU / (1.0 - self.u_high)) * torch.pow(r, -xi_safe - 1.0)
        q_up = torch.where(near_zero, q_exp, q_gpd)
        return Q_up, q_up

    def _lower_tail_quantile(self, u_tail, xL, betaL, xiL):
        r = (u_tail / self.u_low).view(1, 1, -1).clamp_min(self.eps)
        xL = xL.unsqueeze(-1)
        betaL = betaL.unsqueeze(-1).clamp_min(self.eps)
        xiL = xiL.unsqueeze(-1)

        near_zero = xiL.abs() < 1e-6
        xi_safe = torch.where(near_zero, torch.ones_like(xiL), xiL)

        Q_exp = xL + betaL * torch.log(r)
        Q_gpd = xL - (betaL / xi_safe) * (torch.pow(r, -xi_safe) - 1.0)
        Q_lo = torch.where(near_zero, Q_exp, Q_gpd)

        q_exp = betaL / (self.u_low * r)
        q_gpd = (betaL / self.u_low) * torch.pow(r, -xi_safe - 1.0)
        q_lo = torch.where(near_zero, q_exp, q_gpd)
        return Q_lo, q_lo


# Hyperparameters of the head / loss as reported in the paper
# (scripts/run_ANGEL.sh in the supplement repo).
PAPER_HEAD_CONFIG = dict(
    quantile_decomp="spline",
    n_basis=6,                 # --n_cheb 6
    spline_degree=3,
    knot_kind="power-tail",
    knot_p=3.0,
    revin_type="revin",
    u_grid_size=256,           # power-tail grid
    tail_model="gpd",
    gpd_u_low=0.1,
    gpd_u_high=0.9,
    gpd_xi_min=0.0,
    gpd_xi_max=1.0,
    dist_loss="combined",
    rho_combined=0.7,
    twcrps_threshold_low=-10.0,   # bps
    twcrps_threshold_high=10.0,   # bps
    twcrps_side="two_sided",
    twcrps_smooth_h=1.0,          # bps
)


class SplicedSplineQuantileLoss(nn.Module):
    """
    NeuralForecast loss wrapping the I-spline + spliced-GPD quantile head.

    Parameters per (time step, series):
        P = 2 + n_basis (+ 2 if tail_model == "gpd")
          = [b, s_raw, a_0..a_{K-1}, (xi_low_raw, xi_high_raw)]

    Args:
        output_quantiles: quantile levels returned at prediction time.
            Defaults to the training u-grid. If equal to the training grid,
            the head's grid quantiles are returned directly; otherwise they are
            evaluated exactly (GPD tails analytic, body interpolated).
        u_grid: training u-grid (J,). Defaults to power_tails_u(u_grid_size),
            the grid used by our model.
        Remaining args mirror the supplement repo CLI (see PAPER_HEAD_CONFIG).
    """

    def __init__(
        self,
        output_quantiles=None,
        u_grid=None,
        quantile_decomp="spline",
        n_basis=6,
        spline_degree=3,
        knot_kind="power-tail",
        knot_p=3.0,
        revin_type="revin",
        u_grid_size=256,
        tail_model="gpd",
        gpd_u_low=0.1,
        gpd_u_high=0.9,
        gpd_xi_min=0.0,
        gpd_xi_max=1.0,
        dist_loss="combined",
        rho_combined=0.7,
        twcrps_threshold_low=-10.0,
        twcrps_threshold_high=10.0,
        twcrps_side="two_sided",
        twcrps_smooth_h=1.0,
        horizon_weight=None,
    ):
        super().__init__()

        # ---- training u-grid (same construction as models/common.py) ----
        if u_grid is None:
            u = power_tails_u(int(u_grid_size))
        else:
            u = torch.as_tensor(np.asarray(u_grid, dtype=np.float32))
        u = u.float().clamp(min=1e-6, max=1.0 - 1e-6)

        # ---- quantile body ----
        if quantile_decomp == "spline":
            body = ISplineQuantile(
                K=n_basis, u_grid=u, normalize=True, revin_type=revin_type,
                degree=spline_degree, knot_kind=knot_kind, knot_p=knot_p,
            )
        elif quantile_decomp == "chebyshev":
            body = ChebyshevQuantile(
                K=n_basis, u_grid=u, normalize=True, revin_type=revin_type,
            )
        else:
            raise ValueError(f"Unknown quantile_decomp: {quantile_decomp}")

        # ---- tails ----
        if tail_model == "gpd":
            self.head = SafeSplicedGPDQuantile(
                body_quantile=body,
                body_param_dim=2 + n_basis,
                u_low=gpd_u_low,
                u_high=gpd_u_high,
                xi_min=gpd_xi_min,
                xi_max=gpd_xi_max,
                eps=1e-6,
            )
            n_params = 4 + n_basis
        elif tail_model == "none":
            self.head = body
            n_params = 2 + n_basis
        else:
            raise ValueError(f"Unknown tail_model: {tail_model}")

        self.tail_model = tail_model
        self.n_params = n_params

        # ---- objective ----
        if dist_loss not in {"crps", "twcrps", "combined"}:
            raise ValueError(f"Unknown dist_loss: {dist_loss}")
        self.dist_loss = dist_loss
        self.rho_combined = float(rho_combined)
        self.chain = ChainingFunction(
            threshold_low=twcrps_threshold_low,
            threshold_high=twcrps_threshold_high,
            side=twcrps_side,
            smooth_h=twcrps_smooth_h,
        )

        self.horizon_weight = (
            None if horizon_weight is None
            else torch.as_tensor(np.asarray(horizon_weight).flatten(), dtype=torch.float32)
        )

        # ---- NeuralForecast interface attributes ----
        self.is_distribution_output = True
        self.outputsize_multiplier = n_params
        self.return_params = False
        self.has_predicted = False
        self.num_samples = None
        self._set_output_quantiles(output_quantiles)

    # ------------------------------------------------------------------
    # Output quantile levels / column names
    # ------------------------------------------------------------------

    @property
    def u_train(self) -> torch.Tensor:
        return self.head.u

    def _set_output_quantiles(self, q):
        u_train = self.head.u.detach().cpu().numpy().astype(np.float64)
        if q is None:
            q = u_train
        q = np.asarray(q, dtype=np.float64).flatten()
        if np.any(np.diff(q) <= 0):
            raise ValueError("output_quantiles must be strictly increasing.")

        self.quantiles = nn.Parameter(torch.tensor(q, dtype=torch.float32), requires_grad=False)
        self._output_on_train_grid = bool(
            q.shape == u_train.shape and np.allclose(q, u_train, rtol=0.0, atol=1e-7)
        )
        # Index-based names: the 256-point grid has levels too close together
        # for NF's "-lo-xx.xx" names to stay unique. Map back via `self.quantiles`.
        self.output_names = [""] + [f"-q{j:03d}" for j in range(len(q))]

    def update_quantile(self, q=None):
        # Called by NF at predict(); q=None keeps the configured levels.
        if q is not None:
            self._set_output_quantiles(q)
            self.has_predicted = True

    # ------------------------------------------------------------------
    # NF hooks
    # ------------------------------------------------------------------

    def domain_map(self, output: torch.Tensor) -> torch.Tensor:
        """
        output: [B, H, P*N] (param-major, as produced by NF models)
        returns params: [B, H, N, P]
        """
        B, H, PN = output.shape
        if PN % self.n_params != 0:
            raise ValueError(
                f"Backbone output dim {PN} is not a multiple of n_params={self.n_params}."
            )
        N = PN // self.n_params
        return output.reshape(B, H, self.n_params, N).permute(0, 1, 3, 2)

    def _head_quantiles(self, params: torch.Tensor):
        """params [B,H,N,P] (normalised space) -> Q_norm [B,H,N,J]"""
        B, H, N, P = params.shape
        p = params.reshape(B, H * N, P)
        # scale parameter must be > 0 (ANGEL applies softplus here too)
        p = torch.cat([p[..., :1], nn.functional.softplus(p[..., 1:2]), p[..., 2:]], dim=-1)
        Q, _ = self.head(p)
        return Q.reshape(B, H, N, -1), p

    @staticmethod
    def _broadcast_loc_scale(t: torch.Tensor, N: int) -> torch.Tensor:
        # NF passes [B, 1] (univariate) or [B, 1, N] (multivariate) -> [B, 1, N, 1]
        if t.ndim == 2:
            t = t.unsqueeze(-1)
        return t.reshape(t.shape[0], t.shape[1], N).unsqueeze(-1)

    def scale_decouple(self, output: torch.Tensor, loc=None, scale=None, eps: float = 0.0):
        B, H, N, _ = output.shape
        Q_norm, p_flat = self._head_quantiles(output)

        if loc is None:
            loc_b = torch.zeros(B, 1, N, 1, device=output.device, dtype=output.dtype)
        else:
            loc_b = self._broadcast_loc_scale(loc, N)
        if scale is None:
            scale_b = torch.ones(B, 1, N, 1, device=output.device, dtype=output.dtype)
        else:
            scale_b = self._broadcast_loc_scale(scale, N)

        Q = loc_b + scale_b * Q_norm                       # [B,H,N,J], original scale
        return (Q, p_flat, loc_b, scale_b)

    def get_distribution(self, distr_args, **kwargs):
        raise NotImplementedError(
            "SplicedSplineQuantileLoss has no torch Distribution; "
            "use it as both loss and valid_loss."
        )

    def sample(self, distr_args, num_samples=None):
        """
        Exact quantiles (no sampling). Returns (None, mean [B,H,N,1], quants [B,H,N,Q]).
        """
        Q, p_flat, loc_b, scale_b = distr_args
        B, H, N, J = Q.shape

        wu = self.head.wu.to(Q.dtype)
        mean = (Q * wu).sum(-1, keepdim=True) / wu.sum()

        if self._output_on_train_grid:
            quants = Q
        else:
            u_eval = self.quantiles.to(device=Q.device, dtype=Q.dtype)
            if self.tail_model == "gpd":
                Qn = self.head.quantile_at_levels(p_flat, u_eval)          # [B, H*N, A]
                quants = loc_b + scale_b * Qn.reshape(B, H, N, -1)
            else:
                quants = _interp_quantile_grid(Q, self.head.u.to(Q.dtype), u_eval)

        self.distr_mean = mean.squeeze(-1)
        return None, mean, quants

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _weighted_pinball_integral(self, y: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """2 * sum_j w_j * pinball_{u_j}(y - Q_j)  ->  CRPS approximation, [B,H,N]"""
        u = self.head.u.to(Q.dtype)
        wu = self.head.wu.to(Q.dtype)
        e = y.unsqueeze(-1) - Q
        pinball = torch.maximum(u * e, (u - 1.0) * e)
        return 2.0 * (pinball * wu).sum(-1)

    def __call__(self, y: torch.Tensor, distr_args, mask=None):
        Q = distr_args[0]
        B, H, N, _ = Q.shape
        y = y.reshape(B, H, N)

        if self.dist_loss in {"crps", "combined"}:
            crps = self._weighted_pinball_integral(y, Q)
        if self.dist_loss in {"twcrps", "combined"}:
            twcrps = self._weighted_pinball_integral(self.chain(y), self.chain(Q))

        if self.dist_loss == "crps":
            loss_values = crps
        elif self.dist_loss == "twcrps":
            loss_values = twcrps
        else:
            loss_values = self.rho_combined * crps + (1.0 - self.rho_combined) * twcrps

        weights = torch.ones_like(y) if mask is None else mask.reshape(B, H, N).to(y.dtype)
        if self.horizon_weight is not None:
            weights = weights * self.horizon_weight.to(y.device)[None, :, None]

        return (loss_values * weights).sum() / weights.sum().clamp_min(1.0)


def _interp_quantile_grid(Q: torch.Tensor, u: torch.Tensor, u_eval: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of Q [...,J] on grid u (J,) at u_eval (A,), flat extrapolation."""
    u_eval = u_eval.clamp(u[0], u[-1])
    idx = torch.searchsorted(u, u_eval).clamp(1, u.numel() - 1)
    u_lo, u_hi = u[idx - 1], u[idx]
    w = (u_eval - u_lo) / (u_hi - u_lo).clamp_min(1e-12)
    return (1.0 - w) * Q[..., idx - 1] + w * Q[..., idx]


# ============================================================
# NeuralForecast cross-validation output -> comparison pickle
# ============================================================

def cv_df_to_quantile_pickle_dict_from_loss(cv_df, loss, model_name, extra=None):
    """
    Convert NF cross_validation output of a model trained with
    SplicedSplineQuantileLoss into the comparison format:

      true:   (B,H)
      Q:      (B,H,J)   on u_grid = loss.quantiles
      mean:   (B,H)     grid mean of the predictive distribution
      u_grid: (J,)
      ds:     (B,H)
      meta:   cutoff dataframe
    """
    cv_df = cv_df.copy()
    cv_df["ds"] = pd.to_datetime(cv_df["ds"])
    cv_df["cutoff"] = pd.to_datetime(cv_df["cutoff"])
    cv_df = cv_df.sort_values(["cutoff", "ds"]).reset_index(drop=True)

    counts = cv_df.groupby("cutoff").size()
    H = int(counts.iloc[0])
    if not (counts == H).all():
        bad = counts[counts != H]
        raise ValueError(f"Cutoffs with != {H} rows: {bad.head()}")
    B = len(counts)

    u_grid = loss.quantiles.detach().cpu().numpy().astype(np.float64)
    q_cols = [f"{model_name}{n}" for n in loss.output_names[1:]]
    missing = [c for c in q_cols if c not in cv_df.columns]
    if missing:
        raise KeyError(f"Missing quantile columns, e.g. {missing[:5]}")

    Q = cv_df[q_cols].to_numpy(dtype=np.float64).reshape(B, H, len(u_grid))
    # Monotone by construction; this only guards against float round-off.
    Q = np.maximum.accumulate(Q, axis=-1)

    A = {
        "true": cv_df["y"].to_numpy(dtype=np.float64).reshape(B, H),
        "Q": Q,
        "mean": cv_df[model_name].to_numpy(dtype=np.float64).reshape(B, H),
        "u_grid": u_grid,
        "ds": cv_df["ds"].to_numpy(dtype="datetime64[ns]").reshape(B, H),
        "meta": pd.DataFrame({"cutoff": counts.index}),
        "head": "spline_gpd",
    }
    if extra:
        A.update(extra)

    print(
        f"Converted {model_name} spline+GPD forecasts: "
        f"true={A['true'].shape}, Q={A['Q'].shape}, J={len(u_grid)}"
    )
    return A
