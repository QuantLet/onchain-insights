"""
ANGEL — Attentive Neural GatEd eLbo (Learned Sparse) Forecaster
================================================================
Key improvements over SAINT (SparseNHITSiTransformer):

1. CausalTCN target encoder
   Dilated causal convolutions replace NHITS pooling stacks.  The pooling +
   linear-interpolation pipeline in NHITS discards temporal detail and
   introduces a coarse-resolution artefact that hurts distributional
   calibration.  A stack of dilated causal conv blocks at exponentially
   growing receptive fields preserves multi-scale temporal structure and
   produces richer features for the distributional head.

2. Single-stage interpretable variable selection
   SAINT's two-stage gate (sparsemax weights × hard-concrete stochastic gate)
   is opaque: the observable "selection" is the product of two numbers with
   different semantics.  ANGEL replaces this with a single GRN scorer +
   temperature-scaled sparsemax, giving one transparent weight per variable.
   The weight IS the selection probability; variables with weight 0 are
   provably excluded (sparsemax structural property).  A learnable temperature
   parameter controls sparsity tightness.

3. Entropy-based sparsity regularisation (l0_lambda)
   Penalising the entropy of the selection distribution directly encourages
   fewer active variables, without the stochastic L0 approximation.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import RevIN, Baseclass_forecast
from models.custom import (
    sparse_probability_map,
    GatedResidualNetwork,
    SparseTransformerEncoderLayer,
    CrossFusionLayer,
    HorizonOutputHead,
)


# ============================================================
# TCN target encoder
# ============================================================

class CausalTCNBlock(nn.Module):
    """Dilated causal 1-D conv residual block with left-only padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        self._pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.norm = nn.LayerNorm(out_channels)

    def _causal(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (self._pad, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.gelu(self.conv1(self._causal(x))))
        h = self.drop(F.gelu(self.conv2(self._causal(h))))
        out = self.skip(x) + h
        return self.norm(out.transpose(1, 2)).transpose(1, 2)


class TCNTargetEncoder(nn.Module):
    """
    Multi-scale causal TCN encoder for a univariate target series.

    Produces horizon tokens [B, H, D] by combining last-step features (full
    causal context) and a global temporal mean (trend summary) — no
    pooling/interpolation artefacts.

    Receptive field at level i: (kernel_size − 1) × 2^i timesteps.
    With tcn_kernel_size=3 and tcn_n_levels=5 the stack covers 64 past steps.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int,
        tcn_channels: int,
        tcn_kernel_size: int,
        tcn_n_levels: int,
        dropout: float,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.d_model = d_model

        self.input_proj = nn.Conv1d(1, tcn_channels, 1)
        self.blocks = nn.ModuleList([
            CausalTCNBlock(
                in_channels=tcn_channels,
                out_channels=tcn_channels,
                kernel_size=tcn_kernel_size,
                dilation=2 ** i,
                dropout=dropout,
            )
            for i in range(tcn_n_levels)
        ])
        self.horizon_proj = nn.Sequential(
            nn.Linear(2 * tcn_channels, d_model),
            nn.GELU(),
            nn.Linear(d_model, pred_len * d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, y_target: torch.Tensor) -> torch.Tensor:
        # y_target: [B, L]
        x = self.input_proj(y_target.unsqueeze(1))  # [B, C, L]
        for block in self.blocks:
            x = block(x)

        last_step = x[:, :, -1]        # [B, C]  — full causal context
        global_mean = x.mean(dim=-1)   # [B, C]  — trend summary
        ctx = torch.cat([last_step, global_mean], dim=-1)  # [B, 2C]

        horizon = self.horizon_proj(ctx).view(-1, self.pred_len, self.d_model)
        return self.norm(horizon)   # [B, H, D]


# ============================================================
# Interpretable variable selection gate
# ============================================================

class InterpretableVariableSelector(nn.Module):
    """
    Single-stage sparse variable selector.

    Replaces SAINT's two-stage (sparsemax × hard-concrete) gate with a single
    GRN scorer followed by temperature-scaled sparsemax/entmax.  The resulting
    weight IS the selection: one interpretable scalar per variable, exactly
    zero for excluded variables (sparsemax structural property).

    A learnable temperature parameter τ controls sparsity tightness:
      - large τ → soft/uniform distribution
      - small τ → sharp/sparse concentration

    Entropy regularisation (scaled by l0_lambda in the trainer) penalises
    diffuse weights and encourages fewer active variables.
    """

    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        dropout: float,
        selector_activation: str = "sparsemax",
        init_temperature: float = 1.5,
    ):
        super().__init__()
        self.selector_activation = selector_activation

        self.scorer = GatedResidualNetwork(
            input_dim=d_model,
            hidden_dim=hidden_size,
            output_dim=d_model,
            context_dim=d_model,
            dropout=dropout,
        )
        self.score_proj = nn.Linear(d_model, 1)

        # Learnable log-temperature; exp ensures strict positivity
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temperature)))

    def forward(
        self,
        tokens: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # tokens: [B, V, D],  context: [B, D]
        b, v, d = tokens.shape
        ctx = context.unsqueeze(1).expand(b, v, d)

        scored = self.scorer(tokens, context=ctx)         # [B, V, D]
        logits = self.score_proj(scored).squeeze(-1)      # [B, V]

        temp = self.log_temp.exp().clamp(min=0.1, max=10.0)
        weights = sparse_probability_map(logits / temp, kind=self.selector_activation, dim=1)

        out = tokens * weights.unsqueeze(-1)              # [B, V, D]

        # True means variable is selected and may be attended to.
        # With sparsemax/entmax, exact zeros are common.
        # Use a tiny threshold only for numerical safety.
        selected_mask = weights > 1e-8                    # [B, V]

        # Entropy: high when diffuse, low when sparse. Penalising it encourages
        # the model to concentrate mass on fewer variables.
        entropy = -(weights * (weights + 1e-12).log()).sum(dim=-1).mean()

        aux: Dict[str, torch.Tensor] = {
            "selection_logits": logits,
            "selection_weights": weights,
            "selection_mask": selected_mask,

            # Compatibility stubs
            "hard_gates": torch.ones(b, v, device=tokens.device, dtype=tokens.dtype),
            "expected_open": weights,
            "effective_selection_used": weights,
            "effective_mass": weights.sum(dim=-1),
            "selector_temperature": temp.detach(),
            "l0_penalty": entropy,
        }
        return out, aux


class AngelVariateTokenEncoder(nn.Module):
    """
    Covariate branch: per-variable trajectory → token → interpretable sparse
    selection.

    Identical structure to SAINT's VariateTokenEncoder except the selector is
    InterpretableVariableSelector (single-stage, no hard-concrete gate).
    Variate self-attention is optional and disabled by default to preserve
    direct interpretability of the downstream cross-attention maps.
    """

    def __init__(
        self,
        seq_len: int,
        num_vars: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        selector_hidden: int,
        dropout: float,
        attn_activation: str,
        selector_activation: str,
        selector_init_temperature: float,
        use_variate_self_attention: bool = False,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.use_variate_self_attention = use_variate_self_attention

        self.time_norm = nn.LayerNorm(seq_len)
        self.time_proj = nn.Linear(seq_len, d_model)
        self.var_embedding = nn.Embedding(num_vars, d_model)

        self.selector = InterpretableVariableSelector(
            d_model=d_model,
            hidden_size=selector_hidden,
            dropout=dropout,
            selector_activation=selector_activation,
            init_temperature=selector_init_temperature,
        )

        self.layers = nn.ModuleList([
            SparseTransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                attn_activation=attn_activation,
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        x_cov: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x_cov: [B, L, V]
        x = x_cov.transpose(1, 2)       # [B, V, L]
        x = self.time_norm(x)
        tokens = self.time_proj(x)       # [B, V, D]

        var_ids = torch.arange(self.num_vars, device=x_cov.device)
        tokens = tokens + self.var_embedding(var_ids).unsqueeze(0)

        tokens, aux = self.selector(tokens, context=context)

        if self.use_variate_self_attention:
            for layer in self.layers:
                tokens = layer(tokens)

        return tokens, aux


# ============================================================
# ANGEL core module
# ============================================================

class AngelModel(nn.Module):
    """
    ANGEL core nn.Module.

    target  → CausalTCN encoder  →  horizon tokens  [B, H, D]
    covariates → interpretable sparse selector  →  covariate tokens [B, V, D]
    cross-attention fusion (horizon ← cov)
    output head  →  [B, H, out_dim]
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int,
        dropout: float,
        method: str,
        forecast_task=None,
        dist_side=None,
        enc_in: int = None,
        affine: bool = True,
        revin_type: str = "revin",
        n_cheb: int = 2,
        tail_model: str = "none",
        # TCN
        tcn_channels: int = 128,
        tcn_kernel_size: int = 3,
        tcn_n_levels: int = 5,
        # Covariate branch
        cov_n_layers: int = 2,
        cov_n_heads: int = 4,
        cov_d_ff: int = 256,
        selector_hidden: int = 128,
        selector_activation: str = "sparsemax",
        selector_init_temperature: float = 1.5,
        # Fusion
        fusion_n_layers: int = 2,
        fusion_n_heads: int = 4,
        fusion_d_ff: int = 256,
        attn_activation: str = "sparsemax",
    ):
        super().__init__()

        if enc_in is None or enc_in < 1:
            raise ValueError("enc_in must be >= 1")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.method = method
        self.forecast_task = forecast_task
        self.dist_side = dist_side
        self.n_cheb = n_cheb
        self.tail_model = tail_model
        self.d_model = d_model
        self.num_covariates = enc_in - 1

        self.std_activ = nn.Softplus()

        self.revin_target = RevIN(1, affine=affine, mode=revin_type)
        self.revin_cov = (
            RevIN(self.num_covariates, affine=affine, mode=revin_type)
            if self.num_covariates > 0
            else None
        )

        self.target_encoder = TCNTargetEncoder(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            tcn_channels=tcn_channels,
            tcn_kernel_size=tcn_kernel_size,
            tcn_n_levels=tcn_n_levels,
            dropout=dropout,
        )

        if self.num_covariates > 0:
            self.covariate_encoder = AngelVariateTokenEncoder(
                seq_len=seq_len,
                num_vars=self.num_covariates,
                d_model=d_model,
                n_heads=cov_n_heads,
                n_layers=cov_n_layers,
                d_ff=cov_d_ff,
                selector_hidden=selector_hidden,
                dropout=dropout,
                attn_activation=attn_activation,
                selector_activation=selector_activation,
                selector_init_temperature=selector_init_temperature,
            )
        else:
            self.covariate_encoder = None

        self.fusion_layers = nn.ModuleList([
            CrossFusionLayer(
                d_model=d_model,
                n_heads=fusion_n_heads,
                d_ff=fusion_d_ff,
                dropout=dropout,
                attn_activation=attn_activation,
            )
            for _ in range(fusion_n_layers)
        ])
        self.fusion_norm = nn.LayerNorm(d_model)

        self.output_dim = self._get_output_dim()
        self.head = HorizonOutputHead(d_model=d_model, out_dim=self.output_dim, dropout=dropout)

        self.latest_aux: Dict = {}

    def _get_output_dim(self) -> int:
        if self.forecast_task in ["point", None]:
            return 1
        if self.forecast_task in ["quantile", "expectile"]:
            return 2 if self.dist_side == "both" else 1
        if self.forecast_task == "gaussian":
            return 2
        if self.forecast_task == "distribution":
            return 4 + self.n_cheb if self.tail_model == "gpd" else 2 + self.n_cheb
        raise ValueError(f"Unsupported forecast_task: {self.forecast_task}")

    def _split_input(self, x_enc: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        y_target = x_enc[:, :, -1:]
        x_cov = x_enc[:, :, :-1] if self.num_covariates > 0 else None
        return y_target, x_cov

    def _denorm_target(self, x: torch.Tensor) -> torch.Tensor:
        b, h, c = x.shape
        x = x.reshape(b, h * c, 1)
        x = self.revin_target(x, "denorm")
        return x.reshape(b, h, c)

    def _denorm_target_scale(self, x: torch.Tensor) -> torch.Tensor:
        b, h, c = x.shape
        x = x.reshape(b, h * c, 1)
        x = self.revin_target(x, "denorm_scale")
        return x.reshape(b, h, c)

    def _format_output(self, raw_out: torch.Tensor) -> torch.Tensor:
        if self.forecast_task == "distribution":
            mean = self._denorm_target(raw_out[:, :, 0:1])
            std = self.std_activ(self._denorm_target_scale(raw_out[:, :, 1:2]))
            return torch.cat([mean, std, raw_out[:, :, 2:]], dim=-1)
        if self.forecast_task == "gaussian":
            mean = self._denorm_target(raw_out[:, :, 0:1])
            std = self.std_activ(self._denorm_target_scale(raw_out[:, :, 1:2]))
            return torch.cat([mean, std], dim=-1)
        out = self._denorm_target(raw_out)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        y_target, x_cov = self._split_input(x_enc)

        y_norm = self.revin_target(y_target, "norm").squeeze(-1)
        x_cov_norm = self.revin_cov(x_cov, "norm") if x_cov is not None else None

        # 1) TCN target branch → horizon tokens
        target_tokens = self.target_encoder(y_norm)    # [B, H, D]
        target_context = target_tokens.mean(dim=1)     # [B, D]

        # 2) Covariate branch with interpretable sparse selection
        if self.covariate_encoder is not None:
            cov_tokens, cov_aux = self.covariate_encoder(x_cov_norm, target_context)
            selection_mask = cov_aux.get("selection_mask", None)
        else:
            cov_tokens = None
            cov_aux = {
                "selection_logits": None,
                "selection_weights": None,
                "selection_mask": None,
                "hard_gates": None,
                "expected_open": None,
                "effective_selection_used": None,
                "effective_mass": None,
                "l0_penalty": torch.tensor(0.0, device=x_enc.device),
            }
            selection_mask = None

        # 3) Cross-attention fusion: horizon tokens attend to covariate tokens
        fused = target_tokens
        cross_attn_maps = []
        for layer in self.fusion_layers:
            fused, attn_map = layer(
                fused,
                cov_tokens,
                key_padding_mask=None if selection_mask is None else ~selection_mask,
                need_weights=True,
            )
            cross_attn_maps.append(attn_map)

        fused = self.fusion_norm(fused)

        # 4) Output head
        raw_out = self.head(fused)
        out = self._format_output(raw_out)

        self.latest_aux = {
            "target_tokens": target_tokens,
            "target_context": target_context,
            "cov_tokens": cov_tokens,
            "selection_logits": cov_aux.get("selection_logits"),
            "selection_weights": cov_aux.get("selection_weights"),
            "selection_mask": cov_aux.get("selection_mask"),
            "hard_gates": cov_aux.get("hard_gates"),
            "expected_open": cov_aux.get("expected_open"),
            "effective_selection_used": cov_aux.get("effective_selection_used"),
            "effective_mass": cov_aux.get("effective_mass"),
            "l0_penalty": cov_aux["l0_penalty"],
            "cross_attn_maps": cross_attn_maps,
        }
        return out

    def get_auxiliary_losses(self) -> Dict[str, torch.Tensor]:
        if "l0_penalty" in self.latest_aux:
            return {"l0_penalty": self.latest_aux["l0_penalty"]}
        return {"l0_penalty": torch.tensor(0.0, device=next(self.parameters()).device)}

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        if self.method != "forecast":
            raise NotImplementedError("ANGEL currently supports method='forecast' only.")
        return self.forecast(x_enc)


# ============================================================
# Lightning wrapper
# ============================================================

class ANGEL_forecast(Baseclass_forecast):
    def __init__(
        self,
        seq_len, pred_len, d_model, dropout, enc_in, method,
        batch_size, test_batch_size,
        affine, revin_type,
        forecast_task, dist_side, tau_pinball,
        n_cheb, twcrps_threshold_low, twcrps_threshold_high, twcrps_side, twcrps_smooth_h,
        u_grid_size, dist_loss, grid_density, quantile_decomp,
        spline_degree, knot_kind, knot_p,
        tail_model, gpd_u_low, gpd_u_high, gpd_xi_min, gpd_xi_max,
        learning_rate,
        # ANGEL-specific
        tcn_channels, tcn_kernel_size, tcn_n_levels,
        cov_n_layers, cov_n_heads, cov_d_ff,
        selector_hidden, selector_activation, selector_init_temperature,
        fusion_n_layers, fusion_n_heads, fusion_d_ff, attn_activation,
        l0_lambda,
        save_test_diagnostics, diag_top_k_vars, diag_max_plot_samples,
        use_log_price,
        **kwargs,
    ):
        super().__init__(
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            learning_rate=learning_rate,
            method=method,
            forecast_task=forecast_task,
            dist_side=dist_side,
            tau_pinball=tau_pinball,
            n_cheb=n_cheb,
            twcrps_threshold_low=twcrps_threshold_low,
            twcrps_threshold_high=twcrps_threshold_high,
            twcrps_side=twcrps_side,
            twcrps_smooth_h=twcrps_smooth_h,
            u_grid_size=u_grid_size,
            grid_density=grid_density,
            dist_loss=dist_loss,
            revin_type=revin_type,
            quantile_decomp=quantile_decomp,
            spline_degree=spline_degree,
            knot_kind=knot_kind,
            knot_p=knot_p,
            tail_model=tail_model,
            gpd_u_low=gpd_u_low,
            gpd_u_high=gpd_u_high,
            gpd_xi_min=gpd_xi_min,
            gpd_xi_max=gpd_xi_max,
            l0_lambda=l0_lambda,
            save_test_diagnostics=save_test_diagnostics,
            diag_top_k_vars=diag_top_k_vars,
            diag_max_plot_samples=diag_max_plot_samples,
            use_log_price=use_log_price,
        )

        self.model = AngelModel(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            dropout=dropout,
            method=method,
            forecast_task=forecast_task,
            dist_side=dist_side,
            enc_in=enc_in,
            affine=bool(affine),
            revin_type=revin_type,
            n_cheb=n_cheb,
            tail_model=tail_model,
            tcn_channels=tcn_channels,
            tcn_kernel_size=tcn_kernel_size,
            tcn_n_levels=tcn_n_levels,
            cov_n_layers=cov_n_layers,
            cov_n_heads=cov_n_heads,
            cov_d_ff=cov_d_ff,
            selector_hidden=selector_hidden,
            selector_activation=selector_activation,
            selector_init_temperature=selector_init_temperature,
            fusion_n_layers=fusion_n_layers,
            fusion_n_heads=fusion_n_heads,
            fusion_d_ff=fusion_d_ff,
            attn_activation=attn_activation,
        )
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        p = parent_parser.add_argument_group("ANGEL model arguments")

        p.add_argument("--d_model", type=int, default=256)
        p.add_argument("--dropout", type=float, default=0.1)
        p.add_argument("--revin_type", type=str, choices=["revin", "robust"], default="revin")
        p.add_argument("--affine", type=int, choices=[0, 1], default=1)

        # TCN target encoder
        p.add_argument("--tcn_channels", type=int, default=128,
                       help="hidden channels in each TCN level")
        p.add_argument("--tcn_kernel_size", type=int, default=3,
                       help="conv kernel size; dilation doubles each level")
        p.add_argument("--tcn_n_levels", type=int, default=5,
                       help="TCN depth; receptive field = (kernel_size-1)*(2^n_levels - 1)")

        # Covariate encoder
        p.add_argument("--cov_n_layers", type=int, default=2)
        p.add_argument("--cov_n_heads", type=int, default=4)
        p.add_argument("--cov_d_ff", type=int, default=256)
        p.add_argument("--selector_hidden", type=int, default=128)
        p.add_argument(
            "--selector_activation",
            type=str,
            choices=["softmax", "sparsemax", "entmax15", "entmax", "entmax1.5"],
            default="sparsemax",
        )
        p.add_argument(
            "--selector_init_temperature", type=float, default=2.0,
            help="initial sparsemax temperature; higher = softer initial selection",
        )

        # Fusion
        p.add_argument("--fusion_n_layers", type=int, default=2)
        p.add_argument("--fusion_n_heads", type=int, default=4)
        p.add_argument("--fusion_d_ff", type=int, default=256)
        p.add_argument(
            "--attn_activation",
            type=str,
            choices=["softmax", "sparsemax", "entmax15", "entmax", "entmax1.5"],
            default="sparsemax",
        )

        Baseclass_forecast.add_task_specific_args(parent_parser)
        return parent_parser
