# ============================================================
# Ablation study table: performance degradation vs SAINT
#
# Metrics reported:
#   CRPS     - standard continuous ranked probability score
#   twCRPS   - threshold-weighted CRPS (training objective)
#   QL 1%    - pinball loss at tau=0.01 (lower tail)
#   QL 99%   - pinball loss at tau=0.99 (upper tail)
#
# Output:
#   ablation_table.tex  - LaTeX table
#   ablation_table.csv  - raw numbers
#
# SAINT is the reference model. All other models show the
# percentage performance degradation relative to SAINT.
# Positive values mean worse than SAINT.
# ============================================================

import sys
import pickle as pkl
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from benchmark_utils import (
    load_forecast_pickles,
    monotone_Q,
    trapezoid_weights_for_u_np,
    quantiles_from_u_grid,
    get_twcrps_per_horizon,
    pinball_loss_array,
    compute_twcrps_per_horizon_np,
)

# ============================================================
# Edit model paths here
# ============================================================

REFERENCE_MODEL = "Our Model"

model_paths = {
    "Our Model": "3328/33ac86cb6ca74b35af2fdaeb36af462a/preds_test_set.pkl",
    "no twCRPS": "3328/7d14ee95d5cc4e24a5bdd42cce75e3ef/preds_test_set.pkl",
    "no GPD": "3328/c3880ff2526449de975d3572041b16d4/preds_test_set.pkl",
    "Gaussian": "3328/02c3fcf87230438192a9aa5cf095fdf5/preds_test_set.pkl",
    "no sparsity": "3328/e45a84e0147e4b92a15086c50c523a53/preds_test_set.pkl",
    "no liquidity": "3328/490edd09e80349dd94d6a9d470cb81e3/preds_test_set.pkl",
    "no covariates": "3328/31647d6e417d41758ed393084cfb147d/preds_test_set.pkl",
}

# twCRPS parameters — must match training configuration
TWCRPS_KWARGS = dict(
    threshold_low=-10.0,
    threshold_high=10.0,
    side="two_sided",
    smooth_h=1.0,
)

OUT_DIR = Path("./comparison_nn_vs_arima/ablation_table")


# ============================================================
# Metric computation
# ============================================================

def compute_crps_mean(A):
    """Mean CRPS over all observations and horizons."""
    Q = monotone_Q(np.asarray(A["Q"], dtype=np.float64))
    y = np.asarray(A["true"], dtype=np.float64)
    u = np.asarray(A["u_grid"], dtype=np.float64)
    wu = trapezoid_weights_for_u_np(u)

    u3 = u.reshape(1, 1, -1)
    wu3 = wu.reshape(1, 1, -1)

    e = y[:, :, None] - Q
    pinball = np.maximum(u3 * e, (u3 - 1.0) * e)
    loss_bh = 2.0 * np.sum(pinball * wu3, axis=-1)

    return float(np.nanmean(loss_bh))


def compute_twcrps_mean(A, twcrps_kwargs=None):
    """Mean twCRPS over all observations and horizons.

    Uses stored value if available, otherwise computes from Q-grid.
    """
    per_h = get_twcrps_per_horizon(
        A,
        compute_if_missing=True,
        twcrps_kwargs=twcrps_kwargs or TWCRPS_KWARGS,
    )
    if per_h is None:
        return np.nan
    return float(np.nanmean(per_h))


def compute_twcrps_sided_mean(A, side, twcrps_kwargs=None):
    """Mean one-sided twCRPS (lower or upper tail) over all observations and horizons."""
    kw = {**(twcrps_kwargs or TWCRPS_KWARGS), "side": side}
    _, overall = compute_twcrps_per_horizon_np(
        A["Q"], A["true"], A["u_grid"], **kw
    )
    return float(overall)


def compute_ql_mean(A, tau):
    """Mean pinball loss at quantile level tau over all observations and horizons."""
    Q = np.asarray(A["Q"], dtype=np.float64)
    y = np.asarray(A["true"], dtype=np.float64)
    u = np.asarray(A["u_grid"], dtype=np.float64)

    q_pred = quantiles_from_u_grid(Q, u, [tau])[..., 0]  # (B,H)
    loss = pinball_loss_array(y, q_pred, tau)

    return float(np.nanmean(loss))


def compute_all_metrics(A, twcrps_kwargs=None):
    return {
        "CRPS": compute_crps_mean(A),
        "twCRPS": compute_twcrps_mean(A, twcrps_kwargs=twcrps_kwargs),
        "twCRPS low": compute_twcrps_sided_mean(A, side="below", twcrps_kwargs=twcrps_kwargs),
        "twCRPS high": compute_twcrps_sided_mean(A, side="above", twcrps_kwargs=twcrps_kwargs),
        "QL 1%": compute_ql_mean(A, tau=0.01),
        "QL 99%": compute_ql_mean(A, tau=0.99),
    }


# ============================================================
# LaTeX table builder
# ============================================================

METRIC_COLS = ["CRPS", "twCRPS", "twCRPS low", "twCRPS high", "QL 1%", "QL 99%"]


def _fmt_abs(v):
    return f"{v:.4f}"


def _fmt_pct(v):
    sign = "-" if v >= 0 else "+"
    return f"${sign}{abs(v):.1f}\\%$"


def build_metrics_df(runs, reference_model=REFERENCE_MODEL, twcrps_kwargs=None):
    records = {}
    for name, A in runs.items():
        records[name] = compute_all_metrics(A, twcrps_kwargs=twcrps_kwargs)

    df = pd.DataFrame(records).T
    df.index.name = "Model"

    ref = df.loc[reference_model]
    degradation = ((df - ref) / ref.abs() * 100.0).drop(index=reference_model)

    return df, degradation


def build_latex_table(
    df_abs,
    df_pct,
    reference_model=REFERENCE_MODEL,
    caption="Ablation study. Numbers for the reference model are absolute metric values. "
            "All other rows show percentage change relative to the reference ",
    label="tab:ablation",
):
    lines = []

    n_cols = 1 + len(METRIC_COLS)
    col_spec = "l" + "r" * len(METRIC_COLS)

    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")

    # Header
    header = "    Model & " + " & ".join(METRIC_COLS) + r" \\"
    lines.append(header)
    lines.append(r"    \midrule")

    # Reference model row (absolute values)
    ref_vals = [_fmt_abs(df_abs.loc[reference_model, m]) for m in METRIC_COLS]
    ref_row = f"    \\textbf{{{reference_model}}} & " + " & ".join(ref_vals) + r" \\"
    lines.append(ref_row)
    lines.append(r"    \midrule")

    # Ablation model rows (percentage degradation)
    for model in df_pct.index:
        pct_vals = [_fmt_pct(df_pct.loc[model, m]) for m in METRIC_COLS]
        row = f"    {model} & " + " & ".join(pct_vals) + r" \\"
        lines.append(row)

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def make_ablation_table(
    model_paths=model_paths,
    reference_model=REFERENCE_MODEL,
    out_dir=OUT_DIR,
    twcrps_kwargs=None,
    caption="Ablation study. Reference model values are absolute; all other rows show "
            "percentage degradation relative to the reference (positive = worse).",
    label="tab:ablation",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading forecast pickles...")
    runs = load_forecast_pickles(model_paths)

    if reference_model not in runs:
        raise ValueError(
            f"Reference model '{reference_model}' not found in runs. "
            f"Available: {list(runs.keys())}"
        )

    print("Computing metrics...")
    df_abs, df_pct = build_metrics_df(
        runs,
        reference_model=reference_model,
        twcrps_kwargs=twcrps_kwargs or TWCRPS_KWARGS,
    )

    print("\n=== Absolute metric values ===")
    print(df_abs.to_string(float_format="{:.4f}".format))

    print("\n=== Percentage degradation vs", reference_model, "===")
    print(df_pct.to_string(float_format="{:+.2f}%".format))

    df_abs.to_csv(out_dir / "ablation_metrics_abs.csv")
    df_pct.to_csv(out_dir / "ablation_metrics_pct.csv")

    tex = build_latex_table(
        df_abs,
        df_pct,
        reference_model=reference_model,
        caption=caption,
        label=label,
    )

    tex_path = out_dir / "ablation_table.tex"
    tex_path.write_text(tex)
    print(f"\nLaTeX table saved to: {tex_path.resolve()}")
    print("\n" + tex)

    return {
        "runs": runs,
        "df_abs": df_abs,
        "df_pct": df_pct,
        "latex": tex,
        "out_dir": out_dir,
    }


if __name__ == "__main__":
    make_ablation_table()
