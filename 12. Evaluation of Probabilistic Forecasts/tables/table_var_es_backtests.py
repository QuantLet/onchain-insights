"""Backtest saved ANGEL and benchmark VaR and ES forecasts; write CSV and LaTeX tables.

Run: python "12. Probabilistic Forecasting Evaluation/tables/table_var_es_backtests.py"

Each horizon uses one non-overlapping sequence of forecast origins (stride h).
The Basel traffic light thresholds are applied illustratively only to 1% VaR
over the final 250 hourly forecasts; this is not a regulatory capital test.
Z2 is the Acerbi--Szekely direct ES test, calibrated by simulation from each
saved conditional quantile curve; its one-sided alternative is underestimated
tail risk. Results are descriptive at multiple levels without multiplicity
adjustment. The target is depeg_bps, not a trading portfolio P&L.

Outputs (benchmark_tables/var_es_backtests/):
  angel_var_es_backtests.{csv,tex}            ANGEL, every horizon in HORIZONS and level in LEVELS
  var_es_backtests_all_models.csv              ANGEL and every benchmark, all levels (12 h and 24 h)
  var_es_backtests_24h_comparison.{csv,tex}    ANGEL and every benchmark at 0.5% and 1% (12 h and 24 h)
TimeXer and TiDE are reported with both heads: the multi-quantile (MQ) loss head and the head used by ANGEL
(I-spline body + spliced GPD tails, benchmark_scripts/*_forecast.py --head spline_gpd).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from forecast_artifacts import load_forecast_artifact


SECTION = Path(__file__).resolve().parents[1]
LEVELS = (0.002, 0.005, 0.01, 0.02, 0.05)
DISPLAY_LEVELS = (0.005, 0.01, 0.02)
HORIZONS = (12, 24)
MODEL = "ANGEL"
# Insertion order fixes each benchmark's simulation seed (seed + position); new entries go at the end.
BENCHMARK_ARTIFACTS = {
    "ARIMA": "benchmark_outputs/arima/preds_test_set.compact.npz",
    "GARCH": "benchmark_outputs/garch_student_t/preds_test_set.compact.npz",
    "TimeXer (MQ)": "benchmark_outputs/timeXer_hist_exog/preds_test_set.compact.npz",
    "TiDE (MQ)": "benchmark_outputs/tide_hist_exog/preds_test_set.compact.npz",
    "Naive": "benchmark_outputs/naive/preds_test_set.compact.npz",
    "TimeXer (spline-GPD)": "benchmark_outputs/timeXer_hist_exog_spline_gpd/preds_test_set.compact.npz",
    "TiDE (spline-GPD)": "benchmark_outputs/tide_hist_exog_spline_gpd/preds_test_set.compact.npz",
}
# Row order of the comparison table
TABLE_ORDER = (MODEL, "TimeXer (spline-GPD)", "TiDE (spline-GPD)", "TimeXer (MQ)", "TiDE (MQ)", "GARCH", "ARIMA",
               "Naive")


def q_at(q: np.ndarray, grid: np.ndarray, level: float) -> np.ndarray:
    """Linearly interpolate a quantile level, with flat endpoint extension."""
    j = int(np.searchsorted(grid, level))
    if j == 0:
        return q[..., 0]
    if j == len(grid):
        return q[..., -1]
    w = (level - grid[j - 1]) / (grid[j] - grid[j - 1])
    return q[..., j - 1] * (1 - w) + q[..., j] * w


def tail_es(q: np.ndarray, grid: np.ndarray, alpha: float, side: str) -> np.ndarray:
    """Integrate the piecewise-linear Q(u), extended flat to 0 and 1."""
    if side == "lower":
        x = np.r_[0.0, grid[grid < alpha], alpha]
    else:
        x = np.r_[1 - alpha, grid[grid > 1 - alpha], 1.0]
    values = np.stack([q_at(q, grid, float(u)) for u in x], axis=-1)
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return integrate(values, x=x, axis=-1) / alpha


def bernoulli_log_likelihood(successes: int, trials: int, probability: float) -> float:
    if trials == 0:
        return 0.0
    if probability == 0:
        return 0.0 if successes == 0 else -np.inf
    if probability == 1:
        return 0.0 if successes == trials else -np.inf
    return successes * np.log(probability) + (trials - successes) * np.log1p(-probability)


def var_tests(hit: np.ndarray, alpha: float) -> dict:
    """Kupiec UC and Christoffersen first-order Markov IND and CC tests."""
    n = len(hit)
    k = int(hit.sum())
    p_hat = k / n
    lr_uc = 2 * (bernoulli_log_likelihood(k, n, p_hat)
                 - bernoulli_log_likelihood(k, n, alpha))
    prev, current = hit[:-1], hit[1:]
    n00 = int(np.sum(~prev & ~current))
    n01 = int(np.sum(~prev & current))
    n10 = int(np.sum(prev & ~current))
    n11 = int(np.sum(prev & current))
    if n00 + n01 == 0 or n10 + n11 == 0:
        lr_ind = p_ind = lr_cc = p_cc = np.nan
    else:
        p01 = n01 / (n00 + n01)
        p11 = n11 / (n10 + n11)
        p_pool = (n01 + n11) / (n - 1)
        ll_markov = (bernoulli_log_likelihood(n01, n00 + n01, p01)
                     + bernoulli_log_likelihood(n11, n10 + n11, p11))
        ll_iid = bernoulli_log_likelihood(n01 + n11, n - 1, p_pool)
        lr_ind = max(0.0, 2 * (ll_markov - ll_iid))
        p_ind = chi2.sf(lr_ind, 1)
        lr_cc = lr_uc + lr_ind
        p_cc = chi2.sf(lr_cc, 2)
    return dict(n=n, breaches=k, breach_rate=p_hat, kupiec_lr=lr_uc,
                kupiec_p=chi2.sf(lr_uc, 1), christoffersen_ind_lr=lr_ind,
                christoffersen_ind_p=p_ind, christoffersen_cc_lr=lr_cc,
                christoffersen_cc_p=p_cc)


def basel_light(hit: np.ndarray, alpha: float) -> tuple:
    if not np.isclose(alpha, 0.01) or len(hit) < 250:
        return (np.nan, "n/a")
    k = int(hit[-250:].sum())
    return (k, "green" if k <= 4 else "amber" if k <= 9 else "red")


def draw_quantiles(q: np.ndarray, grid: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    """Sample each row's saved conditional distribution at given uniforms."""
    j = np.searchsorted(grid, uniforms)
    left = np.clip(j - 1, 0, len(grid) - 1)
    right = np.clip(j, 0, len(grid) - 1)
    rows = np.arange(len(q))[None, :]
    lo, hi = grid[left], grid[right]
    weight = np.divide(uniforms - lo, hi - lo, out=np.zeros_like(uniforms), where=hi > lo)
    return q[rows, left] * (1 - weight) + q[rows, right] * weight


def z2_test(y: np.ndarray, q: np.ndarray, grid: np.ndarray, alpha: float,
            side: str, var: np.ndarray, es: np.ndarray,
            simulations: int, rng: np.random.Generator) -> tuple:
    """One-sided Acerbi--Szekely Z2 with a conditional-distribution MC p-value."""
    sign = 1 if side == "lower" else -1
    oriented_var = sign * var
    oriented_es = sign * es
    oriented_y = sign * y
    shift = float(np.max(oriented_var) + 1.0)
    denominator = alpha * (shift - oriented_es)
    if np.any(denominator <= 0):
        raise ValueError("Z2 requires positive loss ES after the fixed shift")
    hit = oriented_y < oriented_var
    observed = 1.0 + np.mean(np.where(hit, oriented_y - shift, 0.0) / denominator)
    null = np.empty(simulations)
    for first in range(0, simulations, 100):
        size = min(100, simulations - first)
        uniforms = rng.random((size, len(y)))
        simulated_y = sign * draw_quantiles(q, grid, uniforms)
        simulated_hit = simulated_y < oriented_var[None, :]
        null[first:first + size] = 1.0 + np.mean(
            np.where(simulated_hit, simulated_y - shift, 0.0) / denominator[None, :], axis=1)
    p_value = (1 + int(np.sum(null <= observed))) / (simulations + 1)
    return observed, p_value, shift


def build_table(artifact: dict, simulations: int, seed: int,
                model: str = MODEL, horizons=HORIZONS) -> pd.DataFrame:
    y = np.asarray(artifact["true"], dtype=float)
    q = np.asarray(artifact["Q"], dtype=float)
    grid = np.asarray(artifact["u_grid"], dtype=float)
    cutoff = np.asarray(artifact.get("cutoff_idx", np.arange(len(y))))
    if y.ndim != 2 or q.shape[:2] != y.shape or q.shape[2] != len(grid):
        raise ValueError("Incompatible true, Q, and u_grid shapes")
    if not (np.isfinite(y).all() and np.isfinite(q).all() and np.isfinite(grid).all()):
        raise ValueError("Forecast arrays contain non-finite values")
    if not (np.all(np.diff(grid) > 0) and 0 < grid[0] and grid[-1] < 1):
        raise ValueError("u_grid must be strictly increasing and inside (0,1)")
    if np.any(np.diff(q, axis=-1) < -1e-6):
        raise ValueError("Saved quantile curves cross; cannot simulate a valid null distribution")
    if cutoff.ndim != 1 or len(cutoff) != len(y) or np.any(np.diff(cutoff) <= 0):
        raise ValueError("Forecast origins must be in increasing time order")
    if "cutoff_idx" in artifact and len(cutoff) > 1 and not np.all(np.diff(cutoff) == 3_600_000_000):
        raise ValueError("Expected consecutive hourly forecast origins")
    rng = np.random.default_rng(seed)
    rows = []
    for horizon in horizons:
        if horizon > y.shape[1]:
            continue
        indices = np.arange((len(y) - 1) % horizon, len(y), horizon)
        ys = y[indices, horizon - 1]
        qs = q[indices, horizon - 1]
        for side in ("lower", "upper"):
            for alpha in LEVELS:
                var = q_at(qs, grid, alpha if side == "lower" else 1 - alpha)
                es = tail_es(qs, grid, alpha, side)
                hit = ys < var if side == "lower" else ys > var
                result = var_tests(hit, alpha)
                last_250, light = basel_light(hit, alpha)
                z2, z2_p, shift = z2_test(ys, qs, grid, alpha, side,
                                          var, es, simulations, rng)
                rows.append(dict(model=model, horizon_hours=horizon, side=side,
                                 alpha=alpha, **result, last_250_breaches=last_250,
                                 traffic_light=light, z2=z2, z2_p=z2_p, z2_shift=shift,
                                 kupiec_assessment="reject" if result["kupiec_p"] < .05 else "pass",
                                 cc_assessment=("n/a" if np.isnan(result["christoffersen_cc_p"])
                                                else "reject" if result["christoffersen_cc_p"] < .05 else "pass"),
                                 es_assessment="reject" if z2_p < .05 else "pass"))
    return pd.DataFrame(rows)


def write_latex(table: pd.DataFrame, path: Path, *, models=None,
                horizons=HORIZONS, levels=DISPLAY_LEVELS,
                label="tab:angel_var_es_backtests") -> None:
    """Write p-values in red/green filled cells, split by tail and test."""
    visible = table[table["alpha"].isin(levels)]
    indexed = {(row.model, int(row.horizon_hours), row.side, float(row.alpha)): row
               for row in visible.itertuples()}

    def p_text(value: float) -> str:
        return f"{value:.3g}" if np.isfinite(value) else r"\text{--}"

    def p_cell(value: float) -> str:
        if not np.isfinite(value):
            return r"\cellcolor{gray!20}\text{--}"
        color = "green!35" if value >= .05 else "red!35"
        return rf"\cellcolor{{{color}}}{p_text(value)}"

    models = list(models or visible.model.drop_duplicates())
    lines = [r"% Requires \usepackage[table]{xcolor} and \usepackage{booktabs}.",
             r"\begin{table}[ht]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{6pt}",
             r"\begin{tabular}{l c c c c c c}", r"\toprule",
             r"Model & Horizon & $\alpha$ & VaR lower & VaR upper & ES lower & ES upper \\",
             r"\midrule"]
    for model in models:
        for horizon in horizons:
            for alpha in levels:
                lower = indexed.get((model, horizon, "lower", alpha))
                upper = indexed.get((model, horizon, "upper", alpha))
                if lower is None or upper is None:
                    continue
                lines.append(f"{model} & {horizon}h & {alpha:.1%} & "
                             f"{p_cell(lower.kupiec_p)} & {p_cell(upper.kupiec_p)} & "
                             f"{p_cell(lower.z2_p)} & {p_cell(upper.z2_p)} " + r"\\")
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{VaR unconditional-coverage (Kupiec) and expected-shortfall (Acerbi--Sz\'ekely "
              r"$Z_2$) backtest p-values for lower and upper tails at the selected tail levels. "
              r"Green cells indicate p-values $\geq 0.05$; red cells indicate rejection at 5\%. "
              r"ES p-values are one-sided Monte Carlo values.}",
              rf"\label{{{label}}}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path,
                        default=SECTION / "model_logs/final_retrain_run/preds_test_set.compact.npz")
    parser.add_argument("--output-dir", type=Path, default=SECTION / "benchmark_tables/var_es_backtests")
    parser.add_argument("--simulations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1233)
    args = parser.parse_args()
    if args.simulations < 99:
        parser.error("--simulations must be at least 99 for a 5% test")
    artifact = load_forecast_artifact(args.predictions)
    print(f"{MODEL}: {args.predictions} ({len(artifact.get('covariate_names', []))} covariates)")
    result = build_table(artifact, args.simulations, args.seed)
    reference_y = np.asarray(artifact["true"], dtype=float)
    del artifact
    benchmark_rows = []
    for offset, (model, relative_path) in enumerate(BENCHMARK_ARTIFACTS.items(), start=1):
        artifact_path = SECTION / relative_path
        if not artifact_path.exists():
            print(f"skipped {model}: missing {artifact_path}")
            continue
        artifact = load_forecast_artifact(artifact_path)
        if not np.array_equal(np.asarray(artifact["true"], dtype=float), reference_y):
            raise ValueError(f"{model}: test targets differ from {MODEL}'s; the forecasts are not comparable")
        benchmark_rows.append(build_table(artifact, args.simulations, args.seed + offset,
                                          model=model, horizons=(12, 24)))
        print(f"{model}: {artifact_path}")
        del artifact
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "angel_var_es_backtests.csv"
    tex_path = args.output_dir / "angel_var_es_backtests.tex"
    result.to_csv(csv_path, index=False)
    write_latex(result, tex_path)
    all_models = pd.concat([result[result.horizon_hours.isin((12, 24))], *benchmark_rows], ignore_index=True)
    all_csv = args.output_dir / "var_es_backtests_all_models.csv"
    all_models.to_csv(all_csv, index=False)
    models = [m for m in TABLE_ORDER if m in set(all_models.model)]
    comparison = all_models[all_models.alpha.isin((0.005, 0.01))].copy()
    comparison_csv = args.output_dir / "var_es_backtests_24h_comparison.csv"
    comparison_tex = args.output_dir / "var_es_backtests_24h_comparison.tex"
    comparison.to_csv(comparison_csv, index=False)
    write_latex(comparison, comparison_tex, models=models,
                horizons=(12, 24), levels=(0.005, 0.01),
                label="tab:var_es_backtests_comparison")
    pd.set_option("display.width", 200)
    print(result[["horizon_hours", "side", "alpha", "breaches", "n", "breach_rate",
                  "kupiec_p", "z2", "z2_p", "es_assessment"]].to_string(index=False))
    summary = comparison.assign(model=pd.Categorical(comparison.model, models, ordered=True)).pivot_table(
        index=["model", "horizon_hours", "alpha"], columns="side",
        values=["breaches", "kupiec_p", "z2_p"], observed=True).round(3)
    print(summary.to_string())
    print(f"Saved {csv_path}, {tex_path}, {all_csv}, {comparison_csv}, and {comparison_tex}")


if __name__ == "__main__":
    main()
