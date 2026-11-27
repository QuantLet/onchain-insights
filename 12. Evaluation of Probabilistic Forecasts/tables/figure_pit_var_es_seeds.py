"""PIT / VaR / ES calibration figure of SCQF over training seeds, joint seed-and-time block-bootstrap bands.

Run: python "12. Evaluation of Probabilistic Forecasts/tables/figure_pit_var_es_seeds.py" [--seeds 6|3]

Seed-level version of the figure of tail_diagnostics.ipynb (same PIT inversion, VaR interpolation, ES integration
and horizon-averaged summaries). Every statistic is computed per seed and averaged over seeds (bold line); thin
lines are the individual seeds. Band: 95% joint bootstrap -- each replicate resamples the seeds with replacement
and circular 48-hour blocks of forecast origins (the same blocks for every seed, all 24 horizons of an origin kept
together), so it includes training-seed and time-series sampling uncertainty.
Seeds: the six MLflow experiment 3603 "full" runs (default) or 1000, 1001, 1010 (--seeds 3).
Outputs: PIT_VAR_ES_calibration_seeds{,_3seeds}.png/.pdf and benchmark_tables/var_es_backtests/
         pit_var_es_seed_bands{,_3seeds}.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
SEC = HERE.parent
sys.path.insert(0, str(HERE))
from forecast_artifacts import load_forecast_artifact  # noqa: E402

DEPEG = SEC.parents[1] / "stablecoin-depeg-forecast"
ALPHAS = np.array([0.002, 0.005, 0.01, 0.02, 0.05])
PIT_GRID = np.linspace(0, 1, 401)
N_BOOT, BLOCK, RNG_SEED = 1000, 48, 1233


def q_at(Q, u, p):
    if p <= u[0]:
        return Q[..., 0]
    if p >= u[-1]:
        return Q[..., -1]
    r = np.searchsorted(u, p); l = r - 1
    w = (p - u[l]) / (u[r] - u[l])
    return (1 - w) * Q[..., l] + w * Q[..., r]


def es_grid(Q, u, a, side):
    Qe = np.concatenate([Q[..., :1], Q, Q[..., -1:]], axis=-1)
    ge = np.r_[0.0, u, 1.0]
    if side == "lower":
        m = ge < a
        x = np.r_[ge[m], a]
        v = np.concatenate([Qe[..., m], q_at(Q, u, a)[..., None]], axis=-1)
    else:
        m = ge > 1 - a
        x = np.r_[1 - a, ge[m]]
        v = np.concatenate([q_at(Q, u, 1 - a)[..., None], Qe[..., m]], axis=-1)
    return np.trapezoid(v, x=x, axis=-1) / a


def pit(y, Q, u):
    n, h, J = Q.shape
    out = np.empty((n, h))
    for k in range(h):
        q, yy = Q[:, k], y[:, k]
        i = (q <= yy[:, None]).sum(1)
        l = np.clip(i - 1, 0, J - 2); r = l + 1
        q0, q1 = q[np.arange(n), l], q[np.arange(n), r]
        out[:, k] = np.clip(u[l] + (yy - q0) / (q1 - q0 + 1e-12) * (u[r] - u[l]), 0, 1)
    return out


def per_origin(path, y_ref):
    """Per-origin sufficient statistics of one run (sums over horizons kept per horizon where needed)."""
    a = load_forecast_artifact(path)
    y = np.asarray(a["true"], float)
    if y_ref is not None and not np.array_equal(y, y_ref):
        raise ValueError(f"{path}: test targets differ")
    Q = np.maximum.accumulate(np.asarray(a["Q"], float), axis=-1)
    u = np.asarray(a["u_grid"], float)
    del a
    P = pit(y, Q, u)
    # PIT ECDF counts per origin on the grid: cnt[i, g] = #{h: PIT_ih <= grid_g}
    cnt = np.stack([np.searchsorted(np.sort(P[i]), PIT_GRID, side="right") for i in range(len(P))]).astype(np.float32)
    st = {"pit_cnt": cnt}
    for side in ("lower", "upper"):
        var = np.stack([q_at(Q, u, a if side == "lower" else 1 - a) for a in ALPHAS], -1)   # (n, H, L)
        es = np.stack([es_grid(Q, u, a, side) for a in ALPHAS], -1)
        hit = (y[..., None] <= var) if side == "lower" else (y[..., None] >= var)
        st[f"{side}_hit"] = hit.astype(np.float32)                                         # (n, H, L)
        st[f"{side}_yhit"] = (y[..., None] * hit).astype(np.float32)
        st[f"{side}_eshit"] = (es * hit).astype(np.float32)
    return st, y


def summarize(st, idx):
    """Horizon-averaged statistics of tail_diagnostics.ipynb on the origin sample idx."""
    out = {"pit_ecdf": st["pit_cnt"][idx].sum(0) / (len(idx) * 24)}
    for side in ("lower", "upper"):
        h = st[f"{side}_hit"][idx].sum(0)                              # (H, L)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(h > 0, st[f"{side}_yhit"][idx].sum(0) / h, np.nan)
            p = np.where(h > 0, st[f"{side}_eshit"][idx].sum(0) / h, np.nan)
        out[f"{side}_coverage"] = (h / len(idx)).mean(0)
        out[f"{side}_realized_es"] = np.nanmean(r, 0)
        out[f"{side}_predicted_es"] = np.nanmean(p, 0)
    return out


def blocks(n, rng):
    starts = rng.integers(0, n, size=int(np.ceil(n / BLOCK)))
    return np.concatenate([(s + np.arange(BLOCK)) % n for s in starts])[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", choices=["3", "6"], default="6")
    six = ap.parse_args().seeds == "6"
    sfx = "" if six else "_3seeds"
    man = pd.read_csv(DEPEG / "slides/data/ablation/run_manifest.csv")
    man = man[man.variant.eq("full")]
    if not six:
        man = man[man.seed.isin((1000, 1001, 1010))]
    stats, y_ref, seeds = [], None, []
    for r in man.sort_values("seed").itertuples():
        st, y_ref = per_origin(DEPEG / r.path, y_ref)
        stats.append(st); seeds.append(int(r.seed))
        print(f"loaded SCQF seed {r.seed}", flush=True)
    n, S = len(y_ref), len(stats)
    full = np.arange(n)
    per_seed = [summarize(st, full) for st in stats]
    keys = list(per_seed[0])
    point = {k: np.mean([p[k] for p in per_seed], 0) for k in keys}
    rng = np.random.default_rng(RNG_SEED)
    boot = {k: [] for k in keys}
    for _ in range(N_BOOT):
        idx = blocks(n, rng)
        draw = rng.integers(0, S, size=S)
        cache = {}
        for s in set(draw.tolist()):
            cache[s] = summarize(stats[s], idx)
        for k in keys:
            boot[k].append(np.mean([cache[s][k] for s in draw], 0))
    lo = {k: np.nanpercentile(np.array(v), 2.5, 0) for k, v in boot.items()}
    hi = {k: np.nanpercentile(np.array(v), 97.5, 0) for k, v in boot.items()}

    rows = []
    for k in keys:
        x = PIT_GRID if k == "pit_ecdf" else ALPHAS
        for j, xv in enumerate(x):
            rows.append(dict(statistic=k, x=xv, mean=point[k][j], ci95_low=lo[k][j], ci95_high=hi[k][j],
                             **{f"seed_{s}": p[k][j] for s, p in zip(seeds, per_seed)}))
    csv = SEC / f"benchmark_tables/var_es_backtests/pit_var_es_seed_bands{sfx}.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    # ------------------------------------------------------------------ figure (layout of tail_diagnostics.ipynb)
    band = f"95% joint seed-and-time bootstrap ({S} seeds, {BLOCK}h blocks)"
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(PIT_GRID, PIT_GRID, "k--", lw=1.5, label="Uniform")
    for p in per_seed:
        ax[0].plot(PIT_GRID, p["pit_ecdf"], color="tab:blue", lw=0.6, alpha=0.5)
    ax[0].plot(PIT_GRID, point["pit_ecdf"], color="tab:blue", lw=2.5, label="Empirical PIT (seed mean)")
    ax[0].fill_between(PIT_GRID, lo["pit_ecdf"], hi["pit_ecdf"], color="tab:blue", alpha=0.18, label=band)
    ax[0].plot([], [], color="tab:blue", lw=0.6, alpha=0.5, label="Individual seeds")
    ax[0].set(xlabel="PIT value", ylabel="Empirical CDF", title="PIT ECDF")
    ax[0].legend(frameon=False, fontsize=8, loc="upper left")

    ax[1].plot(ALPHAS, ALPHAS, "k--", label="Nominal")
    for side, mk, c, lab in (("lower", "o", "tab:blue", "Lower"), ("upper", "s", "tab:red", "Upper")):
        k = f"{side}_coverage"
        for p in per_seed:
            ax[1].plot(ALPHAS, p[k], "-", color=c, lw=0.6, alpha=0.5)
        ax[1].plot(ALPHAS, point[k], mk + "-", color=c, label=lab)
        ax[1].fill_between(ALPHAS, lo[k], hi[k], color=c, alpha=0.18)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set(xlabel=r"Nominal tail level $\alpha$", ylabel="Empirical exceedance rate", title="VaR calibration")
    ax[1].legend(frameon=False, fontsize=8)

    for side, mk, c, pc, lab in (("lower", "o", "tab:blue", "tab:cyan", "Lower"),
                                 ("upper", "s", "tab:red", "tab:orange", "Upper")):
        for key, ls, sl, col in (("realized_es", "-", "realized", c), ("predicted_es", "--", "forecast ES", pc)):
            k = f"{side}_{key}"
            ax[2].plot(ALPHAS, point[k], mk + ls, color=col, label=f"{lab} {sl}")
            ax[2].fill_between(ALPHAS, lo[k], hi[k], color=col, alpha=0.12)
    ax[2].set_xscale("log")
    ax[2].set(xlabel=r"Nominal tail level $\alpha$", ylabel="Tail conditional mean", title="ES severity")
    ax[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(SEC / f"PIT_VAR_ES_calibration_seeds{sfx}.{ext}", dpi=300, transparent=True, bbox_inches="tight")
    print("saved", SEC / f"PIT_VAR_ES_calibration_seeds{sfx}.png", "and", csv)
    t = pd.DataFrame(rows)
    print(t[t.statistic.str.contains("coverage|es")].round(4).iloc[:, :5].to_string(index=False))


if __name__ == "__main__":
    main()
