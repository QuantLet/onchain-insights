# ============================================================
# Seed-averaged benchmark tables (Sections 5.2.2 and 5.2.9)
# ============================================================
#
# Same metrics, horizons and tests as table_full_benchmark.py and table_DM_tests.py (their functions are reused
# unchanged), with every trained model evaluated over several training seeds:
#   SCQF     6 seeds, MLflow experiment 3603 "full" runs (the runs of the ablation study)
#   TiDE     6 seeds (1 = release run, 7, 42, 100, 1233, 2026), spline + GPD head (--head spline_gpd)
#   TimeXer  6 seeds (1 = release run, 7, 42, 100, 1233, 2026), spline + GPD head
#   GARCH, ARIMA, Naive: deterministic, one run.
#
# 5.2.2 table: each metric is the mean over seeds of the per-seed score (SD over seeds in *_sd.csv).
# 5.2.9 table: per forecast origin t, the seed-mean loss  Lbar_M(t) = mean_s L_{M,s}(t);  d_t = Lbar_SCQF(t) - Lbar_B(t).
#   DM      HAC (Newey-West) test on d_t, lag as in table_DM_tests.py (horizon-based).
#   CI      joint seed-and-time bootstrap: each replicate resamples every model's seeds with replacement and the
#           forecast origins in moving blocks (block length 48, the same blocks for both models), and recomputes
#           mean(d). The interval therefore includes training-seed randomness as well as time-series sampling.
#   Seeds   robustness: share of the (SCQF seed, benchmark seed) pairs whose own DM p-value (proposed better) < 0.05.
#
#   python tables/table_seed_benchmark.py      # -> benchmark_tables/seed_benchmark/
# Per-run loss series are cached in benchmark_tables/seed_benchmark/cache/ (one run in memory at a time).
# ============================================================

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEC = HERE.parent
sys.path.insert(0, str(HERE))
from benchmark_utils import load_forecast_pickles, _resolve_horizons  # noqa: E402


def _defs(name, marker="# Example usage"):
    """Function definitions of a table script, without its top-level run."""
    src = (HERE / name).read_text()
    src = src[:src.index(marker)]
    mod = types.ModuleType(name.removesuffix(".py"))
    mod.__file__ = str(HERE / name)
    exec(compile(src, str(HERE / name), "exec"), mod.__dict__)
    return mod


FB = _defs("table_full_benchmark.py")
DM = _defs("table_DM_tests.py")

DEPEG = SEC.parents[1] / "stablecoin-depeg-forecast"
MODELS = {
    "SCQF": None,  # exp 3603 "full" runs, from stablecoin-depeg-forecast/slides/data/ablation/run_manifest.csv
    "GARCH": {0: SEC / "benchmark_outputs/garch_student_t/preds_test_set.compact.npz"},
    "TimeXer": {1: SEC / "benchmark_outputs/timeXer_hist_exog/preds_test_set.compact.npz",
                7: SEC / "benchmark_outputs/timeXer_hist_exog_seed7/preds_test_set.pkl",
                42: SEC / "benchmark_outputs/timeXer_hist_exog_seed42/preds_test_set.pkl",
                100: SEC / "benchmark_outputs/timeXer_hist_exog_seed100/preds_test_set.pkl",
                1233: SEC / "benchmark_outputs/timeXer_hist_exog_seed1233/preds_test_set.pkl",
                2026: SEC / "benchmark_outputs/timeXer_hist_exog_seed2026/preds_test_set.pkl"},
    "TiDE": {1: SEC / "benchmark_outputs/tide_hist_exog/preds_test_set.compact.npz",
             7: SEC / "benchmark_outputs/tide_hist_exog_seed7/preds_test_set.pkl",
             42: SEC / "benchmark_outputs/tide_hist_exog_seed42/preds_test_set.pkl",
             100: SEC / "benchmark_outputs/tide_hist_exog_seed100/preds_test_set.pkl",
             1233: SEC / "benchmark_outputs/tide_hist_exog_seed1233/preds_test_set.pkl",
             2026: SEC / "benchmark_outputs/tide_hist_exog_seed2026/preds_test_set.pkl"},
    "ARIMA": {0: SEC / "benchmark_outputs/arima/preds_test_set.compact.npz"},
    "Naive": {0: SEC / "benchmark_outputs/naive/preds_test_set.compact.npz"},
}
OUT = SEC / "benchmark_tables/seed_benchmark"
CACHE = OUT / "cache"
HORIZONS = {"all_horizons": "all", "h1": 1, "h12": 12, "h24": 24}
DM_HORIZONS = {"24h": 24, "1_24h": None}
QLEVELS = (0.01, 0.05, 0.95, 0.99)
TW = dict(tw_threshold_low=-10.0, tw_threshold_high=10.0, tw_smooth_h=1.0)
BLOCK, N_BOOT, SEED = 48, 5000, 1233


SCQF_SEEDS = None      # None: all six; (1000, 1001, 1010): the three-seed version


def scqf_paths():
    man = pd.read_csv(DEPEG / "slides/data/ablation/run_manifest.csv")
    man = man[man.variant.eq("full")]
    if SCQF_SEEDS is not None:
        man = man[man.seed.isin(SCQF_SEEDS)]
        assert len(man) == len(SCQF_SEEDS), man.seed.tolist()
    return {int(r.seed): DEPEG / r.path for r in man.itertuples()}


def evaluate_run(model, seed, path, y_ref):
    """Benchmark-table rows and DM loss series of one run (cached)."""
    tag = f"{model}_seed{seed}"
    f_tab, f_loss = CACHE / f"{tag}_table.csv", CACHE / f"{tag}_loss.parquet"
    if f_tab.exists() and f_loss.exists():
        return pd.read_csv(f_tab), pd.read_parquet(f_loss)
    A = load_forecast_pickles({model: str(path)})[model]
    y = np.asarray(A["true"], dtype=np.float64)
    if y_ref is not None:
        assert y.shape == y_ref.shape and np.allclose(y, y_ref, equal_nan=True), f"{tag}: targets differ"
    rows = []
    for label, h in HORIZONS.items():
        t = FB.build_distributional_benchmark_table(runs={model: A}, horizons_ahead=h, quantile_levels=QLEVELS,
                                                    use_stored_twcrps=False, **TW)
        rows.append(t.assign(horizon_spec=label))
    tab = pd.concat(rows).assign(seed=seed)
    loss = {}
    for hl, h in DM_HORIZONS.items():
        for metric in ("CRPS", "twCRPS"):
            df, _ = DM.build_loss_series_table({model: A}, metric=metric, horizons_ahead=h, **TW)
            loss[f"{metric}|{hl}"] = df[model].to_numpy()
    loss = pd.DataFrame(loss)
    CACHE.mkdir(parents=True, exist_ok=True)
    tab.to_csv(f_tab, index=False)
    loss.to_parquet(f_loss)
    return tab, loss


def joint_bootstrap(Lm, Lb, block, n_boot, rng):
    """Mean of d = seed-mean(Lm) - seed-mean(Lb) under seed resampling and moving-block resampling of origins.
    Lm: (S_m, T), Lb: (S_b, T)."""
    T = Lm.shape[1]
    starts = np.arange(0, T - block + 1)
    n_blocks = int(np.ceil(T / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        sm = Lm[rng.integers(0, len(Lm), len(Lm))].mean(0)
        sb = Lb[rng.integers(0, len(Lb), len(Lb))].mean(0)
        idx = (rng.choice(starts, n_blocks)[:, None] + np.arange(block)[None, :]).ravel()[:T]
        d = sm[idx] - sb[idx]
        out[b] = np.nanmean(d)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS["SCQF"] = scqf_paths()
    ref = load_forecast_pickles({"Naive": str(MODELS["Naive"][0])})["Naive"]
    y_ref = np.asarray(ref["true"], dtype=np.float64)
    h_ref = {hl: _resolve_horizons(ref, h) for hl, h in DM_HORIZONS.items()}
    del ref
    tabs, losses = [], {}
    for model, runs in MODELS.items():
        for seed, path in runs.items():
            if not Path(path).exists():
                print(f"missing, skipped: {model} seed {seed} ({path})")
                continue
            t, l = evaluate_run(model, seed, path, y_ref)
            tabs.append(t.assign(Model=model))
            losses[(model, seed)] = l
            print(f"done: {model} seed {seed}")
    per_seed = pd.concat(tabs, ignore_index=True)
    per_seed.to_csv(OUT / "benchmark_per_seed.csv", index=False)
    metrics = [c for c in per_seed.columns if c not in ("Model", "Horizons", "horizon_spec", "seed")]
    order = list(MODELS)

    # ---------------------------------------------------------------- 5.2.2 tables (seed means)
    for label in HORIZONS:
        g = per_seed[per_seed.horizon_spec.eq(label)].groupby("Model", sort=False)
        mean = g[metrics].mean().reindex(order).reset_index()
        mean.insert(1, "Horizons", g.Horizons.first().reindex(order).to_numpy())
        sd = g[metrics].std(ddof=1).reindex(order).reset_index()
        n = g.seed.nunique().reindex(order).to_numpy()
        mean.to_csv(OUT / f"benchmark_distributional_{label}.csv", index=False)
        sd.assign(seeds=n).to_csv(OUT / f"benchmark_distributional_{label}_sd.csv", index=False)
        imp = FB.compute_main_model_improvements(mean, main_model="SCQF")
        imp.to_csv(OUT / f"benchmark_distributional_improvements_{label}.csv", index=False)
        tex = FB.format_benchmark_table_for_latex(mean, decimals=4)
        tex = tex.replace(r"\bottomrule", r"\bottomrule" + "\n" + rf"\multicolumn{{{len(mean.columns) - 1}}}{{l}}"
                          r"{\footnotesize Trained models: mean over 6 training seeds (SCQF, TiDE, TimeXer); "
                          r"GARCH, ARIMA and Naive are deterministic.} \\")
        (OUT / f"benchmark_distributional_{label}.tex").write_text(tex)
        (OUT / f"benchmark_distributional_{label}.md").write_text(FB.format_benchmark_table_for_markdown(mean, decimals=4))
        print(f"\n{label}\n" + FB.format_benchmark_table_for_markdown(mean, decimals=4))

    # ---------------------------------------------------------------- 5.2.9 tables (seed-mean DM, joint CI)
    for hl in DM_HORIZONS:
        h_idx, hlabel = h_ref[hl]
        lag = int(max(0, h_idx[0])) if len(h_idx) == 1 else int(max(0, np.max(h_idx)))
        rows = []
        for metric in ("CRPS", "twCRPS"):
            key = f"{metric}|{hl}"
            L = {m: np.vstack([losses[(m, s)][key].to_numpy() for s in runs if (m, s) in losses])
                 for m, runs in MODELS.items() if any((m, s) in losses for s in runs)}
            Lm = L["SCQF"]
            for b, Lb in L.items():
                if b == "SCQF":
                    continue
                d = Lm.mean(0) - Lb.mean(0)
                ok = np.isfinite(d)
                dm = DM.diebold_mariano_hac_test(d[ok], lag=lag)
                boot = joint_bootstrap(Lm[:, ok], Lb[:, ok], BLOCK, N_BOOT, np.random.default_rng(SEED))
                pair_p = [DM.diebold_mariano_hac_test((lm - lb)[ok], lag=lag)["DM p-value main better"]
                          for lm in Lm for lb in Lb]
                mm, mb = float(np.mean(Lm.mean(0)[ok])), float(np.mean(Lb.mean(0)[ok]))
                rows.append({
                    "Metric": metric, "Proposed model": "SCQF", "Benchmark": b,
                    "Seeds proposed": len(Lm), "Seeds benchmark": len(Lb),
                    "Mean loss proposed": mm, "Mean loss benchmark": mb,
                    "Mean diff proposed-minus-benchmark": float(np.mean(d[ok])),
                    "Improvement (%)": 100.0 * (mb - mm) / mb if mb else np.nan,
                    "DM stat": dm["DM stat"], "DM p-value two-sided": dm["DM p-value two-sided"],
                    "DM p-value proposed better": dm["DM p-value main better"],
                    "Bootstrap CI low": float(np.quantile(boot, 0.025)),
                    "Bootstrap CI high": float(np.quantile(boot, 0.975)),
                    "Bootstrap p-value proposed better": float(np.mean(boot >= 0.0)),
                    "Seed pairs significant (DM p<0.05)": f"{int(np.sum(np.array(pair_p) < 0.05))}/{len(pair_p)}",
                    "Max pairwise DM p-value": float(np.nanmax(pair_p)),
                    "NW lag": lag, "Block length": BLOCK, "N boot": N_BOOT, "N": int(ok.sum()), "Horizons": hlabel,
                })
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"section_529_pairwise_dm_seed_bootstrap_SCQF_{hl}.csv", index=False)
        tex = DM.format_pairwise_table_for_latex(df, decimals=4)
        (OUT / f"section_529_pairwise_dm_seed_bootstrap_SCQF_{hl}.tex").write_text(tex)
        pd.set_option("display.width", 250)
        print(f"\nDM {hl}\n", df[["Metric", "Benchmark", "Seeds benchmark", "Mean diff proposed-minus-benchmark",
                                 "Bootstrap CI low", "Bootstrap CI high", "Improvement (%)",
                                 "DM p-value proposed better", "Seed pairs significant (DM p<0.05)"]]
              .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
