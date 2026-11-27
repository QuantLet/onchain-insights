"""VaR / ES backtests over training seeds (companion of table_var_es_backtests.py, same tests and settings).

Run: python "12. Evaluation of Probabilistic Forecasts/tables/table_var_es_seed_backtests.py"

A backtest p-value belongs to one forecast system, so seeds are not pooled: every (model, seed) run is backtested
on its own (Kupiec UC for VaR, one-sided Acerbi--Szekely Z2 for ES, non-overlapping origins of stride h, the
functions of table_var_es_backtests.py) and the table reports, per cell, the number of seeds that are not rejected
at 5% and the range of breach counts over seeds. Seeds (as in table_seed_benchmark.py):
  SCQF     the six MLflow experiment 3603 "full" runs
  TiDE     1 (release run), 7, 42, 100, 1233, 2026, spline + GPD head
  TimeXer  1 (release run), 7, 42, 100, 1233, 2026, spline + GPD head
  --seeds 3: SCQF 1000, 1001, 1010 and TiDE / TimeXer 1, 7, 42 (outputs *_3seeds)
  GARCH, ARIMA, Naive: deterministic, one run.
Outputs (benchmark_tables/var_es_backtests/):
  var_es_backtests_per_seed.csv           every model, seed, horizon (12, 24 h), tail and level
  var_es_backtests_seed_summary.{csv,tex} 0.5% and 1%: seeds not rejected / seeds, breach range, median p-value
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEC = HERE.parent
sys.path.insert(0, str(HERE))
from forecast_artifacts import load_forecast_artifact  # noqa: E402
from table_var_es_backtests import build_table  # noqa: E402

DEPEG = SEC.parents[1] / "stablecoin-depeg-forecast"
SCQF_SEEDS = (1000, 1001, 1010)
B = SEC / "benchmark_outputs"
RUNS = {
    "TimeXer": {1: B / "timeXer_hist_exog/preds_test_set.compact.npz",
                7: B / "timeXer_hist_exog_seed7/preds_test_set.pkl",
                42: B / "timeXer_hist_exog_seed42/preds_test_set.pkl",
                100: B / "timeXer_hist_exog_seed100/preds_test_set.pkl",
                1233: B / "timeXer_hist_exog_seed1233/preds_test_set.pkl",
                2026: B / "timeXer_hist_exog_seed2026/preds_test_set.pkl"},
    "TiDE": {1: B / "tide_hist_exog/preds_test_set.compact.npz",
             7: B / "tide_hist_exog_seed7/preds_test_set.pkl",
             42: B / "tide_hist_exog_seed42/preds_test_set.pkl",
             100: B / "tide_hist_exog_seed100/preds_test_set.pkl",
             1233: B / "tide_hist_exog_seed1233/preds_test_set.pkl",
             2026: B / "tide_hist_exog_seed2026/preds_test_set.pkl"},
    "GARCH": {0: B / "garch_student_t/preds_test_set.compact.npz"},
    "ARIMA": {0: B / "arima/preds_test_set.compact.npz"},
    "Naive": {0: B / "naive/preds_test_set.compact.npz"},
}
ORDER = ("SCQF", "TimeXer", "TiDE", "GARCH", "ARIMA", "Naive")
OUT = SEC / "benchmark_tables/var_es_backtests"
SIMULATIONS, SEED = 2000, 1233
LEVELS = (0.005, 0.01)


def scqf_runs():
    man = pd.read_csv(DEPEG / "slides/data/ablation/run_manifest.csv")
    man = man[man.variant.eq("full")]
    if SCQF_SEEDS is not None:
        man = man[man.seed.isin(SCQF_SEEDS)]
        assert len(man) == len(SCQF_SEEDS), man.seed.tolist()
    return {int(r.seed): DEPEG / r.path for r in man.itertuples()}


def main():
    global SCQF_SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", choices=["3", "6"], default="6")
    sfx = ""
    if ap.parse_args().seeds == "3":                    # SCQF 1000/1001/1010, TiDE/TimeXer 1/7/42
        sfx = "_3seeds"
        for m in ("TiDE", "TimeXer"):
            RUNS[m] = {k: v for k, v in RUNS[m].items() if k in (1, 7, 42)}
    else:
        SCQF_SEEDS = None
    runs = {"SCQF": scqf_runs(), **RUNS}
    rows, y_ref, offset = [], None, 0
    for model in ORDER:
        for seed, path in runs[model].items():
            offset += 1
            art = load_forecast_artifact(path)
            y = np.asarray(art["true"], dtype=float)
            if y_ref is None:
                y_ref = y
            elif not np.array_equal(y, y_ref):
                raise ValueError(f"{model} seed {seed}: test targets differ; forecasts are not comparable")
            t = build_table(art, SIMULATIONS, SEED + offset, model=model, horizons=(12, 24))
            rows.append(t.assign(seed=seed))
            print(f"done: {model} seed {seed}", flush=True)
            del art
    per = pd.concat(rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    per.to_csv(OUT / f"var_es_backtests_per_seed{sfx}.csv", index=False)

    sel = per[per.alpha.isin(LEVELS)]
    summ = sel.groupby(["model", "horizon_hours", "alpha", "side"], sort=False).agg(
        seeds=("seed", "nunique"), n=("n", "first"),
        breaches_min=("breaches", "min"), breaches_max=("breaches", "max"), breaches_mean=("breaches", "mean"),
        var_pass=("kupiec_p", lambda p: int((p >= .05).sum())), var_p_median=("kupiec_p", "median"),
        es_pass=("z2_p", lambda p: int((p >= .05).sum())), es_p_median=("z2_p", "median")).reset_index()
    summ.to_csv(OUT / f"var_es_backtests_seed_summary{sfx}.csv", index=False)

    idx = summ.set_index(["model", "horizon_hours", "alpha", "side"])

    def cell(r, kind):
        k, n = int(r[f"{kind}_pass"]), int(r.seeds)
        color = "green!35" if k == n else "red!35" if k == 0 else "orange!35"
        return rf"\cellcolor{{{color}}}{k}/{n}"

    def breaches(r):
        lo, hi = int(r.breaches_min), int(r.breaches_max)
        return f"{lo}" if lo == hi else f"{lo}--{hi}"

    L = [r"% generated by tables/table_var_es_seed_backtests.py -- requires \usepackage[table]{xcolor}, booktabs",
         r"\begin{table}[ht]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{l c c c cc cc cc}", r"\toprule",
         r" & & & & \multicolumn{2}{c}{Breaches} & \multicolumn{2}{c}{VaR (Kupiec)} & "
         r"\multicolumn{2}{c}{ES ($Z_2$)} \\",
         r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
         r"Model & Seeds & Horizon & $\alpha$ & lower & upper & lower & upper & lower & upper \\", r"\midrule"]
    for model in ORDER:
        for h in (12, 24):
            for a in LEVELS:
                lo, up = idx.loc[(model, h, a, "lower")], idx.loc[(model, h, a, "upper")]
                exp = lo.n * a
                L.append(f"{model} & {int(lo.seeds)} & {h}h & {a:.1%} & {breaches(lo)} & {breaches(up)} & "
                         f"{cell(lo, 'var')} & {cell(up, 'var')} & {cell(lo, 'es')} & {cell(up, 'es')} "
                         rf"\\ % expected breaches {exp:.1f} of {int(lo.n)}")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}",
          r"\caption{VaR unconditional-coverage (Kupiec) and expected-shortfall (Acerbi--Sz\'ekely $Z_2$, one-sided "
          r"Monte Carlo) backtests on non-overlapping forecast origins, run separately for every training seed. "
          r"Cells give the number of seeds not rejected at 5\% (green: all, orange: some, red: none); breaches are "
          r"the range over seeds. GARCH, ARIMA and Naive are deterministic (one run). P-values are unadjusted "
          r"across tests.}",
          r"\label{tab:var_es_backtests_seeds}", r"\end{table}"]
    (OUT / f"var_es_backtests_seed_summary{sfx}.tex").write_text("\n".join(L) + "\n")
    pd.set_option("display.width", 220)
    print(summ.drop(columns=["breaches_mean"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
