#!/usr/bin/env bash
set -euo pipefail

CV_SCRIPT="cv_model_comparison.py"
PLOT_SCRIPT="plot_cv_metrics_heatmap.py"
SELECTION_PLOT_SCRIPT="plot_model_selection_by_budget.py"
PAPER_REPORT_SCRIPT="make_paper_ready_reports.py"
FULL_TRAIN_SCRIPT="run_full_training.py"

LOG_DIR="lightning_logs"
EXPERIMENT_NAME="cv_model_comparison_$(date +%Y-%m-%d)"
FULL_EXPERIMENT_NAME="${EXPERIMENT_NAME}_full_retraining"

ALPHAS=(0.1 0.3 0.5 1.0 1.5 2.0)
MODELS=(xgboost lightgbm catboost random_forest)

TARGET_WINDOW=24
# Sensitivity analysis of the realised-depeg definition (basis points).
DEPEG_THRESHOLDS=(10 15 25)
MAX_DEPTH=6
N_ESTIMATORS=800
EARLY_STOPPING_ROUNDS=200
SCALER="robust"
DEPEG_SIDE="both"
EVAL_METRIC="auc"
# Model selection is event-level utility at this operating point.  The full
# list is also evaluated so the selection-frontier plot can show sensitivity.
FALSE_ALERT_BUDGET=2.0
FALSE_ALERT_BUDGETS=(0.5 1.0 2.0)
UTILITY_TOLERANCE=0.01
FALSE_ALERT_COST=0.05
MIN_LEAD_HOURS=1
MIN_LEAD_UTILITY=0.10
# Keep operational scoring aligned with "depeg within TARGET_WINDOW hours".
MAX_LEAD_HOURS="${TARGET_WINDOW}"
UTILITY_TARGET_LEAD_HOURS="${TARGET_WINDOW}"
ALERT_COOLDOWN_HOURS=24
N_BOOTSTRAP=1000

echo "===================================================="
echo "Running CV comparison experiment: ${EXPERIMENT_NAME}"
echo "===================================================="

for TARGET_THRESHOLD in "${DEPEG_THRESHOLDS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    echo "========================================"
    echo "Running CV comparison for threshold=${TARGET_THRESHOLD}bps, alpha=${ALPHA}"
    echo "========================================"

    python "${CV_SCRIPT}" \
      --experiment_name "${EXPERIMENT_NAME}" \
      --run_name "threshold_${TARGET_THRESHOLD}_alpha_${ALPHA}" \
      --alpha "${ALPHA}" \
      --train_pct 0.68 \
      --target \
      --target_window "${TARGET_WINDOW}" \
      --target_threshold "${TARGET_THRESHOLD}" \
      --max_depth "${MAX_DEPTH}" \
      --n_estimators "${N_ESTIMATORS}" \
      --early_stopping_rounds "${EARLY_STOPPING_ROUNDS}" \
      --depeg_side "${DEPEG_SIDE}" \
      --model_names "${MODELS[@]}" \
      --scaler "${SCALER}" \
      --cv_embargo_hours 48 \
      --false_alert_budget_per_month "${FALSE_ALERT_BUDGET}" \
      --false_alert_budgets "${FALSE_ALERT_BUDGETS[@]}" \
      --false_alert_cost "${FALSE_ALERT_COST}" \
      --min_lead_hours "${MIN_LEAD_HOURS}" \
      --min_lead_utility "${MIN_LEAD_UTILITY}" \
      --max_lead_hours "${MAX_LEAD_HOURS}" \
      --utility_target_lead_hours "${UTILITY_TARGET_LEAD_HOURS}" \
      --alert_cooldown_hours "${ALERT_COOLDOWN_HOURS}" \
      --n_bootstrap "${N_BOOTSTRAP}" \
      --utility_tolerance "${UTILITY_TOLERANCE}"
  done
done

echo "All CV runs completed."

python "${PLOT_SCRIPT}" --experiment_name "${EXPERIMENT_NAME}"
python "${SELECTION_PLOT_SCRIPT}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --utility_tolerance "${UTILITY_TOLERANCE}"
python "${PAPER_REPORT_SCRIPT}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --false_alert_budget_per_month "${FALSE_ALERT_BUDGET}" \
  --utility_tolerance "${UTILITY_TOLERANCE}"

echo "===================================================="
echo "Selecting the CV winner by event utility at the false-alert budget"
echo "===================================================="

SELECTED_TSV="${LOG_DIR}/${EXPERIMENT_NAME}/selected_for_full_retraining.tsv"

LOG_DIR="${LOG_DIR}" EXPERIMENT_NAME="${EXPERIMENT_NAME}" SELECTED_TSV="${SELECTED_TSV}" UTILITY_TOLERANCE="${UTILITY_TOLERANCE}" python - <<'PY'
from pathlib import Path
import pandas as pd
import os

log_dir = Path(os.environ["LOG_DIR"])
experiment_name = os.environ["EXPERIMENT_NAME"]
selected_tsv = Path(os.environ["SELECTED_TSV"])
utility_tolerance = float(os.environ["UTILITY_TOLERANCE"])

exp_dir = log_dir / experiment_name
if not exp_dir.exists():
    raise SystemExit(f"Experiment directory not found: {exp_dir}")

summary_files = sorted(
    exp_dir.glob("*_experiment_summary*/artifacts/comparison/model_comparison_summary.csv")
)

if not summary_files:
    raise SystemExit(
        f"No summary CSV files found under {exp_dir}. "
        f"Expected files like */artifacts/comparison/model_comparison_summary.csv"
    )

dfs = []
for fp in summary_files:
    df = pd.read_csv(fp)
    if len(df) == 0:
        continue
    df["source_file"] = str(fp)
    df["source_mtime"] = fp.stat().st_mtime
    dfs.append(df)

if not dfs:
    raise SystemExit("No non-empty summary CSV files found.")

all_df = pd.concat(dfs, ignore_index=True)

required_cols = [
    "model_name",
    "alpha",
    "target_threshold",
    "cv_event_utility_score_mean",
    "cv_event_utility_score_std",
    "cv_timely_event_recall_mean",
    "cv_false_alerts_per_month_mean",
    "selected_model",
]
missing = [c for c in required_cols if c not in all_df.columns]
if missing:
    raise SystemExit(f"Missing expected columns in summary data: {missing}")

all_df["alpha"] = pd.to_numeric(all_df["alpha"], errors="coerce")
all_df["target_threshold"] = pd.to_numeric(all_df["target_threshold"], errors="coerce")
all_df["cv_event_utility_score_mean"] = pd.to_numeric(all_df["cv_event_utility_score_mean"], errors="coerce")
all_df["cv_event_utility_score_std"] = pd.to_numeric(all_df["cv_event_utility_score_std"], errors="coerce")
all_df["cv_timely_event_recall_mean"] = pd.to_numeric(all_df["cv_timely_event_recall_mean"], errors="coerce")
all_df["cv_false_alerts_per_month_mean"] = pd.to_numeric(all_df["cv_false_alerts_per_month_mean"], errors="coerce")
all_df["selected_model"] = all_df["selected_model"].astype(str).str.lower().eq("true")

# Keep the latest row for each (alpha, model_name) in case of reruns
all_df = (
    all_df.sort_values("source_mtime")
          .drop_duplicates(subset=["target_threshold", "alpha", "model_name"], keep="last")
          .reset_index(drop=True)
)

# Each CV summary has already selected a model for one (threshold, alpha)
# combination. Select one full-retraining candidate *within each threshold*.
candidates = all_df[all_df["selected_model"]].dropna(
    subset=["target_threshold", "cv_event_utility_score_mean"]
).copy()
if candidates.empty:
    raise SystemExit("No CV-selected candidates found for full retraining.")

selected_rows = []
for threshold, group in candidates.groupby("target_threshold", sort=True):
    best_utility = group["cv_event_utility_score_mean"].max()
    comparable = group[
        group["cv_event_utility_score_mean"] >= best_utility - utility_tolerance
    ].copy()
    comparable["_recall_sort"] = comparable["cv_timely_event_recall_mean"].fillna(float("-inf"))
    comparable["_fa_sort"] = comparable["cv_false_alerts_per_month_mean"].fillna(float("inf"))
    comparable["_stability_sort"] = comparable["cv_event_utility_score_std"].fillna(float("inf"))
    winner = comparable.sort_values(
        ["_recall_sort", "_fa_sort", "_stability_sort", "cv_event_utility_score_mean"],
        ascending=[False, True, True, False],
    ).head(1).copy()
    winner["selection_reason"] = (
        "threshold_specific_event_utility_then_timely_recall_then_false_alert_burden"
    )
    selected_rows.append(winner)

selected = pd.concat(selected_rows, ignore_index=True)[[
    "target_threshold", "alpha", "model_name", "cv_event_utility_score_mean",
    "cv_event_utility_score_std", "cv_timely_event_recall_mean",
    "cv_false_alerts_per_month_mean", "selection_reason",
]]

selected_tsv.parent.mkdir(parents=True, exist_ok=True)
selected.to_csv(selected_tsv, sep="\t", index=False)

print("\nSelected model-alpha pair for full retraining:\n")
print(selected.to_string(index=False))
print(f"\nSaved selection table to: {selected_tsv}")
PY

echo "===================================================="
echo "Running full retraining for selected candidates"
echo "===================================================="

tail -n +2 "${SELECTED_TSV}" | while IFS=$'\t' read -r TARGET_THRESHOLD ALPHA MODEL CV_UTILITY CV_UTILITY_STD EVENT_RECALL FALSE_ALERTS REASON; do
  echo "----------------------------------------"
  echo "Full retraining: threshold=${TARGET_THRESHOLD}bps, model=${MODEL}, alpha=${ALPHA}, reason=${REASON}"
  echo "CV utility=${CV_UTILITY}, utility std=${CV_UTILITY_STD}, event recall=${EVENT_RECALL}, false alerts/month=${FALSE_ALERTS}"
  echo "----------------------------------------"

  python "${FULL_TRAIN_SCRIPT}" \
    --experiment_name "${FULL_EXPERIMENT_NAME}" \
    --run_name "${MODEL}_threshold_${TARGET_THRESHOLD}_alpha_${ALPHA}_fullfeatures_${REASON}" \
    --alpha "${ALPHA}" \
    --model_name "${MODEL}" \
    --target_window "${TARGET_WINDOW}" \
    --target_threshold "${TARGET_THRESHOLD}" \
    --max_depth "${MAX_DEPTH}" \
    --n_estimators "${N_ESTIMATORS}" \
    --early_stopping_rounds "${EARLY_STOPPING_ROUNDS}" \
    --depeg_side "${DEPEG_SIDE}" \
    --eval_metric "${EVAL_METRIC}" \
    --scaler "${SCALER}" \
    --false_alert_budget_per_month "${FALSE_ALERT_BUDGET}" \
    --false_alert_cost "${FALSE_ALERT_COST}" \
    --min_lead_hours "${MIN_LEAD_HOURS}" \
    --min_lead_utility "${MIN_LEAD_UTILITY}" \
    --max_lead_hours "${MAX_LEAD_HOURS}" \
    --utility_target_lead_hours "${UTILITY_TARGET_LEAD_HOURS}" \
    --alert_cooldown_hours "${ALERT_COOLDOWN_HOURS}"
done

echo "===================================================="
echo "Done."
echo "CV experiment: ${EXPERIMENT_NAME}"
echo "Full retraining experiment: ${FULL_EXPERIMENT_NAME}"
echo "===================================================="
