#!/usr/bin/env bash
set -euo pipefail

CV_SCRIPT="cv_model_comparison.py"
PLOT_SCRIPT="plot_cv_metrics_heatmap.py"
FULL_TRAIN_SCRIPT="run_full_training.py"

LOG_DIR="lightning_logs"
EXPERIMENT_NAME="cv_model_comparison_$(date +%Y-%m-%d)"
FULL_EXPERIMENT_NAME="${EXPERIMENT_NAME}_full_retraining"

ALPHAS=(0.1 0.3 0.5 1.0 1.5 2.0)
MODELS=(xgboost lightgbm catboost random_forest)

TARGET_WINDOW=24
TARGET_THRESHOLD=15
MAX_DEPTH=6
N_ESTIMATORS=800
EARLY_STOPPING_ROUNDS=200
SCALER="robust"
DEPEG_SIDE="both"
EVAL_METRIC="auc"
# Models within this mean OOS AUPRC distance are treated as comparable; Brier
# skill and fold-to-fold AUPRC stability break the tie.
AUPRC_TOLERANCE=0.01

echo "===================================================="
echo "Running CV comparison experiment: ${EXPERIMENT_NAME}"
echo "===================================================="

for ALPHA in "${ALPHAS[@]}"; do
  echo "========================================"
  echo "Running CV comparison for alpha=${ALPHA}"
  echo "========================================"

  python "${CV_SCRIPT}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --run_name "alpha_${ALPHA}" \
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
    --auprc_tolerance "${AUPRC_TOLERANCE}"
done

echo "All CV runs completed."

python "${PLOT_SCRIPT}" --experiment_name "${EXPERIMENT_NAME}"

echo "===================================================="
echo "Selecting the CV winner by OOS AUPRC, Brier skill, and stability"
echo "===================================================="

SELECTED_TSV="${LOG_DIR}/${EXPERIMENT_NAME}/selected_for_full_retraining.tsv"

LOG_DIR="${LOG_DIR}" EXPERIMENT_NAME="${EXPERIMENT_NAME}" SELECTED_TSV="${SELECTED_TSV}" AUPRC_TOLERANCE="${AUPRC_TOLERANCE}" python - <<'PY'
from pathlib import Path
import pandas as pd
import os

log_dir = Path(os.environ["LOG_DIR"])
experiment_name = os.environ["EXPERIMENT_NAME"]
selected_tsv = Path(os.environ["SELECTED_TSV"])
auprc_tolerance = float(os.environ["AUPRC_TOLERANCE"])

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
    "cv_auprc_mean",
    "cv_auprc_std",
    "cv_brier_skill_score_mean",
    "selected_model",
]
missing = [c for c in required_cols if c not in all_df.columns]
if missing:
    raise SystemExit(f"Missing expected columns in summary data: {missing}")

all_df["alpha"] = pd.to_numeric(all_df["alpha"], errors="coerce")
all_df["cv_auprc_mean"] = pd.to_numeric(all_df["cv_auprc_mean"], errors="coerce")
all_df["cv_auprc_std"] = pd.to_numeric(all_df["cv_auprc_std"], errors="coerce")
all_df["cv_brier_skill_score_mean"] = pd.to_numeric(
    all_df["cv_brier_skill_score_mean"], errors="coerce"
)
all_df["selected_model"] = all_df["selected_model"].astype(str).str.lower().eq("true")

# Keep the latest row for each (alpha, model_name) in case of reruns
all_df = (
    all_df.sort_values("source_mtime")
          .drop_duplicates(subset=["alpha", "model_name"], keep="last")
          .reset_index(drop=True)
)

# Each CV summary has already selected its model for that alpha. Select one
# full-retraining candidate across alphas with the same stated hierarchy.
candidates = all_df[all_df["selected_model"]].dropna(subset=["cv_auprc_mean"]).copy()
if candidates.empty:
    raise SystemExit("No CV-selected candidates found for full retraining.")

best_auprc = candidates["cv_auprc_mean"].max()
comparable = candidates[
    candidates["cv_auprc_mean"] >= best_auprc - auprc_tolerance
].copy()
comparable["_bss_sort"] = comparable["cv_brier_skill_score_mean"].fillna(float("-inf"))
comparable["_stability_sort"] = comparable["cv_auprc_std"].fillna(float("inf"))
selected = (
    comparable.sort_values(
        ["_bss_sort", "_stability_sort", "cv_auprc_mean"],
        ascending=[False, True, False],
    )
    .head(1)
    [["alpha", "model_name", "cv_auprc_mean", "cv_auprc_std", "cv_brier_skill_score_mean"]]
    .copy()
)
selected["selection_reason"] = (
    "cv_selected; global_auprc_then_brier_skill_then_auprc_stability"
)

selected_tsv.parent.mkdir(parents=True, exist_ok=True)
selected.to_csv(selected_tsv, sep="\t", index=False)

print("\nSelected model-alpha pair for full retraining:\n")
print(selected.to_string(index=False))
print(f"\nSaved selection table to: {selected_tsv}")
PY

echo "===================================================="
echo "Running full retraining for selected candidates"
echo "===================================================="

tail -n +2 "${SELECTED_TSV}" | while IFS=$'\t' read -r ALPHA MODEL CV_AUPRC CV_AUPRC_STD BRIER_SKILL REASON; do
  echo "----------------------------------------"
  echo "Full retraining: model=${MODEL}, alpha=${ALPHA}, reason=${REASON}"
  echo "CV AUPRC=${CV_AUPRC}, AUPRC std=${CV_AUPRC_STD}, Brier skill=${BRIER_SKILL}"
  echo "----------------------------------------"

  python "${FULL_TRAIN_SCRIPT}" \
    --experiment_name "${FULL_EXPERIMENT_NAME}" \
    --run_name "${MODEL}_alpha_${ALPHA}_fullfeatures_${REASON}" \
    --alpha "${ALPHA}" \
    --model_name "${MODEL}" \
    --target_window "${TARGET_WINDOW}" \
    --target_threshold "${TARGET_THRESHOLD}" \
    --max_depth "${MAX_DEPTH}" \
    --n_estimators "${N_ESTIMATORS}" \
    --early_stopping_rounds "${EARLY_STOPPING_ROUNDS}" \
    --depeg_side "${DEPEG_SIDE}" \
    --eval_metric "${EVAL_METRIC}" \
    --scaler "${SCALER}"
done

echo "===================================================="
echo "Done."
echo "CV experiment: ${EXPERIMENT_NAME}"
echo "Full retraining experiment: ${FULL_EXPERIMENT_NAME}"
echo "===================================================="
