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
    --target \
    --target_window "${TARGET_WINDOW}" \
    --target_threshold "${TARGET_THRESHOLD}" \
    --max_depth "${MAX_DEPTH}" \
    --n_estimators "${N_ESTIMATORS}" \
    --early_stopping_rounds "${EARLY_STOPPING_ROUNDS}" \
    --depeg_side "${DEPEG_SIDE}" \
    --model_names "${MODELS[@]}" \
    --scaler "${SCALER}"
done

echo "All CV runs completed."

python "${PLOT_SCRIPT}" --experiment_name "${EXPERIMENT_NAME}" \

echo "===================================================="
echo "Selecting top 2 by CV AUC and top 2 by CV AUPRC"
echo "===================================================="

SELECTED_TSV="${LOG_DIR}/${EXPERIMENT_NAME}/selected_for_full_retraining.tsv"

LOG_DIR="${LOG_DIR}" EXPERIMENT_NAME="${EXPERIMENT_NAME}" SELECTED_TSV="${SELECTED_TSV}" python - <<'PY'
from pathlib import Path
import pandas as pd
import os

log_dir = Path(os.environ["LOG_DIR"])
experiment_name = os.environ["EXPERIMENT_NAME"]
selected_tsv = Path(os.environ["SELECTED_TSV"])

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

required_cols = ["model_name", "alpha", "cv_auc_mean", "cv_auprc_mean"]
missing = [c for c in required_cols if c not in all_df.columns]
if missing:
    raise SystemExit(f"Missing expected columns in summary data: {missing}")

all_df["alpha"] = pd.to_numeric(all_df["alpha"], errors="coerce")
all_df["cv_auc_mean"] = pd.to_numeric(all_df["cv_auc_mean"], errors="coerce")
all_df["cv_auprc_mean"] = pd.to_numeric(all_df["cv_auprc_mean"], errors="coerce")

# Keep the latest row for each (alpha, model_name) in case of reruns
all_df = (
    all_df.sort_values("source_mtime")
          .drop_duplicates(subset=["alpha", "model_name"], keep="last")
          .reset_index(drop=True)
)

top_auc = (
    all_df.dropna(subset=["cv_auc_mean"])
          .sort_values(["cv_auc_mean", "cv_auprc_mean"], ascending=False)
          .head(2)
          .copy()
)
top_auc["selection_reason"] = "top2_cv_auc"

top_auprc = (
    all_df.dropna(subset=["cv_auprc_mean"])
          .sort_values(["cv_auprc_mean", "cv_auc_mean"], ascending=False)
          .head(2)
          .copy()
)
top_auprc["selection_reason"] = "top2_cv_auprc"

selected = pd.concat([top_auc, top_auprc], ignore_index=True)

if len(selected) == 0:
    raise SystemExit("No valid candidates found for retraining selection.")

# Union duplicates if the same run is top-ranked by both metrics
selected = (
    selected.groupby(["alpha", "model_name"], as_index=False)
            .agg(
                cv_auc_mean=("cv_auc_mean", "max"),
                cv_auprc_mean=("cv_auprc_mean", "max"),
                selection_reason=("selection_reason", lambda s: ",".join(sorted(set(s))))
            )
            .sort_values(["cv_auc_mean", "cv_auprc_mean"], ascending=False)
            .reset_index(drop=True)
)

selected_tsv.parent.mkdir(parents=True, exist_ok=True)
selected.to_csv(selected_tsv, sep="\t", index=False)

print("\nSelected model-alpha pairs for full retraining:\n")
print(selected.to_string(index=False))
print(f"\nSaved selection table to: {selected_tsv}")
PY

echo "===================================================="
echo "Running full retraining for selected candidates"
echo "===================================================="

tail -n +2 "${SELECTED_TSV}" | while IFS=$'\t' read -r ALPHA MODEL CV_AUC CV_AUPRC REASON; do
  echo "----------------------------------------"
  echo "Full retraining: model=${MODEL}, alpha=${ALPHA}, reason=${REASON}"
  echo "CV AUC=${CV_AUC}, CV AUPRC=${CV_AUPRC}"
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