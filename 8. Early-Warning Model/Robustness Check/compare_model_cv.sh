#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EARLY_WARNING_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CV_SCRIPT="${EARLY_WARNING_DIR}/cv_model_comparison.py"
PLOT_SCRIPT="${EARLY_WARNING_DIR}/plot_cv_metrics_heatmap.py"
FULL_TRAIN_SCRIPT="${EARLY_WARNING_DIR}/run_full_training.py"
RUNTIME_DIR="${SCRIPT_DIR}/runtime"
RUNTIME_DATA_DIR="${RUNTIME_DIR}/data"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_LOG="${SCRIPT_DIR}/robustness_check_${RUN_ID}.log"
MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRIPT_DIR}/.matplotlib}"
PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${SCRIPT_DIR}/.pycache}"
export MPLCONFIGDIR
export MPLBACKEND=Agg
export PYTHONPYCACHEPREFIX
export PYTHONUNBUFFERED=1

TARGET_THRESHOLDS=(10 15 20 25)
ALPHAS=(0.1 0.3 0.5 1.0 1.5 2.0)
MODELS=(xgboost lightgbm catboost random_forest)

TARGET_WINDOW=24
MAX_DEPTH=6
N_ESTIMATORS=800
EARLY_STOPPING_ROUNDS=200
SCALER="robust"
DEPEG_SIDE="both"
EVAL_METRIC="auc"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
}
mkdir -p "${MPLCONFIGDIR}"

link_input() {
  local source_path="$1"
  local target_path="$2"

  if [[ ! -e "${source_path}" ]]; then
    echo "Required input not found: ${source_path}" >&2
    exit 1
  fi
  if [[ ! -e "${target_path}" && ! -L "${target_path}" ]]; then
    ln -s "${source_path}" "${target_path}"
  fi
}

# Recreate the expected ./data layout inside Robustness Check. The ownership
# parquet is stored elsewhere in this repository, so stage it here as a symlink
# instead of modifying the original Early-Warning Model input directory.
mkdir -p "${RUNTIME_DATA_DIR}/Curve"
for input_group in AAVE ETH_blocks Uniswap; do
  link_input \
    "${EARLY_WARNING_DIR}/data/${input_group}" \
    "${RUNTIME_DATA_DIR}/${input_group}"
done
for source_path in "${EARLY_WARNING_DIR}/data/Curve/"*; do
  link_input \
    "${source_path}" \
    "${RUNTIME_DATA_DIR}/Curve/$(basename "${source_path}")"
done
link_input \
  "${EARLY_WARNING_DIR}/../4. Stablecoin liquidity ownership/3CRV_lpevents.parquet" \
  "${RUNTIME_DATA_DIR}/Curve/3CRV_lpevents.parquet"

# Keep a complete execution log in Robustness Check while preserving live output.
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "===================================================="
echo "Running threshold robustness check"
echo "Run ID: ${RUN_ID}"
echo "Thresholds (bps): ${TARGET_THRESHOLDS[*]}"
echo "Python: ${PYTHON_BIN}"
echo "Output folder: ${SCRIPT_DIR}"
echo "===================================================="

for TARGET_THRESHOLD in "${TARGET_THRESHOLDS[@]}"; do
  THRESHOLD_DIR="${SCRIPT_DIR}/threshold_${TARGET_THRESHOLD}"
  DATASET_DIR="${THRESHOLD_DIR}/preprocessed_datasets"
  LOG_DIR="${THRESHOLD_DIR}/lightning_logs"
  EXPERIMENT_NAME="cv_model_comparison_threshold_${TARGET_THRESHOLD}_${RUN_ID}"
  FULL_EXPERIMENT_NAME="${EXPERIMENT_NAME}_full_retraining"
  SELECTED_TSV="${LOG_DIR}/${EXPERIMENT_NAME}/selected_for_full_retraining.tsv"

  mkdir -p "${DATASET_DIR}" "${LOG_DIR}"

  {
    echo "run_id=${RUN_ID}"
    echo "target_threshold_bps=${TARGET_THRESHOLD}"
    echo "target_window_hours=${TARGET_WINDOW}"
    echo "alphas=${ALPHAS[*]}"
    echo "models=${MODELS[*]}"
    echo "max_depth=${MAX_DEPTH}"
    echo "n_estimators=${N_ESTIMATORS}"
    echo "early_stopping_rounds=${EARLY_STOPPING_ROUNDS}"
    echo "scaler=${SCALER}"
    echo "depeg_side=${DEPEG_SIDE}"
    echo "eval_metric=${EVAL_METRIC}"
    echo "cv_experiment=${EXPERIMENT_NAME}"
    echo "full_retraining_experiment=${FULL_EXPERIMENT_NAME}"
  } > "${THRESHOLD_DIR}/run_metadata_${RUN_ID}.txt"

  echo
  echo "===================================================="
  echo "Threshold ${TARGET_THRESHOLD} bps: CV comparison"
  echo "Experiment: ${EXPERIMENT_NAME}"
  echo "===================================================="

  for ALPHA in "${ALPHAS[@]}"; do
    echo "========================================"
    echo "Threshold=${TARGET_THRESHOLD} bps, alpha=${ALPHA}"
    echo "========================================"

    (
      cd "${RUNTIME_DIR}"
      "${PYTHON_BIN}" "${CV_SCRIPT}" \
        --dataset_path "${DATASET_DIR}" \
        --log_dir "${LOG_DIR}" \
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
        --scaler "${SCALER}" \
        --cv_embargo_hours 24
    )
  done

  echo "Threshold ${TARGET_THRESHOLD} bps: all CV runs completed."

  # The plotting script expects lightning_logs relative to its working directory.
  (
    cd "${THRESHOLD_DIR}"
    "${PYTHON_BIN}" "${PLOT_SCRIPT}" --experiment_name "${EXPERIMENT_NAME}"
  )

  echo "===================================================="
  echo "Threshold ${TARGET_THRESHOLD} bps: selecting top 2 by CV AUC and CV AUPRC"
  echo "===================================================="

  LOG_DIR="${LOG_DIR}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
  SELECTED_TSV="${SELECTED_TSV}" \
  "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os

import pandas as pd

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
    raise SystemExit(f"No model comparison summary CSV files found under {exp_dir}")

frames = []
for path in summary_files:
    frame = pd.read_csv(path)
    if frame.empty:
        continue
    frame["source_file"] = str(path)
    frame["source_mtime"] = path.stat().st_mtime
    frames.append(frame)

if not frames:
    raise SystemExit("No non-empty model comparison summary CSV files found.")

all_results = pd.concat(frames, ignore_index=True)
required = ["model_name", "alpha", "cv_auc_mean", "cv_auprc_mean"]
missing = [column for column in required if column not in all_results.columns]
if missing:
    raise SystemExit(f"Missing expected columns: {missing}")

for column in ["alpha", "cv_auc_mean", "cv_auprc_mean"]:
    all_results[column] = pd.to_numeric(all_results[column], errors="coerce")

all_results = (
    all_results.sort_values("source_mtime")
    .drop_duplicates(subset=["alpha", "model_name"], keep="last")
    .reset_index(drop=True)
)

top_auc = (
    all_results.dropna(subset=["cv_auc_mean"])
    .sort_values(["cv_auc_mean", "cv_auprc_mean"], ascending=False)
    .head(2)
    .copy()
)
top_auc["selection_reason"] = "top2_cv_auc"

top_auprc = (
    all_results.dropna(subset=["cv_auprc_mean"])
    .sort_values(["cv_auprc_mean", "cv_auc_mean"], ascending=False)
    .head(2)
    .copy()
)
top_auprc["selection_reason"] = "top2_cv_auprc"

selected = pd.concat([top_auc, top_auprc], ignore_index=True)
if selected.empty:
    raise SystemExit("No valid candidates found for full retraining.")

selected = (
    selected.groupby(["alpha", "model_name"], as_index=False)
    .agg(
        cv_auc_mean=("cv_auc_mean", "max"),
        cv_auprc_mean=("cv_auprc_mean", "max"),
        selection_reason=(
            "selection_reason",
            lambda values: ",".join(sorted(set(values))),
        ),
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
  echo "Threshold ${TARGET_THRESHOLD} bps: full retraining"
  echo "===================================================="

  while IFS=$'\t' read -r ALPHA MODEL CV_AUC CV_AUPRC REASON; do
    echo "----------------------------------------"
    echo "Model=${MODEL}, alpha=${ALPHA}, reason=${REASON}"
    echo "CV AUC=${CV_AUC}, CV AUPRC=${CV_AUPRC}"
    echo "----------------------------------------"

    (
      cd "${RUNTIME_DIR}"
      "${PYTHON_BIN}" "${FULL_TRAIN_SCRIPT}" \
        --dataset_path "${DATASET_DIR}" \
        --log_dir "${LOG_DIR}" \
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
    )
  done < <(tail -n +2 "${SELECTED_TSV}")

  echo "Threshold ${TARGET_THRESHOLD} bps completed."
  echo "CV output: ${LOG_DIR}/${EXPERIMENT_NAME}"
  echo "Full retraining output: ${LOG_DIR}/${FULL_EXPERIMENT_NAME}"
done

echo "===================================================="
echo "Building combined robustness-check tables"
echo "===================================================="

SCRIPT_DIR="${SCRIPT_DIR}" \
RUN_ID="${RUN_ID}" \
TARGET_THRESHOLDS="${TARGET_THRESHOLDS[*]}" \
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import json
import os

import pandas as pd

root = Path(os.environ["SCRIPT_DIR"])
run_id = os.environ["RUN_ID"]
thresholds = [int(value) for value in os.environ["TARGET_THRESHOLDS"].split()]

cv_frames = []
selected_frames = []
full_frames = []

for threshold in thresholds:
    threshold_dir = root / f"threshold_{threshold}"
    log_dir = threshold_dir / "lightning_logs"
    experiment = f"cv_model_comparison_threshold_{threshold}_{run_id}"
    full_experiment = f"{experiment}_full_retraining"

    summary_files = sorted(
        (log_dir / experiment).glob(
            "*_experiment_summary*/artifacts/comparison/model_comparison_summary.csv"
        )
    )
    for path in summary_files:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame.insert(0, "target_threshold_bps", threshold)
        frame["source_file"] = str(path)
        frame["source_mtime"] = path.stat().st_mtime
        cv_frames.append(frame)

    selected_path = log_dir / experiment / "selected_for_full_retraining.tsv"
    if selected_path.exists():
        selected = pd.read_csv(selected_path, sep="\t")
        selected.insert(0, "target_threshold_bps", threshold)
        selected_frames.append(selected)

    for metrics_path in sorted((log_dir / full_experiment).glob("*/metrics.json")):
        hparams_path = metrics_path.with_name("hparams.json")
        metrics = json.loads(metrics_path.read_text())
        hparams = json.loads(hparams_path.read_text()) if hparams_path.exists() else {}
        row = {
            "target_threshold_bps": threshold,
            "model_name": hparams.get("model_name"),
            "alpha": hparams.get("alpha"),
            "run_name": metrics_path.parent.name,
            **metrics,
            "source_file": str(metrics_path),
        }
        full_frames.append(pd.DataFrame([row]))

if not cv_frames:
    raise SystemExit("No CV summary rows found for the combined report.")

cv_results = pd.concat(cv_frames, ignore_index=True)
cv_results = (
    cv_results.sort_values("source_mtime")
    .drop_duplicates(
        subset=["target_threshold_bps", "alpha", "model_name"],
        keep="last",
    )
    .drop(columns=["source_mtime"])
    .sort_values(["target_threshold_bps", "alpha", "model_name"])
)
cv_output = root / f"cv_results_all_thresholds_{run_id}.csv"
cv_results.to_csv(cv_output, index=False)

if selected_frames:
    selected_results = pd.concat(selected_frames, ignore_index=True)
    selected_output = root / f"selected_models_all_thresholds_{run_id}.csv"
    selected_results.to_csv(selected_output, index=False)
else:
    selected_output = None

if full_frames:
    full_results = pd.concat(full_frames, ignore_index=True)
    full_results = full_results.sort_values(
        ["target_threshold_bps", "model_name", "alpha"]
    )
    full_output = root / f"full_retraining_metrics_all_thresholds_{run_id}.csv"
    full_results.to_csv(full_output, index=False)
else:
    full_output = None

print(f"Combined CV results: {cv_output}")
if selected_output:
    print(f"Combined selected models: {selected_output}")
if full_output:
    print(f"Combined full-retraining metrics: {full_output}")
PY

echo "===================================================="
echo "Robustness check completed."
echo "Run ID: ${RUN_ID}"
echo "All outputs: ${SCRIPT_DIR}"
echo "Execution log: ${RUN_LOG}"
echo "===================================================="
