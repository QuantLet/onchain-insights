#!/usr/bin/env bash
set -euo pipefail

SCRIPT="cv_model_comparison.py"
EXPERIMENT_NAME="cv_model_comparison"

ALPHAS=(0.1 0.3 0.5 1.0 1.5 2.0)
MODELS=(xgboost lightgbm catboost random_forest)

for ALPHA in "${ALPHAS[@]}"; do
  echo "========================================"
  echo "Running CV comparison for alpha=${ALPHA}"
  echo "========================================"

  python "${SCRIPT}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --run_name "alpha_${ALPHA}" \
    --alpha "${ALPHA}" \
    --target \
    --target_window 24 \
    --target_threshold 15 \
    --depeg_side both \
    --model_names "${MODELS[@]}" \
    --scaler robust 
done

echo "All CV runs completed."