#!/usr/bin/env bash
set -euo pipefail

# Run the complete utility-aware CV workflow for the 15-bp realised-depeg
# definition only. A distinct experiment suffix avoids mixing this result with
# multi-threshold runs made on the same date.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DEPEG_THRESHOLDS_OVERRIDE="15" EXPERIMENT_SUFFIX="15bp" bash ./compare_model_cv.sh
