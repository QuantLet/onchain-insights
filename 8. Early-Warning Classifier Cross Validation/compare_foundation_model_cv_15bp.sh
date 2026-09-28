#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Defaults in foundation_model_cv.py match the 15-bp tree CV run: six alphas,
# five expanding folds, 48-hour embargo, 24-hour warning window, a 0.5–4
# false-alert-episode/month sensitivity range, and event/calendar-block
# bootstrap intervals.
python -u foundation_model_cv.py "$@"
