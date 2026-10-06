#!/usr/bin/env bash
# Run Optuna hyperparameter tuning for the SAINT model.
# All static training/forecast settings are baked into hyperparameter_tuning.py
# (mirroring scripts/run_SaINT.sh); only the search-loop options are exposed
# here.

set -euo pipefail

cd "$(dirname "$0")/.."

N_TRIALS="${N_TRIALS:-10}"
TIMEOUT="${TIMEOUT:-}"                 # seconds; empty = no time limit
STUDY_NAME="${STUDY_NAME:-SAINT_tuning}"
STORAGE="${STORAGE:-sqlite:///optuna_saint.db}"
MONITOR="${MONITOR:-val_loss}"
DIRECTION="${DIRECTION:-minimize}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-stablecoin-paper-optuna}"
SAMPLER_SEED="${SAMPLER_SEED:-1233}"

EXTRA_ARGS=()
if [[ -n "${TIMEOUT}" ]]; then
    EXTRA_ARGS+=(--timeout "${TIMEOUT}")
fi

python hyperparameter_tuning.py \
    --n_trials "${N_TRIALS}" \
    --study_name "${STUDY_NAME}" \
    --storage "${STORAGE}" \
    --monitor "${MONITOR}" \
    --direction "${DIRECTION}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --sampler_seed "${SAMPLER_SEED}" \
    --remote_logging \
    "${EXTRA_ARGS[@]}"
