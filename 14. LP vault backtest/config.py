"""All paths and parameters of the LP vault backtest. Edit this file when the models are re-run.

Paths are relative to the repository root (env var REPO, default: the parent of this folder).
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('REPO', HERE.parent))
OUT = HERE / 'results'            # csv + pickles written by backtest.py
FIG = HERE / 'figures'            # png written by figures.py
GEN = HERE / 'paper' / 'generated'  # LaTeX tables and number macros written by paper_tables.py

# --- pool data -------------------------------------------------------------------------------------
PANEL = REPO / '4. Stablecoin liquidity ownership/USDC_USDT_hourly_metrics.parquet'
POOL = '0x3416cf6c708da44db2624d63ea0aaef7113527c6'      # USDC/USDT 0.01% Uniswap V3
FEE_DATA_START = '2024-03-20 00:00'   # the subgraph has no volume/fees before 2024-03-19 16:00 UTC

# --- early-warning classifier ------------------------------------------------------------------------
KUMO_RUN = REPO / ('8. Early-Warning Classifier Cross Validation/lightning_logs/'
                   'cv_model_comparison_lead6_24_10_15bps_2026-10-19/cv_results_report')
KUMO_SCORES = KUMO_RUN / 'holdout_cache/threshold_15/kumo_tabular_alpha_1/operational_scores.parquet'
KUMO_CUTOFF = 0.09446171671152116      # NP cutoff of the selected specification, fixed before the holdout
KUMO_QUIET = 12                         # quiet-period alert reset (hours)
KUMO_CV_ALERTS = KUMO_RUN / 'cv_alert_outcomes.csv'   # out-of-fold alert onsets (2024 check)
KUMO_CV_FILTER = dict(target_threshold=15, alpha=1.0, model_name='kumo_tabular')
KUMO_CV_EPISODE_H = 21                  # episode length for CV onsets (holdout median; hourly CV scores not stored)
EVENT_BP = 15.0                         # event threshold the classifier is trained for

# --- probabilistic forecasters -------------------------------------------------------------------------
# Full artifacts (origins x 24 horizons x 256 quantiles); extract_quantiles.py reduces them to QDIR.
SEC12 = REPO / '12. Evaluation of Probabilistic Forecasts'
FORECAST_ARTIFACTS = {
    'Our model': SEC12 / 'model_logs/final_retrain_run/preds_test_set.compact.npz',
    'TimeXer': SEC12 / 'benchmark_outputs/timeXer_hist_exog/preds_test_set.compact.npz',
    'TiDE': SEC12 / 'benchmark_outputs/tide_hist_exog/preds_test_set.compact.npz',
    'GARCH': SEC12 / 'benchmark_outputs/garch_student_t/preds_test_set.compact.npz',
}
QDIR = HERE / 'forecast_quantiles'
MAIN_FORECASTER = 'Our model'
FIRST_ORIGIN = '2025-01-15 14:00'       # used only if an artifact carries no timestamps ('ds')
RANGE_ALPHA = 0.2                       # range = [q_{alpha/2}, q_{1-alpha/2}] ...
RANGE_HORIZON = 6                       # ... union over the next RANGE_HORIZON hours

# --- vault and evaluation ------------------------------------------------------------------------------
V0, GAS, SLIP_BPS, POOL_FEE_BPS = 1e6, 10.0, 0.5, 1.0
GAMMA_HALF_WIDTH = 2                    # Gamma-style recentring range (ticks)
STATIC_HALF_WIDTH = 5                   # static range around parity (ticks)
EVENT_WINDOW = (24, 48)                 # hours before / after each event for the event drawdown
N_PLACEBO, SEED = 500, 7
STRESS_WINDOW = ('2023-03-01 00:00', '2023-03-31 23:00')   # USDC depeg scenario (losses only)
STRESS_LEADS = (2, 9)                   # hours before the first breach at which a freeze starts
Y2024 = (FEE_DATA_START, '2024-12-31 23:00')   # out-of-sample check on the classifier's CV folds
STRESS_SUM = ('2023-03-10', '2023-03-20')      # loss window reported for the stress scenario
