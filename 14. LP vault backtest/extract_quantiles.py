"""Reduce full forecast artifacts (Q: origins x 24 x 256) to the 26 quantile levels used by the
LP backtest. Run from the repository root after downloading the section-12 release:

    python "14. LP vault backtest/extract_quantiles.py"

Writes 14. LP vault backtest/forecast_quantiles/q_<name>.npz for every artifact found.
"""
import json
import sys
from pathlib import Path
import numpy as np

IDX = [3, 4, 7, 8, 17, 18, 34, 35, 61, 62, 111, 112, 127, 128, 143, 144, 193, 194, 220, 221,
       237, 238, 247, 248, 251, 252]
HERE = Path(__file__).resolve().parent
SEC = HERE.parent / '12. Evaluation of Probabilistic Forecasts'
ARTIFACTS = {
    'ours': SEC / 'model_logs/final_retrain_run/preds_test_set.compact.npz',
    'timexer': SEC / 'benchmark_outputs/timeXer_hist_exog/preds_test_set.compact.npz',
    'tide': SEC / 'benchmark_outputs/tide_hist_exog/preds_test_set.compact.npz',
    'garch': SEC / 'benchmark_outputs/garch_student_t/preds_test_set.compact.npz',
    'naive': SEC / 'benchmark_outputs/naive/preds_test_set.compact.npz',
    'arima': SEC / 'benchmark_outputs/arima/preds_test_set.compact.npz',
}


def extract(src, dst):
    z = np.load(src)
    md = json.loads(str(z['__metadata__'].item()))
    names = {str(k.get('value')): v for k, v in md['tree']['items']}
    Q = z[names['Q']['name']]
    out = {'Qsel': Q[:, :, IDX].astype(np.float64), 'idx': np.array(IDX)}
    if 'ds' in names and names['ds']['type'] == 'ndarray':
        out['ds'] = z[names['ds']['name']].astype('datetime64[s]').astype(np.int64)
    np.savez_compressed(dst, **out)
    print(f'{src} {Q.shape} -> {dst}')


if __name__ == '__main__':
    od = HERE / 'forecast_quantiles'; od.mkdir(exist_ok=True)
    only = set(sys.argv[1:])
    for name, src in ARTIFACTS.items():
        if only and name not in only:
            continue
        dst = od / f'q_{name}.npz'
        if not src.exists():
            print(f'missing: {src}'); continue
        try:
            extract(src, dst)
        except Exception as e:  # truncated download etc.
            print(f'FAILED {name}: {e!r}')
