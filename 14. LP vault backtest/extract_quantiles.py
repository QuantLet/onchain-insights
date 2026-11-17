"""Reduce the forecast artifacts (origins x 24 x 256 quantiles) to the 26 levels used by the backtest.

    python extract_quantiles.py            # all forecasters in config.FORECAST_ARTIFACTS
    python extract_quantiles.py "Our model"
Writes forecast_quantiles/q_<name>.npz (Qsel, and ds = target timestamps when the artifact stores them).
"""
import json, sys
import numpy as np
import config as C

IDX = [3, 4, 7, 8, 17, 18, 34, 35, 61, 62, 111, 112, 127, 128, 143, 144, 193, 194, 220, 221,
       237, 238, 247, 248, 251, 252]       # pairs bracketing 0.5,1,2.5,5,10,25,50,75,90,95,97.5,99,99.5%


def extract(src, dst):
    z = np.load(src)
    md = json.loads(str(z['__metadata__'].item()))
    names = {str(k.get('value')): v for k, v in md['tree']['items']}
    Q = z[names['Q']['name']]
    out = {'Qsel': Q[:, :, IDX].astype(np.float64)}
    if 'ds' in names and names['ds']['type'] == 'ndarray':
        out['ds'] = z[names['ds']['name']].astype('datetime64[s]').astype(np.int64)
    np.savez_compressed(dst, **out)
    print(f'{src.name} {Q.shape} -> {dst.name}')


if __name__ == '__main__':
    C.QDIR.mkdir(exist_ok=True)
    for name, src in C.FORECAST_ARTIFACTS.items():
        if len(sys.argv) > 1 and name not in sys.argv[1:]:
            continue
        if not src.exists():
            print(f'missing: {src}'); continue
        extract(src, C.QDIR / f"q_{name.lower().replace(' ', '_')}.npz")
