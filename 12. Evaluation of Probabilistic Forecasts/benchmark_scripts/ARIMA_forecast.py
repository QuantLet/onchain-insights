"""Fast probabilistic AutoARIMA benchmark using StatsForecast.

The cross-validation call fits AutoARIMA once and reuses the fitted model
across rolling cutoffs (``refit=False``).  StatsForecast supplies Gaussian
prediction intervals; no residual distribution is fitted here.
"""

import pickle as pkl
import re
from pathlib import Path

import numpy as np
import pandas as pd

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA


DATA_PATH = "./preprocessed_datasets/dataset_alpha_0.1_full.parquet"
OUT_DIR = Path("./benchmark_outputs/arima")
REFERENCE_PICKLE = "3328/a4b021c593f044fabee4a9207a5d090f/preds_test_set.pkl"
FREQ = "h"
HORIZON = 24
TRAIN_FRACTION = 0.7


def load_reference_u_grid(reference_pickle=None):
    """Load the comparison grid, or use the benchmark's default grid."""
    if reference_pickle is not None and Path(reference_pickle).exists():
        with open(reference_pickle, "rb") as f:
            grid = np.asarray(pkl.load(f)["u_grid"], dtype=np.float64)
        print(f"Loaded reference u_grid from {reference_pickle}: J={len(grid)}")
        return grid

    grid = np.array(
        [
            0.001, 0.0025, 0.005, 0.01, 0.02, 0.025, 0.05,
            0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
            0.50,
            0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
            0.95, 0.975, 0.98, 0.99, 0.995, 0.9975, 0.999,
        ],
        dtype=np.float64,
    )
    print(f"Using default u_grid: J={len(grid)}")
    return grid


def central_levels_from_u_grid(u_grid):
    """Convert quantiles to the central interval levels StatsForecast expects."""
    levels = []
    for u in np.asarray(u_grid, dtype=np.float64):
        if np.isclose(u, 0.5):
            continue
        level = 100.0 * abs(2.0 * u - 1.0)
        if 0.0 < level < 100.0:
            levels.append(round(float(level), 6))
    return sorted(set(levels))


def find_interval_col(columns, model_name, side, level):
    """Find a StatsForecast interval column despite numeric formatting differences."""
    prefix = f"{model_name}-{side}-"
    exact = [
        f"{prefix}{level:g}",
        f"{prefix}{float(level):.1f}",
        f"{prefix}{float(level):.2f}",
        f"{prefix}{float(level):.6f}",
    ]
    for name in exact:
        if name in columns:
            return name

    pattern = re.compile(rf"^{re.escape(prefix)}([0-9]+(?:\.[0-9]+)?)$")
    matches = [
        (abs(float(match.group(1)) - float(level)), name)
        for name in columns
        if (match := pattern.match(name)) is not None
    ]
    if matches and min(matches)[0] <= 1e-4:
        return min(matches)[1]
    raise KeyError(f"Missing {prefix}{level} in StatsForecast output.")


def cv_to_quantile_pickle_dict(cv_df, u_grid, model_name="AutoARIMA"):
    """Convert StatsForecast's long CV table to the existing array format."""
    cv_df = cv_df.copy()
    cv_df["ds"] = pd.to_datetime(cv_df["ds"])
    cv_df["cutoff"] = pd.to_datetime(cv_df["cutoff"])

    cutoffs = np.array(sorted(cv_df["cutoff"].unique()))
    if len(cutoffs) == 0:
        raise ValueError("StatsForecast returned no cross-validation windows.")

    horizon_lengths = cv_df.groupby("cutoff").size()
    if horizon_lengths.nunique() != 1:
        raise ValueError("Cross-validation windows have inconsistent horizons.")

    B, H, J = len(cutoffs), int(horizon_lengths.iloc[0]), len(u_grid)
    true = np.empty((B, H), dtype=np.float64)
    quantiles = np.empty((B, H, J), dtype=np.float64)
    ds_mat = np.empty((B, H), dtype="datetime64[ns]")
    levels = central_levels_from_u_grid(u_grid)

    for b, cutoff in enumerate(cutoffs):
        rows = cv_df[cv_df["cutoff"] == cutoff].sort_values("ds")
        if len(rows) != H:
            raise ValueError(f"Cutoff {cutoff} has {len(rows)} rows, expected {H}.")

        true[b] = rows["y"].to_numpy(dtype=np.float64)
        ds_mat[b] = rows["ds"].to_numpy(dtype="datetime64[ns]")
        for j, u in enumerate(u_grid):
            if np.isclose(u, 0.5):
                quantiles[b, :, j] = rows[model_name].to_numpy(dtype=np.float64)
            elif u < 0.5:
                level = 100.0 * (1.0 - 2.0 * u)
                col = find_interval_col(rows.columns, model_name, "lo", level)
                quantiles[b, :, j] = rows[col].to_numpy(dtype=np.float64)
            else:
                level = 100.0 * (2.0 * u - 1.0)
                col = find_interval_col(rows.columns, model_name, "hi", level)
                quantiles[b, :, j] = rows[col].to_numpy(dtype=np.float64)

    # Numerical interval crossings can occur at extreme levels.
    quantiles = np.maximum.accumulate(quantiles, axis=-1)
    return {
        "true": true,
        "Q": quantiles,
        "u_grid": np.asarray(u_grid, dtype=np.float64),
        "ds": ds_mat,
        "meta": pd.DataFrame({"cutoff": pd.to_datetime(cutoffs)}),
        "model_col": model_name,
        "point_col": model_name,
        "interval_prefix": model_name,
        "statsforecast_config": {
            "model": model_name,
            "refit": False,
            "h": H,
        },
    }


def save_pickle(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pkl.dump(value, f)
    print(f"Saved: {path.resolve()}")


def main():
    df = pd.read_parquet(DATA_PATH).copy()
    df["ds"] = df.index.tz_localize(None)
    df["y"] = df["depeg_bps"]
    df["unique_id"] = "stablecoin_depeg"
    y_df = df[["unique_id", "ds", "y"]].reset_index(drop=True)

    h = HORIZON
    test_size = len(y_df) - int(TRAIN_FRACTION * len(y_df))
    train_size = len(y_df) - test_size
    n_windows = test_size - h + 1
    if n_windows < 1:
        raise ValueError("The configured training fraction leaves no CV window.")

    u_grid = load_reference_u_grid(REFERENCE_PICKLE)
    levels = central_levels_from_u_grid(u_grid)
    model = AutoARIMA(
        season_length=24,
        stepwise=True,
        approximation=True,
        alias="AutoARIMA",
    )
    sf = StatsForecast(models=[model], freq=FREQ, n_jobs=-1)

    print(f"Y_df: {y_df.shape}")
    print(f"train_size: {train_size}")
    print(f"test_size: {test_size}")
    print(f"h: {h}")
    print(f"n_windows: {n_windows}")
    print(f"Running StatsForecast AutoARIMA cross-validation: h={h}, n_windows={n_windows}, step_size=1, refit=False")
    cv_df = sf.cross_validation(
        df=y_df,
        h=h,
        n_windows=n_windows,
        step_size=1,
        level=levels,
        refit=False,
    )

    # Keep every rolling forecast origin on the same test segment as GARCH
    # and Naive: test targets start at train_size and advance one row per window.
    ordered_cv = cv_df.assign(
        cutoff=pd.to_datetime(cv_df["cutoff"]),
        ds=pd.to_datetime(cv_df["ds"]),
    ).sort_values(["cutoff", "ds"])
    window_sizes = ordered_cv.groupby("cutoff", sort=True).size().to_numpy()
    if len(window_sizes) != n_windows:
        raise ValueError(
            f"StatsForecast returned {len(window_sizes)} windows, expected {n_windows}."
        )
    if not np.all(window_sizes == h):
        raise ValueError(f"StatsForecast returned window lengths {window_sizes}, expected {h}.")
    actual_ds = ordered_cv["ds"].to_numpy(dtype="datetime64[ns]").reshape(n_windows, h)
    full_ds = pd.to_datetime(y_df["ds"]).to_numpy(dtype="datetime64[ns]")
    expected_ds = np.stack(
        [full_ds[train_size + b : train_size + b + h] for b in range(n_windows)]
    )
    if not np.array_equal(actual_ds, expected_ds):
        raise ValueError("StatsForecast windows do not match the expected 70/30 rolling test split.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv_df.to_parquet(OUT_DIR / "arima_cv.parquet", index=False)
    result = cv_to_quantile_pickle_dict(cv_df, u_grid, model_name="AutoARIMA")
    result["statsforecast_config"].update(
        {
            "freq": FREQ,
            "train_size": train_size,
            "test_size": test_size,
            "n_windows": n_windows,
            "step_size": 1,
            "levels": levels,
        }
    )
    save_pickle(result, OUT_DIR / "preds_test_set.pkl")


if __name__ == "__main__":
    main()
