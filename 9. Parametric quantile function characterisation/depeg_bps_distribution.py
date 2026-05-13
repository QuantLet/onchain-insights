#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm

from tqdm.auto import tqdm
from time import perf_counter

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, GARCH


# ============================================================
# Configuration
# ============================================================

DATA_PATH = Path("./USDC_USDT_hourly_metrics.parquet")
FIG_DIR = Path("./figures")
CACHE_DIR = Path("./cache")

FORCE_REFIT = False

PANDAS_FREQ = "h"
SF_FREQ = "h"

AUTOARIMA_SEASON_LENGTH = 24
GARCH_P = 1
GARCH_Q = 1

QQ_ZOOM_CENTRAL = 0.995   # show central 99.5% in QQ axes
QQ_MARKER_ALPHA = 0.35

TAIL_FOCUS_STANDARD = 0.05
TAIL_FOCUS_EXTREME = 0.01

DPI = 300


# ============================================================
# Plot style
# ============================================================
def zscore_series(x: pd.Series) -> pd.Series:
    x = pd.Series(x).dropna().astype(float)
    s = x.std(ddof=0)
    if s == 0 or not np.isfinite(s):
        raise ValueError("Cannot standardize a series with zero or non-finite std.")
    return (x - x.mean()) / s

def prepare_standardized_diagnostics(arima_resid, garch_resid, garch_std_resid):
    """
    ARIMA:
      - standardize raw residuals by z-score

    GARCH:
      - if standardized residuals are available, use them
      - optionally re-zscore them for plotting consistency
      - otherwise z-score raw residuals as fallback
    """
    arima_z = zscore_series(arima_resid)

    if garch_std_resid is not None and len(garch_std_resid) > 0:
        garch_base = pd.Series(garch_std_resid).dropna().astype(float)
        garch_label = f"GARCH({GARCH_P},{GARCH_Q}) standardized residuals"
    else:
        garch_base = pd.Series(garch_resid).dropna().astype(float)
        garch_label = f"GARCH({GARCH_P},{GARCH_Q}) residuals"

    # optional recenter / rescale for plotting comparability
    garch_z = zscore_series(garch_base)

    return arima_z, garch_z, garch_label

def setup_plot_style():
    # plt.style.use("seaborn-v0_8-white")
    mpl.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })


def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def savefig(fig, filename: str):
    out = FIG_DIR / filename
    fig.savefig(
        out,
        dpi=DPI,
        bbox_inches="tight",
        transparent=True,
        facecolor="none",
    )
    plt.close(fig)
    print(f"Saved figure: {out}")


# ============================================================
# Data loading
# ============================================================

def load_price_series(data_path: Path) -> pd.Series:
    df = pd.read_parquet(data_path).query("feeTier == 100").copy()
    df.index = pd.to_datetime(df["datetime"])
    price = df["depeg_bps"].astype(float).sort_index()

    # remove duplicates if any
    price = price[~price.index.duplicated(keep="last")]

    # # regularize to hourly frequency
    # price = price.asfreq(PANDAS_FREQ)

    # # fill gaps if any
    # if price.isna().any():
    #     price = price.interpolate(method="time").ffill().bfill()

    return price


def to_statsforecast_df(y: pd.Series) -> pd.DataFrame:
    ds = y.index
    if getattr(ds, "tz", None) is not None:
        ds = ds.tz_convert("UTC").tz_localize(None)

    return pd.DataFrame({
        "unique_id": "usdc_usdt_pool",
        "ds": pd.DatetimeIndex(ds),
        "y": y.to_numpy(),
    })

def timed_step(desc, func, *args, **kwargs):
    t0 = perf_counter()
    print(f"[START] {desc}")
    out = func(*args, **kwargs)
    t1 = perf_counter()
    print(f"[DONE ] {desc} in {t1 - t0:.2f}s")
    return out

# ============================================================
# Cached model fitting
# ============================================================

def fit_autoarima_cached(
    y: pd.Series,
    force_refit: bool = False,
    season_length: int = AUTOARIMA_SEASON_LENGTH,
):
    cache_file = CACHE_DIR / "autoarima_fitted.parquet"

    if cache_file.exists() and not force_refit:
        fitted_df = pd.read_parquet(cache_file)
        print(f"Loaded cached AutoARIMA residuals from {cache_file}")
    else:
        df_sf = to_statsforecast_df(y)

        model = AutoARIMA(
            season_length=1,
            stepwise=True,
            approximation=True,
            trace = True,
            alias="AutoARIMA",
        )

        sf = StatsForecast(
            models=[model],
            freq=SF_FREQ,
            n_jobs=-1,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = sf.forecast(df=df_sf, h=1, fitted=True)

        fitted_df = sf.forecast_fitted_values().copy()

        if "y" not in fitted_df.columns:
            fitted_df = fitted_df.merge(df_sf, on=["unique_id", "ds"], how="left")

        reserved = {"unique_id", "ds", "y"}
        model_cols = [c for c in fitted_df.columns if c not in reserved]
        if len(model_cols) != 1:
            raise ValueError(
                f"AutoARIMA: expected one fitted-value column, got {model_cols}"
            )

        fitted_df = fitted_df.rename(columns={model_cols[0]: "fitted"})
        fitted_df["resid"] = fitted_df["y"] - fitted_df["fitted"]
        fitted_df.to_parquet(cache_file, index=False)
        print(f"Saved AutoARIMA residuals to {cache_file}")

    resid = pd.Series(
        fitted_df["resid"].values,
        index=pd.DatetimeIndex(fitted_df["ds"]),
        name="autoarima_resid",
    ).dropna()

    return fitted_df, resid


def fit_garch_cached(
    y: pd.Series,
    force_refit: bool = False,
    p: int = GARCH_P,
    q: int = GARCH_Q,
):
    cache_file = CACHE_DIR / f"garch_{p}_{q}_fitted.parquet"

    if cache_file.exists() and not force_refit:
        fitted_df = pd.read_parquet(cache_file)
        print(f"Loaded cached GARCH residuals from {cache_file}")
    else:
        df_sf = to_statsforecast_df(y)

        model = GARCH(p=p, q=q, alias=f"GARCH({p},{q})")

        sf = StatsForecast(
            models=[model],
            freq=SF_FREQ,
            n_jobs=1,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = sf.forecast(df=df_sf, h=1, fitted=True)

        fitted_df = sf.forecast_fitted_values().copy()

        if "y" not in fitted_df.columns:
            fitted_df = fitted_df.merge(df_sf, on=["unique_id", "ds"], how="left")

        reserved = {"unique_id", "ds", "y"}
        candidate_cols = [c for c in fitted_df.columns if c not in reserved]

        sigma_col = None
        sigma2_col = None
        for c in candidate_cols:
            cl = c.lower()
            if re.search(r"(^|_)(sigma2|variance|var)(_|$)", cl):
                sigma2_col = c
            elif re.search(r"(^|_)(sigma|std|vol|volatility)(_|$)", cl):
                sigma_col = c

        scale_like = {x for x in [sigma_col, sigma2_col] if x is not None}
        mean_cols = [c for c in candidate_cols if c not in scale_like]

        if len(mean_cols) == 0:
            raise ValueError(
                f"GARCH: could not identify fitted mean column. Columns={fitted_df.columns.tolist()}"
            )

        fitted_df = fitted_df.rename(columns={mean_cols[0]: "fitted"})
        fitted_df["resid"] = fitted_df["y"] - fitted_df["fitted"]

        if sigma_col is not None:
            fitted_df = fitted_df.rename(columns={sigma_col: "sigma"})
        elif sigma2_col is not None:
            fitted_df = fitted_df.rename(columns={sigma2_col: "sigma2"})
            fitted_df["sigma"] = np.sqrt(np.maximum(fitted_df["sigma2"], 0))
        else:
            fitted_df["sigma"] = np.nan

        if fitted_df["sigma"].notna().any():
            fitted_df["std_resid"] = fitted_df["resid"] / fitted_df["sigma"]
            fitted_df["std_resid"] = fitted_df["std_resid"].replace([np.inf, -np.inf], np.nan)
        else:
            fitted_df["std_resid"] = np.nan
            print(
                "Warning: statsforecast GARCH fitted values did not expose sigma/sigma2; "
                "falling back to raw residual diagnostics."
            )
            print("Available columns:", fitted_df.columns.tolist())

        fitted_df.to_parquet(cache_file, index=False)
        print(f"Saved GARCH residuals to {cache_file}")

    resid = pd.Series(
        fitted_df["resid"].values,
        index=pd.DatetimeIndex(fitted_df["ds"]),
        name="garch_resid",
    ).dropna()

    std_resid = None
    if "std_resid" in fitted_df.columns and fitted_df["std_resid"].notna().any():
        std_resid = pd.Series(
            fitted_df["std_resid"].values,
            index=pd.DatetimeIndex(fitted_df["ds"]),
            name="garch_std_resid",
        ).dropna()

    return fitted_df, resid, std_resid


# ============================================================
# Distribution helpers
# ============================================================

def fit_reference_distributions(x: pd.Series):
    x = pd.Series(x).dropna().astype(float).values
    norm_params = stats.norm.fit(x)
    t_params = stats.t.fit(x)
    return norm_params, t_params


def cdf_from_params(dist, x, params):
    x = np.asarray(x)
    if len(params) > 2:
        return dist.cdf(x, *params[:-2], loc=params[-2], scale=params[-1])
    return dist.cdf(x, loc=params[0], scale=params[1])


def sf_from_params(dist, x, params):
    x = np.asarray(x)
    if len(params) > 2:
        return dist.sf(x, *params[:-2], loc=params[-2], scale=params[-1])
    return dist.sf(x, loc=params[0], scale=params[1])


# ============================================================
# Plotting primitives
# ============================================================

EMP_COLOR = "black"
GAUSS_COLOR = "#A81C23"
T_COLOR = "#3DA0B4"

def lower_exceedance_t_on_ax(ax, x, t_params, left_focus, title, xlabel="Residual"):
    """
    Lower-tail exceedance plot:
        empirical P(X <= x) vs fitted Student-t CDF
    """
    x = np.sort(pd.Series(x).dropna().astype(float).values)
    n = len(x)
    ecdf = np.arange(1, n + 1) / n

    left_cut = np.quantile(x, left_focus)
    mask = x <= left_cut

    x_left = x[mask]
    emp = ecdf[mask]
    fit_t = cdf_from_params(stats.t, x_left, t_params)

    ax.plot(
        x_left, emp,
        drawstyle="steps-post",
        lw=2.2,
        color=EMP_COLOR,
        label="Empirical"
    )
    ax.plot(
        x_left, fit_t,
        "--",
        lw=2.2,
        color=T_COLOR,
        label="Student-t"
    )

    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("P(X ≤ x)")
    ax.set_title(title)
    ax.grid(alpha=0.3)

def upper_exceedance_t_on_ax(ax, x, t_params, right_focus, title, xlabel="Residual"):
    """
    Upper-tail exceedance plot:
        empirical P(X >= x) vs fitted Student-t survival
    """
    x = np.sort(pd.Series(x).dropna().astype(float).values)
    n = len(x)
    esf = (n - np.arange(n)) / n

    right_cut = np.quantile(x, right_focus)
    mask = x >= right_cut

    x_right = x[mask]
    emp = esf[mask]
    fit_t = sf_from_params(stats.t, x_right, t_params)

    ax.plot(
        x_right, emp,
        drawstyle="steps-post",
        lw=2.2,
        color=EMP_COLOR,
        label="Empirical"
    )
    ax.plot(
        x_right, fit_t,
        "--",
        lw=2.2,
        color=T_COLOR,
        label="Student-t"
    )

    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("P(X ≥ x)")
    ax.set_title(title)
    ax.grid(alpha=0.3)

def qqplot_standardized_square(ax, x, dist, title, t_params=None, zoom_central=QQ_ZOOM_CENTRAL):
    """
    QQ-plot on standardized units.

    Parameters
    ----------
    x : array-like
        Already standardized series (roughly centered/scaled).
    dist : scipy.stats distribution
        stats.norm or stats.t
    t_params : tuple or None
        If dist is Student-t, pass fitted params on standardized x.
    zoom_central : float
        Fraction of central points used to set axis limits, e.g. 0.995
        means axes are based on central 99.5% and extreme points may fall outside.
    """
    x = np.sort(pd.Series(x).dropna().astype(float).values)
    n = len(x)
    p = (np.arange(1, n + 1) - 0.5) / n

    if dist == stats.norm:
        q = stats.norm.ppf(p)
    elif dist == stats.t:
        if t_params is None:
            t_params = stats.t.fit(x, floc=0)
        q = stats.t.ppf(p, *t_params[:-2], loc=t_params[-2], scale=t_params[-1])
    else:
        raise ValueError("This helper currently supports only stats.norm and stats.t")

    ax.scatter(
        q, x,
        s=15,
        alpha=QQ_MARKER_ALPHA,
        facecolor="grey",
        edgecolor="none"
    )

    # robust square limits based on central mass
    alpha = (1.0 - zoom_central) / 2.0
    lo = min(np.quantile(q, alpha), np.quantile(x, alpha))
    hi = max(np.quantile(q, 1 - alpha), np.quantile(x, 1 - alpha))
    pad = 0.05 * (hi - lo)

    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", lw=1.5)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_box_aspect(1)
    ax.set_title(title)
    ax.set_xlabel("Theoretical quantiles (standardized scale)")
    ax.set_ylabel("Empirical quantiles (standardized scale)")

def pit_hist_on_ax(ax, x, dist, params, title, bins=20):
    x = pd.Series(x).dropna().astype(float).values
    u = cdf_from_params(dist, x, params)

    ax.hist(
        u,
        bins=bins,
        density=True,
        color="C0",
        alpha=0.75,
        edgecolor="white",
    )
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.0)
    ax.set_xlim(0, 1)
    ax.set_xlabel("PIT")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(alpha=0.3)

def lower_exceedance_on_ax(ax, x, norm_params, t_params, left_focus, title, xlabel="Residual"):
    x = np.sort(pd.Series(x).dropna().astype(float).values)
    n = len(x)
    ecdf = np.arange(1, n + 1) / n

    left_cut = np.quantile(x, left_focus)
    mask = x <= left_cut

    x_left = x[mask]
    emp = ecdf[mask]

    ax.plot(x_left, emp, drawstyle="steps-post", lw=2.0, color=EMP_COLOR, label="Empirical")
    ax.plot(x_left, cdf_from_params(stats.norm, x_left, norm_params), "--", lw=2.0, color=GAUSS_COLOR, label="Gaussian")
    ax.plot(x_left, cdf_from_params(stats.t, x_left, t_params), "--", lw=2.0, color=T_COLOR, label="Student-t")

    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Lower-tail exceedance  P(X ≤ x)")
    ax.set_title(title)
    ax.grid(alpha=0.3)

def upper_exceedance_on_ax(ax, x, norm_params, t_params, right_focus, title, xlabel="Residual"):
    x = np.sort(pd.Series(x).dropna().astype(float).values)
    n = len(x)
    esf = (n - np.arange(n)) / n

    right_cut = np.quantile(x, right_focus)
    mask = x >= right_cut

    x_right = x[mask]
    emp = esf[mask]

    ax.plot(x_right, emp, drawstyle="steps-post", lw=2.0, color=EMP_COLOR, label="Empirical")
    ax.plot(x_right, sf_from_params(stats.norm, x_right, norm_params), "--", lw=2.0, color=GAUSS_COLOR, label="Gaussian")
    ax.plot(x_right, sf_from_params(stats.t, x_right, t_params), "--", lw=2.0, color=T_COLOR, label="Student-t")

    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Upper-tail exceedance  P(X ≥ x)")
    ax.set_title(title)
    ax.grid(alpha=0.3)

def lower_tail_on_ax(ax, x, norm_params, t_params, left_focus, title):
    x = np.sort(pd.Series(x).dropna().astype(float).values)
    n = len(x)
    ecdf = np.arange(1, n + 1) / n

    left_cut = np.quantile(x, left_focus)
    mask = x <= left_cut

    x_left = x[mask]
    emp_left = ecdf[mask]

    ax.plot(
        x_left, emp_left,
        drawstyle="steps-post",
        lw=2.0,
        color=EMP_COLOR,
        label="Empirical"
    )
    ax.plot(
        x_left,
        cdf_from_params(stats.norm, x_left, norm_params),
        "--",
        lw=2.0,
        color=GAUSS_COLOR,
        label="Gaussian"
    )
    ax.plot(
        x_left,
        cdf_from_params(stats.t, x_left, t_params),
        "--",
        lw=2.0,
        color=T_COLOR,
        label="Student-t"
    )

    ax.set_yscale("log")
    ax.set_xlabel("Residual")
    ax.set_ylabel("P(X ≤ x)")
    ax.set_title(title)
    ax.grid(alpha=0.3)


def upper_tail_on_ax(ax, x, norm_params, t_params, right_focus, title):
    x = np.sort(pd.Series(x).dropna().astype(float).values)
    n = len(x)
    esf = (n - np.arange(n)) / n

    right_cut = np.quantile(x, right_focus)
    mask = x >= right_cut

    x_right = x[mask]
    emp_right = esf[mask]

    ax.plot(
        x_right, emp_right,
        drawstyle="steps-post",
        lw=2.0,
        color=EMP_COLOR,
        label="Empirical"
    )
    ax.plot(
        x_right,
        sf_from_params(stats.norm, x_right, norm_params),
        "--",
        lw=2.0,
        color=GAUSS_COLOR,
        label="Gaussian"
    )
    ax.plot(
        x_right,
        sf_from_params(stats.t, x_right, t_params),
        "--",
        lw=2.0,
        color=T_COLOR,
        label="Student-t"
    )

    ax.set_yscale("log")
    ax.set_xlabel("Residual")
    ax.set_ylabel("P(X ≥ x)")
    ax.set_title(title)
    ax.grid(alpha=0.3)


# ============================================================
# Figure builders
# ============================================================

def make_t_exceedance_figure(
    x,
    name_prefix,
    t_params,
    outname,
    left_focus=TAIL_FOCUS_EXTREME,
    right_focus=1 - TAIL_FOCUS_EXTREME,
    standardized=False,
):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)

    xlabel = "Standardized residual" if standardized else "Residual (bps)"

    lower_exceedance_t_on_ax(
        axes[0],
        x,
        t_params,
        left_focus=left_focus,
        title=f"{name_prefix}: lower-tail exceedance ({int(100*left_focus)}%)",
        xlabel=xlabel,
    )

    upper_exceedance_t_on_ax(
        axes[1],
        x,
        t_params,
        right_focus=right_focus,
        title=f"{name_prefix}: upper-tail exceedance ({int(100*(1-right_focus))}%)",
        xlabel=xlabel,
    )

    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)

    savefig(fig, outname)

def make_qq_figure_standardized(x_std, name_prefix, t_params, outname):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    qqplot_standardized_square(
        axes[0],
        x_std,
        stats.norm,
        f"",
    )
    qqplot_standardized_square(
        axes[1],
        x_std,
        stats.t,
        f"",
        t_params=t_params,
    )

    savefig(fig, outname)

def make_exceedance_figure(x, name_prefix, norm_params, t_params, outname, standardized=False):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    xlabel = "Standardized residual" if standardized else "Residual (bps)"

    lower_exceedance_on_ax(
        axes[0],
        x,
        norm_params,
        t_params,
        left_focus=TAIL_FOCUS_EXTREME,
        title=f"{name_prefix}: lower-tail exceedance (1%)",
        xlabel=xlabel,
    )

    upper_exceedance_on_ax(
        axes[1],
        x,
        norm_params,
        t_params,
        right_focus=1 - TAIL_FOCUS_EXTREME,
        title=f"{name_prefix}: upper-tail exceedance (1%)",
        xlabel=xlabel,
    )

    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)

    savefig(fig, outname)

def make_pit_figure(x, name_prefix, norm_params, t_params, outname):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    pit_hist_on_ax(
        axes[0], x, stats.norm, norm_params,
        f"{name_prefix}: PIT under Gaussian"
    )
    pit_hist_on_ax(
        axes[1], x, stats.t, t_params,
        f"{name_prefix}: PIT under Student-t"
    )

    savefig(fig, outname)


def make_tail_figure(x, name_prefix, norm_params, t_params, left_focus, right_focus, outname):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    lower_tail_on_ax(
        axes[0], x, norm_params, t_params, left_focus,
        f"{name_prefix}: lower tail ({int(100*left_focus)}%)"
    )
    upper_tail_on_ax(
        axes[1], x, norm_params, t_params, right_focus,
        f"{name_prefix}: upper tail ({int(100*(1-right_focus))}%)"
    )

    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)

    savefig(fig, outname)

def make_publication_comparison_figure(
    arima_raw,
    arima_std,
    garch_raw,
    garch_std,
    arima_norm_params_raw,
    arima_t_params_raw,
    arima_t_params_std,
    garch_norm_params_raw,
    garch_t_params_raw,
    garch_t_params_std,
    garch_label,
    outname="comparison_autoarima_vs_garch.png",
):
    fig, axes = plt.subplots(5, 2, figsize=(10, 18), constrained_layout=True)

    # Row 1: standardized QQ vs Gaussian
    qqplot_standardized_square(
        axes[0, 0], arima_std, stats.norm,
        "AutoARIMA residuals: standardized QQ vs Gaussian"
    )
    qqplot_standardized_square(
        axes[0, 1], garch_std, stats.norm,
        f"{garch_label}: standardized QQ vs Gaussian"
    )

    # Row 2: standardized QQ vs Student-t
    qqplot_standardized_square(
        axes[1, 0], arima_std, stats.t,
        "AutoARIMA residuals: standardized QQ vs Student-t",
        t_params=arima_t_params_std,
    )
    qqplot_standardized_square(
        axes[1, 1], garch_std, stats.t,
        f"{garch_label}: standardized QQ vs Student-t",
        t_params=garch_t_params_std,
    )

    # Row 3: PIT under Student-t on raw residual scale
    pit_hist_on_ax(
        axes[2, 0], arima_raw, stats.t, arima_t_params_raw,
        "AutoARIMA residuals: PIT under Student-t"
    )
    pit_hist_on_ax(
        axes[2, 1], garch_raw, stats.t, garch_t_params_raw,
        f"{garch_label}: PIT under Student-t"
    )

    # Row 4: lower exceedance
    lower_exceedance_on_ax(
        axes[3, 0], arima_raw, arima_norm_params_raw, arima_t_params_raw, TAIL_FOCUS_EXTREME,
        "AutoARIMA residuals: lower-tail exceedance (1%)",
        xlabel="Residual (bps)"
    )
    lower_exceedance_on_ax(
        axes[3, 1], garch_raw, garch_norm_params_raw, garch_t_params_raw, TAIL_FOCUS_EXTREME,
        f"{garch_label}: lower-tail exceedance (1%)",
        xlabel="Residual (bps)" if "standardized" not in garch_label.lower() else "Standardized residual"
    )

    # Row 5: upper exceedance
    upper_exceedance_on_ax(
        axes[4, 0], arima_raw, arima_norm_params_raw, arima_t_params_raw, 1 - TAIL_FOCUS_EXTREME,
        "AutoARIMA residuals: upper-tail exceedance (1%)",
        xlabel="Residual (bps)"
    )
    upper_exceedance_on_ax(
        axes[4, 1], garch_raw, garch_norm_params_raw, garch_t_params_raw, 1 - TAIL_FOCUS_EXTREME,
        f"{garch_label}: upper-tail exceedance (1%)",
        xlabel="Residual (bps)" if "standardized" not in garch_label.lower() else "Standardized residual"
    )

    axes[3, 0].legend(frameon=False)
    axes[3, 1].legend(frameon=False)
    axes[4, 0].legend(frameon=False)
    axes[4, 1].legend(frameon=False)

    fig.suptitle(
        "Residual diagnostics: AutoARIMA vs GARCH baseline\n"
        "Standardized QQ-plots and tail exceedance diagnostics",
        y=1.01,
        fontsize=14,
    )

    savefig(fig, outname)

def save_model_figures(x_raw, x_std, name_prefix, norm_params_raw, t_params_raw, t_params_std, stem):
    # standardized QQ
    make_qq_figure_standardized(
        x_std,
        name_prefix,
        t_params_std,
        f"{stem}_qq_standardized.png",
    )

    # PIT on raw scale
    make_pit_figure(
        x_raw,
        name_prefix,
        norm_params_raw,
        t_params_raw,
        f"{stem}_pit.png",
    )

    # Student-t-only exceedance plot on raw scale
    make_t_exceedance_figure(
        x_raw,
        name_prefix,
        t_params_raw,
        outname=f"{stem}_exceedance_t_1pct.png",
        left_focus=TAIL_FOCUS_EXTREME,
        right_focus=1 - TAIL_FOCUS_EXTREME,
        standardized=False,
    )

    # optional: keep a 5% version too
    make_t_exceedance_figure(
        x_raw,
        name_prefix,
        t_params_raw,
        outname=f"{stem}_exceedance_t_5pct.png",
        left_focus=TAIL_FOCUS_STANDARD,
        right_focus=1 - TAIL_FOCUS_STANDARD,
        standardized=False,
    )
    
def save_residual_series(arima_resid, garch_resid, garch_std_resid):
    arima_resid_df = pd.DataFrame({
        "ds": arima_resid.index,
        "resid": arima_resid.values,
    })
    arima_resid_df.to_parquet(CACHE_DIR / "autoarima_residual_series.parquet", index=False)

    garch_resid_df = pd.DataFrame({
        "ds": garch_resid.index,
        "resid": garch_resid.values,
    })

    if garch_std_resid is not None:
        garch_std_df = pd.DataFrame({
            "ds": garch_std_resid.index,
            "std_resid": garch_std_resid.values,
        })
        garch_resid_df = garch_resid_df.merge(garch_std_df, on="ds", how="outer")

    garch_resid_df.to_parquet(
        CACHE_DIR / f"garch_{GARCH_P}_{GARCH_Q}_residual_series.parquet",
        index=False,
    )
# ============================================================
# Main
# ============================================================
def main():
    setup_plot_style()
    ensure_dirs()

    steps = [
        "Load price series",
        "Fit/load AutoARIMA",
        "Fit/load GARCH",
        "Fit reference distributions",
        "Save AutoARIMA figures",
        "Save GARCH figures",
        "Save comparison figure",
        "Save residual parquet files",
    ]

    with tqdm(total=len(steps), desc="Diagnostics pipeline") as pbar:
        price = timed_step("Load price series", load_price_series, DATA_PATH)
        pbar.update(1)

        autoarima_fitted, arima_resid = timed_step(
            "Fit/load AutoARIMA",
            fit_autoarima_cached,
            price,
            FORCE_REFIT,
            AUTOARIMA_SEASON_LENGTH,
        )
        pbar.update(1)

        garch_fitted, garch_resid, garch_std_resid = timed_step(
            "Fit/load GARCH",
            fit_garch_cached,
            price,
            FORCE_REFIT,
            GARCH_P,
            GARCH_Q,
        )
        pbar.update(1)

        # ----------------------------------------------------
        # Prepare standardized series for QQ plots
        # ----------------------------------------------------
        arima_std, garch_std, garch_label = timed_step(
            "Prepare standardized diagnostic series",
            prepare_standardized_diagnostics,
            arima_resid,
            garch_resid,
            garch_std_resid,
        )

        # Raw-scale fitted distributions for PIT / exceedance
        arima_norm_params_raw, arima_t_params_raw = timed_step(
            "Fit raw-scale distributions for AutoARIMA residuals",
            fit_reference_distributions,
            arima_resid,
        )

        # For GARCH:
        #   - use standardized residuals if available for all diagnostics
        #   - otherwise raw residuals fallback
        if garch_std_resid is not None and len(garch_std_resid) > 0:
            garch_raw = garch_std_resid
        else:
            garch_raw = garch_resid

        garch_norm_params_raw, garch_t_params_raw = timed_step(
            "Fit raw-scale distributions for GARCH diagnostics",
            fit_reference_distributions,
            garch_raw,
        )

        # Standardized-scale fitted Student-t for QQ only
        arima_t_params_std = timed_step(
            "Fit standardized Student-t for AutoARIMA QQ",
            lambda x: stats.t.fit(pd.Series(x).dropna().values, floc=0),
            arima_std,
        )

        garch_t_params_std = timed_step(
            "Fit standardized Student-t for GARCH QQ",
            lambda x: stats.t.fit(pd.Series(x).dropna().values, floc=0),
            garch_std,
        )
        pbar.update(1)

        timed_step(
            "Save AutoARIMA figures",
            save_model_figures,
            arima_resid,          # raw
            arima_std,            # standardized for QQ
            "AutoARIMA residuals",
            arima_norm_params_raw,
            arima_t_params_raw,
            arima_t_params_std,
            "autoarima",
        )
        pbar.update(1)

        timed_step(
            "Save GARCH figures",
            save_model_figures,
            garch_raw,            # std_resid if available, else raw resid
            garch_std,            # standardized for QQ
            garch_label,
            garch_norm_params_raw,
            garch_t_params_raw,
            garch_t_params_std,
            "garch",
        )
        pbar.update(1)

        timed_step(
            "Save publication comparison figure",
            make_publication_comparison_figure,
            arima_resid,
            arima_std,
            garch_raw,
            garch_std,
            arima_norm_params_raw,
            arima_t_params_raw,
            arima_t_params_std,
            garch_norm_params_raw,
            garch_t_params_raw,
            garch_t_params_std,
            garch_label,
            "comparison_autoarima_vs_garch.png",
        )
        pbar.update(1)

        timed_step(
            "Save residual parquet files",
            save_residual_series,
            arima_resid,
            garch_resid,
            garch_std_resid,
        )
        pbar.update(1)

    print("\nDone.")
    print(f"Figures saved in: {FIG_DIR.resolve()}")
    print(f"Residual cache saved in: {CACHE_DIR.resolve()}")

if __name__ == "__main__":
    main()