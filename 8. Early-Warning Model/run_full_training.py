# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------
from xgboost import XGBClassifier

import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

import matplotlib.pyplot as plt

import shap
from sklearn.preprocessing import StandardScaler, RobustScaler
from utils.build_dataset import build_dataset, add_dataset_args
import argparse

import json
import shutil
import joblib
from pathlib import Path
from datetime import datetime


# -------------------------------------------------------------------
# JSON helper
# -------------------------------------------------------------------
def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return str(obj)


# -------------------------------------------------------------------
# Time-to-depeg / lift helpers
# -------------------------------------------------------------------
def infer_depeg_event_now(df, args):
    """
    Infer whether current timestamp is in depeg.

    For your target definition:
        target = 1 if price deviation reaches > threshold within target_window.

    This helper reconstructs the current event:
        abs(depeg_bps) >= target_threshold

    unless depeg_side specifies positive/negative only.
    """

    candidate_cols = [
        "depeg_event",
        "is_depeg",
        "event",
        "depeg_now",
        "target_now",
    ]

    for col in candidate_cols:
        if col in df.columns:
            return df[col].astype(bool)

    if "depeg_bps" not in df.columns:
        raise ValueError(
            "Cannot infer current depeg event because `depeg_bps` is missing."
        )

    threshold = getattr(args, "target_threshold", None)
    if threshold is None:
        threshold = 15.0

    threshold = float(threshold)
    side = str(getattr(args, "depeg_side", "abs")).lower()

    depeg_bps = df["depeg_bps"]

    if side in ["up", "upper", "above", "positive", "pos"]:
        event_now = depeg_bps >= abs(threshold)
    elif side in ["down", "lower", "below", "negative", "neg"]:
        event_now = depeg_bps <= -abs(threshold)
    else:
        event_now = depeg_bps.abs() >= abs(threshold)

    return event_now.astype(bool)


def compute_earliest_future_depeg_hour(df, args, max_horizon):
    """
    For each row t, compute earliest h in [1, max_horizon] such that
    depeg occurs at t + h.

    Returns NaN if no depeg occurs within max_horizon.
    """

    event_now = infer_depeg_event_now(df, args)

    earliest_hour = pd.Series(np.nan, index=df.index, dtype=float)

    for h in range(1, max_horizon + 1):
        future_event_h = event_now.shift(-h).fillna(False)
        mask = earliest_hour.isna() & future_event_h
        earliest_hour.loc[mask] = h

    return earliest_hour


def default_time_to_depeg_bins(max_horizon):
    """
    Non-overlapping time-to-depeg bins.

    For max_horizon=24:
        1-5h
        6-10h
        11-15h
        16-20h
        21-24h
    """

    candidate_bins = [
        ("1-4h", 1, 4),
        ("5-9h", 5, 9),
        ("10-14h", 10, 14),
        ("15-19h", 15, 19),
        ("20-24h", 20, 24),
    ]

    bins = []

    for name, lo, hi in candidate_bins:
        if lo <= max_horizon:
            bins.append((name, lo, min(hi, max_horizon)))

    return bins


def compute_time_to_depeg_bin_metrics(
    df,
    y_test,
    proba_test,
    test_start_idx,
    threshold,
    args,
    max_horizon=None,
    bins=None,
):
    """
    Compute precision, recall, F1, base rate, alert-conditioned event rate,
    and lift by time-to-depeg bin.

    Definitions per bin:

        event_bin:
            earliest future depeg happens inside that bin.

        alert:
            proba_test >= threshold

        recall_bin:
            P(alert | event_bin)
            = alerts that caught events in bin / all events in bin

        precision_bin:
            P(event_bin | alert)
            = alerts that correspond to events in bin / all alerts

        base_event_rate:
            P(event_bin)

        alert_conditioned_event_rate:
            P(event_bin | alert)

        lift:
            P(event_bin | alert) / P(event_bin)

    Note:
        precision_bin values across bins sum to total precision for any
        depeg within max_horizon, assuming bins cover the full horizon.
    """

    if max_horizon is None:
        max_horizon = int(getattr(args, "target_window", 24) or 24)

    if bins is None:
        bins = default_time_to_depeg_bins(max_horizon)

    proba_test = np.asarray(proba_test)
    y_test_reset = y_test.reset_index(drop=True).astype(int)

    earliest_hour = compute_earliest_future_depeg_hour(
        df=df,
        args=args,
        max_horizon=max_horizon,
    )

    earliest_test = earliest_hour.iloc[
        test_start_idx:test_start_idx + len(proba_test)
    ].reset_index(drop=True)

    # Exclude last rows where full future horizon is not observable.
    absolute_test_indices = np.arange(test_start_idx, test_start_idx + len(proba_test))
    valid_full_horizon = absolute_test_indices + max_horizon < len(df)

    alert_mask = proba_test >= threshold

    valid_mask = valid_full_horizon
    n_valid = int(valid_mask.sum())
    n_alerts = int((alert_mask & valid_mask).sum())

    rows = []

    for bin_name, lo, hi in bins:
        event_bin_mask = (
            valid_mask
            & earliest_test.notna().values
            & (earliest_test.values >= lo)
            & (earliest_test.values <= hi)
        )

        n_events_bin = int(event_bin_mask.sum())
        n_alert_events_bin = int((alert_mask & event_bin_mask).sum())

        recall = (
            n_alert_events_bin / n_events_bin
            if n_events_bin > 0
            else np.nan
        )

        precision = (
            n_alert_events_bin / n_alerts
            if n_alerts > 0
            else np.nan
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if pd.notna(precision)
            and pd.notna(recall)
            and precision + recall > 0
            else np.nan
        )

        base_event_rate = (
            n_events_bin / n_valid
            if n_valid > 0
            else np.nan
        )

        alert_conditioned_event_rate = precision

        lift = (
            alert_conditioned_event_rate / base_event_rate
            if pd.notna(alert_conditioned_event_rate)
            and pd.notna(base_event_rate)
            and base_event_rate > 0
            else np.nan
        )

        rows.append({
            "time_to_depeg_bin": bin_name,
            "min_hours_ahead": int(lo),
            "max_hours_ahead": int(hi),
            "threshold": float(threshold),

            "n_valid": int(n_valid),
            "n_alerts": int(n_alerts),
            "alert_rate": float(n_alerts / n_valid) if n_valid > 0 else np.nan,

            "n_events_bin": int(n_events_bin),
            "n_alert_events_bin": int(n_alert_events_bin),

            "precision": float(precision) if pd.notna(precision) else np.nan,
            "recall": float(recall) if pd.notna(recall) else np.nan,
            "f1": float(f1) if pd.notna(f1) else np.nan,

            "base_event_rate": float(base_event_rate) if pd.notna(base_event_rate) else np.nan,
            "alert_conditioned_event_rate": (
                float(alert_conditioned_event_rate)
                if pd.notna(alert_conditioned_event_rate)
                else np.nan
            ),
            "lift": float(lift) if pd.notna(lift) else np.nan,
        })

    # Diagnostic: target positives where reconstructed future crossing is missing.
    missing_mask = (
        valid_mask
        & (y_test_reset.values == 1)
        & earliest_test.isna().values
    )

    rows.append({
        "time_to_depeg_bin": "positive_target_but_no_reconstructed_crossing",
        "min_hours_ahead": None,
        "max_hours_ahead": None,
        "threshold": float(threshold),

        "n_valid": int(n_valid),
        "n_alerts": int(n_alerts),
        "alert_rate": float(n_alerts / n_valid) if n_valid > 0 else np.nan,

        "n_events_bin": int(missing_mask.sum()),
        "n_alert_events_bin": int((alert_mask & missing_mask).sum()),

        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,

        "base_event_rate": np.nan,
        "alert_conditioned_event_rate": np.nan,
        "lift": np.nan,
    })

    return pd.DataFrame(rows), earliest_test


def compute_lift_heatmap_by_score_bucket(
    df,
    proba_test,
    test_start_idx,
    args,
    max_horizon=None,
    bins=None,
    top_fracs=(0.01, 0.05, 0.10, 0.20),
):
    """
    Compute lift by cumulative top-score buckets and time-to-depeg bins.

    Example buckets:
        top 1%, top 5%, top 10%, top 20%

    lift =
        P(event_bin | score in top bucket) / P(event_bin)
    """

    if max_horizon is None:
        max_horizon = int(getattr(args, "target_window", 24) or 24)

    if bins is None:
        bins = default_time_to_depeg_bins(max_horizon)

    proba_test = np.asarray(proba_test)
    n_test = len(proba_test)

    earliest_hour = compute_earliest_future_depeg_hour(
        df=df,
        args=args,
        max_horizon=max_horizon,
    )

    earliest_test = earliest_hour.iloc[
        test_start_idx:test_start_idx + n_test
    ].reset_index(drop=True)

    absolute_test_indices = np.arange(test_start_idx, test_start_idx + n_test)
    valid_full_horizon = absolute_test_indices + max_horizon < len(df)

    n_valid = int(valid_full_horizon.sum())

    rows = []

    for frac in top_fracs:
        k = max(1, int(np.ceil(frac * n_test)))

        top_idx = np.argsort(proba_test)[::-1][:k]
        bucket_mask = np.zeros(n_test, dtype=bool)
        bucket_mask[top_idx] = True
        bucket_mask = bucket_mask & valid_full_horizon

        n_bucket = int(bucket_mask.sum())
        bucket_name = f"top_{int(frac * 100)}pct"

        for bin_name, lo, hi in bins:
            event_bin_mask = (
                valid_full_horizon
                & earliest_test.notna().values
                & (earliest_test.values >= lo)
                & (earliest_test.values <= hi)
            )

            n_events_bin = int(event_bin_mask.sum())
            n_bucket_events_bin = int((bucket_mask & event_bin_mask).sum())

            base_event_rate = (
                n_events_bin / n_valid
                if n_valid > 0
                else np.nan
            )

            bucket_event_rate = (
                n_bucket_events_bin / n_bucket
                if n_bucket > 0
                else np.nan
            )

            lift = (
                bucket_event_rate / base_event_rate
                if pd.notna(bucket_event_rate)
                and pd.notna(base_event_rate)
                and base_event_rate > 0
                else np.nan
            )

            rows.append({
                "score_bucket": bucket_name,
                "top_fraction": float(frac),
                "time_to_depeg_bin": bin_name,
                "min_hours_ahead": int(lo),
                "max_hours_ahead": int(hi),

                "n_valid": int(n_valid),
                "n_bucket": int(n_bucket),
                "n_events_bin": int(n_events_bin),
                "n_bucket_events_bin": int(n_bucket_events_bin),

                "base_event_rate": float(base_event_rate) if pd.notna(base_event_rate) else np.nan,
                "bucket_event_rate": float(bucket_event_rate) if pd.notna(bucket_event_rate) else np.nan,
                "lift": float(lift) if pd.notna(lift) else np.nan,
            })

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Plot helpers
# -------------------------------------------------------------------
def plot_metric_by_time_to_depeg_bin(
    metric_df,
    metric_col,
    title,
    ylabel,
    logger,
    artifact_path,
    color="steelblue",
    ylim=None,
):
    plot_df = metric_df[
        metric_df["time_to_depeg_bin"]
        != "positive_target_but_no_reconstructed_crossing"
    ].copy()

    if len(plot_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        plot_df["time_to_depeg_bin"],
        plot_df[metric_col],
        color=color,
        alpha=0.85,
    )

    for i, row in plot_df.reset_index(drop=True).iterrows():
        value = row[metric_col]

        if pd.notna(value):
            label = f"{value:.3f}\nn={int(row['n_events_bin'])}"
            y = float(value) + 0.03 if ylim == (0, 1.08) else float(value)
            ax.text(
                i,
                y,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title(title)
    ax.set_xlabel("Time until first threshold crossing")
    ax.set_ylabel(ylabel)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    logger.save_figure(fig, artifact_path, dpi=200)
    plt.close(fig)

def log_random_forest_mdi_importance(model, feature_names, logger, top_n=30):
    """
    Log Random Forest feature importance using Mean Decrease in Impurity, MDI.

    In sklearn RandomForestClassifier, model.feature_importances_ is the
    normalized total reduction of the splitting criterion brought by each feature,
    averaged over all trees.

    For classification with the default criterion='gini', this is mean decrease
    in Gini impurity.
    """

    if not hasattr(model, "feature_importances_"):
        logger.save_text(
            "Model does not expose feature_importances_. Cannot compute MDI.",
            "plots/feature_importance/random_forest_mdi_error.txt",
        )
        return

    mdi = pd.Series(
        model.feature_importances_,
        index=feature_names,
        name="mdi_importance",
    ).sort_values(ascending=False)

    top_mdi = mdi.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, 0.3 * len(top_mdi))))

    top_mdi.iloc[::-1].plot(
        kind="barh",
        ax=ax,
        color="forestgreen",
        alpha=0.85,
    )

    ax.set_title("Random Forest Feature Importance, Mean Decrease in Impurity")
    ax.set_xlabel("Mean decrease in impurity, normalized")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()

    logger.save_figure(
        fig,
        "plots/feature_importance/random_forest_mdi_importance.png",
        dpi=200,
    )

    plt.close(fig)

    mdi_df = mdi.reset_index()
    mdi_df.columns = ["feature", "mdi_importance"]

    logger.save_dataframe(
        mdi_df,
        "plots/feature_importance/random_forest_mdi_importance.parquet",
    )

    logger.save_dataframe(
        mdi_df,
        "plots/feature_importance/random_forest_mdi_importance.csv",
    )

def plot_lift_by_time_to_depeg_bin(metric_df, logger):
    plot_df = metric_df[
        metric_df["time_to_depeg_bin"]
        != "positive_target_but_no_reconstructed_crossing"
    ].copy()

    if len(plot_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        plot_df["time_to_depeg_bin"],
        plot_df["lift"],
        color="darkorange",
        alpha=0.85,
    )

    max_lift = plot_df["lift"].dropna().max()
    if pd.isna(max_lift):
        max_lift = 1.0

    for i, row in plot_df.reset_index(drop=True).iterrows():
        if pd.notna(row["lift"]):
            label = (
                f"{row['lift']:.1f}x\n"
                f"{int(row['n_alert_events_bin'])}/{int(row['n_events_bin'])}"
            )
            ax.text(
                i,
                float(row["lift"]) + 0.05 * max(max_lift, 1.0),
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Lift by Time Until Depeg")
    ax.set_xlabel("Time until first threshold crossing")
    ax.set_ylabel("Lift vs random timestamp")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    logger.save_figure(fig, "plots/lift_by_time_to_depeg_bin.png", dpi=200)
    plt.close(fig)


def plot_event_rate_after_alert_vs_base(metric_df, logger):
    plot_df = metric_df[
        metric_df["time_to_depeg_bin"]
        != "positive_target_but_no_reconstructed_crossing"
    ].copy()

    if len(plot_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(plot_df))
    width = 0.38

    ax.bar(
        x - width / 2,
        100 * plot_df["base_event_rate"],
        width,
        label="Depeg probability",
        color="lightgray",
        edgecolor="black",
    )

    ax.bar(
        x + width / 2,
        100 * plot_df["alert_conditioned_event_rate"],
        width,
        label="Depeg probability conditional on alert",
        color="steelblue",
        edgecolor="black",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["time_to_depeg_bin"])

    ax.set_ylabel("Depeg probability (%)")
    ax.set_xlabel("Time until first threshold crossing")
    ax.set_title("Depeg Probability With and Without Model Alert")

    ax.grid(axis="y", alpha=0.3)

    # Legend below the plot, not inset
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.27, -0.15),
        ncols=2,
        frameon=False,
    )

    # Add bottom margin so legend is not clipped
    fig.subplots_adjust(bottom=0.25)

    logger.save_figure(
        fig,
        "plots/depeg_probability_conditional_on_alert_by_time_to_depeg.png",
        dpi=200,
    )

    plt.close(fig)


def plot_lift_heatmap(lift_heatmap_df, logger):
    if len(lift_heatmap_df) == 0:
        return

    # Order score buckets manually
    desired_row_order = ["top_1pct", "top_5pct", "top_10pct", "top_20pct"]

    # Order time bins by their numeric lower bound, not alphabetically
    bin_order_df = (
        lift_heatmap_df[["time_to_depeg_bin", "min_hours_ahead"]]
        .drop_duplicates()
        .sort_values("min_hours_ahead")
    )

    desired_col_order = bin_order_df["time_to_depeg_bin"].tolist()

    heatmap = lift_heatmap_df.pivot(
        index="score_bucket",
        columns="time_to_depeg_bin",
        values="lift",
    )

    heatmap = heatmap.reindex(
        index=[x for x in desired_row_order if x in heatmap.index],
        columns=[x for x in desired_col_order if x in heatmap.columns],
    )

    if heatmap.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    im = ax.imshow(
        heatmap.values,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(heatmap.columns)))
    ax.set_yticks(np.arange(len(heatmap.index)))
    ax.set_xticklabels(heatmap.columns)
    ax.set_yticklabels(heatmap.index)

    ax.set_xlabel("Time until first threshold crossing")
    ax.set_ylabel("Model score bucket")
    ax.set_title("Lift by Model Score Bucket and Time Until Depeg")

    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            val = heatmap.iloc[i, j]
            if pd.notna(val):
                ax.text(
                    j,
                    i,
                    f"{val:.1f}x",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=9,
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Lift vs random timestamp")

    fig.tight_layout()

    logger.save_figure(
        fig,
        "plots/lift_heatmap_by_score_bucket_and_time_to_depeg.png",
        dpi=200,
    )

    plt.close(fig)


# -------------------------------------------------------------------
# Local logger
# -------------------------------------------------------------------
class LocalLightningLogger:
    def __init__(self, base_dir="lightning_logs", experiment_name="default", run_name=None):
        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name or "default"

        exp_dir = self.base_dir / self.experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        self.run_dir = self._make_run_dir(exp_dir, run_name)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.artifact_dir = self.run_dir / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.hparams_path = self.run_dir / "hparams.json"
        self.metrics_path = self.run_dir / "metrics.json"
        self.metrics_history_path = self.run_dir / "metrics.jsonl"

        self._metrics_latest = {}

        self._write_json(
            self.run_dir / "run_info.json",
            {
                "experiment_name": self.experiment_name,
                "run_name": self.run_dir.name,
                "created_at": datetime.utcnow().isoformat() + "Z",
            },
        )

    def _make_run_dir(self, exp_dir: Path, run_name: str | None):
        if run_name:
            candidate = exp_dir / run_name
            if not candidate.exists():
                return candidate

            i = 1
            while (exp_dir / f"{run_name}_{i}").exists():
                i += 1
            return exp_dir / f"{run_name}_{i}"

        versions = []
        for p in exp_dir.glob("version_*"):
            try:
                versions.append(int(p.name.split("_")[-1]))
            except Exception:
                pass

        next_version = 0 if not versions else max(versions) + 1
        return exp_dir / f"version_{next_version}"

    def _write_json(self, path: Path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=_json_default)

    def log_params(self, params: dict):
        existing = {}
        if self.hparams_path.exists():
            with open(self.hparams_path, "r") as f:
                existing = json.load(f)

        existing.update({k: v for k, v in params.items() if v is not None})
        self._write_json(self.hparams_path, existing)

    def log_metric(self, key: str, value, step=None):
        value = float(value) if value is not None and pd.notna(value) else None
        self._metrics_latest[key] = value
        self._write_json(self.metrics_path, self._metrics_latest)

        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "key": key,
            "value": value,
            "step": step,
        }

        with open(self.metrics_history_path, "a") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")

    def log_metrics(self, metrics: dict, step=None):
        for k, v in metrics.items():
            self.log_metric(k, v, step=step)

    def save_figure(self, fig, artifact_path, dpi=150):
        dst = self.artifact_dir / artifact_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dst, dpi=dpi, bbox_inches="tight", transparent=True)

    def save_current_fig(self, filename, artifact_subdir="plots", dpi=200):
        dst = self.artifact_dir / artifact_subdir / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(dst, dpi=dpi, bbox_inches="tight", transparent=True)
        plt.close()

    def log_artifact(self, src_path, artifact_subdir=""):
        src = Path(src_path)
        dst = self.artifact_dir / artifact_subdir / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    def save_text(self, text: str, artifact_path: str):
        dst = self.artifact_dir / artifact_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            f.write(text)

    def save_json(self, payload, artifact_path: str):
        dst = self.artifact_dir / artifact_path
        self._write_json(dst, payload)

    def save_dataframe(self, df: pd.DataFrame, artifact_path: str):
        dst = self.artifact_dir / artifact_path
        dst.parent.mkdir(parents=True, exist_ok=True)

        if str(dst).endswith(".parquet"):
            df.to_parquet(dst, index=False)
        elif str(dst).endswith(".csv"):
            df.to_csv(dst, index=False)
        else:
            raise ValueError("artifact_path must end with .parquet or .csv")

    def save_pickle(self, obj, artifact_path: str):
        dst = self.artifact_dir / artifact_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, dst)


def log_fig(logger, fig, artifact_path):
    logger.save_figure(fig, artifact_path)


def _log_current_fig(logger, filename, artifact_subdir="plots", dpi=200):
    logger.save_current_fig(filename, artifact_subdir=artifact_subdir, dpi=dpi)


# -------------------------------------------------------------------
# Model helpers
# -------------------------------------------------------------------
def build_model(args, pos_weight):
    if args.model_name == "xgboost":
        return XGBClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_weight=1.0,
            objective="binary:logistic",
            eval_metric="auc",
            early_stopping_rounds=args.early_stopping_rounds,
            scale_pos_weight=pos_weight,
            random_state=1233,
            n_jobs=args.n_jobs,
        )

    if args.model_name == "lightgbm":
        return LGBMClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            num_leaves=args.num_leaves,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_weight=1.0,
            objective="binary",
            scale_pos_weight=pos_weight,
            random_state=1233,
            n_jobs=args.n_jobs,
            verbosity=-1,
        )

    if args.model_name == "catboost":
        return CatBoostClassifier(
            iterations=args.n_estimators,
            learning_rate=args.learning_rate,
            depth=args.max_depth,
            loss_function="Logloss",
            eval_metric="AUC",
            class_weights=[1.0, float(pos_weight)],
            random_seed=1233,
            verbose=False,
        )

    if args.model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            class_weight={0: 1.0, 1: float(pos_weight)},
            random_state=1233,
            n_jobs=args.n_jobs,
            max_features=args.rf_max_features,
        )

    raise ValueError(f"Unsupported model_name: {args.model_name}")


def fit_model(model, args, X_train, y_train, X_val, y_val):
    if args.model_name == "xgboost":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )

    elif args.model_name == "lightgbm":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)],
        )

    elif args.model_name == "catboost":
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=False,
        )

    elif args.model_name == "random_forest":
        model.fit(X_train, y_train)

    else:
        raise ValueError(f"Unsupported model_name: {args.model_name}")


def get_best_iteration(model, model_name):
    if model_name == "xgboost":
        bi = getattr(model, "best_iteration", None)
    elif model_name == "lightgbm":
        bi = getattr(model, "best_iteration_", None)
    elif model_name == "catboost":
        bi = model.get_best_iteration()
    else:
        bi = None

    return -1 if bi is None else int(bi)


def save_model_to_local(model, model_name, logger, X_train):
    logger.save_pickle(model, "models/model.joblib")

    models_dir = logger.artifact_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        if model_name == "xgboost":
            model.save_model(str(models_dir / "model.json"))
        elif model_name == "lightgbm":
            booster = getattr(model, "booster_", None)
            if booster is not None:
                booster.save_model(str(models_dir / "model.txt"))
        elif model_name == "catboost":
            model.save_model(str(models_dir / "model.cbm"))
    except Exception as e:
        logger.save_text(
            f"Native model save failed: {e}",
            "models/native_save_error.txt",
        )

    signature = {
        "input_columns": list(X_train.columns),
        "input_dtypes": {c: str(X_train[c].dtype) for c in X_train.columns},
        "output": "predict_proba[:, 1] float",
    }

    logger.save_json(signature, "models/signature.json")
    logger.save_dataframe(X_train.head(5), "models/input_example.parquet")


def get_native_feature_importance(model, model_name, feature_names):
    if model_name == "catboost":
        importances = model.get_feature_importance()
    else:
        importances = getattr(model, "feature_importances_", None)

    if importances is None:
        return None

    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


def log_native_feature_importance(model, model_name, feature_names, logger):
    imp = get_native_feature_importance(model, model_name, feature_names)
    if imp is None:
        return

    top_imp = imp.head(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_imp.iloc[::-1].plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"{model_name} native feature importance")
    ax.set_xlabel("importance")
    fig.tight_layout()

    logger.save_figure(
        fig,
        "plots/feature_importance/feature_importance_native.png",
        dpi=200,
    )
    plt.close(fig)

    imp_df = top_imp.reset_index()
    imp_df.columns = ["feature", "importance"]

    logger.save_dataframe(
        imp_df,
        "plots/feature_importance/feature_importance_native_top20.parquet",
    )


# -------------------------------------------------------------------
# SHAP helpers
# -------------------------------------------------------------------
def normalize_shap_output(shap_output, X, expected_value=None, positive_class_idx=1):
    feature_names = list(X.columns)
    data = X.values

    if isinstance(shap_output, shap.Explanation):
        values = shap_output.values
        base_values = shap_output.base_values

        if values.ndim == 2:
            return shap_output

        if values.ndim == 3:
            values = values[:, :, positive_class_idx]

            if isinstance(base_values, np.ndarray):
                if base_values.ndim == 2:
                    base_values = base_values[:, positive_class_idx]
                elif (
                    base_values.ndim == 1
                    and len(base_values) > positive_class_idx
                    and len(base_values) != len(values)
                ):
                    base_values = np.repeat(
                        base_values[positive_class_idx],
                        values.shape[0],
                    )

            return shap.Explanation(
                values=values,
                base_values=base_values,
                data=data,
                feature_names=feature_names,
            )

    if isinstance(shap_output, list):
        values = shap_output[positive_class_idx] if len(shap_output) > 1 else shap_output[0]

        if isinstance(expected_value, (list, np.ndarray)):
            base_value = (
                expected_value[positive_class_idx]
                if len(expected_value) > positive_class_idx
                else expected_value[0]
            )
        else:
            base_value = expected_value

        if np.isscalar(base_value):
            base_value = np.repeat(base_value, len(X))

        return shap.Explanation(
            values=values,
            base_values=base_value,
            data=data,
            feature_names=feature_names,
        )

    arr = np.asarray(shap_output)

    if arr.ndim == 3:
        arr = arr[:, :, positive_class_idx]

    return shap.Explanation(
        values=arr,
        data=data,
        feature_names=feature_names,
    )


def compute_shap_explanation(model, X_explain, shap_type):
    explainer = shap.TreeExplainer(model, feature_perturbation=shap_type)
    raw_shap = explainer(X_explain)

    shap_values = normalize_shap_output(
        raw_shap,
        X_explain,
        expected_value=getattr(explainer, "expected_value", None),
        positive_class_idx=1,
    )

    return explainer, shap_values


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_dataset_args(parser)

    training_args = parser.add_argument_group("Training arguments")
    training_args.add_argument("--remote_logging", action="store_true", help="deprecated / ignored")
    training_args.add_argument("--experiment_name", type=str, default="default")
    training_args.add_argument("--run_name", type=str, default=None)
    training_args.add_argument("--log_dir", type=str, default="lightning_logs")
    training_args.add_argument("--test_size", type=float, default=0.30)
    training_args.add_argument("--val_size", type=float, default=0.15)
    training_args.add_argument("--scaler", type=str, default="standard", choices=["standard", "robust", "none"])
    training_args.add_argument(
        "--model_name",
        type=str,
        default="xgboost",
        choices=["xgboost", "lightgbm", "catboost", "random_forest"],
    )

    model_args = parser.add_argument_group("Model arguments")
    model_args.add_argument("--learning_rate", type=float, default=0.01)
    model_args.add_argument("--early_stopping_rounds", type=int, default=200)
    model_args.add_argument("--eval_metric", type=str, default="auc")
    model_args.add_argument("--n_estimators", type=int, default=800)
    model_args.add_argument("--shap_type", type=str, default="interventional")
    model_args.add_argument("--max_depth", type=int, default=6)
    model_args.add_argument("--num_leaves", type=int, default=31)
    model_args.add_argument("--rf_max_features", type=str, default="sqrt")
    model_args.add_argument("--n_jobs", type=int, default=-1)

    args = parser.parse_args()

    dict_args = vars(args).copy()
    dict_args["target"] = True
    dataset_path = build_dataset(**dict_args)

    logger = LocalLightningLogger(
        base_dir=args.log_dir,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )

    print(f"Local logs will be saved to: {logger.run_dir}")

    # ---------------------------
    # Load and prepare data
    # ---------------------------
    dataset = pd.read_parquet(dataset_path)
    dataset["timestamp"] = dataset.index

    for k in range(1, 8):
        dataset[f"depeg_bps_lag{k}h"] = dataset["depeg_bps"].shift(k)

    dataset = dataset.dropna()

    TIME_COL = "timestamp"
    TARGET_COL = "target"

    df = dataset.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    FEATURES = [c for c in df.columns if c not in [TIME_COL, TARGET_COL]]

    X = df[FEATURES]
    y = df[TARGET_COL].astype(int)

    n = len(df)
    test_size = int(args.test_size * n)
    val_size = int(args.val_size * n)

    train_end = n - test_size
    val_end = train_end
    train_end2 = train_end - val_size

    X_train, y_train = X.iloc[:train_end2], y.iloc[:train_end2]
    X_val, y_val = X.iloc[train_end2:val_end], y.iloc[train_end2:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    # ---------------------------
    # Scaling
    # ---------------------------
    num_cols = X_train.columns
    scaler = None

    if args.scaler == "standard":
        scaler = StandardScaler()

        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_val_scaled[num_cols] = scaler.transform(X_val[num_cols])
        X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

        X_train, X_val, X_test = X_train_scaled, X_val_scaled, X_test_scaled

    elif args.scaler == "robust":
        scaler = RobustScaler()

        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_val_scaled[num_cols] = scaler.transform(X_val[num_cols])
        X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

        X_train, X_val, X_test = X_train_scaled, X_val_scaled, X_test_scaled

    # ---------------------------
    # Class weight
    # ---------------------------
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    w_pos = float(n_neg / max(n_pos, 1))

    print(f"Train positives={n_pos}, negatives={n_neg}, w_pos={w_pos:.3f}")

    model = build_model(args, pos_weight=w_pos)

    # ---------------------------
    # Log params
    # ---------------------------
    params_to_log = {
        "model_name": args.model_name,
        "dataset_path": dataset_path,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "scaler": args.scaler,
        "learning_rate": args.learning_rate,
        "early_stopping_rounds": args.early_stopping_rounds,
        "eval_metric": args.eval_metric,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "num_leaves": args.num_leaves,
        "rf_max_features": args.rf_max_features,
        "target_window": getattr(args, "target_window", None),
        "target_threshold": getattr(args, "target_threshold", None),
        "depeg_side": getattr(args, "depeg_side", None),
        "dynamic_threshold": int(getattr(args, "dynamic_threshold", False)),
        "alpha": getattr(args, "alpha", None),
        "scale_pos_weight_used": float(w_pos),
        "n_features": len(FEATURES),
    }

    logger.log_params(params_to_log)

    # ---------------------------
    # Train
    # ---------------------------
    fit_model(model, args, X_train, y_train, X_val, y_val)
    logger.log_metric("best_iteration", get_best_iteration(model, args.model_name))

    # ---------------------------
    # Evaluate
    # ---------------------------
    proba_test = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba_test)
    auprc = average_precision_score(y_test, proba_test)

    logger.log_metric("test_roc_auc", float(auc))
    logger.log_metric("test_auprc", float(auprc))

    # Threshold by Youden J
    fpr, tpr, thresholds = roc_curve(y_test, proba_test)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    thresh = float(thresholds[best_idx])

    logger.log_metric("best_threshold_youdenJ", thresh)
    logger.log_metric("tpr_at_best_threshold", float(tpr[best_idx]))
    logger.log_metric("fpr_at_best_threshold", float(fpr[best_idx]))

    yhat = (proba_test >= thresh).astype(int)

    # ---------------------------
    # ROC + PR plot
    # ---------------------------
    prec, rec, _ = precision_recall_curve(y_test, proba_test)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax[0].plot([0, 1], [0, 1], "--", color="gray")
    ax[0].set_title("ROC Curve")
    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].legend()

    ax[1].plot(rec, prec, label=f"AUPRC={auprc:.3f}")
    ax[1].set_title("Precision-Recall Curve")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].legend()

    log_fig(logger, fig, "plots/roc_pr.png")
    plt.close(fig)

    # ---------------------------
    # Confusion matrix
    # ---------------------------
    cm = confusion_matrix(y_test, yhat)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])
    ax.set_xlim(-0.5, cm.shape[1] - 0.5)
    ax.set_ylim(cm.shape[0] - 0.5, -0.5)

    for (i, j_), v in np.ndenumerate(cm):
        ax.text(j_, i, str(v), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    log_fig(logger, fig, "plots/confusion_matrix.png")
    plt.close(fig)

    clf_report = classification_report(y_test, yhat, output_dict=True)
    logger.save_json(clf_report, "reports/classification_report.json")

    # ---------------------------
    # Time-to-depeg bin metrics
    # ---------------------------
    max_horizon = int(getattr(args, "target_window", 24) or 24)

    ttd_metrics, earliest_test_depeg_hour = compute_time_to_depeg_bin_metrics(
        df=df,
        y_test=y_test,
        proba_test=proba_test,
        test_start_idx=val_end,
        threshold=thresh,
        args=args,
        max_horizon=max_horizon,
    )

    logger.save_dataframe(
        ttd_metrics,
        "reports/time_to_depeg_bin_metrics.parquet",
    )

    logger.save_dataframe(
        ttd_metrics,
        "reports/time_to_depeg_bin_metrics.csv",
    )

    # Log bin metrics
    for _, row in ttd_metrics.iterrows():
        bin_name = str(row["time_to_depeg_bin"])
        safe_bin = (
            bin_name
            .replace(" ", "_")
            .replace("-", "_")
            .replace(">", "gt")
            .replace("<", "lt")
        )

        for metric in ["precision", "recall", "f1", "lift", "base_event_rate", "alert_conditioned_event_rate"]:
            if metric in row and pd.notna(row[metric]):
                logger.log_metric(f"{metric}_time_to_depeg_{safe_bin}", row[metric])

        if pd.notna(row["n_events_bin"]):
            logger.log_metric(f"support_time_to_depeg_{safe_bin}", row["n_events_bin"])

    # Separate precision plot by time-to-depeg bin
    plot_metric_by_time_to_depeg_bin(
        metric_df=ttd_metrics,
        metric_col="precision",
        title="Precision by Time Until Depeg",
        ylabel="Precision contribution: P(bin | alert)",
        logger=logger,
        artifact_path="plots/precision_by_time_to_depeg_bin.png",
        color="seagreen",
        ylim=(0, 1.08),
    )

    # Separate recall plot by time-to-depeg bin
    plot_metric_by_time_to_depeg_bin(
        metric_df=ttd_metrics,
        metric_col="recall",
        title="Recall by Time Until Depeg",
        ylabel="Recall: P(alert | bin)",
        logger=logger,
        artifact_path="plots/recall_by_time_to_depeg_bin.png",
        color="steelblue",
        ylim=(0, 1.08),
    )

    # Lift plots
    plot_lift_by_time_to_depeg_bin(ttd_metrics, logger)
    plot_event_rate_after_alert_vs_base(ttd_metrics, logger)

    # Lift heatmap by score bucket
    lift_heatmap_df = compute_lift_heatmap_by_score_bucket(
        df=df,
        proba_test=proba_test,
        test_start_idx=val_end,
        args=args,
        max_horizon=max_horizon,
        top_fracs=(0.01, 0.05, 0.10, 0.20),
    )

    logger.save_dataframe(
        lift_heatmap_df,
        "reports/lift_heatmap_by_score_bucket_and_time_to_depeg.parquet",
    )

    logger.save_dataframe(
        lift_heatmap_df,
        "reports/lift_heatmap_by_score_bucket_and_time_to_depeg.csv",
    )

    plot_lift_heatmap(lift_heatmap_df, logger)

    # ---------------------------
    # Save scaler + model
    # ---------------------------
    if scaler is not None:
        logger.save_pickle(scaler, "models/preprocess_scaler.joblib")

    save_model_to_local(
        model=model,
        model_name=args.model_name,
        logger=logger,
        X_train=X_train,
    )

    logger.save_text("\n".join(FEATURES), "meta/features.txt")

    # ---------------------------
    # SHAP
    # ---------------------------
    X_explain = X_test.copy()

    explainer, shap_values = compute_shap_explanation(
        model=model,
        X_explain=X_explain,
        shap_type=args.shap_type,
    )

    # Global beeswarm out-of-sample
    std_per_feature = shap_values.values.std(axis=0)
    order = np.argsort(std_per_feature)[::-1]

    shap.plots.beeswarm(
        shap_values,
        max_display=10,
        show=False,
    )

    _log_current_fig(
        logger,
        "shap_beeswarm_global.png",
        artifact_subdir="plots/shap",
    )

    # Global beeswarm in-sample
    raw_train_shap = explainer(X_train)

    shap_values_train = normalize_shap_output(
        raw_train_shap,
        X_train,
        expected_value=getattr(explainer, "expected_value", None),
        positive_class_idx=1,
    )

    std_per_feature_train = shap_values_train.values.std(axis=0)
    order_train = np.argsort(std_per_feature_train)[::-1]

    shap.plots.beeswarm(
        shap_values_train,
        order=order_train,
        max_display=10,
        show=False,
    )

    _log_current_fig(
        logger,
        "shap_beeswarm_global_insample.png",
        artifact_subdir="plots/shap",
    )

    # Beeswarm above threshold
    warn_mask = proba_test >= thresh
    sv_warn = shap_values[warn_mask]

    if sv_warn.values.shape[0] > 0:
        shap.plots.beeswarm(
            sv_warn,
            max_display=10,
            show=False,
        )

        _log_current_fig(
            logger,
            "shap_beeswarm_above_threshold.png",
            artifact_subdir="plots/shap",
        )

        pos_mean = np.clip(sv_warn.values, 0, None).mean(axis=0)
        order_pos = np.argsort(pos_mean)[::-1]

        shap.plots.beeswarm(
            sv_warn,
            order=order_pos,
            max_display=10,
            show=False,
        )

        _log_current_fig(
            logger,
            "shap_beeswarm_above_threshold_ordered_by_pos_mean.png",
            artifact_subdir="plots/shap",
        )

    # Native feature importance
    log_native_feature_importance(
        model,
        args.model_name,
        FEATURES,
        logger,
    )
    if args.model_name == "random_forest":
        log_random_forest_mdi_importance(
            model=model,
            feature_names=FEATURES,
            logger=logger,
            top_n=30,
        )

    # ---------------------------
    # Predictions over time
    # ---------------------------
    test_dates = df["timestamp"].iloc[val_end:].reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, 6))
    axi = ax.twinx()

    for i in range(len(y_test_reset)):
        if int(y_test_reset.iloc[i]) == 1:
            ax.axvspan(
                test_dates.iloc[i],
                test_dates.iloc[min(i + 1, len(test_dates) - 1)],
                color="lightgrey",
                alpha=0.2,
                zorder=0,
            )

    ax.plot(
        test_dates,
        proba_test,
        label="Predicted probability",
        color="royalblue",
        lw=2,
        zorder=2,
    )

    # Use raw, unscaled depeg_bps from df, not X_test.
    axi.plot(
        test_dates,
        df["depeg_bps"].iloc[val_end:].reset_index(drop=True),
        label="Depeg BPS",
        color="crimson",
        lw=1.5,
        alpha=0.9,
        zorder=1,
    )

    ax.axhline(
        thresh,
        color="royalblue",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"Threshold={thresh:.3f}",
    )

    ax.set_ylabel(
        f"Probability of depeg in next {max_horizon} hours",
        color="royalblue",
    )

    axi.set_ylabel("Depeg BPS", color="crimson")

    ax.tick_params(axis="y", labelcolor="royalblue")
    axi.tick_params(axis="y", labelcolor="crimson")

    ax.legend(loc="upper left")
    fig.tight_layout()

    logger.save_figure(
        fig,
        "plots/timeseries/predictions_over_time.png",
        dpi=300,
    )

    plt.close(fig)

    # ---------------------------
    # SHAP scatter per top features
    # ---------------------------
    shap_spread = np.abs(shap_values.values).std(axis=0)
    top10_idx = np.argsort(shap_spread)[::-1][:10]

    feature_names = (
        list(shap_values.feature_names)
        if shap_values.feature_names is not None
        else list(X_test.columns)
    )

    for idx in top10_idx:
        fname = feature_names[idx]
        safe_fname = (
            fname.replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace(":", "_")
        )

        shap.plots.scatter(
            shap_values[:, idx],
            color=shap_values,
            show=False,
        )

        plt.title(f"SHAP scatter: {fname}")

        _log_current_fig(
            logger,
            f"shap_scatter_{safe_fname}.png",
            artifact_subdir="plots/shap",
        )

        shap.plots.scatter(
            shap_values[:, idx],
            color=proba_test,
            show=False,
        )

        plt.title(f"SHAP scatter colored by proba: {fname}")

        _log_current_fig(
            logger,
            f"shap_scatter_{safe_fname}_colored_by_proba.png",
            artifact_subdir="plots/shap",
        )

    # ---------------------------
    # Save predictions
    # ---------------------------
    pred_df = pd.DataFrame({
        "timestamp": df["timestamp"].iloc[val_end:].reset_index(drop=True),
        "y_true": y_test.reset_index(drop=True).astype(int),
        "proba_depeg": proba_test.astype(float),
        "y_pred_at_best_threshold": yhat.astype(int),
        "earliest_depeg_hour_ahead": earliest_test_depeg_hour,
    })

    logger.save_dataframe(
        pred_df,
        "predictions/test_pred_proba.parquet",
    )

    print(f"Done. All logs/artifacts saved to: {logger.run_dir}")
