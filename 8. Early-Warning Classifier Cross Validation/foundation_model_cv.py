"""Run TabPFN-3.5 and Causilo through the existing depeg CV evaluator.

The script consumes the same preprocessed alpha-specific Parquet files as the
tree-model experiment. It reuses its folds, validation-only operating-threshold
selection, event utility, false-alert budgets, and event/calendar-block CIs.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import re
from datetime import datetime, timezone
from pathlib import Path
import sys
from threading import Event, Lock, Thread
from time import monotonic

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SHAP_DIR = REPO_ROOT / "9. SHAP explanations of Early Warning Model"
# The repository move placed the dataset builder in folder 9. The shared CV
# module imports it at module load time, though this runner only reads Parquet.
if not (SCRIPT_DIR / "utils").is_dir() and (SHAP_DIR / "utils").is_dir():
    sys.path.insert(0, str(SHAP_DIR))

from cv_model_comparison import LocalLightningLogger, run_expanding_window_cv  # noqa: E402


class ProgressReporter:
    """Timestamped stage updates plus heartbeats during long blocking calls."""

    def __init__(self, label: str, log_path: Path, interval_seconds: float):
        self.label = label
        self.log_path = log_path
        self.interval_seconds = interval_seconds
        self.started = monotonic()
        self.stage_started = self.started
        self.stage = "starting"
        self.stop_event = Event()
        self.lock = Lock()
        self.thread = Thread(target=self._heartbeat, daemon=True)

    def _write_locked(self, message: str) -> None:
        elapsed = monotonic() - self.started
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = f"[{timestamp}] [{self.label}] +{elapsed:,.0f}s {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def update(self, stage: str) -> None:
        with self.lock:
            self.stage = stage
            self.stage_started = monotonic()
            self._write_locked(stage)

    def _heartbeat(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            with self.lock:
                stage_elapsed = monotonic() - self.stage_started
                self._write_locked(f"still running: {self.stage} ({stage_elapsed:,.0f}s in stage)")

    def __enter__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.update("run started")
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.update("run complete" if exc_type is None else f"run failed: {exc_type.__name__}: {exc}")
        return False


class BatchedClassifier:
    """Limit prediction-batch memory without changing fit rows or predictions."""

    def __init__(self, estimator, batch_size: int, progress_every_batches: int = 1):
        self.estimator = estimator
        self.batch_size = batch_size
        self.progress_every_batches = progress_every_batches
        self.progress_callback = None
        self.fold = None
        self.n_folds = None
        self.prediction_stage = "prediction"

    def set_progress_context(self, callback, fold: int, n_folds: int) -> None:
        self.progress_callback = callback
        self.fold = fold
        self.n_folds = n_folds

    def set_prediction_stage(self, stage: str) -> None:
        self.prediction_stage = stage

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.estimator.fit(X, y)
        classes = np.asarray(getattr(self.estimator, "classes_", np.unique(y)))
        if not np.array_equal(classes, [0, 1]):
            raise ValueError(f"Expected binary class order [0, 1], got {classes.tolist()}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        chunks = []
        started = monotonic()
        n_batches = (len(X) + self.batch_size - 1) // self.batch_size
        for batch_number, start in enumerate(range(0, len(X), self.batch_size), start=1):
            batch = np.asarray(
                self.estimator.predict_proba(X.iloc[start:start + self.batch_size]),
                dtype=float,
            )
            if batch.shape != (min(self.batch_size, len(X) - start), 2):
                raise ValueError(f"Unexpected probability shape: {batch.shape}")
            if not np.isfinite(batch).all():
                raise ValueError("Foundation model returned non-finite probabilities")
            chunks.append(batch)
            if self.progress_callback is not None and (
                batch_number % self.progress_every_batches == 0 or batch_number == n_batches
            ):
                processed = min(start + self.batch_size, len(X))
                elapsed = monotonic() - started
                eta = elapsed * (len(X) - processed) / max(processed, 1)
                self.progress_callback(
                    f"fold {self.fold}/{self.n_folds}: {self.prediction_stage} prediction "
                    f"batch {batch_number}/{n_batches}, {processed:,}/{len(X):,} rows "
                    f"({elapsed:,.0f}s elapsed, ~{eta:,.0f}s remaining)"
                )
        return np.concatenate(chunks, axis=0)


def build_foundation_model(model_name, args, pos_weight, use_early_stopping):
    # No class-weight or early-stopping tuning: both are frozen pretrained
    # estimators. Validation remains entirely held out for threshold selection.
    del pos_weight, use_early_stopping
    if model_name == "tabpfn_3_5":
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        estimator = TabPFNClassifier.create_default_for_version(
            ModelVersion.V3_5, device=args.tabpfn_device,
            random_state=args.random_state,
        )
    elif model_name == "causilo":
        from causilo import CausiloClassifier

        estimator = CausiloClassifier(
            n_estimators=args.causilo_n_estimators,
            random_state=args.random_state,
            device=args.causilo_device,
        )
    else:
        raise ValueError(f"Unsupported foundation model: {model_name}")
    return BatchedClassifier(estimator, args.predict_batch_size, args.progress_every_batches)


def fit_foundation_model(*, model, model_name, args, X_train, y_train, X_val, y_val):
    del model_name, args, X_val, y_val
    model.fit(X_train, y_train)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def require_cuda_device(device_name: str) -> dict[str, str | int]:
    """Fail before CV if the requested CUDA device is absent or unusable."""
    if not re.fullmatch(r"cuda(?::\d+)?", device_name):
        raise ValueError(
            f"GPU-only benchmark requires a CUDA device such as cuda:0, got {device_name!r}"
        )
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the GPU-only benchmark") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to PyTorch; refusing to run on CPU")
    index = int(device_name.split(":", 1)[1]) if ":" in device_name else 0
    if index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested {device_name}, but PyTorch sees only {torch.cuda.device_count()} CUDA device(s)"
        )
    # An allocation and synchronization catch driver/runtime errors before any
    # expensive model fit. Explicit estimator device settings then prevent a
    # model's default 'auto' selection from silently falling back to CPU.
    try:
        probe = torch.ones(1, device=f"cuda:{index}")
        torch.cuda.synchronize(index)
        del probe
    except Exception as exc:
        raise RuntimeError(f"Cannot allocate on {device_name}: {exc}") from exc
    return {"device": f"cuda:{index}", "gpu_name": torch.cuda.get_device_name(index), "visible_index": index}


def load_cv_frame(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Preprocessed dataset not found: {path}")
    df = pd.read_parquet(path)
    df["timestamp"] = df.index
    for lag in range(1, 8):
        df[f"depeg_bps_lag{lag}h"] = df["depeg_bps"].shift(lag)
    df = df.dropna().copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    if not df["timestamp"].is_unique:
        raise ValueError(f"Duplicate timestamps in {path}")
    if not set(df["target"].astype(int).unique()).issubset({0, 1}):
        raise ValueError(f"Target is not binary in {path}")
    feature_cols = [col for col in df if col not in ("timestamp", "target")]
    non_numeric = df[feature_cols].select_dtypes(exclude=[np.number, "bool"]).columns
    if len(non_numeric):
        raise ValueError(f"Non-numeric features in {path}: {non_numeric.tolist()}")
    return df, feature_cols


def select_within_alpha(summary_df: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    best = summary_df["cv_event_utility_score_mean"].max()
    if pd.isna(best):
        raise RuntimeError("No boundary-complete events were scored in the CV folds")
    result = summary_df.copy()
    result["within_utility_tolerance"] = result["cv_event_utility_score_mean"] >= best - tolerance
    candidates = result.loc[result["within_utility_tolerance"]].copy()
    candidates = candidates.sort_values(
        ["cv_timely_event_recall_mean", "cv_false_alerts_per_month_mean",
         "cv_event_utility_score_std", "cv_event_utility_score_mean"],
        ascending=[False, True, True, False],
        na_position="last",
    )
    result["selected_model"] = result["model_name"].eq(candidates.iloc[0]["model_name"])
    result["selection_rank"] = np.nan
    result.loc[candidates.index, "selection_rank"] = np.arange(1, len(candidates) + 1)
    result["selection_policy"] = (
        "Mean outer-fold event utility at fixed validation false-alert budget; "
        "within utility tolerance: higher recall, lower false-alert burden, "
        "lower utility variation"
    )
    return result.sort_values("cv_event_utility_score_mean", ascending=False)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, default=SHAP_DIR / "preprocessed_datasets")
    parser.add_argument("--log_dir", type=Path, default=SCRIPT_DIR / "lightning_logs")
    parser.add_argument("--experiment_name", default=f"cv_foundation_models_{datetime.now(timezone.utc):%Y-%m-%d}_15bp")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.3, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--model_names", nargs="+", choices=["tabpfn_3_5", "causilo"],
                        default=["tabpfn_3_5", "causilo"])
    parser.add_argument("--target_window", type=int, default=24)
    parser.add_argument("--target_threshold", type=int, default=15)
    parser.add_argument("--depeg_side", choices=["both", "up", "down"], default="both")
    parser.add_argument("--dynamic_threshold", action="store_true")
    parser.add_argument("--scaler", choices=["none", "standard", "robust"], default="robust")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--cv_test_frac", type=float, default=0.30)
    parser.add_argument("--cv_min_train_frac", type=float, default=0.68)
    parser.add_argument("--cv_val_frac", type=float, default=0.30)
    parser.add_argument("--cv_embargo_hours", type=int, default=48)
    parser.add_argument("--false_alert_budget_per_month", type=float, default=2.0)
    parser.add_argument("--false_alert_budgets", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--false_alert_cost", type=float, default=0.05)
    parser.add_argument("--min_lead_hours", type=float, default=1.0)
    parser.add_argument("--max_lead_hours", type=float, default=None)
    parser.add_argument("--utility_target_lead_hours", type=float, default=None)
    parser.add_argument("--min_lead_utility", type=float, default=0.10)
    parser.add_argument("--utility_power", type=float, default=1.0)
    parser.add_argument("--alert_cooldown_hours", type=float, default=24.0)
    parser.add_argument("--threshold_grid_size", type=int, default=201)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap_block_hours", type=float, default=168.0)
    parser.add_argument("--random_state", type=int, default=1233)
    parser.add_argument("--utility_tolerance", type=float, default=0.01)
    parser.add_argument("--tabpfn_device", default="cuda:0", help="CUDA device for TabPFN 3.5; CPU/auto are rejected")
    parser.add_argument("--causilo_device", default="cuda:0", help="CUDA device for Causilo; CPU/auto are rejected")
    parser.add_argument("--causilo_n_estimators", type=int, default=8)
    parser.add_argument("--predict_batch_size", type=int, default=512)
    parser.add_argument("--progress_interval_seconds", type=float, default=60.0,
                        help="heartbeat interval during long fit/scoring stages")
    parser.add_argument("--progress_every_batches", type=int, default=1,
                        help="report every N completed prediction batches")
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if args.target_window < 1 or args.target_threshold <= 0:
        parser.error("target_window and target_threshold must be positive")
    if args.predict_batch_size < 1 or args.causilo_n_estimators < 1:
        parser.error("predict_batch_size and causilo_n_estimators must be positive")
    if args.progress_interval_seconds <= 0 or args.progress_every_batches < 1:
        parser.error("progress_interval_seconds and progress_every_batches must be positive")
    if args.n_bootstrap < 0 or args.threshold_grid_size < 2:
        parser.error("n_bootstrap must be non-negative and threshold_grid_size >= 2")
    if args.max_lead_hours is None:
        args.max_lead_hours = float(args.target_window)
    if args.utility_target_lead_hours is None:
        args.utility_target_lead_hours = args.max_lead_hours
    if not (0 <= args.min_lead_hours <= args.max_lead_hours):
        parser.error("Require 0 <= min_lead_hours <= max_lead_hours")
    if args.utility_target_lead_hours < args.min_lead_hours:
        parser.error("utility_target_lead_hours must be at least min_lead_hours")
    if not 0 <= args.min_lead_utility <= 1 or args.utility_power <= 0:
        parser.error("Invalid utility start value or power")
    if args.false_alert_budget_per_month < 0 or any(b < 0 for b in args.false_alert_budgets):
        parser.error("False-alert budgets must be non-negative")
    args.effective_embargo_hours = max(args.cv_embargo_hours, args.target_window)
    versions = {name: package_version(name) for name in ("tabpfn", "causilo")}
    for model_name in args.model_names:
        package = "tabpfn" if model_name == "tabpfn_3_5" else "causilo"
        if versions[package] == "not installed":
            parser.error(f"{package} is not installed in this Python environment")

    gpu_info = {}
    for model_name in args.model_names:
        device_name = args.tabpfn_device if model_name == "tabpfn_3_5" else args.causilo_device
        try:
            gpu_info[model_name] = require_cuda_device(device_name)
        except (ValueError, RuntimeError) as exc:
            parser.error(str(exc))

    print(f"Foundation model package versions: {versions}", flush=True)
    print(f"Verified CUDA devices: {gpu_info}", flush=True)
    for alpha in args.alphas:
        args.alpha = alpha
        dataset_name = (
            f"dataset_alpha_{alpha}_full_binarytarget_win-{args.target_window}_"
            f"thresh-{args.target_threshold}_{args.depeg_side}_dynamic-{args.dynamic_threshold}.parquet"
        )
        dataset_path = args.dataset_dir / dataset_name
        df, feature_cols = load_cv_frame(dataset_path)
        print(f"alpha={alpha:g}: {len(df)} rows, {len(feature_cols)} features from {dataset_path}", flush=True)
        base_run_name = f"threshold_{args.target_threshold}_alpha_{alpha}"
        summaries = []
        for model_name in args.model_names:
            print(f"Running {model_name}, alpha={alpha:g}, threshold={args.target_threshold} bp", flush=True)
            logger = LocalLightningLogger(
                base_dir=args.log_dir, experiment_name=args.experiment_name,
                run_name=f"{base_run_name}_{model_name}_alpha_{alpha}",
            )
            logger.log_params({
                **{k: v for k, v in vars(args).items() if k not in ("dataset_dir", "log_dir")},
                "dataset_path": str(dataset_path), "model_name": model_name,
                "n_rows": len(df), "n_features": len(feature_cols),
                "package_versions": versions,
                "gpu_info": gpu_info[model_name],
                "tabpfn_version": "3.5" if model_name == "tabpfn_3_5" else None,
                "validation_used_for_fit": False,
            })
            with ProgressReporter(
                f"{model_name} alpha={alpha:g}",
                logger.artifact_dir / "cv_progress.log",
                args.progress_interval_seconds,
            ) as progress:
                summaries.append(run_expanding_window_cv(
                    df=df, feature_cols=feature_cols, target_col="target",
                    model_name=model_name, args=args, logger=logger,
                    model_factory=build_foundation_model,
                    fit_callback=fit_foundation_model,
                    progress_callback=progress.update,
                ))
            gc.collect()

        summary_df = select_within_alpha(pd.DataFrame(summaries), args.utility_tolerance)
        experiment_logger = LocalLightningLogger(
            base_dir=args.log_dir, experiment_name=args.experiment_name,
            run_name=f"{base_run_name}_experiment_summary",
        )
        experiment_logger.log_params({
            "alpha": alpha, "target_threshold": args.target_threshold,
            "dataset_path": str(dataset_path), "models_compared": args.model_names,
            "false_alert_budget_per_month": args.false_alert_budget_per_month,
            "effective_embargo_hours": args.effective_embargo_hours,
            "package_versions": versions,
            "gpu_info": gpu_info,
        })
        experiment_logger.save_dataframe(
            summary_df, "comparison/model_comparison_summary.csv",
        )
        experiment_logger.save_dataframe(
            summary_df, "comparison/model_comparison_summary.parquet",
        )
        selected = summary_df.loc[summary_df["selected_model"], "model_name"].iloc[0]
        experiment_logger.save_json({
            "selected_model": selected,
            "false_alert_budget_per_month": args.false_alert_budget_per_month,
            "utility_tolerance": args.utility_tolerance,
        }, "comparison/selected_model.json")
        print(summary_df[["model_name", "alpha", "cv_event_utility_score_mean",
                          "cv_timely_event_recall_mean", "cv_false_alerts_per_month_mean"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
