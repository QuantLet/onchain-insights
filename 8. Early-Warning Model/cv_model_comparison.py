# Import models
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler

from utils.build_dataset import build_dataset, add_dataset_args

import argparse
import json
import shutil
import joblib
from pathlib import Path
from datetime import datetime
from typing import Optional


# -------------------------------------------------------------------
# Local logger writing to lightning_logs/
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

def safe_auprc(y_true, proba):
    y_true = pd.Series(y_true).astype(int)

    # AP is not meaningful if there are no positives
    if (y_true == 1).sum() == 0:
        return np.nan

    return average_precision_score(y_true, proba)


def compute_fold_metrics(y_true, proba):
    """
    Computes:
      - ROC AUC
      - AUPRC
      - best threshold by Youden J
      - TPR/FPR at best threshold
      - lift at best threshold

    Lift definition used here:
      precision_at_best_threshold / prevalence
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).astype(float)

    prevalence = float(y_true.mean()) if len(y_true) > 0 else np.nan

    auc = safe_auc(y_true, proba)
    auprc = safe_auprc(y_true, proba)

    best_threshold = np.nan
    tpr_best = np.nan
    fpr_best = np.nan
    precision_best = np.nan
    lift_best = np.nan

    # ROC-based threshold metrics require both classes
    if len(np.unique(y_true)) >= 2:
        fpr, tpr, thresholds = roc_curve(y_true, proba)
        j = tpr - fpr
        best_idx = int(np.argmax(j))

        best_threshold = float(thresholds[best_idx])
        tpr_best = float(tpr[best_idx])
        fpr_best = float(fpr[best_idx])

        yhat = (proba >= best_threshold).astype(int)
        predicted_positives = int(yhat.sum())

        if predicted_positives > 0:
            precision_best = float(y_true[yhat == 1].mean())
            if prevalence > 0:
                lift_best = float(precision_best / prevalence)

    return {
        "fold_auc": None if np.isnan(auc) else float(auc),
        "fold_auprc": None if np.isnan(auprc) else float(auprc),
        "best_threshold_youdenJ": None if np.isnan(best_threshold) else float(best_threshold),
        "tpr_at_best_threshold": None if np.isnan(tpr_best) else float(tpr_best),
        "fpr_at_best_threshold": None if np.isnan(fpr_best) else float(fpr_best),
        "precision_at_best_threshold": None if np.isnan(precision_best) else float(precision_best),
        "lift_at_best_threshold": None if np.isnan(lift_best) else float(lift_best),
        "test_prevalence": None if np.isnan(prevalence) else float(prevalence),
    }
def make_overlapping_expanding_window_splits(
    n_samples,
    n_splits=5,
    test_frac=0.30,
    min_train_frac=0.20,
    embargo_periods=0,
):
    """
    Create expanding-window CV splits with:
      - fixed test size = test_frac * n_samples for every fold
      - overlapping test windows allowed
      - train always strictly before test
      - embargo_periods rows dropped between train end and test start

    Example with n=1000, test_frac=0.30, min_train_frac=0.20, n_splits=5,
    embargo_periods=0:
      fold 1: train [0:200), test [200:500)
      fold 2: train [0:325), test [325:625)
      fold 3: train [0:450), test [450:750)
      fold 4: train [0:575), test [575:875)
      fold 5: train [0:700), test [700:1000)
    """
    if not (0 < test_frac < 1):
        raise ValueError("test_frac must be in (0, 1)")
    if not (0 < min_train_frac < 1):
        raise ValueError("min_train_frac must be in (0, 1)")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if embargo_periods < 0:
        raise ValueError("embargo_periods must be >= 0")

    test_size = int(np.ceil(test_frac * n_samples))
    min_train_size = int(np.ceil(min_train_frac * n_samples))
    # The latest a training window can end so the embargoed test fits
    max_train_end = n_samples - test_size - embargo_periods

    if max_train_end <= 0:
        raise ValueError(
            f"Dataset too small for test_frac={test_frac} with embargo_periods={embargo_periods}; "
            f"test_size={test_size}, n_samples={n_samples}"
        )

    if min_train_size >= max_train_end:
        raise ValueError(
            f"min_train_frac={min_train_frac} leaves no room for {n_splits} folds. "
            f"Need min_train_size < n_samples - test_size - embargo_periods "
            f"({min_train_size} < {max_train_end})."
        )

    # Evenly spaced train_end values from earliest to latest feasible point
    train_ends = np.linspace(min_train_size, max_train_end, n_splits)
    train_ends = np.round(train_ends).astype(int)

    # Ensure strictly increasing unique split points
    train_ends = np.unique(train_ends)
    if len(train_ends) < n_splits:
        raise ValueError(
            f"Could not construct {n_splits} unique folds. "
            f"Try fewer splits or a smaller min_train_frac."
        )

    train_ends = train_ends[:n_splits]

    splits = []
    for train_end in train_ends:
        test_start = train_end + embargo_periods
        test_end = test_start + test_size

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        splits.append((train_idx, test_idx))

    if len(splits) != n_splits:
        raise ValueError(f"Expected {n_splits} splits, got {len(splits)}")

    return splits

class LocalLightningLogger:
    def __init__(self, base_dir="lightning_logs", experiment_name="default", run_name: Optional[str] = None):
        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name or "default"

        self.exp_dir = self.base_dir / self.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.run_dir = self._make_run_dir(self.exp_dir, run_name)
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
            }
        )

    def _make_run_dir(self, exp_dir: Path, run_name: Optional[str]):
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
        if value is None or (isinstance(value, float) and np.isnan(value)):
            stored_value = None
        else:
            stored_value = float(value)

        self._metrics_latest[key] = stored_value
        self._write_json(self.metrics_path, self._metrics_latest)

        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "key": key,
            "value": stored_value,
            "step": step,
        }
        with open(self.metrics_history_path, "a") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")

    def save_figure(self, fig, artifact_path, dpi=150):
        dst = self.artifact_dir / artifact_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dst, dpi=dpi, bbox_inches="tight", transparent=True)

    def save_dataframe(self, df: pd.DataFrame, artifact_path: str):
        dst = self.artifact_dir / artifact_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if str(dst).endswith(".parquet"):
            df.to_parquet(dst, index=False)
        elif str(dst).endswith(".csv"):
            df.to_csv(dst, index=False)
        else:
            raise ValueError("artifact_path must end with .parquet or .csv")

    def save_json(self, payload, artifact_path: str):
        self._write_json(self.artifact_dir / artifact_path, payload)

    def save_text(self, text: str, artifact_path: str):
        dst = self.artifact_dir / artifact_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            f.write(text)

    def save_pickle(self, obj, artifact_path: str):
        dst = self.artifact_dir / artifact_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, dst)

    def copy_artifact(self, src_path, artifact_subdir=""):
        src = Path(src_path)
        dst = self.artifact_dir / artifact_subdir / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


# -------------------------------------------------------------------
# Model helpers
# -------------------------------------------------------------------
def build_model(model_name, args, pos_weight, use_early_stopping=True):
    if model_name == "xgboost":
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
            early_stopping_rounds=args.early_stopping_rounds if use_early_stopping else None,
            scale_pos_weight=pos_weight,
            random_state=1233,
            n_jobs=args.n_jobs,
        )

    elif model_name == "lightgbm":
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

    elif model_name == "catboost":
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

    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            class_weight={0: 1.0, 1: float(pos_weight)},
            random_state=1233,
            n_jobs=args.n_jobs,
            max_features=args.rf_max_features,
        )

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")


def fit_model(model, model_name, args, X_train, y_train, X_val=None, y_val=None):
    use_val = X_val is not None and y_val is not None and len(X_val) > 0 and pd.Series(y_val).nunique() > 1

    if model_name == "xgboost":
        if use_val:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            model.fit(X_train, y_train, verbose=False)

    elif model_name == "lightgbm":
        if use_val:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)]
            )
        else:
            model.fit(X_train, y_train)

    elif model_name == "catboost":
        if use_val:
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                use_best_model=True,
                verbose=False
            )
        else:
            model.fit(X_train, y_train, verbose=False)

    elif model_name == "random_forest":
        model.fit(X_train, y_train)

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")


# -------------------------------------------------------------------
# CV helpers
# -------------------------------------------------------------------
def apply_scaling(X_train, X_val, X_test, scaler_name):
    if scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "robust":
        scaler = RobustScaler()
    elif scaler_name == "none":
        return X_train.copy(), X_val.copy() if X_val is not None else None, X_test.copy(), None
    else:
        raise ValueError(f"Unsupported scaler: {scaler_name}")

    cols = X_train.columns

    X_train_s = X_train.copy()
    X_test_s = X_test.copy()
    X_train_s[cols] = scaler.fit_transform(X_train[cols])
    X_test_s[cols] = scaler.transform(X_test[cols])

    if X_val is not None:
        X_val_s = X_val.copy()
        X_val_s[cols] = scaler.transform(X_val[cols])
    else:
        X_val_s = None

    return X_train_s, X_val_s, X_test_s, scaler


def split_train_val_tail(X_train_full, y_train_full, val_frac, embargo_periods=0):
    n = len(X_train_full)
    val_n = max(1, int(n * val_frac))

    if val_n >= n:
        val_n = max(1, n // 5)

    split_idx = n - val_n
    if split_idx <= 0:
        return X_train_full, y_train_full, None, None

    val_start = split_idx + embargo_periods
    if val_start >= n:
        # Embargo eats the entire val window — fall back to no early stopping
        return X_train_full.iloc[:split_idx], y_train_full.iloc[:split_idx], None, None

    X_train = X_train_full.iloc[:split_idx]
    y_train = y_train_full.iloc[:split_idx]
    X_val = X_train_full.iloc[val_start:]
    y_val = y_train_full.iloc[val_start:]

    return X_train, y_train, X_val, y_val


def safe_auc(y_true, proba):
    if pd.Series(y_true).nunique() < 2:
        return np.nan
    return roc_auc_score(y_true, proba)


def run_expanding_window_cv(df, feature_cols, target_col, model_name, args, logger):
    X = df[feature_cols].copy()
    y = df[target_col].astype(int).copy()

    splits = make_overlapping_expanding_window_splits(
        n_samples=len(X),
        n_splits=args.n_splits,
        test_frac=args.cv_test_frac,
        min_train_frac=args.cv_min_train_frac,
        embargo_periods=args.cv_embargo_hours,
    )

    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train_full = X.iloc[train_idx].copy()
        y_train_full = y.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_test = y.iloc[test_idx].copy()

        # Tail validation split from the training fold (with the same embargo)
        X_train, y_train, X_val, y_val = split_train_val_tail(
            X_train_full, y_train_full, val_frac=args.cv_val_frac,
            embargo_periods=args.cv_embargo_hours,
        )

        X_train_s, X_val_s, X_test_s, scaler = apply_scaling(
            X_train, X_val, X_test, scaler_name=args.scaler
        )

        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        w_pos = float(n_neg / max(n_pos, 1))

        use_early_stopping = (
            X_val_s is not None and
            y_val is not None and
            pd.Series(y_val).nunique() > 1
        )

        model = build_model(
            model_name=model_name,
            args=args,
            pos_weight=w_pos,
            use_early_stopping=use_early_stopping
        )

        fit_model(
            model=model,
            model_name=model_name,
            args=args,
            X_train=X_train_s,
            y_train=y_train,
            X_val=X_val_s if use_early_stopping else None,
            y_val=y_val if use_early_stopping else None
        )

        proba_test = model.predict_proba(X_test_s)[:, 1]
        fold_metrics = compute_fold_metrics(y_test, proba_test)

        row = {
            "fold": fold,
            "model_name": model_name,
            "alpha": args.alpha,
            "train_size_full": len(X_train_full),
            "train_size_fit": len(X_train),
            "val_size": 0 if X_val is None else len(X_val),
            "test_size": len(X_test),
            "train_pos": int((y_train_full == 1).sum()),
            "train_neg": int((y_train_full == 0).sum()),
            "test_pos": int((y_test == 1).sum()),
            "test_neg": int((y_test == 0).sum()),
            "scale_pos_weight": w_pos,
            "train_end_timestamp": df.iloc[train_idx]["timestamp"].iloc[-1],
            "embargo_hours": args.cv_embargo_hours,
            "test_start": df.iloc[test_idx]["timestamp"].iloc[0],
            "test_end": df.iloc[test_idx]["timestamp"].iloc[-1],
            **fold_metrics,
        }
        fold_rows.append(row)

        logger.log_metric(f"fold_{fold}_auc", fold_metrics["fold_auc"])
        logger.log_metric(f"fold_{fold}_auprc", fold_metrics["fold_auprc"])
        logger.log_metric(f"fold_{fold}_lift_at_best_threshold", fold_metrics["lift_at_best_threshold"])
        logger.log_metric(f"fold_{fold}_best_threshold_youdenJ", fold_metrics["best_threshold_youdenJ"])
        logger.log_metric(f"fold_{fold}_tpr_at_best_threshold", fold_metrics["tpr_at_best_threshold"])
        logger.log_metric(f"fold_{fold}_fpr_at_best_threshold", fold_metrics["fpr_at_best_threshold"])

        print(
            f"[{model_name}] fold={fold} "
            f"train={len(train_idx)} test={len(test_idx)} "
            f"auc={fold_metrics['fold_auc']} "
            f"auprc={fold_metrics['fold_auprc']} "
            f"lift={fold_metrics['lift_at_best_threshold']}"
        )

    fold_df = pd.DataFrame(fold_rows)

    def col_mean_std(df, col):
        vals = pd.to_numeric(df[col], errors="coerce").dropna().astype(float)
        if len(vals) == 0:
            return np.nan, np.nan
        return float(vals.mean()), float(vals.std(ddof=0))

    cv_auc_mean, cv_auc_std = col_mean_std(fold_df, "fold_auc")
    cv_auprc_mean, cv_auprc_std = col_mean_std(fold_df, "fold_auprc")
    cv_lift_mean, cv_lift_std = col_mean_std(fold_df, "lift_at_best_threshold")
    cv_best_threshold_mean, cv_best_threshold_std = col_mean_std(fold_df, "best_threshold_youdenJ")
    cv_tpr_best_mean, cv_tpr_best_std = col_mean_std(fold_df, "tpr_at_best_threshold")
    cv_fpr_best_mean, cv_fpr_best_std = col_mean_std(fold_df, "fpr_at_best_threshold")

    logger.log_metric("cv_auc_mean", cv_auc_mean)
    logger.log_metric("cv_auc_std", cv_auc_std)
    logger.log_metric("five_fold_cv_auc", cv_auc_mean)

    logger.log_metric("cv_auprc_mean", cv_auprc_mean)
    logger.log_metric("cv_auprc_std", cv_auprc_std)

    logger.log_metric("cv_lift_at_best_threshold_mean", cv_lift_mean)
    logger.log_metric("cv_lift_at_best_threshold_std", cv_lift_std)

    logger.log_metric("cv_best_threshold_youdenJ_mean", cv_best_threshold_mean)
    logger.log_metric("cv_best_threshold_youdenJ_std", cv_best_threshold_std)

    logger.log_metric("cv_tpr_at_best_threshold_mean", cv_tpr_best_mean)
    logger.log_metric("cv_tpr_at_best_threshold_std", cv_tpr_best_std)

    logger.log_metric("cv_fpr_at_best_threshold_mean", cv_fpr_best_mean)
    logger.log_metric("cv_fpr_at_best_threshold_std", cv_fpr_best_std)

    logger.save_dataframe(fold_df, "cv/fold_metrics.parquet")
    logger.save_dataframe(fold_df, "cv/fold_metrics.csv")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(fold_df["fold"], fold_df["fold_auc"], marker="o", color="royalblue")
    ax.set_title(
        f"{model_name} - 5-fold expanding CV AUC\n"
        f"fixed test size = {int(args.cv_test_frac * 100)}% of dataset"
    )
    ax.set_xlabel("Fold")
    ax.set_ylabel("AUC")
    ax.set_xticks(fold_df["fold"].tolist())
    ax.grid(alpha=0.3)
    fig.tight_layout()
    logger.save_figure(fig, "plots/cv/fold_auc.png", dpi=200)
    plt.close(fig)

    return {
        "model_name": model_name,
        "alpha": args.alpha,
        "n_splits": args.n_splits,
        "cv_test_frac": args.cv_test_frac,
        "cv_min_train_frac": args.cv_min_train_frac,

        "cv_auc_mean": None if np.isnan(cv_auc_mean) else cv_auc_mean,
        "cv_auc_std": None if np.isnan(cv_auc_std) else cv_auc_std,

        "cv_auprc_mean": None if np.isnan(cv_auprc_mean) else cv_auprc_mean,
        "cv_auprc_std": None if np.isnan(cv_auprc_std) else cv_auprc_std,

        "cv_lift_at_best_threshold_mean": None if np.isnan(cv_lift_mean) else cv_lift_mean,
        "cv_lift_at_best_threshold_std": None if np.isnan(cv_lift_std) else cv_lift_std,

        "cv_best_threshold_youdenJ_mean": None if np.isnan(cv_best_threshold_mean) else cv_best_threshold_mean,
        "cv_best_threshold_youdenJ_std": None if np.isnan(cv_best_threshold_std) else cv_best_threshold_std,

        "cv_tpr_at_best_threshold_mean": None if np.isnan(cv_tpr_best_mean) else cv_tpr_best_mean,
        "cv_tpr_at_best_threshold_std": None if np.isnan(cv_tpr_best_std) else cv_tpr_best_std,

        "cv_fpr_at_best_threshold_mean": None if np.isnan(cv_fpr_best_mean) else cv_fpr_best_mean,
        "cv_fpr_at_best_threshold_std": None if np.isnan(cv_fpr_best_std) else cv_fpr_best_std,

        **{f"fold_{int(r['fold'])}_auc": r["fold_auc"] for _, r in fold_df.iterrows()},
        **{f"fold_{int(r['fold'])}_auprc": r["fold_auprc"] for _, r in fold_df.iterrows()},
    }

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_dataset_args(parser)

    training_args = parser.add_argument_group("Training / logging arguments")
    training_args.add_argument(
        "--experiment_name",
        type=str,
        default="cv_model_comparison",
        help="experiment folder under lightning_logs/"
    )
    training_args.add_argument(
        "--run_name",
        type=str,
        default="expanding_window_cv",
        help="base run name"
    )
    training_args.add_argument(
        "--log_dir",
        type=str,
        default="lightning_logs",
        help="local logging folder"
    )
    training_args.add_argument(
        "--model_names",
        nargs="+",
        default=["xgboost", "lightgbm", "catboost", "random_forest"],
        choices=["xgboost", "lightgbm", "catboost", "random_forest"],
        help="which models to compare"
    )
    training_args.add_argument(
        "--scaler",
        type=str,
        default="none",
        choices=["none", "standard", "robust"],
        help="feature scaler"
    )

    cv_args = parser.add_argument_group("Cross-validation arguments")
    cv_args.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="number of expanding-window CV folds"
    )
    cv_args.add_argument(
    "--cv_test_frac",
    type=float,
    default=0.30,
    help="fraction of the full dataset used as test set in every fold"
    )
    cv_args.add_argument(
        "--cv_min_train_frac",
        type=float,
        default=0.68,
        help="minimum fraction of the full dataset used as training set in the first fold"
    )
    cv_args.add_argument(
        "--cv_val_frac",
        type=float,
        default=0.30,
        help="fraction of each training fold used as tail validation set for early stopping"
    )
    cv_args.add_argument(
        "--cv_embargo_hours",
        type=int,
        default=24,
        help="number of hourly periods to drop between train/val and train/test boundaries (leakage embargo)"
    )

    model_args = parser.add_argument_group("Model arguments")
    model_args.add_argument("--learning_rate", type=float, default=0.01, help="learning rate")
    model_args.add_argument("--early_stopping_rounds", type=int, default=200, help="early stopping rounds")
    model_args.add_argument("--eval_metric", type=str, default="auc", help="evaluation metric")
    model_args.add_argument("--n_estimators", type=int, default=800, help="number of trees")
    model_args.add_argument("--max_depth", type=int, default=6, help="maximum tree depth")
    model_args.add_argument("--num_leaves", type=int, default=31, help="LightGBM only")
    model_args.add_argument("--rf_max_features", type=str, default="sqrt", help="RandomForest only")
    model_args.add_argument("--n_jobs", type=int, default=-1, help="parallel jobs")

    args = parser.parse_args()

    # Build dataset using your existing pipeline
    dict_args = vars(args).copy()
    dict_args["target"] = True
    dataset_path = build_dataset(**dict_args)

    # Load and prepare dataset exactly in the same spirit as your training script
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

    feature_cols = [c for c in df.columns if c not in [TIME_COL, TARGET_COL]]

    # Shared experiment-level metadata
    experiment_logger = LocalLightningLogger(
        base_dir=args.log_dir,
        experiment_name=args.experiment_name,
        run_name=f"{args.run_name}_experiment_summary"
    )

    experiment_logger.log_params({
        "dataset_path": dataset_path,
        "alpha": args.alpha,
        "n_splits": args.n_splits,
        "cv_val_frac": args.cv_val_frac,
        "cv_embargo_hours": args.cv_embargo_hours,
        "scaler": args.scaler,
        "models_compared": args.model_names,
        "n_rows": len(df),
        "n_features": len(feature_cols),
    })

    print(f"Experiment logs will be saved under: {experiment_logger.exp_dir}")

    summaries = []

    for model_name in args.model_names:
        print(f"\nRunning expanding-window CV for model={model_name}")

        logger = LocalLightningLogger(
            base_dir=args.log_dir,
            experiment_name=args.experiment_name,
            run_name=f"{args.run_name}_{model_name}_alpha_{args.alpha}"
        )

        logger.log_params({
            "model_name": model_name,
            "alpha": args.alpha,
            "dataset_path": dataset_path,
            "n_splits": args.n_splits,
            "cv_val_frac": args.cv_val_frac,
            "cv_embargo_hours": args.cv_embargo_hours,
            "scaler": args.scaler,
            "learning_rate": args.learning_rate,
            "early_stopping_rounds": args.early_stopping_rounds,
            "eval_metric": args.eval_metric,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "num_leaves": args.num_leaves,
            "rf_max_features": args.rf_max_features,
            "target_window": args.target_window,
            "target_threshold": args.target_threshold,
            "depeg_side": args.depeg_side,
            "dynamic_threshold": int(args.dynamic_threshold),
            "n_features": len(feature_cols),
        })

        summary = run_expanding_window_cv(
            df=df,
            feature_cols=feature_cols,
            target_col=TARGET_COL,
            model_name=model_name,
            args=args,
            logger=logger
        )
        summaries.append(summary)

    # Save overall comparison
    summary_df = pd.DataFrame(summaries).sort_values("cv_auc_mean", ascending=False)
    experiment_logger.save_dataframe(summary_df, "comparison/model_comparison_summary.parquet")
    experiment_logger.save_dataframe(summary_df, "comparison/model_comparison_summary.csv")
    experiment_logger.save_json(
        summary_df.to_dict(orient="records"),
        "comparison/model_comparison_summary.json"
    )

    # Overall comparison plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(summary_df["model_name"], summary_df["cv_auc_mean"], color="steelblue")
    ax.set_title(f"5-fold expanding-window CV AUC comparison (alpha={args.alpha})")
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean CV AUC")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    experiment_logger.save_figure(fig, "plots/model_comparison_auc.png", dpi=200)
    plt.close(fig)

    print("\nDone.")
    print(f"Summary saved to: {experiment_logger.run_dir}")
    print(summary_df[["model_name", "alpha", "cv_auc_mean", "cv_auc_std"]])