# Import models
import xgboost
from xgboost import XGBClassifier
from xgboost import plot_importance

import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt

import shap
from sklearn.preprocessing import StandardScaler, RobustScaler
from utils.build_dataset import build_dataset, add_dataset_args
import argparse

import os
import json
import shutil
import joblib
from pathlib import Path
from datetime import datetime


# -------------------------------------------------------------------
# Local logger that writes to lightning_logs/
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
            }
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
        value = float(value) if value is not None else None
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

    elif args.model_name == "lightgbm":
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

    elif args.model_name == "catboost":
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

    elif args.model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            class_weight={0: 1.0, 1: float(pos_weight)},
            random_state=1233,
            n_jobs=args.n_jobs,
            max_features=args.rf_max_features,
        )

    else:
        raise ValueError(f"Unsupported model_name: {args.model_name}")


def fit_model(model, args, X_train, y_train, X_val, y_val):
    if args.model_name == "xgboost":
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )

    elif args.model_name == "lightgbm":
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)]
        )

    elif args.model_name == "catboost":
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=False
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
    # universal sklearn/joblib dump
    logger.save_pickle(model, "models/model.joblib")

    # native format when available
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
        logger.save_text(f"Native model save failed: {e}", "models/native_save_error.txt")

    # simple signature replacement
    signature = {
        "input_columns": list(X_train.columns),
        "input_dtypes": {c: str(X_train[c].dtype) for c in X_train.columns},
        "output": "predict_proba[:, 1] (float)",
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

    logger.save_figure(fig, "plots/feature_importance/feature_importance_native.png", dpi=200)
    plt.close(fig)

    logger.save_dataframe(
        top_imp.reset_index().rename(columns={"index": "feature", 0: "importance"}),
        "plots/feature_importance/feature_importance_native_top20.parquet"
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
                elif base_values.ndim == 1 and len(base_values) > positive_class_idx and len(base_values) != len(values):
                    base_values = np.repeat(base_values[positive_class_idx], values.shape[0])

            return shap.Explanation(
                values=values,
                base_values=base_values,
                data=data,
                feature_names=feature_names
            )

    if isinstance(shap_output, list):
        values = shap_output[positive_class_idx] if len(shap_output) > 1 else shap_output[0]

        if isinstance(expected_value, (list, np.ndarray)):
            base_value = expected_value[positive_class_idx] if len(expected_value) > positive_class_idx else expected_value[0]
        else:
            base_value = expected_value

        if np.isscalar(base_value):
            base_value = np.repeat(base_value, len(X))

        return shap.Explanation(
            values=values,
            base_values=base_value,
            data=data,
            feature_names=feature_names
        )

    arr = np.asarray(shap_output)
    if arr.ndim == 3:
        arr = arr[:, :, positive_class_idx]

    return shap.Explanation(
        values=arr,
        data=data,
        feature_names=feature_names
    )


def compute_shap_explanation(model, X_explain, shap_type):
    explainer = shap.TreeExplainer(model, feature_perturbation=shap_type)
    raw_shap = explainer(X_explain)
    shap_values = normalize_shap_output(
        raw_shap,
        X_explain,
        expected_value=getattr(explainer, "expected_value", None),
        positive_class_idx=1
    )
    return explainer, shap_values


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_dataset_args(parser)

    training_args = parser.add_argument_group('Training arguments')
    training_args.add_argument('--remote_logging', action='store_true', help='deprecated / ignored; logging is local only')
    training_args.add_argument('--experiment_name', type=str, default="default", help='experiment name under lightning_logs/')
    training_args.add_argument('--run_name', type=str, default=None, help='run name under the experiment')
    training_args.add_argument('--log_dir', type=str, default='lightning_logs', help='local folder where logs/artifacts are stored')
    training_args.add_argument('--test_size', type=float, default=0.30, help='proportion of dataset to use as test set')
    training_args.add_argument('--val_size', type=float, default=0.15, help='proportion of dataset to use as validation set')
    training_args.add_argument('--scaler', type=str, default='standard', help='standard | robust | none')
    training_args.add_argument('--model_name', type=str, default='xgboost', choices=['xgboost', 'lightgbm', 'catboost', 'random_forest'], help='which model to train')

    model_args = parser.add_argument_group('Model arguments')
    model_args.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    model_args.add_argument('--early_stopping_rounds', type=int, default=200, help='early stopping rounds')
    model_args.add_argument('--eval_metric', type=str, default='auc', help='evaluation metric')
    model_args.add_argument('--n_estimators', type=int, default=800, help='number of trees')
    model_args.add_argument('--shap_type', type=str, default='tree_path_dependent', help='type of SHAP explainer to use')
    model_args.add_argument('--max_depth', type=int, default=6, help='maximum tree depth')
    model_args.add_argument('--num_leaves', type=int, default=31, help='LightGBM only')
    model_args.add_argument('--rf_max_features', type=str, default='sqrt', help='RandomForest only')
    model_args.add_argument('--n_jobs', type=int, default=-1, help='parallel jobs where supported')

    args = parser.parse_args()
    dict_args = vars(args).copy()
    dict_args['target'] = True
    dataset_path = build_dataset(**dict_args)

    logger = LocalLightningLogger(
        base_dir=args.log_dir,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )
    print(f"Local logs will be saved to: {logger.run_dir}")

    dataset = pd.read_parquet(dataset_path)
    dataset['timestamp'] = dataset.index
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

    num_cols = X_train.columns
    scaler = None

    if args.scaler == 'standard':
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_val_scaled[num_cols] = scaler.transform(X_val[num_cols])
        X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
        X_train, X_val, X_test = X_train_scaled, X_val_scaled, X_test_scaled

    elif args.scaler == 'robust':
        scaler = RobustScaler()
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_val_scaled[num_cols] = scaler.transform(X_val[num_cols])
        X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
        X_train, X_val, X_test = X_train_scaled, X_val_scaled, X_test_scaled

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    w_pos = float(n_neg / max(n_pos, 1))
    print(f"Train positives={n_pos}, negatives={n_neg}, w_pos={w_pos:.3f}")

    model = build_model(args, pos_weight=w_pos)

    # ---------------------------
    # log params
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
        "alpha": args.alpha,
        "scale_pos_weight_used": float(w_pos),
        "n_features": len(FEATURES),
    }
    logger.log_params(params_to_log)

    # ---------------------------
    # train
    # ---------------------------
    fit_model(model, args, X_train, y_train, X_val, y_val)
    logger.log_metric("best_iteration", get_best_iteration(model, args.model_name))

    # ---------------------------
    # evaluate
    # ---------------------------
    proba_test = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba_test)
    auprc = average_precision_score(y_test, proba_test)

    logger.log_metric("test_roc_auc", float(auc))
    logger.log_metric("test_auprc", float(auprc))

    # threshold by Youden J
    fpr, tpr, thresholds = roc_curve(y_test, proba_test)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    thresh = float(thresholds[best_idx])

    logger.log_metric("best_threshold_youdenJ", thresh)
    logger.log_metric("tpr_at_best_threshold", float(tpr[best_idx]))
    logger.log_metric("fpr_at_best_threshold", float(fpr[best_idx]))

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
    # confusion matrix
    # ---------------------------
    yhat = (proba_test >= thresh).astype(int)
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

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    log_fig(logger, fig, "plots/confusion_matrix.png")
    plt.close(fig)

    # classification report
    clf_report = classification_report(y_test, yhat, output_dict=True)
    logger.save_json(clf_report, "reports/classification_report.json")

    # ---------------------------
    # save scaler + model
    # ---------------------------
    if scaler is not None:
        logger.save_pickle(scaler, "models/preprocess_scaler.joblib")

    save_model_to_local(
        model=model,
        model_name=args.model_name,
        logger=logger,
        X_train=X_train,
    )

    # feature list
    logger.save_text("\n".join(FEATURES), "meta/features.txt")

    # ---------------------------
    # SHAP
    # ---------------------------
    X_explain = X_test.copy()
    explainer, shap_values = compute_shap_explanation(
        model=model,
        X_explain=X_explain,
        shap_type=args.shap_type
    )

    # 1) SHAP beeswarm (global)
    std_per_feature = shap_values.values.std(axis=0)
    order = np.argsort(std_per_feature)[::-1]
    shap.plots.beeswarm(shap_values, order=order, max_display=10, show=False)
    _log_current_fig(logger, "shap_beeswarm_global.png", artifact_subdir="plots/shap")

    raw_train_shap = explainer(X_train)
    shap_values_train = normalize_shap_output(
        raw_train_shap,
        X_train,
        expected_value=getattr(explainer, "expected_value", None),
        positive_class_idx=1
    )
    std_per_feature_train = shap_values_train.values.std(axis=0)
    order = np.argsort(std_per_feature_train)[::-1]
    shap.plots.beeswarm(shap_values_train, order=order, max_display=10, show=False)
    _log_current_fig(logger, "shap_beeswarm_global_insample.png", artifact_subdir="plots/shap")

    # 2) SHAP beeswarm above threshold
    warn_mask = proba_test >= thresh
    sv_warn = shap_values[warn_mask]
    if sv_warn.values.shape[0] > 0:
        shap.plots.beeswarm(sv_warn, max_display=10, show=False)
        _log_current_fig(logger, "shap_beeswarm_above_threshold.png", artifact_subdir="plots/shap")

        # 3) ordered by mean positive contribution
        pos_mean = np.clip(sv_warn.values, 0, None).mean(axis=0)
        order_pos = np.argsort(pos_mean)[::-1]
        shap.plots.beeswarm(sv_warn, order=order_pos, max_display=10, show=False)
        _log_current_fig(
            logger,
            "shap_beeswarm_above_threshold_ordered_by_pos_mean.png",
            artifact_subdir="plots/shap"
        )

    # 4) native feature importance
    log_native_feature_importance(model, args.model_name, FEATURES, logger)

    # 5) predictions over time
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

    ax.plot(test_dates, proba_test, label="Predicted Probability", color="royalblue", lw=2, zorder=2)
    axi.plot(
        test_dates,
        X_test["depeg_bps"].reset_index(drop=True),
        label="Depeg BPS",
        color="crimson",
        lw=1.5,
        alpha=0.9,
        zorder=1
    )

    ax.set_ylabel("probability of depeg in the next 24 hours", color="royalblue")
    axi.set_ylabel("Depeg BPS", color="crimson")
    ax.tick_params(axis="y", labelcolor="royalblue")
    axi.tick_params(axis="y", labelcolor="crimson")

    fig.tight_layout()
    logger.save_figure(fig, "plots/timeseries/predictions_over_time.png", dpi=300)
    plt.close(fig)

    # ---------------------------
    # SHAP scatter per top features
    # ---------------------------
    mean_abs = np.abs(shap_values.values).std(axis=0)
    top10_idx = np.argsort(mean_abs)[::-1][:10]
    feature_names = list(shap_values.feature_names) if shap_values.feature_names is not None else list(X_test.columns)

    for idx in top10_idx:
        fname = feature_names[idx]

        shap.plots.scatter(shap_values[:, idx], color=shap_values, show=False)
        plt.title(f"SHAP scatter: {fname}")
        _log_current_fig(logger, f"shap_scatter_{fname}.png", artifact_subdir="plots/shap")

        shap.plots.scatter(shap_values[:, idx], color=proba_test, show=False)
        plt.title(f"SHAP scatter (colored by proba): {fname}")
        _log_current_fig(logger, f"shap_scatter_{fname}_colored_by_proba.png", artifact_subdir="plots/shap")

    # ---------------------------
    # save predictions
    # ---------------------------
    pred_df = pd.DataFrame({
        "timestamp": df["timestamp"].iloc[val_end:].reset_index(drop=True),
        "y_true": y_test.reset_index(drop=True).astype(int),
        "proba_depeg": proba_test.astype(float),
    })
    logger.save_dataframe(pred_df, "predictions/test_pred_proba.parquet")

    print(f"Done. All logs/artifacts saved to: {logger.run_dir}")
