#!/usr/bin/env python3
"""Create false-alert-budget CV, holdout, and economic-ablation reports.

Usage (from this directory or repository root):
  python cv_results_report.py
  python cv_results_report.py --experiment_name EXPERIMENT --log_dir lightning_logs

The chosen model at each depeg threshold is the supported model/alpha with the
highest pooled recall over unique realized events across OOF test folds. Each
fold's cutoff is selected on validation data using the requested threshold policy.
Foundation models are supported; this report does not compute SHAP values.

All stages use a 24-hour quiet-period alert reset. Empirical and event-UCB policies maximize
validation event recall; NP uses a negative-score order statistic to bound FPR.
Recall counts unique observed events, lead time pools their first timely alerts,
and false-alert burden pools counts over hours with complete future follow-up.
Events at the last observed hour remain eligible; unresolved alerts are censored.
Repeated timely episodes never become false alerts. The holdout starts at cv_end;
its embargo is taken from the preceding development period. Ablations refit their
full-feature control and replay saved validation/test timestamps using the same
scorer, without calling the legacy utility selector.

By default, use NP thresholds on cached CV probabilities, select models by event
recall then false-alert burden, and report the selected-model holdout. Holdout
scores are reused when available; auto mode otherwise fits the selected model.
--holdout_mode cached prohibits fitting; --cv_only skips holdout altogether.
ROC/NP-ROC plots, descriptive full/partial NP areas, and grouped LaTeX tables
are written separately for depeg thresholds 10 and 15 when those runs exist.
Use --run_ablations to refit feature variants for each policy's selected model.
--fit_models is a compatibility alias for empirical-policy ablations and holdout refitting.
Fitting requires the model helpers (including Kumo/TabICLv2 adapters and economic
feature blocks) from the environment that produced the original experiment.
Rescoring existing probabilities does not change the model's original training
target; holdout and ablation refits rebuild the requested onset-based target.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from early_warning_evaluation import depeg_event_starts

DEFAULT_EXPERIMENT = "cv_model_comparison_lead6_24_10_15bps_2026-10-19"
BUDGET = 5.0
DEFAULT_MIN_LEAD_HOURS = 2.0
DEFAULT_MAX_LEAD_HOURS = 24.0
DEFAULT_HOLDOUT_EMBARGO_HOURS = 24.0
SUPPORTED_MODELS = {"xgboost", "lightgbm", "catboost", "random_forest", "kumo_tabular", "tabicl_v2"}
ALERT_COOLDOWN_HOURS = 12.0
HOURS_PER_MONTH = 730.5
HOUR = pd.Timedelta(hours=1)


def model_helpers():
    """Import model dependencies only when actually fitting a model."""
    shap_dir = HERE.parent / "9. SHAP explanations of Early Warning Model"
    if not (HERE / "utils").is_dir() and (shap_dir / "utils").is_dir():
        if str(shap_dir) not in sys.path:
            sys.path.insert(0, str(shap_dir))
    return importlib.import_module("cv_model_comparison")


def utc_times(values):
    ts = pd.DatetimeIndex(pd.to_datetime(values, utc=True)).as_unit("ns")
    if ts.hasnans or not ts.is_unique or not ts.is_monotonic_increasing:
        raise ValueError("Timestamps must be unique, chronological and non-missing")
    if len(ts) and np.any(ts.asi8 % HOUR.value):
        raise ValueError("This report expects observations on the UTC hourly grid")
    return ts


def complete_windows(starts, ends, available):
    """Whether every hourly timestamp in each inclusive interval is observed."""
    available = utc_times(available)
    starts, ends = pd.DatetimeIndex(starts), pd.DatetimeIndex(ends)
    needed = ((ends.asi8 - starts.asi8) // HOUR.value + 1)
    counts = available.searchsorted(ends, side="right") - available.searchsorted(starts, side="left")
    return (needed > 0) & (counts == needed)


def alert_times(timestamps, probability, cutoff, cooldown_hours=ALERT_COOLDOWN_HOURS):
    """Start an episode only after ``cooldown_hours`` observed quiet hours.

    Above-threshold crossings belong to the same episode until the score has
    stayed below cutoff for the full quiet period. The first alert also needs
    that observed quiet history, censoring episodes with an unknown onset.
    Missing timestamps or scores break the quiet streak; they never reset an
    episode. This is a quiet-period reset, not reminders every 24 hours.
    """
    if not np.isfinite(cooldown_hours) or cooldown_hours < 1:
        raise ValueError("alert_cooldown_hours must be finite and at least one hour")
    ts = utc_times(timestamps)
    p = np.asarray(probability, dtype=float)
    if len(p) != len(ts):
        raise ValueError("Each timestamp must have one probability")
    if not len(ts):
        return pd.DatetimeIndex([], tz="UTC")
    adjacent = np.r_[False, np.diff(ts.asi8) == HOUR.value]
    quiet_hours, starts = 0, []
    for index, score in enumerate(p):
        if not adjacent[index]:
            quiet_hours = 0
        if np.isfinite(score) and score < cutoff:
            quiet_hours += 1
        else:
            if np.isfinite(score) and score >= cutoff and quiet_hours >= cooldown_hours:
                starts.append(index)
            quiet_hours = 0
    return ts[starts]


def score_events(frame, probability, cutoff, event_starts,
                 min_lead_hours=DEFAULT_MIN_LEAD_HOURS,
                 max_lead_hours=DEFAULT_MAX_LEAD_HOURS,
                 cooldown_hours=ALERT_COOLDOWN_HOURS, *, outcome_context=None):
    """Score observed events and mature alerts without using future partitions.

    Event onsets may extend to the final observed hour. Recall requires a full
    2--24h prediction window. All alerts inside *any* observed event's warning
    window are timely, including repeated alerts and warnings for left-boundary
    events excluded from recall. A false alert needs its complete future window
    observed; unresolved alerts are censored. Its denominator uses exactly the
    hours with that full follow-up. Context is always capped at this split's end.
    """
    if not 0 < min_lead_hours < max_lead_hours:
        raise ValueError("Require 0 < min_lead_hours < max_lead_hours")
    if any(float(x) != int(x) for x in (min_lead_hours, max_lead_hours)):
        raise ValueError("Lead times must be whole hours on this hourly dataset")
    ts = utc_times(frame["timestamp"])
    p = np.asarray(probability, dtype=float)
    if not len(ts) or len(p) != len(ts):
        raise ValueError("A nonempty frame and one score per row are required")
    score_start, score_end = ts[0], ts[-1]
    context = frame if outcome_context is None else outcome_context
    context_ts = utc_times(context["timestamp"])
    price_ok = np.isfinite(pd.to_numeric(context["depeg_bps"], errors="coerce"))
    known_prices = context_ts[price_ok & (context_ts <= score_end)]
    all_events = utc_times(event_starts)
    all_events = all_events[(all_events <= score_end) & all_events.isin(known_prices)]
    minimum, maximum = pd.Timedelta(hours=min_lead_hours), pd.Timedelta(hours=max_lead_hours)
    observed_scores = ts[np.isfinite(p)]
    events = all_events[
        (all_events - maximum >= score_start)
        & complete_windows(all_events - maximum, all_events - minimum, observed_scores)
    ]
    alerts = alert_times(ts, p, cutoff, cooldown_hours)
    timely = np.zeros(len(alerts), dtype=bool)
    for event in all_events:
        timely |= (alerts >= event - maximum) & (alerts <= event - minimum)
    mature_hours = complete_windows(ts + HOUR, ts + maximum, known_prices) & np.isfinite(p)
    exposure_times = ts[mature_hours]
    mature_alerts = alerts.isin(exposure_times)
    false_times = alerts[~timely & mature_alerts]
    event_outcomes = []
    for event in events:
        candidates = alerts[(alerts >= event - maximum) & (alerts <= event - minimum)]
        lead = (event - candidates[0]).total_seconds() / 3600 if len(candidates) else np.nan
        event_outcomes.append({"event_time": event, "detected": float(len(candidates) > 0), "lead_hours": lead})
    detected = np.asarray([r["detected"] for r in event_outcomes], dtype=float)
    leads = np.asarray([r["lead_hours"] for r in event_outcomes], dtype=float)
    hours = len(exposure_times)
    return {
        "recall": float(detected.mean()) if len(detected) else np.nan,
        "events": len(events), "detected": int(detected.sum()),
        "lead_hours": float(np.nanmedian(leads)) if np.isfinite(leads).any() else np.nan,
        "false_alerts": len(false_times), "false_alert_exposure_hours": hours,
        "timely_alert_episodes": int(timely.sum()),
        "right_censored_alert_episodes": int((~timely & ~mature_alerts).sum()),
        "alert_cooldown_hours": cooldown_hours,
        "_event_outcomes": event_outcomes,
        "false_alerts_per_month": len(false_times) * HOURS_PER_MONTH / hours if hours else np.nan,
        "_alert_outcomes": [{"alert_time": a, "status": "timely" if timely[j] else
                             "false" if mature_alerts[j] else "right_censored"}
                            for j, a in enumerate(alerts)],
        "_bootstrap_inputs": {
            "detected": detected, "lead_hours": leads,
            "false_alert_times": false_times,
            "exposure_times": exposure_times,
        },
    }


def choose_cutoff(validation, probability, events, budget, min_lead_hours=DEFAULT_MIN_LEAD_HOURS, max_lead_hours=DEFAULT_MAX_LEAD_HOURS, grid_size=201, cooldown_hours=ALERT_COOLDOWN_HOURS, *, outcome_context=None):
    p = np.asarray(probability, dtype=float)
    finite = p[np.isfinite(p)]
    if not len(finite):
        raise ValueError("Validation probabilities contain no finite values")
    if len(np.unique(finite)) <= grid_size:
        candidates = np.unique(finite)
    else:
        candidates = np.unique(np.quantile(finite, np.linspace(0, 1, grid_size)))
    candidates = np.unique(np.r_[candidates, np.nextafter(float(finite.max()), np.inf)])
    feasible = []
    for cutoff in candidates:
        metrics = score_events(validation, p, float(cutoff), events, min_lead_hours, max_lead_hours, cooldown_hours, outcome_context=outcome_context)
        if not metrics["events"] or not metrics["false_alert_exposure_hours"]:
            raise ValueError("Validation requires eligible events and fully observed alert follow-up")
        if metrics["false_alerts_per_month"] <= budget + 1e-12:
            feasible.append((float(cutoff), metrics))
    if not feasible:
        raise RuntimeError("No validation cutoff satisfies the alert budget")
    # Recall first; on ties use fewer false alerts, then the more conservative cutoff.
    return max(feasible, key=lambda item: (
        -np.inf if pd.isna(item[1]["recall"]) else item[1]["recall"],
        -item[1]["false_alerts_per_month"], item[0],
    ))


def eventwise_recall_ci(detected, seed=1233, n_bootstrap=5000):
    """Percentile CI from resampling unique realized event outcomes."""
    values = np.asarray(detected, dtype=float)
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))


def discover_runs(log_root: Path, experiment: str):
    exp_dir = log_root / experiment
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    rows = []
    for run_dir in sorted(exp_dir.iterdir()):
        hparams_path = run_dir / "hparams.json"
        score_path = run_dir / "artifacts/cv/operational_scores.parquet"
        context_path = run_dir / "artifacts/cv/realised_depeg_context.parquet"
        metrics_path = run_dir / "artifacts/cv/fold_metrics.csv"
        # Ignore summary and ablation/report folders; only original CV runs
        # with saved OOF probabilities and their own hparams are candidates.
        if (not run_dir.is_dir() or "experiment_summary" in run_dir.name
                or "economic_ablation" in run_dir.name or "report_" in run_dir.name
                or not hparams_path.is_file() or not score_path.is_file()
                or not context_path.is_file()):
            continue
        hp = json.loads(hparams_path.read_text())
        model = str(hp.get("model_name", "")).lower()
        if model not in SUPPORTED_MODELS or "alpha" not in hp or "target_threshold" not in hp:
            continue
        metrics = pd.read_csv(metrics_path) if metrics_path.is_file() else pd.DataFrame()
        rows.append({
            "run_dir": str(run_dir), "model_name": model,
            "alpha": float(hp["alpha"]), "target_threshold": int(hp["target_threshold"]),
            "cv_auc_mean": pd.to_numeric(metrics.get("fold_auc"), errors="coerce").mean() if "fold_auc" in metrics else np.nan,
            "cv_auprc_mean": pd.to_numeric(metrics.get("fold_auprc"), errors="coerce").mean() if "fold_auprc" in metrics else np.nan,
            "cv_brier_score_mean": pd.to_numeric(metrics.get("fold_brier_score"), errors="coerce").mean() if "fold_brier_score" in metrics else np.nan,
            "cv_brier_skill_score_mean": pd.to_numeric(metrics.get("fold_brier_skill_score"), errors="coerce").mean() if "fold_brier_skill_score" in metrics else np.nan,
        })
    if not rows:
        raise FileNotFoundError(f"No supported individual CV run folders found under {exp_dir}")
    return pd.DataFrame(rows), exp_dir


def pooled_summary(event_rows, exposure_times, false_times, *, seed=1233,
                   n_bootstrap=1000, block_hours=168):
    """Pool event outcomes and calendar exposure before computing statistics."""
    detected = np.asarray([r["detected"] for r in event_rows], dtype=float)
    leads = np.asarray([r["lead_hours"] for r in event_rows], dtype=float)
    exposure = utc_times(exposure_times)
    false_times = utc_times(false_times)
    if not false_times.isin(exposure).all():
        raise ValueError("Every false alert must have fully observed follow-up")
    result = {
        "cv_event_recall": float(detected.mean()) if len(detected) else np.nan,
        "cv_events_detected": int(detected.sum()), "cv_eligible_events": len(detected),
        "cv_duplicate_event_assignments": 0,
        "cv_median_lead_hours": float(np.nanmedian(leads)) if np.isfinite(leads).any() else np.nan,
        "cv_false_alerts": len(false_times), "cv_false_alert_exposure_hours": len(exposure),
        "cv_false_alerts_per_month": HOURS_PER_MONTH * len(false_times) / len(exposure) if len(exposure) else np.nan,
    }
    # Bootstrap paired event outcomes, rather than averaging fold medians.
    rng = np.random.default_rng(seed)
    event_draws, lead_draws, false_draws = [], [], []
    if block_hours <= 0 or n_bootstrap < 1:
        raise ValueError("Positive block length and bootstrap count are required")
    block_ns = pd.Timedelta(hours=block_hours).value
    exposure_blocks = exposure.asi8 // block_ns
    block_ids, exposure_counts = np.unique(exposure_blocks, return_counts=True)
    false_counts = np.zeros(len(block_ids))
    if len(false_times):
        positions = np.searchsorted(block_ids, false_times.asi8 // block_ns)
        false_counts = np.bincount(positions, minlength=len(block_ids))
    for _ in range(n_bootstrap):
        if len(detected):
            indices = rng.integers(0, len(detected), size=len(detected))
            event_draws.append(float(detected[indices].mean()))
            selected_leads = leads[indices]
            if np.isfinite(selected_leads).any():
                lead_draws.append(float(np.nanmedian(selected_leads)))
        if len(block_ids):
            indices = rng.integers(0, len(block_ids), size=len(block_ids))
            false_draws.append(float(HOURS_PER_MONTH * false_counts[indices].sum() / exposure_counts[indices].sum()))
    for metric, draws in [("event_recall", event_draws), ("median_lead_hours", lead_draws),
                          ("false_alerts_per_month", false_draws)]:
        result[f"cv_{metric}_ci_lower"] = float(np.quantile(draws, .025)) if len(draws) else np.nan
        result[f"cv_{metric}_ci_upper"] = float(np.quantile(draws, .975)) if len(draws) else np.nan
    return result


def read_scores(run_dir):
    scores = pd.read_parquet(Path(run_dir) / "artifacts/cv/operational_scores.parquet")
    scores["timestamp"] = pd.to_datetime(scores["timestamp"], utc=True)
    return scores


def fold_signature(scores):
    return [(int(fold), split, tuple(pd.to_datetime(part.timestamp, utc=True).sort_values()))
            for (fold, split), part in scores.groupby(["fold", "split"], sort=True)
            if split in {"validation", "test"}]


def run_cv_budget(all_cv, exp_dir, min_lead_hours, max_lead_hours, *,
                  detail_dir=None, n_bootstrap=1000, threshold_selector=None, budget=BUDGET):
    from threshold_selection import hourly_metrics
    selector = choose_cutoff if threshold_selector is None else threshold_selector
    folds, summaries, event_records, alert_records = [], [], [], []
    reference_events, reference_folds = {}, {}
    supported = all_cv[all_cv.model_name.astype(str).str.lower().isin(SUPPORTED_MODELS)]
    for row in supported.to_dict("records"):
        run_dir = Path(row["run_dir"])
        hp = json.loads((run_dir / "hparams.json").read_text())
        scores = read_scores(run_dir)
        context = pd.read_parquet(run_dir / "artifacts/cv/realised_depeg_context.parquet")
        context["timestamp"] = pd.to_datetime(context["timestamp"], utc=True)
        context = context.sort_values("timestamp").reset_index(drop=True)
        utc_times(context.timestamp)
        threshold = int(row["target_threshold"])
        signature = fold_signature(scores)
        if threshold in reference_folds and signature != reference_folds[threshold]:
            raise ValueError("Models must use identical validation/test timestamps")
        reference_folds[threshold] = signature
        events = depeg_event_starts(context, timestamp_col="timestamp", depeg_col="depeg_bps",
                                    threshold_bps=threshold, depeg_side="both", dynamic_threshold=False,
                                    reset_hours=float(hp.get("depeg_event_reset_hours", 24)))
        config = {"target_threshold": threshold, "alpha": float(row["alpha"]),
                  "model_name": str(row["model_name"])}
        outcomes, exposures, false_alerts, test_times = [], [], [], []
        for fold in sorted(scores["fold"].unique()):
            part = scores[scores.fold.eq(fold)]
            val = part[part.split.eq("validation")].sort_values("timestamp").reset_index(drop=True)
            test = part[part.split.eq("test")].sort_values("timestamp").reset_index(drop=True)
            if val.empty or test.empty:
                raise ValueError(f"Missing validation or test timeline in {run_dir}, fold {fold}")
            if val.timestamp.iloc[-1] + pd.Timedelta(hours=24) >= test.timestamp.iloc[0]:
                raise ValueError("Validation and test require 24 unused embargo hours")
            test_times.extend(test.timestamp)
            val.attrs["depeg_threshold"] = threshold
            val.attrs["event_reset_hours"] = float(hp.get("depeg_event_reset_hours", 24))
            val.attrs["selection_metadata"] = {**config, "fold": int(fold)}
            cutoff, vm = selector(val, val.probability.to_numpy(), events, budget,
                                  min_lead_hours, max_lead_hours, outcome_context=context)
            vm.update(hourly_metrics(val, val.probability.to_numpy(), cutoff, context, threshold,
                                     min_lead_hours, max_lead_hours, val.attrs["event_reset_hours"]))
            tm = score_events(test, test.probability.to_numpy(), cutoff, events,
                              min_lead_hours, max_lead_hours, outcome_context=context)
            tm.update(hourly_metrics(test, test.probability.to_numpy(), cutoff, context, threshold,
                                     min_lead_hours, max_lead_hours, val.attrs["event_reset_hours"]))
            current_events = tm.pop("_event_outcomes")
            current_alerts = tm.pop("_alert_outcomes")
            inputs = tm.pop("_bootstrap_inputs")
            outcomes.extend(current_events)
            exposures.extend(inputs["exposure_times"])
            false_alerts.extend(inputs["false_alert_times"])
            event_records.extend({**config, "fold": int(fold), **r} for r in current_events)
            alert_records.extend({**config, "fold": int(fold), **r} for r in current_alerts)
            folds.append({**config, "fold": int(fold), "cutoff": cutoff,
                          "validation_calendar_hours": (val.timestamp.iloc[-1]-val.timestamp.iloc[0]).total_seconds()/3600+1,
                          "test_calendar_hours": (test.timestamp.iloc[-1]-test.timestamp.iloc[0]).total_seconds()/3600+1,
                          **{f"validation_{k}": v for k, v in vm.items() if not k.startswith("_")},
                          **{f"test_{k}": v for k, v in tm.items()}})
        if pd.DatetimeIndex(test_times).duplicated().any():
            raise ValueError("Outer test folds overlap; pooled calendar burden would double-count time")
        event_times = tuple(r["event_time"] for r in sorted(outcomes, key=lambda r:r["event_time"]))
        if len(set(event_times)) != len(event_times):
            raise ValueError("An event was assigned to multiple outer test folds")
        if not event_times:
            raise ValueError(f"No eligible outer-test events in {run_dir}")
        if threshold in reference_events and event_times != reference_events[threshold]:
            raise ValueError("Configurations have different eligible events; compare on a common population")
        reference_events[threshold] = event_times
        pooled = pooled_summary(outcomes, pd.DatetimeIndex(sorted(exposures)),
                                pd.DatetimeIndex(sorted(false_alerts)), n_bootstrap=n_bootstrap)
        config_folds = [f for f in folds if all(f[k] == v for k, v in config.items())]
        for name in ("tp", "fp", "tn", "fn", "observed", "censored"):
            pooled[f"cv_hourly_{name}"] = sum(f[f"test_hourly_{name}"] for f in config_folds)
        positive = pooled["cv_hourly_tp"] + pooled["cv_hourly_fn"]
        negative = pooled["cv_hourly_fp"] + pooled["cv_hourly_tn"]
        pooled["cv_hourly_tpr"] = pooled["cv_hourly_tp"] / positive if positive else np.nan
        pooled["cv_hourly_fpr"] = pooled["cv_hourly_fp"] / negative if negative else np.nan
        pooled["cv_no_alert_rule_folds"] = sum(not np.isfinite(f["cutoff"]) for f in config_folds)
        summaries.append({**row, **pooled, "alert_cooldown_hours": ALERT_COOLDOWN_HOURS})
    fold_df, summary = pd.DataFrame(folds), pd.DataFrame(summaries)
    if summary.empty:
        raise ValueError("No supported model score artifacts were found")
    winners = [part.sort_values(["cv_event_recall", "cv_false_alerts_per_month", "model_name", "alpha"],
                               ascending=[False, True, True, True], kind="mergesort").iloc[0]
               for _, part in summary.groupby("target_threshold")]
    selected = pd.DataFrame(winners).sort_values("target_threshold").reset_index(drop=True)
    if detail_dir is not None:
        Path(detail_dir).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(event_records).to_csv(Path(detail_dir) / "cv_event_outcomes.csv", index=False)
        pd.DataFrame(alert_records, columns=["target_threshold", "alpha", "model_name", "fold", "alert_time", "status"]).to_csv(Path(detail_dir) / "cv_alert_outcomes.csv", index=False)
    return fold_df, summary, selected


def build_target(frame, context, threshold, min_lead_hours=2, max_lead_hours=24, reset_hours=24):
    """Event-onset target; censor current breaches, too-late rows and unknown futures."""
    ts = utc_times(frame.timestamp)
    ctx_ts = utc_times(context.timestamp)
    valid_prices = ctx_ts[np.isfinite(pd.to_numeric(context.depeg_bps, errors="coerce"))]
    onsets = depeg_event_starts(context, timestamp_col="timestamp", depeg_col="depeg_bps",
                                threshold_bps=threshold, depeg_side="both", dynamic_threshold=False,
                                reset_hours=reset_hours)
    observed = complete_windows(ts + HOUR, ts + pd.Timedelta(hours=max_lead_hours), valid_prices)
    labels = np.zeros(len(ts), dtype=float)
    late = np.zeros(len(ts), dtype=bool)
    for event in onsets:
        lead = (event - ts).total_seconds() / 3600
        labels[(lead >= min_lead_hours) & (lead <= max_lead_hours)] = 1
        late |= (lead > 0) & (lead < min_lead_hours)
    breached = np.abs(pd.to_numeric(frame.depeg_bps, errors="coerce").to_numpy()) >= threshold
    observed &= ~breached & ~late & np.isfinite(pd.to_numeric(frame.depeg_bps, errors="coerce"))
    reason = np.full(len(ts), "", dtype=object)
    reason[~observed] = "unobserved_future"
    reason[late] = "inside_minimum_lead"
    reason[breached] = "already_depegged"
    return labels, observed, reason


def set_targets(frame, context, threshold, min_lead_hours, max_lead_hours, reset_hours):
    result = frame.copy()
    labels, observed, reason = build_target(frame, context, threshold, min_lead_hours, max_lead_hours, reset_hours)
    result["target"], result["target_observed"], result["target_censor_reason"] = labels, observed, reason
    return result


def recreate_dataset(row, exp_dir, min_lead_hours=2, max_lead_hours=24):
    run_dir = Path(row["run_dir"])
    hp = json.loads((run_dir / "hparams.json").read_text())
    dataset_path = Path(hp["dataset_path"])
    if not dataset_path.is_absolute():
        candidates = [HERE / dataset_path, Path.cwd() / dataset_path, run_dir / dataset_path]
        dataset_path = next((candidate for candidate in candidates if candidate.is_file()), dataset_path)
    dataset = pd.read_parquet(dataset_path)
    if "timestamp" not in dataset:
        dataset["timestamp"] = dataset.index
    dataset["timestamp"] = pd.to_datetime(dataset.timestamp, utc=True)
    dataset = dataset.sort_values("timestamp").reset_index(drop=True)
    ts = utc_times(dataset.timestamp)
    # Preserve realized-price coverage before dropping incomplete feature rows.
    context = dataset[["timestamp", "depeg_bps"]].copy()
    series = pd.Series(dataset.depeg_bps.to_numpy(), index=ts)
    for lag in range(1, 8):
        dataset[f"depeg_bps_lag{lag}h"] = series.reindex(ts - lag * HOUR).to_numpy()
    feature_cols = [c for c in dataset if c not in {"timestamp", "target", "target_observed", "target_censor_reason"}]
    dataset = dataset.dropna(subset=feature_cols).reset_index(drop=True)
    dataset = set_targets(dataset, context, int(row["target_threshold"]), min_lead_hours,
                          max_lead_hours, float(hp.get("depeg_event_reset_hours", 24)))
    return dataset, hp, context


def chronological_holdout_start(dataset, row, hparams, embargo_hours=DEFAULT_HOLDOUT_EMBARGO_HOURS):
    """The holdout starts at cv_end; its embargo is taken BEFORE that date."""
    if embargo_hours < 0:
        raise ValueError("holdout embargo hours must be non-negative")
    timestamps = utc_times(dataset.timestamp)
    run_dir = Path(row["run_dir"])
    manifest_path = run_dir / "artifacts/cv/split_manifest.csv"
    cv_end = hparams.get("cv_end")
    manifest_end = None
    if manifest_path.is_file():
        manifest = pd.read_csv(manifest_path)
        if "test_end" in manifest and not manifest.empty:
            manifest_end = pd.to_datetime(manifest.test_end, utc=True).max()
    if cv_end is None and manifest_end is None:
        raise ValueError("A cv_end or CV split manifest must define the final holdout")
    boundary = pd.to_datetime(cv_end, utc=True) if cv_end is not None else manifest_end + HOUR
    if manifest_end is not None and manifest_end >= boundary:
        raise ValueError("CV test rows extend into the reserved final holdout")
    index = int(timestamps.searchsorted(boundary))
    if index <= 0 or index >= len(timestamps):
        raise ValueError("The configured holdout boundary leaves no development or test data")
    return index


def model_arguments(hparams):
    return SimpleNamespace(
        n_estimators=int(hparams.get("n_estimators", 800)),
        learning_rate=float(hparams.get("learning_rate", .01)),
        max_depth=int(hparams.get("max_depth", 6)), num_leaves=int(hparams.get("num_leaves", 31)),
        early_stopping_rounds=int(hparams.get("early_stopping_rounds", 200)), n_jobs=-1,
        rf_max_features=hparams.get("rf_max_features", "sqrt"), rf_early_stopping_step=25,
        foundation_device=hparams.get("foundation_device", "auto"),
        foundation_num_estimators=int(hparams.get("foundation_num_estimators", 8)),
        foundation_query_batch_rows=int(hparams.get("foundation_query_batch_rows", 512)),
        kumo_size=hparams.get("kumo_size", "small"),
        random_state=1233, alert_cooldown_hours=ALERT_COOLDOWN_HOURS,
    )


def train_holdout(row, dataset, hparams, holdout_embargo_hours=DEFAULT_HOLDOUT_EMBARGO_HOURS, *,
                  outcome_context=None, min_lead_hours=2, max_lead_hours=24):
    cv = model_helpers()
    threshold, alpha, model_name = int(row.target_threshold), float(row.alpha), str(row.model_name)
    if model_name.lower() not in SUPPORTED_MODELS:
        raise ValueError(f"Holdout selection uses an unsupported model, received {model_name}")
    feature_cols = [c for c in dataset.columns if c not in {"timestamp", "target", "target_observed", "target_censor_reason"}]
    holdout_start = chronological_holdout_start(dataset, row, hparams, holdout_embargo_hours)
    boundary = pd.to_datetime(hparams.get("cv_end", dataset.timestamp.iloc[holdout_start]), utc=True)
    dev_end = boundary - pd.Timedelta(hours=holdout_embargo_hours)
    dev = dataset[dataset.timestamp < dev_end].copy().reset_index(drop=True)
    test = dataset.iloc[holdout_start:].copy().reset_index(drop=True)
    if dev.empty:
        raise ValueError("No development rows remain before the holdout embargo")
    context = dataset if outcome_context is None else outcome_context
    dev_context = context[context.timestamp <= dev.timestamp.iloc[-1]]
    dev = set_targets(dev, dev_context, threshold, min_lead_hours, max_lead_hours,
                      float(hparams.get("depeg_event_reset_hours", 24)))
    val_n = max(1, int(len(dev) * float(hparams.get("cv_val_frac", .30))))
    val_start = dev.timestamp.iloc[len(dev) - val_n]
    val = dev[dev.timestamp >= val_start].copy()
    fit_timeline = dev[dev.timestamp < val_start - pd.Timedelta(hours=24)].copy()
    if fit_timeline.empty:
        raise ValueError("The 24-hour fitting/validation embargo leaves no fitting rows")
    fit_mask = fit_timeline.target_observed.to_numpy(bool)
    val_mask = val.target_observed.to_numpy(bool)
    test_mask = test.target_observed.to_numpy(bool)
    X_fit = fit_timeline.loc[fit_mask, feature_cols].copy()
    y_fit = fit_timeline.loc[fit_mask, "target"].astype(int)
    X_val = val[feature_cols].copy()
    y_val = val.loc[val_mask, "target"].astype(int)
    X_test = test[feature_cols].copy()
    y_test = test.loc[test_mask, "target"].astype(int)
    X_fit_s, X_val_s, X_test_s, _ = cv.apply_scaling(X_fit, X_val, X_test, hparams.get("scaler", "none"))
    pos_weight = float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))
    args = model_arguments(hparams)
    if y_fit.nunique() < 2:
        raise ValueError("The holdout fitting partition must contain both target classes")
    model = cv.build_model(model_name, args, pos_weight, use_early_stopping=y_val.nunique() > 1)
    model = cv.fit_model(model, model_name, args, X_fit_s, y_fit,
                         X_val=X_val_s.loc[val_mask].copy(), y_val=y_val)
    return model, dev, val, X_val_s, test, X_test_s, y_test, test_mask, feature_cols, args


def plot_roc_pr(holdout_data, outdir):
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for item in holdout_data:
        y, p = item["y"], item["probability"]
        name = f"{item['threshold']} bp: {item['model']} (α={item['alpha']:g})"
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, lw=1.8, label=f"{name} (AUC {roc_auc_score(y,p):.3f})")
        precision, recall, _ = precision_recall_curve(y, p)
        axes[1].plot(recall, precision, lw=1.8, label=f"{name} (AP {average_precision_score(y,p):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="0.6", lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Holdout ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Holdout precision–recall")
    for ax in axes: ax.grid(alpha=.22); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); save_figure(fig, outdir / "holdout_roc_pr")


def save_figure(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_event_recall(holdout_data, outdir):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(holdout_data))
    values = [d["event_metrics"]["recall"] for d in holdout_data]
    colors = plt.get_cmap("tab10")(np.arange(len(x)) % 10)
    bars = ax.bar(x, values, color=colors)
    ax.set_xticks(x, [f"{d['threshold']} bp\n{d['model']} α={d['alpha']:g}" for d in holdout_data])
    ax.set_ylim(0, 1); ax.set_ylabel("Holdout timely event recall")
    ax.set_title("Holdout event detection at validation-selected cutoff")
    ax.grid(axis="y", alpha=.22)
    for bar, d in zip(bars, holdout_data):
        m=d["event_metrics"]
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.025,
                f"{m['detected']}/{m['events']}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); save_figure(fig, outdir / "holdout_event_recall")



def latex_escape(value):
    return str(value).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_cv_latex(table, destination, threshold, min_lead_hours, max_lead_hours, budget,
                   *, policy="empirical", np_fpr=None, np_delta=.05, sample="cv"):
    """Publication layout requested by the paper; captions match the actual policy."""
    order = ["catboost", "xgboost", "lightgbm", "random_forest", "kumo_tabular", "tabicl_v2"]
    names = {"catboost": "Catboost", "xgboost": "XGBoost", "lightgbm": "LightGBM",
             "random_forest": "Random Forest", "kumo_tabular": "Kumo", "tabicl_v2": "TabICL-v2"}
    rows = table.copy()
    rows["_order"] = rows.model_name.map({name: i for i, name in enumerate(order)}).fillna(len(order))
    rows = rows.sort_values(["_order", "alpha"], kind="mergesort").reset_index(drop=True)
    specifications = [
        ("cv_event_recall", "cv_event_recall_ci_lower", "cv_event_recall_ci_upper", 3, True),
        ("cv_false_alerts_per_month", "cv_false_alerts_per_month_ci_lower", "cv_false_alerts_per_month_ci_upper", 2, False),
        ("cv_median_lead_hours", "cv_median_lead_hours_ci_lower", "cv_median_lead_hours_ci_upper", 1, True),
        ("cv_auc_mean", None, None, 3, True), ("cv_auprc_mean", None, None, 3, True),
        ("cv_brier_score_mean", None, None, 3, False)]
    styles = {}
    for key, low, high, digits, higher in specifications:
        values = pd.to_numeric(rows[key], errors="coerce").round(digits)
        for rank, value in enumerate(sorted(values.dropna().unique(), reverse=higher)[:2]):
            wrapper = r"\textcolor{red}{\textbf{%s}}" if rank == 0 else r"\textcolor{blue}{\underline{%s}}"
            for index in rows.index[values.eq(value)]:
                styles[(index, key)] = wrapper
    def cell(row, spec):
        key, low, high, digits, _ = spec
        value = row[key]
        if pd.isna(value):
            return "NA"
        text = f"{value:.{digits}f}"
        if low and pd.notna(row.get(low)) and pd.notna(row.get(high)):
            text += f" [{row[low]:.{digits}f}, {row[high]:.{digits}f}]"
        wrapper = styles.get((row.name, key))
        return wrapper % text if wrapper else text
    if policy.startswith("np_fpr_") or policy == "neyman_pearson":
        if np_fpr is None:
            np_fpr = float(policy.removeprefix("np_fpr_").replace("p", "."))
        selection = (fr"Probability cutoffs are selected on preceding validation data using the Neyman--Pearson order-statistic rule with hourly FPR limit $\eta={np_fpr:g}$ and nominal violation tolerance $\delta_{{\mathrm{{NP}}}}={np_delta:g}$. "
                     "The nominal NP assumptions are not guaranteed by temporal data. The hourly FPR limit is distinct from false-alert episodes per month. ")
    elif policy == "event_ucb":
        selection = f"Probability cutoffs maximize validation event recall subject to an upper false-alert-rate estimate not exceeding {budget:g} episodes per month. "
    else:
        selection = (f"Probability cutoffs are selected separately on each preceding validation fold to maximize event recall within {min_lead_hours:g}–{max_lead_hours:g} hours under a {budget:g}-false-alerts-per-month budget. ")
    if sample == "holdout":
        count = int(rows.cv_events_detected.iloc[0]) if "cv_events_detected" in rows else None
        total = int(rows.cv_eligible_events.iloc[0]) if "cv_eligible_events" in rows else None
        heading = f"Final holdout performance of the CV-selected model at the {threshold}-basis-point depeg threshold. "
        selection += "The specification is fixed by CV event recall, with lower false-alert burden breaking ties; no holdout outcome selects the model or cutoff. "
        if count is not None:
            selection += f"The selected model detects {count} of {total} eligible holdout events. "
    else:
        heading = f"Operational out-of-sample comparison at the {threshold}-basis-point depeg threshold. "
        selection += "Models are selected by event-level recall, with false alerts per month as the secondary criterion. "
    caption = (heading + selection +
               f"Timely warnings precede onset by {min_lead_hours:g}–{max_lead_hours:g} hours. "
               "Recall and pooled median-lead confidence intervals resample unique realized events; alert-burden intervals resample calendar blocks. "
               "Alert burden pools false-alert counts over fully observed follow-up hours. "
               f"Above-threshold crossings form one episode until {ALERT_COOLDOWN_HOURS:g} consecutive observed below-threshold hours reset it. "
               "The first alert requires the same observed quiet history; missing hours or scores break that history. "
               "All timely episodes are recognized, while each event contributes once to recall. AUC and AUPRC are secondary row-level ranking metrics. "
               "Best displayed values in each metric are highlighted in bold red and second-best distinct displayed values in underlined blue.")
    label = "holdout_model_results" if sample == "holdout" else "cv_model_alpha_results"
    if sample == "cv":
        caption += " Ranking metrics in this table are fold means; ROC plots additionally report pooled OOF AUC."
    lines = [r"\begin{table}[pos=!htbp,align=\centering]", r"\caption{" + caption + "}",
             r"\label{tab:" + label + f"_{threshold}" + "}", r"\resizebox{\textwidth}{!}{",
             r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Model & $\alpha$ & Event recall & False alerts/month & Median lead (h) & AUC & AUPRC & Brier score \\",
             r"\midrule"]
    for number, (model, group) in enumerate(rows.groupby("model_name", sort=False)):
        if number:
            lines.append(r"\midrule")
        for position, (_, row) in enumerate(group.iterrows()):
            model_cell = (fr"\multirow{{{len(group)}}}{{*}}{{{names.get(model, latex_escape(model))}}}" if position == 0 else "")
            cells = [model_cell, f"{row.alpha:g}", *[cell(row, spec) for spec in specifications]]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table}", ""]
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(destination).write_text("\n".join(lines))


def fit_fold_probabilities(fit, val, test, features, hp, context, threshold,
                           min_lead_hours, max_lead_hours):
    """Fit only on uncensored historical labels; return full-timeline scores."""
    cv = model_helpers()
    reset = float(hp.get("depeg_event_reset_hours", 24))
    fit_context = context[context.timestamp < val.timestamp.iloc[0]]
    val_context = context[context.timestamp <= val.timestamp.iloc[-1]]
    fit = set_targets(fit, fit_context, threshold, min_lead_hours, max_lead_hours, reset)
    val_labels = set_targets(val, val_context, threshold, min_lead_hours, max_lead_hours, reset)
    test_labels = set_targets(test, context[context.timestamp <= test.timestamp.iloc[-1]],
                              threshold, min_lead_hours, max_lead_hours, reset)
    fit_mask = fit.target_observed.to_numpy(bool)
    val_mask = val_labels.target_observed.to_numpy(bool)
    y_fit = fit.loc[fit_mask, "target"].astype(int)
    y_val = val_labels.loc[val_mask, "target"].astype(int)
    if y_fit.nunique() < 2:
        raise ValueError("The historical fitting partition must contain both target classes")
    X_fit, X_val, X_test, _ = cv.apply_scaling(fit.loc[fit_mask, features], val[features],
                                             test[features], hp.get("scaler", "none"))
    args = model_arguments(hp)
    early_stopping = y_val.nunique() > 1
    weight = float((y_fit == 0).sum() / (y_fit == 1).sum())
    model = cv.build_model(hp["model_name"], args, weight, use_early_stopping=early_stopping)
    model = cv.fit_model(model, hp["model_name"], args, X_fit, y_fit,
                         X_val=X_val.loc[val_mask], y_val=y_val)
    predictions = []
    for stage, values in [("validation", X_val), ("test", X_test)]:
        if hasattr(model, "set_prediction_stage"):
            model.set_prediction_stage(stage)
        predictions.append(np.asarray(model.predict_proba(values))[:, 1])
    # Diagnostics are computed on the supervised 2--24h target, independently
    # of operational alert grouping or how many timely episodes match an event.
    mask = test_labels.target_observed.to_numpy(bool)
    y = test_labels.loc[mask, "target"].to_numpy(int)
    prob = predictions[1][mask]
    metrics = cv.compute_fold_metrics(y, prob)
    return predictions[0], predictions[1], metrics


def run_ablations(selected, exp_dir, output_dir, min_lead_hours, max_lead_hours, enabled=True,
                  *, threshold_selector=None, policy="empirical", args=None, prediction_cache=None):
    """Fit every variant, then apply exactly the main report's scoring policy.

    Replay the saved validation/test timestamps, with a 24-hour fitting embargo.
    Only estimator construction, fitting and scaling use the model backend.
    No legacy utility selector is called. Refit the full-feature control too.
    """
    if not enabled:
        print("Economic ablations skipped by request.")
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_cache = {} if prediction_cache is None else prediction_cache
    cv = model_helpers()
    if not hasattr(cv, "economic_feature_blocks"):
        raise RuntimeError("Use the cv_model_comparison.py version accompanying the original script; economic_feature_blocks is missing")
    rows_out, curve_groups, curve_areas = [], {}, []
    for selected_row in selected.to_dict("records"):
        dataset, hp, context = recreate_dataset(selected_row, exp_dir, min_lead_hours, max_lead_hours)
        features = [c for c in dataset if c not in {"timestamp", "target", "target_observed", "target_censor_reason"}]
        blocks = cv.economic_feature_blocks(features)
        variants = [("full", "none", features)]
        price = [f for f in features if f == "depeg_bps" or f.startswith("depeg_bps_lag")]
        if price and len(price) < len(features):
            variants.append(("price_only_reference", "all_economic_blocks", price))
        for name, members in blocks.items():
            keep = [f for f in features if f not in set(members)]
            if keep:
                variants.append((f"without_{name}", name, keep))
        source_scores = read_scores(selected_row["run_dir"])
        reference_signature = fold_signature(source_scores)
        reference_events = None
        source_hp = {**hp, "model_name": selected_row["model_name"],
                     "alpha": selected_row["alpha"], "target_threshold": selected_row["target_threshold"],
                     "warning_window_hours": [min_lead_hours, max_lead_hours],
                     "alert_cooldown_hours": ALERT_COOLDOWN_HOURS}
        indexed = dataset.set_index("timestamp", drop=False)
        for feature_set, removed, columns in variants:
            run_name = f"report_{int(selected_row['target_threshold'])}_{selected_row['model_name']}_{feature_set}_alpha_{selected_row['alpha']:g}"
            run_dir = output_dir / "ablation_runs" / run_name
            artifact_dir = run_dir / "artifacts/cv"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "hparams.json").write_text(json.dumps(source_hp, default=str, indent=2))
            fold_rows, prediction_frames = [], []
            cache_key = (str(Path(selected_row["run_dir"]).resolve()), tuple(columns),
                         min_lead_hours, max_lead_hours)
            for fold in sorted(source_scores.fold.unique()):
                part = source_scores[source_scores.fold.eq(fold)]
                val_times = utc_times(part[part.split.eq("validation")].sort_values("timestamp").timestamp)
                test_times = utc_times(part[part.split.eq("test")].sort_values("timestamp").timestamp)
                missing = val_times.union(test_times).difference(indexed.index)
                if len(missing):
                    raise ValueError("Recreated dataset is missing original fold timestamps")
                val = indexed.loc[val_times].reset_index(drop=True)
                test = indexed.loc[test_times].reset_index(drop=True)
                fit = dataset[dataset.timestamp < val.timestamp.iloc[0] - pd.Timedelta(hours=24)].copy()
                fold_key = (cache_key, int(fold))
                if fold_key not in prediction_cache:
                    prediction_cache[fold_key] = fit_fold_probabilities(
                        fit, val, test, columns, source_hp, context,
                        int(selected_row["target_threshold"]), min_lead_hours, max_lead_hours)
                val_prob, test_prob, metrics = prediction_cache[fold_key]
                fold_rows.append({"fold": int(fold), **metrics,
                                  "fit_start": fit.timestamp.iloc[0], "fit_end": fit.timestamp.iloc[-1],
                                  "validation_start": val.timestamp.iloc[0], "validation_end": val.timestamp.iloc[-1],
                                  "test_start": test.timestamp.iloc[0], "test_end": test.timestamp.iloc[-1]})
                for split, frame, probability in [("validation", val, val_prob), ("test", test, test_prob)]:
                    prediction_frames.append(pd.DataFrame({"fold": int(fold), "split": split,
                                                           "timestamp": frame.timestamp,
                                                           "depeg_bps": frame.depeg_bps,
                                                           "probability": probability}))
            pd.concat(prediction_frames, ignore_index=True).to_parquet(artifact_dir / "operational_scores.parquet", index=False)
            pd.DataFrame(fold_rows).to_csv(artifact_dir / "fold_metrics.csv", index=False)
            if fold_signature(read_scores(run_dir)) != reference_signature:
                raise ValueError("Ablation validation/test timestamps differ from the source experiment")
            # Replace any feature-filtered context with the complete realized-price
            # timeline. This cannot leak labels: the scorer caps context per split.
            context.to_parquet(run_dir / "artifacts/cv/realised_depeg_context.parquet", index=False)
            fold_metrics = pd.read_csv(run_dir / "artifacts/cv/fold_metrics.csv")
            meta = {**selected_row, "run_dir": str(run_dir)}
            for metric, source in [("cv_auc_mean", "fold_auc"), ("cv_auprc_mean", "fold_auprc"),
                                   ("cv_brier_score_mean", "fold_brier_score"),
                                   ("cv_brier_skill_score_mean", "fold_brier_skill_score")]:
                meta[metric] = pd.to_numeric(fold_metrics[source], errors="coerce").mean() if source in fold_metrics else np.nan
            details = run_dir / "corrected_report"
            fold_report, report, _ = run_cv_budget(
                pd.DataFrame([meta]), exp_dir, min_lead_hours, max_lead_hours,
                detail_dir=details, threshold_selector=threshold_selector,
                budget=args.false_alert_budget if args is not None else BUDGET,
                n_bootstrap=args.summary_bootstrap if args is not None else 1000)
            fold_report["threshold_policy"] = policy
            report["threshold_policy"] = policy
            fold_report.to_csv(details / "fold_results.csv", index=False)
            report.to_csv(details / "model_comparison.csv", index=False)
            result = report.iloc[0]
            event_times = tuple(pd.read_csv(details / "cv_event_outcomes.csv").event_time)
            if reference_events is not None and event_times != reference_events:
                raise ValueError("Ablations must share exactly the same eligible realized events")
            reference_events = event_times
            rows_out.append({**result.to_dict(), "threshold_policy": policy,
                             "threshold_bp": int(selected_row["target_threshold"]),
                             "alpha": float(selected_row["alpha"]), "model": selected_row["model_name"],
                             "feature_set": feature_set, "removed_block": removed, "n_features": len(columns),
                             "ROC AUC": result.cv_auc_mean, "AUPRC": result.cv_auprc_mean,
                             "Event recall": result.cv_event_recall,
                             "Events detected": int(result.cv_events_detected),
                             "Eligible events": int(result.cv_eligible_events),
                             "Median lead hours": result.cv_median_lead_hours,
                             "False alerts/month": result.cv_false_alerts_per_month})
            if args is not None and not args.skip_plots:
                from report_curves import save_curve_record
                scores = read_scores(run_dir)
                parts = [(part[part.split.eq("validation")].sort_values("timestamp").reset_index(drop=True),
                          part[part.split.eq("test")].sort_values("timestamp").reset_index(drop=True))
                         for _, part in scores.groupby("fold", sort=True)]
                key = (int(selected_row["target_threshold"]), selected_row["model_name"],
                       float(selected_row["alpha"]))
                plot_dir = output_dir / "plots" / f"{key[0]}bp_{key[1]}_alpha_{key[2]:g}"
                record = save_curve_record(parts, context,
                    {**meta, "depeg_event_reset_hours": source_hp.get("depeg_event_reset_hours", 24)},
                    args, plot_dir / feature_set)
                record["label"] = feature_set
                curve_groups.setdefault(key, []).append(record)
                curve_areas.append({"threshold_policy": policy, "target_threshold": key[0],
                                    "model_name": key[1], "alpha": key[2],
                                    "feature_set": feature_set, **record["areas"]})
    frame = pd.DataFrame(rows_out)
    for threshold, group in frame.groupby("threshold_bp"):
        model = str(group.model.iloc[0])
        destination = output_dir / f"economic_ablations_{int(threshold)}bp_{model}.tex"
        columns = ["feature_set", "removed_block", "n_features", "ROC AUC", "AUPRC",
                   "Event recall", "Events detected", "Eligible events", "Median lead hours", "False alerts/month"]
        group[columns].to_latex(destination, index=False, escape=True, float_format="%.3f",
                       caption=f"Feature ablations under the {latex_escape(policy)} threshold policy. The full-feature control is refitted on the same fold windows as the ablations.",
                       label=f"tab:ablation_{int(threshold)}_{model}")
        group.to_csv(destination.with_suffix(".csv"), index=False)
    frame.to_csv(output_dir / "economic_ablations_all.csv", index=False)
    if curve_groups:
        from report_curves import plot_curve_records
        for (threshold, model, alpha), records in curve_groups.items():
            plot_dir = output_dir / "plots" / f"{threshold}bp_{model}_alpha_{alpha:g}"
            plot_curve_records(records, plot_dir, threshold,
                               f"CV ablations: {model} (α={alpha:g})", args.nproc_partial_max)
        pd.DataFrame(curve_areas).to_csv(output_dir / "plots/curve_areas.csv", index=False)
    return frame


def run_threshold_comparison(all_cv, exp_dir, output, args):
    """Compare policies exclusively from stored validation/test probabilities."""
    from threshold_selection import choose_neyman_pearson, choose_event_ucb
    methods = args.threshold_methods or ["neyman_pearson"]
    policies = []
    candidate_trace = []
    if "empirical" in methods:
        def empirical(*values, **keywords):
            cutoff, metrics = choose_cutoff(*values, **keywords, grid_size=args.threshold_grid_size)
            metrics.update({"threshold_method": "empirical", "selection_status": "selected"})
            return cutoff, metrics
        policies.append(("empirical", empirical))
    if "neyman_pearson" in methods:
        for target in args.np_fpr:
            def np_selector(*values, _target=target, **keywords):
                return choose_neyman_pearson(*values, **keywords, fpr=_target,
                                             delta=args.np_delta,
                                             negative_spacing_hours=args.np_negative_spacing_hours,
                                             reset_hours=values[0].attrs["event_reset_hours"])
            policies.append((f"np_fpr_{target:g}".replace(".", "p"), np_selector))
    if "event_ucb" in methods:
        def ucb_selector(*values, **keywords):
            candidates = []
            result = choose_event_ucb(*values, **keywords, confidence=args.ucb_confidence,
                                       block_hours=args.ucb_block_hours,
                                       n_bootstrap=args.ucb_bootstrap,
                                       grid_size=args.threshold_grid_size, candidate_rows=candidates)
            candidate_trace.extend({**values[0].attrs["selection_metadata"], **row} for row in candidates)
            return result
        policies.append(("event_ucb", ucb_selector))
    comparison, diagnostics, holdout_results = [], [], []
    ablation_cache = {}
    run_ablation_models = getattr(args, "run_ablations", False) and not args.skip_ablations
    if not run_ablation_models:
        print("CV feature ablations disabled; use --run_ablations to fit and report feature variants.", flush=True)
    root = output / "threshold_comparison"
    root.mkdir(parents=True, exist_ok=True)
    primary_policy = next((name for name, _ in policies if name.startswith("np_fpr_")), policies[0][0])
    for policy, selector in policies:
        print(f"Scoring cached probabilities: {policy}", flush=True)
        destination = root / policy
        folds, summary, selected = run_cv_budget(
            all_cv, exp_dir, args.min_lead_hours, args.max_lead_hours, detail_dir=destination,
            n_bootstrap=args.summary_bootstrap, threshold_selector=selector, budget=args.false_alert_budget)
        folds["threshold_policy"], summary["threshold_policy"] = policy, policy
        selected["threshold_policy"] = policy
        folds.to_csv(destination / "fold_results.csv", index=False)
        summary.to_csv(destination / "model_comparison.csv", index=False)
        selected.to_csv(destination / "selected_models.csv", index=False)
        for threshold, part in summary.groupby("target_threshold"):
            filename = f"cv_results_{int(threshold)}bp.tex"
            write_cv_latex(part, destination / filename, int(threshold), args.min_lead_hours,
                           args.max_lead_hours, args.false_alert_budget, policy=policy, np_delta=args.np_delta)
            if policy == primary_policy:
                write_cv_latex(part, output / filename, int(threshold), args.min_lead_hours,
                               args.max_lead_hours, args.false_alert_budget, policy=policy, np_delta=args.np_delta)
        if policy == primary_policy:
            selected.to_csv(output / "selected_models.csv", index=False)
        if not args.skip_plots:
            from report_curves import make_cv_plots
            make_cv_plots(summary, destination / "plots", args)
        if run_ablation_models:
            print(f"Running CV feature ablations: {policy}", flush=True)
            run_ablations(selected, exp_dir, destination / "ablations",
                          args.min_lead_hours, args.max_lead_hours,
                          threshold_selector=selector, policy=policy, args=args,
                          prediction_cache=ablation_cache)
        if not args.cv_only and args.holdout_mode != "skip":
            from report_holdout import evaluate_selected_holdout
            # Keep holdout fitting separate from CV and optional ablation refits.
            holdout = evaluate_selected_holdout(selected, selector, policy, exp_dir, output,
                                                args, SimpleNamespace(**globals()))
            holdout_results.append(holdout)
            if policy == primary_policy:
                holdout.to_csv(output / "holdout_results.csv", index=False)
        comparison.append(summary)
        diagnostics.append(folds)
        if policy == "empirical":
            # Preserve legacy CV report filenames for existing notebook consumers.
            folds.to_csv(output / "five_alert_budget_fold_results.csv", index=False)
            summary.to_csv(output / "five_alert_budget_model_comparison.csv", index=False)
            selected.to_csv(output / "selected_models_five_alert_budget.csv", index=False)
            for threshold, part in summary.groupby("target_threshold"):
                write_cv_latex(part, output / f"cv_results_{int(threshold)}bp_five_alert_budget.tex",
                               int(threshold), args.min_lead_hours, args.max_lead_hours, args.false_alert_budget)
    pd.concat(comparison, ignore_index=True).to_csv(root / "all_methods_model_comparison.csv", index=False)
    pd.concat(diagnostics, ignore_index=True).to_csv(root / "all_methods_fold_results.csv", index=False)
    if holdout_results:
        pd.concat(holdout_results, ignore_index=True).to_csv(root / "all_methods_holdout_results.csv", index=False)
    if candidate_trace:
        pd.DataFrame(candidate_trace).to_csv(root / "event_ucb_validation_candidates.csv", index=False)
    settings = {
        "base_cv_cached_probabilities_only": True,
        "cached_cv_probabilities_only": not run_ablation_models,
        "cv_model_fit_or_predict_calls": run_ablation_models,
        "cached_probabilities_only": not run_ablation_models and (args.cv_only or args.holdout_mode in {"cached", "skip"}),
        "model_fit_or_predict_calls": "ablation_refits_and_configured_holdout" if run_ablation_models else ("selected_holdout_only_if_cache_missing" if not args.cv_only and args.holdout_mode == "auto" else args.holdout_mode == "fit"),
        "ablations_enabled": run_ablation_models,
        "primary_policy": primary_policy, "holdout_mode": "skip" if args.cv_only else args.holdout_mode,
        "minimum_lead_hours": args.min_lead_hours, "maximum_lead_hours": args.max_lead_hours,
        "threshold_methods": methods, "hourly_np_fpr_targets": args.np_fpr,
        "np_violation_tolerance": args.np_delta, "np_negative_spacing_hours": args.np_negative_spacing_hours,
        "np_assumptions": "iid negatives, stable class-conditional scores, and calibration independent of fitted scorer; not verified",
        "np_early_stopping_caveat": "Saved validation may have selected training iterations, violating independent calibration",
        "event_false_alert_budget_per_month": args.false_alert_budget,
        "event_ucb_confidence": args.ucb_confidence, "event_ucb_block_hours": args.ucb_block_hours,
        "event_ucb_bootstrap_draws": args.ucb_bootstrap,
        "event_ucb_scope": "Pointwise calendar-block bootstrap upper percentile with positive Poisson count floor; no postselection or distribution-shift guarantee",
        "model_selection": "Pooled event recall, then false-alert burden; hourly metrics are also reported",
        "hourly_population": "Full-future-observed rows excluding current breaches and the final hour before onset",
        "event_population": "Unique event onsets with full 2--24h score coverage; full timelines retained for episodes",
        "alert_cooldown_hours": ALERT_COOLDOWN_HOURS,
        "nproc_definition": "nominal_hourly_FPR_limit_vs_heldout_TPR_at_validation_NP_cutoffs; empirical block-bootstrap bands",
        "nproc_area_definition": "trapezoidal full integral on [0,1]; partial integral on [0,partial_max]; normalized partial divided by partial_max",
        "nproc_partial_max": args.nproc_partial_max,
    }
    (root / "policy_settings.json").write_text(json.dumps(settings, indent=2))
    print(f"Saved cached-CV threshold comparison under {root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--log_dir", type=Path, default=HERE / "lightning_logs")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--cv_only", action="store_true", help="Rescore saved CV predictions without training holdout or ablation models")
    parser.add_argument("--fit_models", action="store_true", help="Compatibility alias: enable ablations and refit holdout; default policy is empirical")
    parser.add_argument("--run_ablations", action="store_true", help="Refit feature variants for each policy's CV-selected model and save policy-specific ablation reports and NP-ROC plots")
    parser.add_argument("--threshold_methods", nargs="+", choices=["empirical", "neyman_pearson", "event_ucb"],
                        help="Cached-CV threshold policies; default: neyman_pearson")
    parser.add_argument("--depeg_thresholds", type=int, nargs="+", default=[10, 15], help="Depeg thresholds to report separately")
    parser.add_argument("--holdout_mode", choices=["auto", "cached", "fit", "skip"], default="auto",
                        help="Selected-model holdout: auto reuses scores or fits once; cached forbids prediction; skip disables holdout")
    parser.add_argument("--holdout_cache_dir", type=Path, default=None)
    parser.add_argument("--skip_plots", action="store_true", help="Skip ROC/NP-ROC plots and curve-area diagnostics")
    parser.add_argument("--nproc_partial_max", type=float, default=.1)
    parser.add_argument("--nproc_grid_size", type=int, default=51)
    parser.add_argument("--curve_bootstrap", type=int, default=500, help="Calendar-block draws for descriptive NP-ROC TPR bands")
    parser.add_argument("--np_fpr", nargs="+", type=float, default=[.01], help="Hourly conditional FPR target(s), not alert episodes/month")
    parser.add_argument("--np_delta", type=float, default=.05, help="Nominal NP violation probability under iid assumptions")
    parser.add_argument("--np_negative_spacing_hours", type=float, default=1, help="Optional spacing of negative calibration rows; not a proof of independence")
    parser.add_argument("--false_alert_budget", type=float, default=BUDGET)
    parser.add_argument("--ucb_confidence", type=float, default=.95)
    parser.add_argument("--ucb_block_hours", type=float, default=168)
    parser.add_argument("--ucb_bootstrap", type=int, default=2000)
    parser.add_argument("--summary_bootstrap", type=int, default=1000)
    parser.add_argument("--threshold_grid_size", type=int, default=201)
    parser.add_argument("--skip_ablations", action="store_true", help="Do not run economic block CV ablations")
    parser.add_argument("--holdout_embargo_hours", type=float, default=DEFAULT_HOLDOUT_EMBARGO_HOURS,
                        help="Unused gap BEFORE cv_end, preserving the configured holdout start (default: 24 hours)")
    parser.add_argument("--min_lead_hours", type=float, default=DEFAULT_MIN_LEAD_HOURS,
                        help=f"Minimum warning lead time used for event scoring (default: {DEFAULT_MIN_LEAD_HOURS:g})")
    parser.add_argument("--max_lead_hours", type=float, default=DEFAULT_MAX_LEAD_HOURS,
                        help="Maximum warning lead time used for event scoring (default: 24)")
    args = parser.parse_args()
    if args.fit_models:
        args.run_ablations = not args.skip_ablations
        args.holdout_mode = "fit"
        args.threshold_methods = args.threshold_methods or ["empirical"]
    if args.cv_only and (args.run_ablations or args.fit_models):
        parser.error("--cv_only cannot be combined with --run_ablations or --fit_models")
    if args.run_ablations and args.skip_ablations:
        parser.error("Choose either --run_ablations or --skip_ablations")
    if any(not 0 < value < 1 for value in args.np_fpr) or not 0 < args.np_delta < 1:
        parser.error("NP FPR and delta must be strictly between zero and one")
    if args.np_negative_spacing_hours < 1 or args.false_alert_budget < 0:
        parser.error("NP spacing must be >=1 hour and alert budget nonnegative")
    if not .5 < args.ucb_confidence < 1 or args.ucb_block_hours <= 0 or args.ucb_bootstrap < 100:
        parser.error("Require .5<ucb_confidence<1, positive blocks and >=100 UCB bootstrap draws")
    if args.summary_bootstrap < 1 or args.threshold_grid_size < 2:
        parser.error("Require positive summary bootstrap count and >=2 threshold candidates")
    if not 0 < args.nproc_partial_max <= 1 or args.nproc_grid_size < 5 or args.curve_bootstrap < 0:
        parser.error("Require 0<nproc_partial_max<=1, >=5 NP-ROC grid points and nonnegative curve bootstrap draws")
    if args.min_lead_hours <= 0 or args.max_lead_hours <= args.min_lead_hours:
        parser.error("Require 0 < min_lead_hours < max_lead_hours")
    if any(float(x) != int(x) for x in (args.min_lead_hours, args.max_lead_hours)):
        parser.error("Lead times must be whole hours")
    if args.holdout_embargo_hours < 0:
        parser.error("--holdout_embargo_hours must be non-negative")
    log_root = args.log_dir if args.log_dir.is_absolute() else (HERE / args.log_dir).resolve()
    output = args.output_dir or (log_root / args.experiment_name / "cv_results_report")
    output = output.resolve(); output.mkdir(parents=True, exist_ok=True)
    all_cv, exp_dir = discover_runs(log_root, args.experiment_name)
    all_cv = all_cv[all_cv.target_threshold.isin(args.depeg_thresholds)].copy()
    if all_cv.empty:
        parser.error("No cached CV runs match the requested depeg thresholds")
    missing = sorted(set(args.depeg_thresholds)-set(all_cv.target_threshold))
    if missing:
        print(f"No cached runs for depeg threshold(s) {missing}; no results will be fabricated.", flush=True)
    run_threshold_comparison(all_cv, exp_dir, output, args)


if __name__ == "__main__":
    main()
