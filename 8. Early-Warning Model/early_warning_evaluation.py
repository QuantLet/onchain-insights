"""Operational evaluation for depeg early-warning models.

The supervised label may remain "a depeg within H hours", but a warning system is
deployed through alerts rather than row classifications.  These helpers therefore
score a probability threshold as event-level, timely warnings and count false
*alert episodes* after a refractory period.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


HOURS_PER_MONTH = 24.0 * 30.4375


def _as_utc_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True)


def depeg_event_starts(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    depeg_col: str,
    threshold_bps: float,
    depeg_side: str,
    dynamic_threshold: bool,
) -> pd.DatetimeIndex:
    """Return the first timestamp of each contiguous realised depeg episode.

    Dynamic events reproduce the historical rolling-quantile rule used to build
    the classification label.  The function deliberately uses realised prices
    only for post-hoc evaluation; they are never model inputs at alert time.
    """
    if frame.empty:
        return pd.DatetimeIndex([], tz="UTC")
    if depeg_col not in frame:
        raise KeyError(f"{depeg_col!r} is required for event-level evaluation")

    ordered = frame[[timestamp_col, depeg_col]].copy().sort_values(timestamp_col)
    ts = _as_utc_timestamp(ordered[timestamp_col])
    x = pd.to_numeric(ordered[depeg_col], errors="coerce")

    if depeg_side == "up":
        hard = x >= threshold_bps
    elif depeg_side == "down":
        hard = x <= -threshold_bps
    elif depeg_side == "both":
        hard = x.abs() >= threshold_bps
    else:
        raise ValueError(f"Unsupported depeg_side: {depeg_side}")

    event_mask = hard.fillna(False)
    if dynamic_threshold:
        upper = x.rolling(30 * 24, min_periods=1).quantile(0.9975)
        lower = x.rolling(30 * 24, min_periods=1).quantile(0.0025)
        eligible = ts >= (ts.iloc[0] + pd.Timedelta(days=30))
        if depeg_side == "up":
            dynamic = x > upper
        elif depeg_side == "down":
            dynamic = x < lower
        else:
            dynamic = (x > upper) | (x < lower)
        event_mask = event_mask | (dynamic & eligible).fillna(False)

    starts = event_mask & ~event_mask.shift(fill_value=False)
    return pd.DatetimeIndex(ts.loc[starts])


def alert_episode_times(
    timestamps: Iterable[pd.Timestamp],
    probabilities: Iterable[float],
    threshold: float,
    cooldown_hours: float,
) -> pd.DatetimeIndex:
    """Collapse above-threshold rows into alert episodes with a cooldown."""
    ts = _as_utc_timestamp(pd.Series(timestamps))
    prob = np.asarray(probabilities, dtype=float)
    cooldown = pd.Timedelta(hours=cooldown_hours)
    next_allowed = None
    episodes = []
    for time, score in zip(ts, prob):
        if score >= threshold and (next_allowed is None or time >= next_allowed):
            episodes.append(time)
            next_allowed = time + cooldown
    # ``DatetimeIndex(list_of_timestamps)`` can infer a tz-naive dtype under
    # some pandas versions.  The realised event timestamps are explicitly UTC,
    # so force the same representation for episode/window comparisons.
    return pd.DatetimeIndex(pd.to_datetime(episodes, utc=True))


def lead_time_utility(
    lead_hours: float,
    min_lead_hours: float,
    target_lead_hours: float,
    min_lead_utility: float,
) -> float:
    """Zero before the valid window; rise linearly to one at the target lead."""
    if lead_hours < min_lead_hours:
        return 0.0
    if target_lead_hours <= min_lead_hours:
        return 1.0
    progress = min(1.0, (lead_hours - min_lead_hours) / (target_lead_hours - min_lead_hours))
    return float(min_lead_utility + (1.0 - min_lead_utility) * progress)


def evaluate_early_warning(
    frame: pd.DataFrame,
    probabilities: Iterable[float],
    threshold: float,
    *,
    timestamp_col: str,
    depeg_col: str,
    threshold_bps: float,
    depeg_side: str,
    dynamic_threshold: bool,
    min_lead_hours: float,
    max_lead_hours: float,
    target_lead_hours: float,
    min_lead_utility: float,
    cooldown_hours: float,
    false_alert_cost: float,
    event_context_frame: pd.DataFrame | None = None,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Evaluate one fixed threshold without using labels to choose it.

    Boundary-incomplete windows are excluded: an eligible event must have its
    full warning window inside the evaluated frame, and the final
    ``min_lead_hours`` cannot start a new scored alert episode.  This avoids
    crediting or penalising alerts whose outcome lies outside the fold.
    """
    if max_lead_hours < min_lead_hours:
        raise ValueError("max_lead_hours must be >= min_lead_hours")
    if frame.empty:
        return _empty_metrics(threshold), _empty_bootstrap_inputs()

    ordered = frame.copy().sort_values(timestamp_col).reset_index(drop=True)
    ts = _as_utc_timestamp(ordered[timestamp_col])
    prob = np.asarray(probabilities, dtype=float)
    if len(prob) != len(ordered):
        raise ValueError("probabilities and frame must contain the same number of rows")

    score_start = ts.iloc[0]
    score_end = ts.iloc[-1] - pd.Timedelta(hours=min_lead_hours)
    if score_end < score_start:
        return _empty_metrics(threshold), _empty_bootstrap_inputs()

    # The dynamic event rule has a trailing 30-day quantile.  Supply historical
    # context without expanding the scoring interval, so its state at a fold
    # boundary matches the state it would have had in deployment.
    event_context = ordered if event_context_frame is None else event_context_frame.copy()
    event_starts = depeg_event_starts(
        event_context,
        timestamp_col=timestamp_col,
        depeg_col=depeg_col,
        threshold_bps=threshold_bps,
        depeg_side=depeg_side,
        dynamic_threshold=dynamic_threshold,
    )
    eligible_events = event_starts[
        (event_starts - pd.Timedelta(hours=max_lead_hours) >= score_start)
        & (event_starts <= score_end)
    ]
    scored_rows = (ts >= score_start) & (ts <= score_end)
    alert_times = alert_episode_times(
        ts.loc[scored_rows], prob[scored_rows.to_numpy()], threshold, cooldown_hours
    )

    event_utilities = []
    detected = []
    lead_hours = []
    timely_window_alerts = set()
    for event_time in eligible_events:
        window_start = event_time - pd.Timedelta(hours=max_lead_hours)
        window_end = event_time - pd.Timedelta(hours=min_lead_hours)
        in_window = alert_times[(alert_times >= window_start) & (alert_times <= window_end)]
        if len(in_window):
            timely_window_alerts.update(in_window.tolist())
            first_alert = in_window[0]
            lead = float((event_time - first_alert).total_seconds() / 3600.0)
            lead_hours.append(lead)
            detected.append(1.0)
            event_utilities.append(
                lead_time_utility(lead, min_lead_hours, target_lead_hours, min_lead_utility)
            )
        else:
            detected.append(0.0)
            event_utilities.append(0.0)

    false_alert_times = [t for t in alert_times if t not in timely_window_alerts]
    duration_hours = max((score_end - score_start).total_seconds() / 3600.0 + 1.0, 1.0)
    fa_per_month = len(false_alert_times) / duration_hours * HOURS_PER_MONTH
    event_utility = float(np.mean(event_utilities)) if event_utilities else np.nan
    utility = event_utility - false_alert_cost * fa_per_month if event_utilities else np.nan
    precision = len(timely_window_alerts) / len(alert_times) if len(alert_times) else np.nan

    metrics = {
        "alert_threshold": float(threshold),
        "n_events": int(len(eligible_events)),
        "events_detected": int(sum(detected)),
        "timely_event_recall": float(np.mean(detected)) if detected else np.nan,
        "median_lead_hours": float(np.median(lead_hours)) if lead_hours else np.nan,
        "lead_hours_iqr_low": float(np.quantile(lead_hours, 0.25)) if lead_hours else np.nan,
        "lead_hours_iqr_high": float(np.quantile(lead_hours, 0.75)) if lead_hours else np.nan,
        "alert_episodes": int(len(alert_times)),
        "false_alert_episodes": int(len(false_alert_times)),
        "false_alerts_per_month": float(fa_per_month),
        "alert_episode_precision": float(precision) if not np.isnan(precision) else np.nan,
        "event_utility": event_utility,
        "event_utility_score": utility,
        "evaluation_hours": float(duration_hours),
    }
    bootstrap_inputs = {
        "event_utilities": np.asarray(event_utilities, dtype=float),
        "detected": np.asarray(detected, dtype=float),
        "lead_hours": np.asarray(lead_hours, dtype=float),
        # Keep the timezone. Casting to numpy datetime64 silently strips UTC and
        # later causes naive/aware comparisons in the block bootstrap.
        "false_alert_times": pd.DatetimeIndex(false_alert_times),
        "score_start": score_start,
        "duration_hours": np.asarray(duration_hours),
    }
    return metrics, bootstrap_inputs


def choose_threshold_by_utility(
    frame: pd.DataFrame,
    probabilities: Iterable[float],
    false_alert_budget_per_month: float,
    threshold_grid_size: int,
    **evaluation_kwargs,
) -> Tuple[float, Dict[str, float]]:
    """Select a threshold on validation data under a pre-specified FA budget."""
    scores = np.asarray(probabilities, dtype=float)
    finite = scores[np.isfinite(scores)]
    if len(finite) == 0:
        raise ValueError("No finite validation probabilities available for threshold selection")
    if threshold_grid_size < 2:
        raise ValueError("threshold_grid_size must be at least 2")

    if len(np.unique(finite)) <= threshold_grid_size:
        thresholds = np.unique(finite)
    else:
        thresholds = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, threshold_grid_size)))
    # A no-alert option prevents a model from being forced to violate the budget.
    thresholds = np.unique(np.r_[thresholds, np.nextafter(float(finite.max()), np.inf)])

    candidates = []
    for threshold in thresholds:
        metrics, _ = evaluate_early_warning(frame, scores, float(threshold), **evaluation_kwargs)
        if metrics["false_alerts_per_month"] <= false_alert_budget_per_month + 1e-12:
            candidates.append(metrics)
    if not candidates:
        # The no-alert candidate should always be feasible, but retain a safe fallback.
        metrics, _ = evaluate_early_warning(frame, scores, float(thresholds[-1]), **evaluation_kwargs)
        return float(thresholds[-1]), metrics

    def sort_key(metrics: Dict[str, float]):
        utility = metrics["event_utility_score"]
        recall = metrics["timely_event_recall"]
        return (
            -np.inf if np.isnan(utility) else utility,
            -np.inf if np.isnan(recall) else recall,
            -metrics["false_alerts_per_month"],
            metrics["alert_threshold"],
        )

    winner = max(candidates, key=sort_key)
    return float(winner["alert_threshold"]), winner


def event_block_bootstrap_ci(
    bootstrap_inputs: Dict[str, np.ndarray],
    *,
    false_alert_cost: float,
    block_hours: float,
    n_bootstrap: int,
    random_state: int,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Paired event/block bootstrap confidence intervals.

    Detected-event contributions are resampled at the event level, while false
    alert episodes are resampled as contiguous calendar blocks.  Resampling rows
    would understate uncertainty because hourly observations are serially
    dependent.
    """
    event_utilities = bootstrap_inputs["event_utilities"]
    detected = bootstrap_inputs["detected"]
    lead_hours = bootstrap_inputs["lead_hours"]
    duration_hours = float(bootstrap_inputs["duration_hours"])
    false_times = pd.to_datetime(bootstrap_inputs["false_alert_times"], utc=True)
    score_start = pd.Timestamp(bootstrap_inputs["score_start"])
    if score_start.tzinfo is None:
        score_start = score_start.tz_localize("UTC")
    else:
        score_start = score_start.tz_convert("UTC")

    if len(event_utilities) == 0 or n_bootstrap <= 0:
        return {}, {}
    if block_hours <= 0:
        raise ValueError("block_hours must be positive")

    n_blocks = max(1, int(np.ceil(duration_hours / block_hours)))
    block_duration = np.full(n_blocks, block_hours, dtype=float)
    block_duration[-1] = duration_hours - block_hours * (n_blocks - 1)
    block_duration[-1] = max(block_duration[-1], 1e-12)
    false_by_block = np.zeros(n_blocks, dtype=float)
    if len(false_times):
        offsets = (false_times - score_start).total_seconds() / 3600.0
        indices = np.clip((offsets // block_hours).astype(int), 0, n_blocks - 1)
        false_by_block = np.bincount(indices, minlength=n_blocks).astype(float)

    rng = np.random.default_rng(random_state)
    samples = {
        "event_utility_score": np.empty(n_bootstrap),
        "timely_event_recall": np.empty(n_bootstrap),
        "false_alerts_per_month": np.empty(n_bootstrap),
        "median_lead_hours": np.full(n_bootstrap, np.nan),
    }
    n_events = len(event_utilities)
    for draw in range(n_bootstrap):
        event_idx = rng.integers(0, n_events, size=n_events)
        block_idx = rng.integers(0, n_blocks, size=n_blocks)
        fa_rate = (
            false_by_block[block_idx].sum() / block_duration[block_idx].sum() * HOURS_PER_MONTH
        )
        samples["false_alerts_per_month"][draw] = fa_rate
        samples["timely_event_recall"][draw] = detected[event_idx].mean()
        samples["event_utility_score"][draw] = event_utilities[event_idx].mean() - false_alert_cost * fa_rate
        detected_leads = lead_hours[rng.integers(0, len(lead_hours), size=len(lead_hours))] if len(lead_hours) else []
        if len(detected_leads):
            samples["median_lead_hours"][draw] = np.median(detected_leads)

    ci = {}
    for name, values in samples.items():
        finite = values[np.isfinite(values)]
        if len(finite):
            ci[f"{name}_ci_lower"] = float(np.quantile(finite, 0.025))
            ci[f"{name}_ci_upper"] = float(np.quantile(finite, 0.975))
    return ci, samples


def _empty_metrics(threshold: float) -> Dict[str, float]:
    return {
        "alert_threshold": float(threshold), "n_events": 0, "events_detected": 0,
        "timely_event_recall": np.nan, "median_lead_hours": np.nan,
        "lead_hours_iqr_low": np.nan, "lead_hours_iqr_high": np.nan,
        "alert_episodes": 0, "false_alert_episodes": 0,
        "false_alerts_per_month": np.nan, "alert_episode_precision": np.nan,
        "event_utility": np.nan, "event_utility_score": np.nan,
        "evaluation_hours": 0.0,
    }


def _empty_bootstrap_inputs() -> Dict[str, np.ndarray]:
    return {
        "event_utilities": np.asarray([], dtype=float),
        "detected": np.asarray([], dtype=float),
        "lead_hours": np.asarray([], dtype=float),
        "false_alert_times": pd.DatetimeIndex([], tz="UTC"),
        "score_start": pd.Timestamp("1970-01-01", tz="UTC"),
        "duration_hours": np.asarray(0.0),
    }
