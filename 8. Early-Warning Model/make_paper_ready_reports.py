"""Create publication-ready threshold and decision-sensitivity reports.

All inputs are outer chronological-fold reports created by cv_model_comparison.py.
The model/alpha is reselected separately for every realised-depeg threshold and
false-alert budget, so a figure never implies that one operating point was
chosen using another operating point's test performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SELECTION_METRICS = [
    "cv_event_utility_score_mean",
    "cv_timely_event_recall_mean",
    "cv_false_alerts_per_month_mean",
    "cv_event_utility_score_std",
]


def _read_latest(files, keys: list[str]) -> pd.DataFrame:
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["source_mtime"] = path.stat().st_mtime
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No non-empty report files were found.")
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("source_mtime")
        .drop_duplicates(subset=keys, keep="last")
        .reset_index(drop=True)
    )


def load_reports(log_dir: Path, experiment_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    experiment_dir = log_dir / experiment_name
    primary = _read_latest(
        experiment_dir.glob("*_experiment_summary*/artifacts/comparison/model_comparison_summary.csv"),
        ["target_threshold", "alpha", "model_name"],
    )
    decisions = _read_latest(
        experiment_dir.glob("*/artifacts/cv/utility_by_false_alert_budget_summary.csv"),
        ["target_threshold", "alpha", "model_name", "false_alert_budget_per_month"],
    )
    required = {"target_threshold", "alpha", "model_name", *SELECTION_METRICS}
    for name, frame in [("primary summaries", primary), ("budget summaries", decisions)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} are missing columns: {sorted(missing)}")
    primary["target_threshold"] = pd.to_numeric(primary["target_threshold"], errors="coerce")
    decisions["target_threshold"] = pd.to_numeric(decisions["target_threshold"], errors="coerce")
    decisions["false_alert_budget_per_month"] = pd.to_numeric(
        decisions["false_alert_budget_per_month"], errors="coerce"
    )
    return primary, decisions


def select_operating_points(reports: pd.DataFrame, group_columns: list[str], tolerance: float) -> pd.DataFrame:
    winners = []
    for group_key, candidates in reports.groupby(group_columns, sort=True, dropna=False):
        candidates = candidates.dropna(subset=["cv_event_utility_score_mean"]).copy()
        if candidates.empty:
            continue
        leader = candidates["cv_event_utility_score_mean"].max()
        comparable = candidates[
            candidates["cv_event_utility_score_mean"] >= leader - tolerance
        ].copy()
        comparable["_recall"] = comparable["cv_timely_event_recall_mean"].fillna(-np.inf)
        comparable["_false_alerts"] = comparable["cv_false_alerts_per_month_mean"].fillna(np.inf)
        comparable["_stability"] = comparable["cv_event_utility_score_std"].fillna(np.inf)
        selected = comparable.sort_values(
            ["_recall", "_false_alerts", "_stability", "cv_event_utility_score_mean"],
            ascending=[False, True, True, False],
        ).iloc[0].to_dict()
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        selected.update(dict(zip(group_columns, group_key)))
        selected["utility_tolerance"] = tolerance
        winners.append(selected)
    if not winners:
        raise RuntimeError("No threshold/budget combination had a finite outer-fold utility.")
    return pd.DataFrame(winners).sort_values(group_columns).reset_index(drop=True)


def _ci_label(row: pd.Series, metric: str, digits: int = 3) -> str:
    estimate = row.get(f"cv_{metric}_mean")
    low = row.get(f"cv_{metric}_ci_lower")
    high = row.get(f"cv_{metric}_ci_upper")
    if pd.isna(estimate):
        return "NA"
    if pd.notna(low) and pd.notna(high):
        return f"{estimate:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"
    return f"{estimate:.{digits}f}"


def format_table(table: pd.DataFrame) -> pd.DataFrame:
    output = table.copy()
    output["selected_specification"] = output.apply(
        lambda row: f"{row['model_name']} (α={row['alpha']:g})", axis=1
    )
    output["event_utility_95ci"] = output.apply(
        _ci_label, axis=1, metric="event_utility_score"
    )
    output["timely_recall_95ci"] = output.apply(
        _ci_label, axis=1, metric="timely_event_recall"
    )
    output["false_alerts_per_month_95ci"] = output.apply(
        _ci_label, axis=1, metric="false_alerts_per_month", digits=2
    )
    output["median_lead_hours_95ci"] = output.apply(
        _ci_label, axis=1, metric="median_lead_hours", digits=1
    )
    columns = [
        "target_threshold", "false_alert_budget_per_month", "selected_specification",
        "event_utility_95ci", "timely_recall_95ci", "false_alerts_per_month_95ci",
        "median_lead_hours_95ci",
    ]
    return output[[column for column in columns if column in output.columns]]


def _error_values(table: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    point = table[f"cv_{metric}_mean"].to_numpy(dtype=float)
    low = pd.to_numeric(table.get(f"cv_{metric}_ci_lower"), errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(table.get(f"cv_{metric}_ci_upper"), errors="coerce").to_numpy(dtype=float)
    if np.isnan(low).all() or np.isnan(high).all():
        std = pd.to_numeric(table[f"cv_{metric}_std"], errors="coerce").fillna(0.0).to_numpy()
        return std, std
    return np.maximum(point - low, 0.0), np.maximum(high - point, 0.0)


def save_figure(fig, destination: Path) -> None:
    fig.savefig(destination.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sensitivity(winners: pd.DataFrame, destination: Path) -> None:
    winners = winners.sort_values("target_threshold")
    x = winners["target_threshold"].to_numpy()
    panels = [
        ("event_utility_score", "Event utility"),
        ("timely_event_recall", "Timely event recall"),
        ("false_alerts_per_month", "False-alert episodes/month"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    for axis, (metric, label) in zip(axes, panels):
        y = winners[f"cv_{metric}_mean"].to_numpy(dtype=float)
        lower, upper = _error_values(winners, metric)
        axis.errorbar(x, y, yerr=np.vstack([lower, upper]), marker="o", capsize=4, linewidth=1.6)
        axis.set_xlabel("Realised-depeg threshold (bps)")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    for _, row in winners.iterrows():
        axes[0].annotate(
            f"{row['model_name']}\nα={row['alpha']:g}",
            (row["target_threshold"], row["cv_event_utility_score_mean"]),
            xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8,
        )
    fig.suptitle("Sensitivity to the realised-depeg definition", y=1.02)
    fig.tight_layout()
    save_figure(fig, destination)


def plot_decision_sensitivity(winners: pd.DataFrame, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for threshold, subset in winners.groupby("target_threshold", sort=True):
        subset = subset.sort_values("false_alert_budget_per_month")
        y = subset["cv_event_utility_score_mean"].to_numpy(dtype=float)
        lower, upper = _error_values(subset, "event_utility_score")
        ax.errorbar(
            subset["false_alert_budget_per_month"], y,
            yerr=np.vstack([lower, upper]), marker="o", capsize=4, linewidth=1.6,
            label=f"{threshold:g} bps",
        )
    ax.axhline(0, color="0.45", linewidth=0.8)
    ax.set_xlabel("False-alert budget (episodes/month)")
    ax.set_ylabel("Selected model’s outer-fold event utility")
    ax.set_title("Utility decision sensitivity")
    ax.legend(title="Depeg threshold", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, destination)


def write_table(table: pd.DataFrame, destination: Path) -> None:
    table.to_csv(destination.with_suffix(".csv"), index=False)
    destination.with_suffix(".tex").write_text(
        table.to_latex(index=False, escape=True, na_rep="NA", longtable=len(table) > 12)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create threshold and utility-decision sensitivity reports")
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--log_dir", default="lightning_logs")
    parser.add_argument("--false_alert_budget_per_month", type=float, default=2.0)
    parser.add_argument("--utility_tolerance", type=float, default=0.01)
    args = parser.parse_args()

    output_dir = Path(args.log_dir) / args.experiment_name / "paper_ready"
    output_dir.mkdir(parents=True, exist_ok=True)
    _, decisions = load_reports(Path(args.log_dir), args.experiment_name)
    decision_winners = select_operating_points(
        decisions, ["target_threshold", "false_alert_budget_per_month"], args.utility_tolerance
    )
    primary_budget_winners = decision_winners.loc[
        np.isclose(decision_winners["false_alert_budget_per_month"], args.false_alert_budget_per_month)
    ].copy()
    if primary_budget_winners.empty:
        raise ValueError(
            f"No results found for false-alert budget {args.false_alert_budget_per_month:g}/month."
        )

    write_table(format_table(primary_budget_winners), output_dir / "table_threshold_sensitivity")
    write_table(format_table(decision_winners), output_dir / "table_utility_decision_sensitivity")
    decision_winners.to_csv(output_dir / "selected_model_by_threshold_and_budget_raw.csv", index=False)
    plot_threshold_sensitivity(primary_budget_winners, output_dir / "figure_threshold_sensitivity")
    plot_decision_sensitivity(decision_winners, output_dir / "figure_utility_decision_sensitivity")
    print(f"Saved paper-ready figures and tables to: {output_dir}")


if __name__ == "__main__":
    main()
