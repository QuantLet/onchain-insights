"""Plot which model/alpha wins as the false-alert budget changes.

Consumes the fold-level operational reports written by ``cv_model_comparison.py``.
The plotted utility is computed on outer chronological test folds; thresholds were
chosen on their preceding validation periods, so this is not an in-sample
threshold sweep.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_budget_reports(base_dir: Path, experiment_name: str) -> pd.DataFrame:
    exp_dir = base_dir / experiment_name
    reports = sorted(exp_dir.glob("*/artifacts/cv/utility_by_false_alert_budget.csv"))
    if not reports:
        raise FileNotFoundError(
            f"No utility-by-budget reports found under {exp_dir}. Run cv_model_comparison.py first."
        )

    frames = []
    for report in reports:
        frame = pd.read_csv(report)
        required = {
            "fold", "model_name", "alpha", "false_alert_budget_per_month",
            "test_event_utility_score", "test_timely_event_recall",
            "test_false_alerts_per_month",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{report} is missing columns: {sorted(missing)}")
        frame["source_mtime"] = report.stat().st_mtime
        frames.append(frame)

    reports_df = pd.concat(frames, ignore_index=True)
    # Keep one rerun per model/alpha/fold/budget, rather than accidentally
    # treating repeated command invocations as additional CV evidence.
    return (
        reports_df.sort_values("source_mtime")
        .drop_duplicates(
            subset=["model_name", "alpha", "fold", "false_alert_budget_per_month"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def select_by_budget(reports: pd.DataFrame, utility_tolerance: float) -> pd.DataFrame:
    aggregate = (
        reports.groupby(["false_alert_budget_per_month", "model_name", "alpha"], as_index=False)
        .agg(
            mean_outer_test_utility=("test_event_utility_score", "mean"),
            std_outer_test_utility=("test_event_utility_score", "std"),
            mean_timely_event_recall=("test_timely_event_recall", "mean"),
            mean_false_alerts_per_month=("test_false_alerts_per_month", "mean"),
            outer_folds=("fold", "nunique"),
        )
    )
    winners = []
    for budget, candidates in aggregate.groupby("false_alert_budget_per_month", sort=True):
        leader = candidates["mean_outer_test_utility"].max()
        comparable = candidates[
            candidates["mean_outer_test_utility"] >= leader - utility_tolerance
        ].copy()
        winner = comparable.sort_values(
            [
                "mean_timely_event_recall",
                "mean_false_alerts_per_month",
                "std_outer_test_utility",
                "mean_outer_test_utility",
            ],
            ascending=[False, True, True, False],
            na_position="last",
        ).iloc[0].to_dict()
        winner["selection_policy"] = (
            "mean outer-fold event utility; within utility tolerance: timely recall, "
            "false-alert burden, then utility stability"
        )
        winner["utility_tolerance"] = utility_tolerance
        winners.append(winner)
    return pd.DataFrame(winners).sort_values("false_alert_budget_per_month")


def plot_selection(winners: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = winners["false_alert_budget_per_month"].to_numpy()
    y = winners["mean_outer_test_utility"].to_numpy()
    err = winners["std_outer_test_utility"].fillna(0.0).to_numpy()
    ax.errorbar(x, y, yerr=err, marker="o", linewidth=2, capsize=4, color="darkorange")
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    for _, row in winners.iterrows():
        ax.annotate(
            f"{row['model_name']}\nα={row['alpha']:g}",
            (row["false_alert_budget_per_month"], row["mean_outer_test_utility"]),
            xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9,
        )
    ax.set_xlabel("False-alert budget (episodes/month)")
    ax.set_ylabel("Mean outer-fold event utility")
    ax.set_title("Selected model evolves with the operational false-alert budget")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot operational model selection by false-alert budget")
    parser.add_argument("--experiment_name", default="cv_model_comparison")
    parser.add_argument("--log_dir", default="lightning_logs")
    parser.add_argument("--utility_tolerance", type=float, default=0.01)
    args = parser.parse_args()

    output_dir = Path(args.log_dir) / args.experiment_name / "plots_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = load_budget_reports(Path(args.log_dir), args.experiment_name)
    winners = select_by_budget(reports, args.utility_tolerance)
    winners.to_csv(output_dir / "selection_by_false_alert_budget.csv", index=False)
    plot_selection(winners, output_dir / "selection_by_false_alert_budget.png")
    print(winners.to_string(index=False))
    print(f"Saved {output_dir / 'selection_by_false_alert_budget.png'}")


if __name__ == "__main__":
    main()
