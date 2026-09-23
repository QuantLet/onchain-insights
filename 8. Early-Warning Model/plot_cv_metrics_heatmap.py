from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
BASE_DIR = Path("lightning_logs")



# ------------------------------------------------------------
# Load experiment summary CSVs produced by the CV script
# ------------------------------------------------------------
def load_cv_summaries(base_dir: Path, experiment_name: str) -> pd.DataFrame:
    exp_dir = base_dir / experiment_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    summary_files = list(
        exp_dir.glob("*_experiment_summary*/artifacts/comparison/model_comparison_summary.csv")
    )

    if not summary_files:
        raise FileNotFoundError(
            f"No model_comparison_summary.csv files found under {exp_dir}"
        )

    dfs = []
    for fp in summary_files:
        try:
            df = pd.read_csv(fp)
            if len(df) == 0:
                continue

            df["source_file"] = str(fp)
            df["source_mtime"] = fp.stat().st_mtime
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {fp}: {e}")

    if not dfs:
        raise ValueError("No readable summary CSVs found.")

    all_df = pd.concat(dfs, ignore_index=True)

    # Normalize threshold and alpha before deduplicating reruns.
    if "alpha" not in all_df.columns:
        raise ValueError("Expected column 'alpha' not found in summary CSVs.")
    if "model_name" not in all_df.columns:
        raise ValueError("Expected column 'model_name' not found in summary CSVs.")

    all_df["alpha"] = pd.to_numeric(all_df["alpha"], errors="coerce")
    if "target_threshold" not in all_df.columns:
        raise ValueError("Expected column 'target_threshold' not found in summary CSVs.")
    all_df["target_threshold"] = pd.to_numeric(all_df["target_threshold"], errors="coerce")

    # If you have reruns, keep the latest result per event definition/model/alpha.
    all_df = (
        all_df.sort_values("source_mtime")
              .drop_duplicates(subset=["target_threshold", "alpha", "model_name"], keep="last")
              .reset_index(drop=True)
    )

    return all_df


# ------------------------------------------------------------
# Heatmap helper
# ------------------------------------------------------------
def build_pivot(df: pd.DataFrame, value_col: str, model_order=None) -> pd.DataFrame:
    pivot = df.pivot_table(
        values=value_col,
        index="model_name",
        columns="alpha",
        aggfunc="mean",
    )

    pivot = pivot.sort_index(axis=1)

    if model_order is not None:
        keep = [m for m in model_order if m in pivot.index]
        pivot = pivot.loc[keep]
    else:
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    return pivot


def plot_single_heatmap(pivot_df: pd.DataFrame, title: str, cbar_label: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(
        pivot_df.values,
        cmap="coolwarm",
        aspect="auto",
        interpolation="bilinear"
    )

    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_xticklabels([f"{x:.2f}" for x in pivot_df.columns])
    ax.set_yticklabels(pivot_df.index)

    ax.set_xlabel("Alpha (α)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Model", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, rotation=270, labelpad=20, fontsize=12, fontweight="bold")

    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            value = pivot_df.values[i, j]
            if not np.isnan(value):
                ax.text(
                    j, i, f"{value:.3f}",
                    ha="center", va="center",
                    color="black", fontsize=11, fontweight="bold"
                )

    ax.set_xticks(np.arange(len(pivot_df.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(pivot_df.index)) - 0.5, minor=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)


def plot_three_heatmaps(
    pivot_auc: pd.DataFrame,
    pivot_auprc: pd.DataFrame,
    pivot_utility: pd.DataFrame,
    output_path: Path
):
    fig, axes = plt.subplots(1, 3, figsize=(22, 8), constrained_layout=True)

    heatmaps = [
        (pivot_auc, "Mean CV AUC", "AUC"),
        (pivot_auprc, "Mean CV AP", "Average precision"),
        (pivot_utility, "Mean OOS event utility", "Utility"),
    ]

    for ax, (pivot_df, title, cbar_label) in zip(axes, heatmaps):
        im = ax.imshow(
            pivot_df.values,
            cmap="coolwarm",
            aspect="auto",
            interpolation="bilinear"
        )

        ax.set_xticks(np.arange(len(pivot_df.columns)))
        ax.set_yticks(np.arange(len(pivot_df.index)))
        ax.set_xticklabels([f"{x:.2f}" for x in pivot_df.columns], rotation=45, ha="right")
        ax.set_yticklabels(pivot_df.index)

        ax.set_xlabel("Alpha (α)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Model", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=16)

        for i in range(len(pivot_df.index)):
            for j in range(len(pivot_df.columns)):
                value = pivot_df.values[i, j]
                if not np.isnan(value):
                    ax.text(
                        j, i, f"{value:.3f}",
                        ha="center", va="center",
                        color="black", fontsize=10, fontweight="bold"
                    )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, rotation=270, labelpad=16, fontsize=11, fontweight="bold")

    plt.savefig(output_path, dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)


def render_threshold_heatmaps(summary_df: pd.DataFrame, output_dir: Path, threshold: float) -> None:
    """Render one alpha/model grid per realised-depeg definition.

    Combining different target thresholds in a single heatmap would average
    different classification tasks and is therefore intentionally avoided.
    """
    model_order = (
        summary_df.groupby("model_name")["cv_event_utility_score_mean"]
        .mean().sort_values(ascending=False).index.tolist()
    )
    pivot_auc = build_pivot(summary_df, "cv_auc_mean", model_order=model_order)
    pivot_auprc = build_pivot(summary_df, "cv_auprc_mean", model_order=model_order)
    pivot_utility = build_pivot(summary_df, "cv_event_utility_score_mean", model_order=model_order)
    label = f"{threshold:g} bps"
    plot_single_heatmap(pivot_auc, f"Mean CV AUC by model and alpha ({label})", "AUC", output_dir / "heatmap_cv_auc.png")
    plot_single_heatmap(pivot_auprc, f"Mean CV AP by model and alpha ({label})", "Average precision", output_dir / "heatmap_cv_auprc.png")
    plot_single_heatmap(pivot_utility, f"Mean OOS event utility by model and alpha ({label})", "Utility", output_dir / "heatmap_cv_event_utility.png")
    plot_three_heatmaps(
        pivot_auc, pivot_auprc, pivot_utility, output_dir / "heatmap_cv_metrics_combined.png"
    )
    pivot_auc.to_csv(output_dir / "pivot_cv_auc.csv")
    pivot_auprc.to_csv(output_dir / "pivot_cv_auprc.csv")
    pivot_utility.to_csv(output_dir / "pivot_cv_event_utility.csv")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot CV metrics heatmaps for model comparison")
    parser.add_argument("--experiment_name", default="cv_model_comparison")
    args = parser.parse_args()
    output_dir = BASE_DIR / args.experiment_name / "plots_summary"
    summary_df = load_cv_summaries(BASE_DIR, args.experiment_name)
    required_cols = {
        "model_name", "alpha", "target_threshold", "cv_auc_mean", "cv_auprc_mean",
        "cv_event_utility_score_mean",
    }
    missing = required_cols - set(summary_df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in summary data: {sorted(missing)}")

    for threshold, threshold_df in summary_df.groupby("target_threshold", sort=True):
        threshold_dir = output_dir / f"threshold_{float(threshold):g}bps"
        threshold_dir.mkdir(parents=True, exist_ok=True)
        render_threshold_heatmaps(threshold_df, threshold_dir, float(threshold))
    print(f"Saved threshold-specific plots and pivot tables to: {output_dir}")
