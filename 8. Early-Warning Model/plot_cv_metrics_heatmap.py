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

    # Normalize alpha
    if "alpha" not in all_df.columns:
        raise ValueError("Expected column 'alpha' not found in summary CSVs.")
    if "model_name" not in all_df.columns:
        raise ValueError("Expected column 'model_name' not found in summary CSVs.")

    all_df["alpha"] = pd.to_numeric(all_df["alpha"], errors="coerce")

    # If you have reruns, keep the latest result per (alpha, model_name)
    all_df = (
        all_df.sort_values("source_mtime")
              .drop_duplicates(subset=["alpha", "model_name"], keep="last")
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
    pivot_lift: pd.DataFrame,
    output_path: Path
):
    fig, axes = plt.subplots(1, 3, figsize=(22, 8), constrained_layout=True)

    heatmaps = [
        (pivot_auc, "Mean CV AUC", "AUC"),
        (pivot_auprc, "Mean CV AUPRC", "AUPRC"),
        (pivot_lift, "Mean CV Lift @ Best Threshold", "Lift"),
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


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot CV metrics heatmaps for model comparison")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="cv_model_comparison",
        help="Name of the experiment to load CV summaries from")
    args = parser.parse_args()
    EXPERIMENT_NAME = args.experiment_name
    OUTPUT_DIR = BASE_DIR / EXPERIMENT_NAME / "plots_summary"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    summary_df = load_cv_summaries(BASE_DIR, EXPERIMENT_NAME)

    required_cols = [
        "model_name",
        "alpha",
        "cv_auc_mean",
        "cv_auprc_mean",
        "cv_lift_at_best_threshold_mean",
    ]
    missing = [c for c in required_cols if c not in summary_df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns in summary data: {missing}\n"
            f"Available columns: {list(summary_df.columns)}"
        )

    # Sort models by overall mean CV AUC for consistent ordering across all heatmaps
    model_order = (
        summary_df.groupby("model_name")["cv_auc_mean"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    pivot_auc = build_pivot(summary_df, "cv_auc_mean", model_order=model_order)
    pivot_auprc = build_pivot(summary_df, "cv_auprc_mean", model_order=model_order)
    pivot_lift = build_pivot(summary_df, "cv_lift_at_best_threshold_mean", model_order=model_order)

    # Save individual heatmaps
    plot_single_heatmap(
        pivot_auc,
        title="Mean 5-Fold CV AUC by Model and Alpha",
        cbar_label="AUC",
        output_path=OUTPUT_DIR / "heatmap_cv_auc.png"
    )

    plot_single_heatmap(
        pivot_auprc,
        title="Mean 5-Fold CV AUPRC by Model and Alpha",
        cbar_label="AUPRC",
        output_path=OUTPUT_DIR / "heatmap_cv_auprc.png"
    )

    plot_single_heatmap(
        pivot_lift,
        title="Mean 5-Fold CV Lift @ Best Threshold by Model and Alpha",
        cbar_label="Lift",
        output_path=OUTPUT_DIR / "heatmap_cv_lift_at_best_threshold.png"
    )

    # Save combined figure
    plot_three_heatmaps(
        pivot_auc,
        pivot_auprc,
        pivot_lift,
        output_path=OUTPUT_DIR / "heatmap_cv_metrics_combined.png"
    )

    # Save pivot tables too
    pivot_auc.to_csv(OUTPUT_DIR / "pivot_cv_auc.csv")
    pivot_auprc.to_csv(OUTPUT_DIR / "pivot_cv_auprc.csv")
    pivot_lift.to_csv(OUTPUT_DIR / "pivot_cv_lift_at_best_threshold.csv")

    print("\nPivot Table: Mean CV AUC by Model and Alpha")
    print("=" * 80)
    print(pivot_auc.to_string())

    print("\nPivot Table: Mean CV AUPRC by Model and Alpha")
    print("=" * 80)
    print(pivot_auprc.to_string())

    print("\nPivot Table: Mean CV Lift @ Best Threshold by Model and Alpha")
    print("=" * 80)
    print(pivot_lift.to_string())

    print(f"\nSaved plots and pivot tables to: {OUTPUT_DIR}")