"""
Analyze the train/validation/test parquet files produced by preprocess.py:
which chromosome went to which split, split size ratios, class balance,
and coverage / methylation-ratio distributions.

Usage (no arguments needed):

    python preprocessing/analyze_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd

from config.project_config import DATASET_DIR, RESULTS_DIR

TABLES_DIR = RESULTS_DIR / "tables" / "dataset_analysis"
FIGURES_DIR = RESULTS_DIR / "figures" / "dataset_analysis"

SPLIT_NAMES = ("train", "validation", "test")
SPLIT_COLORS = {"train": "#1f77b4", "validation": "#ff7f0e", "test": "#2ca02c"}


def load_splits() -> dict[str, pd.DataFrame]:
    splits = {}

    for split_name in SPLIT_NAMES:
        path = DATASET_DIR / f"{split_name}.parquet"

        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist. Run preprocessing/preprocess.py first.")

        splits[split_name] = pd.read_parquet(path)

    return splits


def save_table(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def chromosome_split_table(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Which chromosome ended up in which split, and how many CpGs it carried."""
    rows = []

    for split_name, dataframe in splits.items():
        counts = dataframe["chrom"].value_counts()

        for chrom, count in counts.items():
            rows.append({"chrom": chrom, "split": split_name, "cpg_count": int(count)})

    table = pd.DataFrame(rows)

    total = table["cpg_count"].sum()
    table["fraction_of_total"] = table["cpg_count"] / total

    # natural chromosome order: chr1..chr22, chrX
    def chrom_sort_key(chrom: str) -> tuple[int, str]:
        suffix = chrom.removeprefix("chr")
        return (int(suffix), "") if suffix.isdigit() else (999, suffix)

    table = table.sort_values("chrom", key=lambda col: col.map(chrom_sort_key)).reset_index(drop=True)

    return table


def split_size_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    total_rows = sum(len(dataframe) for dataframe in splits.values())

    rows = []

    for split_name, dataframe in splits.items():
        rows.append(
            {
                "split": split_name,
                "row_count": len(dataframe),
                "chromosome_count": dataframe["chrom"].nunique(),
                "fraction_of_total": len(dataframe) / total_rows,
                "unmethylated_count": int(dataframe["label"].eq(0).sum()),
                "methylated_count": int(dataframe["label"].eq(1).sum()),
                "methylated_fraction": float(dataframe["label"].mean()),
                "mean_coverage": float(dataframe["coverage"].mean()),
                "median_coverage": float(dataframe["coverage"].median()),
                "mean_consensus_methylation_ratio": float(dataframe["consensus_methylation_ratio"].mean()),
            }
        )

    return pd.DataFrame(rows)


def plot_chromosome_split_assignment(chrom_table: pd.DataFrame) -> None:
    pivot = chrom_table.pivot_table(
        index="chrom", columns="split", values="cpg_count", fill_value=0
    ).reindex(columns=SPLIT_NAMES, fill_value=0)

    pivot = pivot.loc[chrom_table["chrom"].drop_duplicates()]

    fig, ax = plt.subplots(figsize=(13, 6))

    bottom = None
    for split_name in SPLIT_NAMES:
        ax.bar(
            pivot.index,
            pivot[split_name],
            bottom=bottom,
            label=split_name,
            color=SPLIT_COLORS[split_name],
        )
        bottom = pivot[split_name] if bottom is None else bottom + pivot[split_name]

    ax.set_title("CpG Count per Chromosome, Colored by Split Assignment")
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("Number of CpGs")
    ax.tick_params(axis="x", rotation=60)
    ax.legend()
    save_figure(fig, FIGURES_DIR / "chromosome_split_assignment")


def plot_split_sizes(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    bars = ax.bar(summary["split"], summary["row_count"], color=[SPLIT_COLORS[s] for s in summary["split"]])

    for bar, fraction in zip(bars, summary["fraction_of_total"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{fraction * 100:.2f}%",
            ha="center",
            va="bottom",
        )

    ax.set_title("Chromosome-Disjoint Dataset Split Sizes")
    ax.set_xlabel("Split")
    ax.set_ylabel("Number of CpGs")
    save_figure(fig, FIGURES_DIR / "split_sizes")


def plot_class_distribution(summary: pd.DataFrame) -> None:
    unmethylated_pct = (1 - summary["methylated_fraction"]) * 100
    methylated_pct = summary["methylated_fraction"] * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.bar(summary["split"], unmethylated_pct, label="Unmethylated")
    ax.bar(summary["split"], methylated_pct, bottom=unmethylated_pct, label="Methylated")

    ax.set_title("Class Distribution Across Dataset Splits")
    ax.set_xlabel("Split")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    save_figure(fig, FIGURES_DIR / "class_distribution_by_split")


def plot_distributions(splits: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for split_name, dataframe in splits.items():
        ax.hist(
            dataframe["coverage"].clip(upper=dataframe["coverage"].quantile(0.99)),
            bins=60,
            histtype="step",
            linewidth=1.8,
            density=True,
            label=split_name,
            color=SPLIT_COLORS[split_name],
        )
    ax.set_title("Consensus Coverage Distribution by Split (99th pct clipped)")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Density")
    ax.legend()
    save_figure(fig, FIGURES_DIR / "coverage_distribution_by_split")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for split_name, dataframe in splits.items():
        ax.hist(
            dataframe["consensus_methylation_ratio"],
            bins=60,
            histtype="step",
            linewidth=1.8,
            density=True,
            label=split_name,
            color=SPLIT_COLORS[split_name],
        )
    ax.set_title("Consensus Methylation-Ratio Distribution by Split")
    ax.set_xlabel("Consensus Methylation Ratio")
    ax.set_ylabel("Density")
    ax.legend()
    save_figure(fig, FIGURES_DIR / "methylation_ratio_distribution_by_split")


def main() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )

    print("=" * 70)
    print("Analyzing preprocessed dataset (train/validation/test)")
    print("=" * 70)

    splits = load_splits()

    for split_name, dataframe in splits.items():
        print(f"{split_name}: {len(dataframe):,} rows, {dataframe['chrom'].nunique()} chromosomes")

    chrom_table = chromosome_split_table(splits)
    save_table(chrom_table, TABLES_DIR / "chromosome_split_assignment.csv")

    summary = split_size_summary(splits)
    save_table(summary, TABLES_DIR / "split_summary.csv")

    plot_chromosome_split_assignment(chrom_table)
    plot_split_sizes(summary)
    plot_class_distribution(summary)
    plot_distributions(splits)

    print("\nAnalysis completed.")
    print(f"Tables:  {TABLES_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print()
    print("Chromosome -> split assignment:")
    print(chrom_table.to_string(index=False))
    print()
    print("Split summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
