"""
Exploratory analysis of the raw HepG2 WGBS bedMethyl replicates.

Reads both replicates in full, validates and profiles every column, checks
inter-replicate concordance, and saves every table/figure under
results/tables/wgbs_analysis and results/figures/wgbs_analysis.

Nothing here decides how the two replicates get merged for training — that
decision is made in preprocess.py, informed by what this script reports
(coverage distribution, replicate correlation, chromosome list, etc).

Usage (no arguments needed):

    python preprocessing/analyze_data.py
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.project_config import RESULTS_DIR, WGBS_REPLICATE_ACCESSIONS, WGBS_REPLICATE_PATHS

TABLES_DIR = RESULTS_DIR / "tables" / "wgbs_analysis"
FIGURES_DIR = RESULTS_DIR / "figures" / "wgbs_analysis"

# Standard ENCODE bedMethyl columns for WGBS CpG methylation
# (https://www.encodeproject.org/data-standards/wgbs/). If a downloaded file
# does not match this 11-column layout, the assertion below fails loudly
# instead of silently mis-parsing the file.
BEDMETHYL_COLUMNS = [
    "chrom",
    "chrom_start",
    "chrom_end",
    "name",
    "score",
    "strand",
    "thick_start",
    "thick_end",
    "item_rgb",
    "coverage",
    "percent_methylated",
]


# ============================================================
# Loading and validation
# ============================================================

def load_bedmethyl(path: Path) -> pd.DataFrame:
    """Load and validate one ENCODE WGBS/RRBS bedMethyl replicate file."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run preprocessing/download_data.py first."
        )

    # Some bedMethyl files (seen on GM12878 RRBS downloads, not on the
    # HepG2 WGBS ones this was originally written for) have a leading
    # UCSC "track name=... description=... ..." header line before the
    # actual tab-separated data. Left unskipped, it gets parsed as a data
    # row - its fields don't line up with the 11-column layout, corrupting
    # every column's dtype (pandas then can't parse chrom_start/etc. as
    # numeric). Detect and skip it instead.
    with gzip.open(path, "rt") as file:
        first_line = file.readline()
    rows_to_skip = 1 if first_line.startswith("track") else 0

    dataframe = pd.read_csv(
        path,
        sep="\t",
        header=None,
        compression="gzip",
        skiprows=rows_to_skip,
    )

    if dataframe.shape[1] != len(BEDMETHYL_COLUMNS):
        raise ValueError(
            f"{path.name}: expected {len(BEDMETHYL_COLUMNS)} bedMethyl columns, "
            f"found {dataframe.shape[1]}. The ENCODE bedMethyl layout may have "
            "changed — inspect the raw file before continuing."
        )

    dataframe.columns = BEDMETHYL_COLUMNS

    if dataframe.empty:
        raise ValueError(f"{path.name}: file is empty.")

    dataframe["chrom_start"] = pd.to_numeric(dataframe["chrom_start"], errors="raise").astype("int64")
    dataframe["chrom_end"] = pd.to_numeric(dataframe["chrom_end"], errors="raise").astype("int64")
    dataframe["score"] = pd.to_numeric(dataframe["score"], errors="raise").astype("int64")
    dataframe["coverage"] = pd.to_numeric(dataframe["coverage"], errors="raise").astype("int64")
    dataframe["percent_methylated"] = pd.to_numeric(
        dataframe["percent_methylated"], errors="raise"
    ).astype("float64")

    if (dataframe["chrom_end"] - dataframe["chrom_start"]).ne(1).any():
        raise ValueError(f"{path.name}: found intervals that are not 1 bp wide (single-base CpG calls expected).")

    if not dataframe["strand"].isin(["+", "-"]).all():
        raise ValueError(f"{path.name}: unexpected strand values: {sorted(dataframe['strand'].unique())}")

    if (dataframe["coverage"] < 0).any():
        raise ValueError(f"{path.name}: negative coverage values found.")

    if ((dataframe["percent_methylated"] < 0) | (dataframe["percent_methylated"] > 100)).any():
        raise ValueError(f"{path.name}: percent_methylated outside [0, 100] found.")

    return dataframe


# ============================================================
# Output helpers
# ============================================================

def save_table(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


# ============================================================
# Per-replicate analysis
# ============================================================

def analyze_replicate(dataframe: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    """Profile every column of one replicate; save tables and figures."""

    chrom_counts = (
        dataframe["chrom"]
        .value_counts()
        .rename_axis("chrom")
        .reset_index(name="cpg_count")
        .sort_values("chrom")
    )
    save_table(chrom_counts, TABLES_DIR / f"{label}_chrom_counts.csv")

    strand_counts = (
        dataframe["strand"]
        .value_counts()
        .rename_axis("strand")
        .reset_index(name="cpg_count")
    )
    save_table(strand_counts, TABLES_DIR / f"{label}_strand_counts.csv")

    name_counts = (
        dataframe["name"]
        .value_counts()
        .rename_axis("name")
        .reset_index(name="cpg_count")
    )
    save_table(name_counts, TABLES_DIR / f"{label}_name_field_counts.csv")

    # "score" is frequently coverage capped at 1000 in ENCODE bedMethyl files;
    # report how often it actually differs from the real "coverage" column.
    score_vs_coverage_mismatch = int((dataframe["score"] != dataframe["coverage"].clip(upper=1000)).sum())

    coverage_summary = pd.DataFrame(
        {
            "metric": [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "p25",
                "p75",
                "p95",
                "p99",
                "max",
                "zero_coverage_count",
                "score_ne_min(coverage,1000)_count",
            ],
            "value": [
                len(dataframe),
                dataframe["coverage"].mean(),
                dataframe["coverage"].median(),
                dataframe["coverage"].std(),
                dataframe["coverage"].min(),
                dataframe["coverage"].quantile(0.25),
                dataframe["coverage"].quantile(0.75),
                dataframe["coverage"].quantile(0.95),
                dataframe["coverage"].quantile(0.99),
                dataframe["coverage"].max(),
                int((dataframe["coverage"] == 0).sum()),
                score_vs_coverage_mismatch,
            ],
        }
    )
    save_table(coverage_summary, TABLES_DIR / f"{label}_coverage_summary.csv")

    methylation_summary = pd.DataFrame(
        {
            "metric": [
                "mean_percent_methylated",
                "median_percent_methylated",
                "fully_unmethylated_count",
                "fully_unmethylated_fraction",
                "fully_methylated_count",
                "fully_methylated_fraction",
                "intermediate_count",
                "intermediate_fraction",
            ],
            "value": [
                dataframe["percent_methylated"].mean(),
                dataframe["percent_methylated"].median(),
                int((dataframe["percent_methylated"] == 0).sum()),
                float((dataframe["percent_methylated"] == 0).mean()),
                int((dataframe["percent_methylated"] == 100).sum()),
                float((dataframe["percent_methylated"] == 100).mean()),
                int(dataframe["percent_methylated"].between(0, 100, inclusive="neither").sum()),
                float(dataframe["percent_methylated"].between(0, 100, inclusive="neither").mean()),
            ],
        }
    )
    save_table(methylation_summary, TABLES_DIR / f"{label}_methylation_summary.csv")

    # Figure: coverage distribution (log scale)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.hist(np.log10(dataframe["coverage"] + 1), bins=80)
    ax.set_title(f"{label}: Raw WGBS Coverage Distribution")
    ax.set_xlabel("Log10(1 + Coverage)")
    ax.set_ylabel("Number of CpG Records")
    save_figure(fig, FIGURES_DIR / f"{label}_coverage_distribution")

    # Figure: percent methylated distribution
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.hist(dataframe["percent_methylated"], bins=100)
    ax.set_title(f"{label}: Percent-Methylated Distribution")
    ax.set_xlabel("Percent Methylated")
    ax.set_ylabel("Number of CpG Records")
    save_figure(fig, FIGURES_DIR / f"{label}_percent_methylated_distribution")

    # Figure: CpG count per chromosome
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(chrom_counts["chrom"], chrom_counts["cpg_count"])
    ax.set_title(f"{label}: CpG Record Count per Chromosome")
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("Number of CpG Records")
    ax.tick_params(axis="x", rotation=60)
    save_figure(fig, FIGURES_DIR / f"{label}_chrom_counts")

    return {
        "label": label,
        "row_count": len(dataframe),
        "chromosome_count": dataframe["chrom"].nunique(),
        "mean_coverage": float(dataframe["coverage"].mean()),
        "median_coverage": float(dataframe["coverage"].median()),
        "mean_percent_methylated": float(dataframe["percent_methylated"].mean()),
    }


# ============================================================
# Replicate concordance
# ============================================================

def compare_replicates(
    dataframe_1: pd.DataFrame,
    dataframe_2: pd.DataFrame,
    label_1: str,
    label_2: str,
) -> dict[str, float | int]:
    """Check how well the two WGBS replicates agree at shared CpG positions."""

    join_columns = ["chrom", "chrom_start", "chrom_end", "strand"]

    merged = dataframe_1[join_columns + ["coverage", "percent_methylated"]].merge(
        dataframe_2[join_columns + ["coverage", "percent_methylated"]],
        on=join_columns,
        how="outer",
        suffixes=(f"_{label_1}", f"_{label_2}"),
        indicator=True,
    )

    overlap_counts = merged["_merge"].value_counts().rename_axis("presence").reset_index(name="cpg_count")
    save_table(overlap_counts, TABLES_DIR / "replicate_position_overlap.csv")

    both_covered = merged[
        merged["_merge"].eq("both")
        & merged[f"coverage_{label_1}"].gt(0)
        & merged[f"coverage_{label_2}"].gt(0)
    ]

    pearson_r = float(
        both_covered[f"percent_methylated_{label_1}"].corr(
            both_covered[f"percent_methylated_{label_2}"], method="pearson"
        )
    )
    spearman_r = float(
        both_covered[f"percent_methylated_{label_1}"].corr(
            both_covered[f"percent_methylated_{label_2}"], method="spearman"
        )
    )

    concordance_summary = pd.DataFrame(
        {
            "metric": [
                f"positions_only_in_{label_1}",
                f"positions_only_in_{label_2}",
                "positions_in_both",
                "positions_covered_in_both_(coverage>0)",
                "pearson_r_percent_methylated",
                "spearman_r_percent_methylated",
            ],
            "value": [
                int(merged["_merge"].eq("left_only").sum()),
                int(merged["_merge"].eq("right_only").sum()),
                int(merged["_merge"].eq("both").sum()),
                len(both_covered),
                pearson_r,
                spearman_r,
            ],
        }
    )
    save_table(concordance_summary, TABLES_DIR / "replicate_concordance_summary.csv")

    # Figure: scatter of percent-methylated, rep1 vs rep2, at shared covered positions
    sample = both_covered.sample(n=min(len(both_covered), 200_000), random_state=42)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(
        sample[f"percent_methylated_{label_1}"],
        sample[f"percent_methylated_{label_2}"],
        s=2,
        alpha=0.15,
    )
    ax.set_title(f"Replicate Concordance (Pearson r = {pearson_r:.3f})")
    ax.set_xlabel(f"% Methylated ({label_1})")
    ax.set_ylabel(f"% Methylated ({label_2})")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    save_figure(fig, FIGURES_DIR / "replicate_concordance_scatter")

    return {
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "positions_covered_in_both": len(both_covered),
    }


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

    labels = [f"rep{i + 1}_{accession}" for i, accession in enumerate(WGBS_REPLICATE_ACCESSIONS)]

    print("=" * 70)
    print("HepG2 WGBS raw-data analysis")
    print("=" * 70)

    dataframes = []

    for label, path in zip(labels, WGBS_REPLICATE_PATHS):
        print(f"\nLoading {label}: {path}")
        dataframe = load_bedmethyl(path)
        print(f"  Rows: {len(dataframe):,}  Columns: {list(dataframe.columns)}")
        dataframes.append(dataframe)

    replicate_summaries = []

    for label, dataframe in zip(labels, dataframes):
        print(f"\nAnalyzing {label}...")
        replicate_summaries.append(analyze_replicate(dataframe, label))

    save_table(pd.DataFrame(replicate_summaries), TABLES_DIR / "replicate_summaries.csv")

    print("\nComparing replicates...")
    concordance = compare_replicates(dataframes[0], dataframes[1], labels[0], labels[1])

    print("\nAnalysis completed.")
    print(f"Tables:  {TABLES_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print()
    print(pd.DataFrame(replicate_summaries).to_string(index=False))
    print()
    print(f"Replicate Pearson r:  {concordance['pearson_r']:.4f}")
    print(f"Replicate Spearman r: {concordance['spearman_r']:.4f}")
    print(f"Positions covered in both replicates: {concordance['positions_covered_in_both']:,}")


if __name__ == "__main__":
    main()
