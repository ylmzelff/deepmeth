from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RAW_COLUMNS = [
    "chrom",
    "position",
    "base",
    "strand",
    "coverage",
    "count_m",
    "count_u",
    "methylation_ratio",
    "sequence_context",
    "context_type",
]

NUMERIC_COLUMNS = [
    "position",
    "coverage",
    "count_m",
    "count_u",
    "methylation_ratio",
]

DEFAULT_INPUT_DIR = Path("/content/deepmeth/data/raw/GSE65364/extracted")
DEFAULT_OUTPUT_DIR = Path(
    "/content/drive/MyDrive/1001_BioSeq_LLM/"
    "deepmeth_backup/GSE65364/all_cells_analysis"
)


def infer_cell_id(path: Path) -> str:
    """Extract Ca_01 ... Ca_26 from the filename."""
    match = re.search(r"(Ca_\d{2})", path.name)
    if match is None:
        raise ValueError(f"Cell ID could not be inferred from: {path.name}")
    return match.group(1)


def load_raw_file(path: Path) -> pd.DataFrame:
    """Load and validate one GSE65364 RRBS file."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=RAW_COLUMNS,
        compression="gzip",
    )

    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if df[NUMERIC_COLUMNS].isna().any().any():
        columns = df[NUMERIC_COLUMNS].columns[
            df[NUMERIC_COLUMNS].isna().any()
        ].tolist()
        raise ValueError(
            f"{path.name}: missing or non-numeric values in {columns}"
        )

    df["base"] = df["base"].astype(str).str.upper().str.strip()
    df["strand"] = df["strand"].astype(str).str.strip()
    df["representation"] = df["base"] + "/" + df["strand"]

    unexpected = sorted(set(df["representation"]) - {"C/+", "G/-"})
    if unexpected:
        raise ValueError(
            f"{path.name}: unexpected base/strand combinations: {unexpected}"
        )

    calculated_coverage = df["count_m"] + df["count_u"]
    mismatch_count = int(
        (~np.isclose(df["coverage"], calculated_coverage)).sum()
    )
    if mismatch_count:
        raise ValueError(
            f"{path.name}: {mismatch_count} coverage values do not equal "
            "count_m + count_u"
        )

    return df


def build_merged_method(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Canonical strand merge: C/+ -> p and G/- -> p - 1."""
    strand_df = raw_df.copy()

    strand_df["canonical_position"] = np.where(
        strand_df["representation"].eq("C/+"),
        strand_df["position"],
        strand_df["position"] - 1,
    ).astype("int64")

    strand_df["is_c_plus"] = (
        strand_df["representation"].eq("C/+").astype("int8")
    )
    strand_df["is_g_minus"] = (
        strand_df["representation"].eq("G/-").astype("int8")
    )

    merged_df = (
        strand_df.groupby(
            ["chrom", "canonical_position"],
            as_index=False,
            sort=False,
        )
        .agg(
            raw_record_count=("position", "size"),
            c_plus_record_count=("is_c_plus", "sum"),
            g_minus_record_count=("is_g_minus", "sum"),
            count_m=("count_m", "sum"),
            count_u=("count_u", "sum"),
        )
    )

    merged_df["coverage"] = merged_df["count_m"] + merged_df["count_u"]
    merged_df["methylation_ratio"] = (
        merged_df["count_m"] / merged_df["coverage"].replace(0, np.nan)
    )

    merged_df["strand_support"] = np.select(
        [
            merged_df["c_plus_record_count"].gt(0)
            & merged_df["g_minus_record_count"].gt(0),
            merged_df["c_plus_record_count"].gt(0),
            merged_df["g_minus_record_count"].gt(0),
        ],
        [
            "Both C/+ and G/-",
            "C/+ only",
            "G/- only",
        ],
        default="Unexpected",
    )

    tie_mask = merged_df["count_m"].eq(merged_df["count_u"])
    tie_df = merged_df.loc[tie_mask].copy().reset_index(drop=True)

    labeled_df = merged_df.loc[~tie_mask].copy().reset_index(drop=True)
    labeled_df["label"] = (
        labeled_df["count_m"].gt(labeled_df["count_u"]).astype("int8")
    )
    labeled_df["methylation_state"] = labeled_df["label"].map(
        {0: "Unmethylated", 1: "Methylated"}
    )

    return merged_df, labeled_df, tie_df


def build_c_plus_method(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sensitivity method using only C/+ records."""
    c_plus_df = raw_df.loc[
        raw_df["representation"].eq("C/+")
    ].copy()

    c_plus_df = (
        c_plus_df.groupby(
            ["chrom", "position"],
            as_index=False,
            sort=False,
        )
        .agg(
            raw_record_count=("position", "size"),
            count_m=("count_m", "sum"),
            count_u=("count_u", "sum"),
        )
        .rename(columns={"position": "canonical_position"})
    )

    c_plus_df["coverage"] = c_plus_df["count_m"] + c_plus_df["count_u"]
    c_plus_df["methylation_ratio"] = (
        c_plus_df["count_m"] / c_plus_df["coverage"].replace(0, np.nan)
    )

    tie_mask = c_plus_df["count_m"].eq(c_plus_df["count_u"])
    tie_df = c_plus_df.loc[tie_mask].copy().reset_index(drop=True)

    labeled_df = c_plus_df.loc[~tie_mask].copy().reset_index(drop=True)
    labeled_df["label"] = (
        labeled_df["count_m"].gt(labeled_df["count_u"]).astype("int8")
    )
    labeled_df["methylation_state"] = labeled_df["label"].map(
        {0: "Unmethylated", 1: "Methylated"}
    )

    return labeled_df, tie_df


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyze_cell(raw_path: Path, output_root: Path) -> dict[str, float | int | str]:
    """Run all exploratory checks for one cell and save its results."""
    cell_id = infer_cell_id(raw_path)
    cell_dir = output_root / cell_id
    tables_dir = cell_dir / "tables"
    figures_dir = cell_dir / "figures"

    raw_df = load_raw_file(raw_path)
    merged_all_df, merged_labeled_df, merged_tie_df = build_merged_method(
        raw_df
    )
    c_plus_labeled_df, c_plus_tie_df = build_c_plus_method(raw_df)

    representation_counts = (
        raw_df["representation"]
        .value_counts()
        .rename_axis("Representation")
        .reset_index(name="Record Count")
    )
    representation_counts["Percentage"] = (
        representation_counts["Record Count"] / len(raw_df) * 100
    ).round(2)
    save_table(
        representation_counts,
        tables_dir / "01_representation_counts.csv",
    )

    ratio_difference = (
        raw_df["methylation_ratio"]
        - raw_df["count_m"] / raw_df["coverage"].replace(0, np.nan)
    ).abs()

    raw_qc_summary = pd.DataFrame(
        {
            "Metric": [
                "Raw records",
                "Mean coverage",
                "Median coverage",
                "Maximum coverage",
                "Mean methylated reads",
                "Mean unmethylated reads",
                "Mean methylation ratio",
                "Fully unmethylated records",
                "Fully methylated records",
                "Ratio differences above 1e-8",
            ],
            "Value": [
                len(raw_df),
                raw_df["coverage"].mean(),
                raw_df["coverage"].median(),
                raw_df["coverage"].max(),
                raw_df["count_m"].mean(),
                raw_df["count_u"].mean(),
                raw_df["methylation_ratio"].mean(),
                int(raw_df["methylation_ratio"].eq(0).sum()),
                int(raw_df["methylation_ratio"].eq(1).sum()),
                int(ratio_difference.gt(1e-8).sum()),
            ],
        }
    )
    save_table(raw_qc_summary, tables_dir / "02_raw_qc_summary.csv")

    strand_support_counts = (
        merged_all_df["strand_support"]
        .value_counts()
        .rename_axis("Strand Support")
        .reset_index(name="CpG Count")
    )
    strand_support_counts["Percentage"] = (
        strand_support_counts["CpG Count"] / len(merged_all_df) * 100
    ).round(2)
    save_table(
        strand_support_counts,
        tables_dir / "03_strand_support_distribution.csv",
    )

    pre_post_summary = pd.DataFrame(
        {
            "Metric": [
                "Number of records",
                "Mean coverage",
                "Median coverage",
                "Maximum coverage",
                "Mean methylation ratio",
                "Median methylation ratio",
            ],
            "Before Merging": [
                len(raw_df),
                raw_df["coverage"].mean(),
                raw_df["coverage"].median(),
                raw_df["coverage"].max(),
                raw_df["methylation_ratio"].mean(),
                raw_df["methylation_ratio"].median(),
            ],
            "After Merging": [
                len(merged_all_df),
                merged_all_df["coverage"].mean(),
                merged_all_df["coverage"].median(),
                merged_all_df["coverage"].max(),
                merged_all_df["methylation_ratio"].mean(),
                merged_all_df["methylation_ratio"].median(),
            ],
        }
    )
    save_table(
        pre_post_summary,
        tables_dir / "04_pre_post_merge_summary.csv",
    )

    method_comparison = pd.DataFrame(
        {
            "Method": [
                "Method 1: Merged strands",
                "Method 2: C/+ only",
            ],
            "Labeled CpG Count": [
                len(merged_labeled_df),
                len(c_plus_labeled_df),
            ],
            "Unmethylated Count": [
                int(merged_labeled_df["label"].eq(0).sum()),
                int(c_plus_labeled_df["label"].eq(0).sum()),
            ],
            "Methylated Count": [
                int(merged_labeled_df["label"].eq(1).sum()),
                int(c_plus_labeled_df["label"].eq(1).sum()),
            ],
            "Mean Coverage": [
                merged_labeled_df["coverage"].mean(),
                c_plus_labeled_df["coverage"].mean(),
            ],
            "Median Coverage": [
                merged_labeled_df["coverage"].median(),
                c_plus_labeled_df["coverage"].median(),
            ],
        }
    )
    method_comparison["Methylated Percentage"] = (
        method_comparison["Methylated Count"]
        / method_comparison["Labeled CpG Count"]
        * 100
    ).round(2)
    save_table(
        method_comparison,
        tables_dir / "05_method_comparison.csv",
    )

    common_df = merged_labeled_df[
        [
            "chrom",
            "canonical_position",
            "coverage",
            "methylation_ratio",
            "label",
        ]
    ].rename(
        columns={
            "coverage": "merged_coverage",
            "methylation_ratio": "merged_ratio",
            "label": "merged_label",
        }
    ).merge(
        c_plus_labeled_df[
            [
                "chrom",
                "canonical_position",
                "coverage",
                "methylation_ratio",
                "label",
            ]
        ].rename(
            columns={
                "coverage": "c_plus_coverage",
                "methylation_ratio": "c_plus_ratio",
                "label": "c_plus_label",
            }
        ),
        on=["chrom", "canonical_position"],
        how="inner",
        validate="one_to_one",
    )

    common_df["same_label"] = (
        common_df["merged_label"].eq(common_df["c_plus_label"])
    )
    common_df["coverage_gain"] = (
        common_df["merged_coverage"] - common_df["c_plus_coverage"]
    )
    common_df["label_transition"] = (
        common_df["c_plus_label"].astype(str)
        + " -> "
        + common_df["merged_label"].astype(str)
    )

    same_label_count = int(common_df["same_label"].sum())
    label_agreement = (
        same_label_count / len(common_df) * 100
        if len(common_df)
        else np.nan
    )

    common_summary = pd.DataFrame(
        {
            "Metric": [
                "Common labeled CpGs",
                "Common CpGs with same label",
                "Common CpGs with different labels",
                "Label agreement percentage",
                "Mean merged coverage",
                "Mean C/+ coverage",
                "Mean coverage gain",
                "Common CpGs with additional G/- coverage",
            ],
            "Value": [
                len(common_df),
                same_label_count,
                int((~common_df["same_label"]).sum()),
                label_agreement,
                common_df["merged_coverage"].mean(),
                common_df["c_plus_coverage"].mean(),
                common_df["coverage_gain"].mean(),
                int(common_df["coverage_gain"].gt(0).sum()),
            ],
        }
    )
    save_table(
        common_summary,
        tables_dir / "06_common_cpg_summary.csv",
    )

    label_transitions = (
        common_df["label_transition"]
        .value_counts()
        .rename_axis("Label Transition")
        .reset_index(name="CpG Count")
    )
    label_transitions["Percentage"] = (
        label_transitions["CpG Count"] / len(common_df) * 100
    ).round(4)
    save_table(
        label_transitions,
        tables_dir / "07_label_transition_summary.csv",
    )

    changed_labels = common_df.loc[~common_df["same_label"]].copy()
    save_table(
        changed_labels,
        tables_dir / "08_changed_label_cpg_sites.csv",
    )

    # Figure 1: base/strand representations
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(
        representation_counts["Representation"],
        representation_counts["Record Count"],
    )
    ax.bar_label(
        bars,
        labels=[
            f"{count:,}\n({pct:.2f}%)"
            for count, pct in zip(
                representation_counts["Record Count"],
                representation_counts["Percentage"],
            )
        ],
        padding=3,
        fontsize=9,
    )
    ax.set_title(f"{cell_id}: Base-Strand Representation")
    ax.set_xlabel("Representation")
    ax.set_ylabel("Number of Raw Records")
    save_figure(fig, figures_dir / "01_representation_counts.png")

    # Figure 2: raw coverage
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(np.log10(raw_df["coverage"] + 1), bins=60)
    ax.set_title(f"{cell_id}: Raw Coverage Distribution")
    ax.set_xlabel("Log10(1 + Coverage)")
    ax.set_ylabel("Number of Raw Records")
    save_figure(fig, figures_dir / "02_raw_coverage_distribution.png")

    # Figure 3: methylated and unmethylated reads
    count_m_log = np.log10(raw_df["count_m"] + 1)
    count_u_log = np.log10(raw_df["count_u"] + 1)
    bins = np.linspace(
        min(count_m_log.min(), count_u_log.min()),
        max(count_m_log.max(), count_u_log.max()),
        60,
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(
        count_m_log,
        bins=bins,
        histtype="step",
        linewidth=1.8,
        label="Methylated reads",
    )
    ax.hist(
        count_u_log,
        bins=bins,
        histtype="step",
        linewidth=1.8,
        label="Unmethylated reads",
    )
    ax.set_title(f"{cell_id}: Raw Read-Count Distributions")
    ax.set_xlabel("Log10(1 + Read Count)")
    ax.set_ylabel("Number of Raw Records")
    ax.legend(frameon=False)
    save_figure(fig, figures_dir / "03_read_count_distributions.png")

    # Figure 4: coverage vs ratio
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    density = ax.hexbin(
        np.log10(raw_df["coverage"].clip(lower=1)),
        raw_df["methylation_ratio"],
        gridsize=60,
        mincnt=1,
        bins="log",
    )
    ax.set_title(f"{cell_id}: Coverage vs. Methylation Ratio")
    ax.set_xlabel("Log10 Coverage")
    ax.set_ylabel("Methylation Ratio")
    ax.set_ylim(0, 1)
    colorbar = fig.colorbar(density, ax=ax)
    colorbar.set_label("Log10 Number of Records")
    save_figure(fig, figures_dir / "04_coverage_vs_ratio.png")

    # Figure 5: strand support
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar(
        strand_support_counts["Strand Support"],
        strand_support_counts["CpG Count"],
    )
    ax.bar_label(
        bars,
        labels=[
            f"{count:,}\n({pct:.2f}%)"
            for count, pct in zip(
                strand_support_counts["CpG Count"],
                strand_support_counts["Percentage"],
            )
        ],
        padding=3,
        fontsize=8,
    )
    ax.set_title(f"{cell_id}: Strand Support of Canonical CpGs")
    ax.set_xlabel("Strand Support")
    ax.set_ylabel("Number of Physical CpG Sites")
    ax.tick_params(axis="x", rotation=12)
    save_figure(fig, figures_dir / "05_strand_support.png")

    # Figure 6: coverage before/after merge
    raw_log_cov = np.log10(raw_df["coverage"] + 1)
    merged_log_cov = np.log10(merged_all_df["coverage"] + 1)
    bins = np.linspace(
        min(raw_log_cov.min(), merged_log_cov.min()),
        max(raw_log_cov.max(), merged_log_cov.max()),
        60,
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.hist(
        raw_log_cov,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        label="Before merging",
    )
    ax.hist(
        merged_log_cov,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        label="After merging",
    )
    ax.set_title(f"{cell_id}: Coverage Before and After Merging")
    ax.set_xlabel("Log10(1 + Coverage)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    save_figure(fig, figures_dir / "06_pre_post_coverage.png")

    # Figure 7: ratio before/after merge
    ratio_bins = np.linspace(0, 1, 51)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.hist(
        raw_df["methylation_ratio"],
        bins=ratio_bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        label="Before merging",
    )
    ax.hist(
        merged_all_df["methylation_ratio"].dropna(),
        bins=ratio_bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        label="After merging",
    )
    ax.set_title(
        f"{cell_id}: Methylation Ratio Before and After Merging"
    )
    ax.set_xlabel("Methylation Ratio")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    save_figure(fig, figures_dir / "07_pre_post_ratio.png")

    # Figure 8: dataset size by method
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    bars = ax.bar(
        method_comparison["Method"],
        method_comparison["Labeled CpG Count"],
    )
    ax.bar_label(
        bars,
        labels=[
            f"{count:,}"
            for count in method_comparison["Labeled CpG Count"]
        ],
        padding=3,
    )
    ax.set_title(f"{cell_id}: Dataset Size by Method")
    ax.set_xlabel("Preprocessing Method")
    ax.set_ylabel("Number of Labeled CpGs")
    ax.tick_params(axis="x", rotation=10)
    save_figure(fig, figures_dir / "08_method_dataset_size.png")

    # Figure 9: label agreement
    agreement_counts = pd.DataFrame(
        {
            "Agreement": ["Same label", "Different label"],
            "CpG Count": [
                same_label_count,
                int((~common_df["same_label"]).sum()),
            ],
        }
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    bars = ax.bar(
        agreement_counts["Agreement"],
        agreement_counts["CpG Count"],
    )
    ax.bar_label(
        bars,
        labels=[
            f"{count:,}"
            for count in agreement_counts["CpG Count"]
        ],
        padding=3,
    )
    ax.set_title(f"{cell_id}: Label Agreement Between Methods")
    ax.set_xlabel("Agreement Status")
    ax.set_ylabel("Number of Common CpGs")
    save_figure(fig, figures_dir / "09_label_agreement.png")

    return {
        "cell_id": cell_id,
        "raw_records": len(raw_df),
        "c_plus_records": int(raw_df["representation"].eq("C/+").sum()),
        "g_minus_records": int(raw_df["representation"].eq("G/-").sum()),
        "physical_cpgs": len(merged_all_df),
        "both_strands": int(
            merged_all_df["strand_support"]
            .eq("Both C/+ and G/-")
            .sum()
        ),
        "c_plus_only": int(
            merged_all_df["strand_support"].eq("C/+ only").sum()
        ),
        "g_minus_only": int(
            merged_all_df["strand_support"].eq("G/- only").sum()
        ),
        "merged_ties": len(merged_tie_df),
        "merged_labeled_cpgs": len(merged_labeled_df),
        "c_plus_ties": len(c_plus_tie_df),
        "c_plus_labeled_cpgs": len(c_plus_labeled_df),
        "merged_methylated_percentage": round(
            merged_labeled_df["label"].mean() * 100,
            4,
        ),
        "c_plus_methylated_percentage": round(
            c_plus_labeled_df["label"].mean() * 100,
            4,
        ),
        "common_labeled_cpgs": len(common_df),
        "label_agreement_percentage": round(label_agreement, 4),
        "mean_merged_coverage": merged_labeled_df["coverage"].mean(),
        "mean_c_plus_coverage": c_plus_labeled_df["coverage"].mean(),
    }


def find_raw_files(
    input_dir: Path,
    pattern: str,
    selected_cells: list[str] | None,
) -> list[Path]:
    files = sorted(input_dir.glob(pattern))
    if selected_cells:
        selected = set(selected_cells)
        files = [
            path for path in files
            if infer_cell_id(path) in selected
        ]
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run raw-data and preprocessing-method analyses for "
            "GSE65364 HCC cells."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--pattern",
        default="*_Ca_*_RRBS.single.CpG.txt.gz",
    )
    parser.add_argument(
        "--cells",
        nargs="*",
        default=None,
        help="Optional subset, for example: --cells Ca_01 Ca_02",
    )
    parser.add_argument(
        "--expected-cells",
        type=int,
        default=26,
        help="Expected file count when --cells is not used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = find_raw_files(
        input_dir=args.input_dir,
        pattern=args.pattern,
        selected_cells=args.cells,
    )

    if not raw_files:
        raise FileNotFoundError(
            f"No raw files found in {args.input_dir} "
            f"with pattern {args.pattern}"
        )

    if args.cells is None and len(raw_files) != args.expected_cells:
        raise RuntimeError(
            f"Expected {args.expected_cells} files, found {len(raw_files)}."
        )

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

    summaries = []
    for index, raw_path in enumerate(raw_files, start=1):
        cell_id = infer_cell_id(raw_path)
        print(f"[{index}/{len(raw_files)}] Analyzing {cell_id}: {raw_path.name}")
        summaries.append(analyze_cell(raw_path, args.output_dir))

    all_cells_summary = pd.DataFrame(summaries).sort_values("cell_id")
    save_table(
        all_cells_summary,
        args.output_dir / "all_cells_analysis_summary.csv",
    )

    # Overall figure: final labeled dataset size
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    x = np.arange(len(all_cells_summary))
    width = 0.38
    ax.bar(
        x - width / 2,
        all_cells_summary["merged_labeled_cpgs"],
        width=width,
        label="Merged strands",
    )
    ax.bar(
        x + width / 2,
        all_cells_summary["c_plus_labeled_cpgs"],
        width=width,
        label="C/+ only",
    )
    ax.set_title("Final Labeled CpG Count Across HCC Cells")
    ax.set_xlabel("HCC Cell")
    ax.set_ylabel("Number of Labeled CpG Sites")
    ax.set_xticks(x)
    ax.set_xticklabels(
        all_cells_summary["cell_id"],
        rotation=60,
        ha="right",
    )
    ax.legend(frameon=False)
    save_figure(
        fig,
        args.output_dir / "all_cells_dataset_size_comparison.png",
    )

    # Overall figure: label agreement
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.plot(
        all_cells_summary["cell_id"],
        all_cells_summary["label_agreement_percentage"],
        marker="o",
    )
    ax.set_title("Label Agreement Between Methods Across HCC Cells")
    ax.set_xlabel("HCC Cell")
    ax.set_ylabel("Label Agreement (%)")
    ax.tick_params(axis="x", rotation=60)
    save_figure(
        fig,
        args.output_dir / "all_cells_label_agreement.png",
    )

    print("\nAnalysis completed.")
    print(f"Output directory: {args.output_dir}")
    print(
        all_cells_summary[
            [
                "cell_id",
                "raw_records",
                "merged_labeled_cpgs",
                "c_plus_labeled_cpgs",
                "label_agreement_percentage",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
