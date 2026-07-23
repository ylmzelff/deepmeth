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
DEFAULT_OUTPUT_DIR = Path("/content/deepmeth/data/wp5_1/analysis")
DEFAULT_PATTERN = "*_Ca_*_RRBS.single.CpG.txt.gz"

FIRST_HCC_CELL = 1
LAST_HCC_CELL = 25


def infer_cell_id(path: Path) -> str:
    """Extract a cell identifier such as Ca_01 from a filename."""
    match = re.search(r"(Ca_\d{2})", path.name)
    if match is None:
        raise ValueError(f"Cell ID could not be inferred from: {path.name}")
    return match.group(1)


def cell_number(cell_id: str) -> int:
    return int(cell_id.split("_")[1])


def is_wp51_hcc_cell(cell_id: str) -> bool:
    number = cell_number(cell_id)
    return FIRST_HCC_CELL <= number <= LAST_HCC_CELL


def load_raw_file(path: Path) -> pd.DataFrame:
    """Load and validate one GSE65364 scRRBS file."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=RAW_COLUMNS,
        compression="gzip",
    )

    if df.empty:
        raise ValueError(f"{path.name}: file is empty.")

    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if df[NUMERIC_COLUMNS].isna().any().any():
        invalid_columns = df[NUMERIC_COLUMNS].columns[
            df[NUMERIC_COLUMNS].isna().any()
        ].tolist()
        raise ValueError(
            f"{path.name}: missing or non-numeric values in {invalid_columns}"
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


def build_wp51_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the WP5.1 canonical strand merge and binary-label rule.

    Ties are retained and assigned label 0.
    """
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

    merged_df["count_m"] = merged_df["count_m"].astype("int64")
    merged_df["count_u"] = merged_df["count_u"].astype("int64")
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

    merged_df["is_tie"] = (
        merged_df["count_m"].eq(merged_df["count_u"]).astype("int8")
    )
    merged_df["label"] = (
        merged_df["count_m"].gt(merged_df["count_u"]).astype("int8")
    )
    merged_df["methylation_state"] = merged_df["label"].map(
        {0: "Unmethylated", 1: "Methylated"}
    )

    if int(merged_df["count_m"].sum()) != int(raw_df["count_m"].sum()):
        raise RuntimeError("Methylated-read total changed during merging.")
    if int(merged_df["count_u"].sum()) != int(raw_df["count_u"].sum()):
        raise RuntimeError("Unmethylated-read total changed during merging.")
    if merged_df.duplicated(["chrom", "canonical_position"]).any():
        raise RuntimeError("Duplicated canonical CpG coordinates remain.")
    if not merged_df.loc[merged_df["is_tie"].eq(1), "label"].eq(0).all():
        raise RuntimeError("Tie records must have label 0.")

    return merged_df


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyze_cell(
    raw_path: Path,
    output_root: Path,
) -> dict[str, float | int | str]:
    """Run WP5.1 exploratory analysis for one HCC cell."""
    cell_id = infer_cell_id(raw_path)
    cell_dir = output_root / cell_id
    tables_dir = cell_dir / "tables"
    figures_dir = cell_dir / "figures"

    raw_df = load_raw_file(raw_path)
    processed_df = build_wp51_dataset(raw_df)

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
        processed_df["strand_support"]
        .value_counts()
        .rename_axis("Strand Support")
        .reset_index(name="CpG Count")
    )
    strand_support_counts["Percentage"] = (
        strand_support_counts["CpG Count"] / len(processed_df) * 100
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
                len(processed_df),
                processed_df["coverage"].mean(),
                processed_df["coverage"].median(),
                processed_df["coverage"].max(),
                processed_df["methylation_ratio"].mean(),
                processed_df["methylation_ratio"].median(),
            ],
        }
    )
    save_table(
        pre_post_summary,
        tables_dir / "04_pre_post_merge_summary.csv",
    )

    label_distribution = (
        processed_df.groupby(
            ["label", "methylation_state"],
            as_index=False,
        )
        .agg(CpG_Count=("canonical_position", "size"))
    )
    label_distribution["Percentage"] = (
        label_distribution["CpG_Count"] / len(processed_df) * 100
    ).round(2)
    save_table(
        label_distribution,
        tables_dir / "05_label_distribution.csv",
    )

    final_qc_summary = pd.DataFrame(
        {
            "Metric": [
                "Physical CpG sites",
                "Tie sites retained as label 0",
                "Unmethylated CpGs",
                "Methylated CpGs",
                "Unmethylated percentage",
                "Methylated percentage",
                "Mean merged coverage",
                "Median merged coverage",
                "Maximum merged coverage",
                "Duplicated canonical CpGs",
            ],
            "Value": [
                len(processed_df),
                int(processed_df["is_tie"].sum()),
                int(processed_df["label"].eq(0).sum()),
                int(processed_df["label"].eq(1).sum()),
                processed_df["label"].eq(0).mean() * 100,
                processed_df["label"].eq(1).mean() * 100,
                processed_df["coverage"].mean(),
                processed_df["coverage"].median(),
                processed_df["coverage"].max(),
                int(
                    processed_df.duplicated(
                        ["chrom", "canonical_position"]
                    ).sum()
                ),
            ],
        }
    )
    save_table(final_qc_summary, tables_dir / "06_final_qc_summary.csv")

    # Figure 1: raw base/strand representation.
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(
        representation_counts["Representation"],
        representation_counts["Record Count"],
    )
    ax.bar_label(
        bars,
        labels=[
            f"{count:,}\n({percentage:.2f}%)"
            for count, percentage in zip(
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

    # Figure 2: raw coverage distribution.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(np.log10(raw_df["coverage"] + 1), bins=60)
    ax.set_title(f"{cell_id}: Raw Coverage Distribution")
    ax.set_xlabel("Log10(1 + Coverage)")
    ax.set_ylabel("Number of Raw Records")
    save_figure(fig, figures_dir / "02_raw_coverage_distribution.png")

    # Figure 3: coverage before and after strand merging.
    raw_log_coverage = np.log10(raw_df["coverage"] + 1)
    merged_log_coverage = np.log10(processed_df["coverage"] + 1)
    bins = np.linspace(
        min(raw_log_coverage.min(), merged_log_coverage.min()),
        max(raw_log_coverage.max(), merged_log_coverage.max()),
        60,
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.hist(
        raw_log_coverage,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        label="Before merging",
    )
    ax.hist(
        merged_log_coverage,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        label="After merging",
    )
    ax.set_title(f"{cell_id}: Coverage Before and After Strand Merging")
    ax.set_xlabel("Log10(1 + Coverage)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    save_figure(fig, figures_dir / "03_pre_post_coverage.png")

    # Figure 4: methylation ratio before and after merging.
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
        processed_df["methylation_ratio"].dropna(),
        bins=ratio_bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        label="After merging",
    )
    ax.set_title(f"{cell_id}: Methylation Ratio Before and After Merging")
    ax.set_xlabel("Methylation Ratio")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    save_figure(fig, figures_dir / "04_pre_post_ratio.png")

    # Figure 5: strand support.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar(
        strand_support_counts["Strand Support"],
        strand_support_counts["CpG Count"],
    )
    ax.bar_label(
        bars,
        labels=[
            f"{count:,}\n({percentage:.2f}%)"
            for count, percentage in zip(
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

    # Figure 6: final binary label distribution.
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    bars = ax.bar(
        label_distribution["methylation_state"],
        label_distribution["CpG_Count"],
    )
    ax.bar_label(
        bars,
        labels=[
            f"{count:,}\n({percentage:.2f}%)"
            for count, percentage in zip(
                label_distribution["CpG_Count"],
                label_distribution["Percentage"],
            )
        ],
        padding=3,
        fontsize=9,
    )
    ax.set_title(f"{cell_id}: WP5.1 Binary Label Distribution")
    ax.set_xlabel("Methylation State")
    ax.set_ylabel("Number of Physical CpG Sites")
    save_figure(fig, figures_dir / "06_label_distribution.png")

    return {
        "cell_id": cell_id,
        "raw_records": len(raw_df),
        "c_plus_records": int(raw_df["representation"].eq("C/+").sum()),
        "g_minus_records": int(raw_df["representation"].eq("G/-").sum()),
        "physical_cpgs": len(processed_df),
        "both_strands": int(
            processed_df["strand_support"].eq("Both C/+ and G/-").sum()
        ),
        "c_plus_only": int(
            processed_df["strand_support"].eq("C/+ only").sum()
        ),
        "g_minus_only": int(
            processed_df["strand_support"].eq("G/- only").sum()
        ),
        "ties_retained_as_label_0": int(processed_df["is_tie"].sum()),
        "unmethylated_count": int(processed_df["label"].eq(0).sum()),
        "methylated_count": int(processed_df["label"].eq(1).sum()),
        "unmethylated_percentage": round(
            processed_df["label"].eq(0).mean() * 100,
            4,
        ),
        "methylated_percentage": round(
            processed_df["label"].eq(1).mean() * 100,
            4,
        ),
        "mean_merged_coverage": float(processed_df["coverage"].mean()),
        "median_merged_coverage": float(processed_df["coverage"].median()),
    }


def find_raw_files(
    input_dir: Path,
    pattern: str,
    selected_cells: list[str] | None,
) -> list[Path]:
    files = sorted(input_dir.glob(pattern))

    wp51_files = []
    for path in files:
        cell_id = infer_cell_id(path)
        if is_wp51_hcc_cell(cell_id):
            wp51_files.append(path)

    if selected_cells:
        invalid_cells = sorted(
            cell for cell in selected_cells if not is_wp51_hcc_cell(cell)
        )
        if invalid_cells:
            raise ValueError(
                "Only Ca_01-Ca_25 are valid for WP5.1. Invalid selection: "
                + ", ".join(invalid_cells)
            )

        selected = set(selected_cells)
        wp51_files = [
            path for path in wp51_files if infer_cell_id(path) in selected
        ]

    return sorted(wp51_files, key=lambda path: cell_number(infer_cell_id(path)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the 25 GSE65364 human HCC scRRBS cells for "
            "DeepMeth Work Package 5.1."
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
        default=DEFAULT_PATTERN,
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
        default=25,
        help="Expected file count when --cells is not used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = find_raw_files(
        input_dir=args.input_dir,
        pattern=args.pattern,
        selected_cells=args.cells,
    )

    if not raw_files:
        raise FileNotFoundError(
            f"No WP5.1 HCC files found in {args.input_dir} "
            f"with pattern {args.pattern}"
        )

    if args.cells is None and len(raw_files) != args.expected_cells:
        raise RuntimeError(
            f"Expected {args.expected_cells} Ca_01-Ca_25 files, "
            f"found {len(raw_files)}."
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

    summaries: list[dict[str, float | int | str]] = []

    for index, raw_path in enumerate(raw_files, start=1):
        cell_id = infer_cell_id(raw_path)
        print(
            f"[{index:02d}/{len(raw_files):02d}] "
            f"Analyzing {cell_id}: {raw_path.name}"
        )
        summaries.append(analyze_cell(raw_path, args.output_dir))

    all_cells_summary = (
        pd.DataFrame(summaries)
        .sort_values("cell_id")
        .reset_index(drop=True)
    )
    save_table(
        all_cells_summary,
        args.output_dir / "wp5_1_all_cells_analysis_summary.csv",
    )

    # Overall figure 1: final physical CpG counts.
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    ax.bar(
        all_cells_summary["cell_id"],
        all_cells_summary["physical_cpgs"],
    )
    ax.set_title("Final Physical CpG Count Across the 25 HCC Cells")
    ax.set_xlabel("HCC Cell")
    ax.set_ylabel("Number of Physical CpG Sites")
    ax.tick_params(axis="x", rotation=60)
    save_figure(
        fig,
        args.output_dir / "wp5_1_all_cells_physical_cpg_count.png",
    )

    # Overall figure 2: methylated percentage across cells.
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.plot(
        all_cells_summary["cell_id"],
        all_cells_summary["methylated_percentage"],
        marker="o",
    )
    ax.set_title("Methylated CpG Percentage Across the 25 HCC Cells")
    ax.set_xlabel("HCC Cell")
    ax.set_ylabel("Methylated CpGs (%)")
    ax.tick_params(axis="x", rotation=60)
    save_figure(
        fig,
        args.output_dir / "wp5_1_all_cells_methylated_percentage.png",
    )

    # Overall figure 3: tie records retained as label 0.
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.bar(
        all_cells_summary["cell_id"],
        all_cells_summary["ties_retained_as_label_0"],
    )
    ax.set_title("Equal Read-Count CpGs Retained as Unmethylated")
    ax.set_xlabel("HCC Cell")
    ax.set_ylabel("Number of Tie CpG Sites")
    ax.tick_params(axis="x", rotation=60)
    save_figure(
        fig,
        args.output_dir / "wp5_1_all_cells_tie_counts.png",
    )

    print("\nWP5.1 analysis completed.")
    print(f"Output directory: {args.output_dir}")
    print(
        all_cells_summary[
            [
                "cell_id",
                "raw_records",
                "physical_cpgs",
                "ties_retained_as_label_0",
                "unmethylated_count",
                "methylated_count",
                "methylated_percentage",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
