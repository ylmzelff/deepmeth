from __future__ import annotations

import argparse
import re
from pathlib import Path

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

# The data directory may be a symbolic link to Google Drive.
DEFAULT_INPUT_DIR = Path("/content/deepmeth/data/raw/GSE65364/extracted")
DEFAULT_OUTPUT_DIR = Path("/content/deepmeth/data/wp5_1/processed")
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
    """Convert Ca_01 to 1."""
    return int(cell_id.split("_")[1])


def is_wp51_hcc_cell(cell_id: str) -> bool:
    """Return True only for the 25 HCC cells used in WP5.1."""
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

    if (df[["coverage", "count_m", "count_u"]] < 0).any().any():
        raise ValueError(f"{path.name}: negative read counts were found.")

    return df


def preprocess_cell(
    raw_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    """
    Preprocess one HCC cell for Work Package 5.1.

    The two strand representations of the same physical CpG are merged
    within the same cell:

        C/+ position p     -> canonical position p
        G/- position p + 1 -> canonical position p

    WP5.1 binary-label rule:

        count_m > count_u  -> label 1 (Methylated)
        count_m <= count_u -> label 0 (Unmethylated)

    Equal-count records are retained and assigned label 0.
    Different cells are never merged with one another.
    """
    cell_id = infer_cell_id(raw_path)
    if not is_wp51_hcc_cell(cell_id):
        raise ValueError(
            f"{cell_id} is outside the WP5.1 HCC range Ca_01-Ca_25."
        )

    raw_df = load_raw_file(raw_path)

    raw_df["canonical_position"] = np.where(
        raw_df["representation"].eq("C/+"),
        raw_df["position"],
        raw_df["position"] - 1,
    ).astype("int64")

    raw_df["is_c_plus"] = raw_df["representation"].eq("C/+").astype("int8")
    raw_df["is_g_minus"] = raw_df["representation"].eq("G/-").astype("int8")

    merged_df = (
        raw_df.groupby(
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

    # Project rule: equality belongs to the unmethylated class.
    merged_df["is_tie"] = (
        merged_df["count_m"].eq(merged_df["count_u"]).astype("int8")
    )
    merged_df["label"] = (
        merged_df["count_m"].gt(merged_df["count_u"]).astype("int8")
    )
    merged_df["methylation_state"] = merged_df["label"].map(
        {0: "Unmethylated", 1: "Methylated"}
    )
    merged_df.insert(0, "cell_id", cell_id)

    processed_columns = [
        "cell_id",
        "chrom",
        "canonical_position",
        "strand_support",
        "raw_record_count",
        "c_plus_record_count",
        "g_minus_record_count",
        "count_m",
        "count_u",
        "coverage",
        "methylation_ratio",
        "is_tie",
        "label",
        "methylation_state",
    ]

    processed_df = (
        merged_df[processed_columns]
        .sort_values(["chrom", "canonical_position"])
        .reset_index(drop=True)
    )

    tie_audit_df = (
        processed_df.loc[processed_df["is_tie"].eq(1)]
        .copy()
        .reset_index(drop=True)
    )

    # Validation: strand merging must preserve all read counts.
    if int(processed_df["count_m"].sum()) != int(raw_df["count_m"].sum()):
        raise RuntimeError(f"{cell_id}: methylated-read total changed.")

    if int(processed_df["count_u"].sum()) != int(raw_df["count_u"].sum()):
        raise RuntimeError(f"{cell_id}: unmethylated-read total changed.")

    duplicate_count = int(
        processed_df.duplicated(
            subset=["chrom", "canonical_position"]
        ).sum()
    )
    if duplicate_count:
        raise RuntimeError(
            f"{cell_id}: {duplicate_count} duplicated canonical CpGs remain."
        )

    if not processed_df["label"].isin([0, 1]).all():
        raise RuntimeError(f"{cell_id}: invalid labels were produced.")

    if not tie_audit_df.empty and not tie_audit_df["label"].eq(0).all():
        raise RuntimeError(f"{cell_id}: tie records must have label 0.")

    summary = {
        "cell_id": cell_id,
        "raw_record_count": len(raw_df),
        "c_plus_raw_count": int(raw_df["representation"].eq("C/+").sum()),
        "g_minus_raw_count": int(raw_df["representation"].eq("G/-").sum()),
        "merged_physical_cpg_count": len(processed_df),
        "both_strands_count": int(
            processed_df["strand_support"].eq("Both C/+ and G/-").sum()
        ),
        "c_plus_only_count": int(
            processed_df["strand_support"].eq("C/+ only").sum()
        ),
        "g_minus_only_count": int(
            processed_df["strand_support"].eq("G/- only").sum()
        ),
        "tie_count_retained_as_label_0": int(processed_df["is_tie"].sum()),
        "final_labeled_cpg_count": len(processed_df),
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
        "mean_coverage": float(processed_df["coverage"].mean()),
        "median_coverage": float(processed_df["coverage"].median()),
        "maximum_coverage": int(processed_df["coverage"].max()),
    }

    return processed_df, tie_audit_df, summary


def find_raw_files(
    input_dir: Path,
    pattern: str,
    selected_cells: list[str] | None,
) -> list[Path]:
    """Find only the 25 HCC files required for WP5.1."""
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
            "Preprocess the 25 GSE65364 human HCC scRRBS cells for "
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
    parser.add_argument(
        "--save-combined",
        action="store_true",
        help="Also save all processed cell-level CpGs in one compressed CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    cells_dir = args.output_dir / "cells"
    tie_audit_dir = args.output_dir / "tie_audit"
    cells_dir.mkdir(parents=True, exist_ok=True)
    tie_audit_dir.mkdir(parents=True, exist_ok=True)

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

    summaries: list[dict[str, float | int | str]] = []
    combined_frames: list[pd.DataFrame] = []

    for index, raw_path in enumerate(raw_files, start=1):
        cell_id = infer_cell_id(raw_path)
        print(
            f"[{index:02d}/{len(raw_files):02d}] "
            f"Preprocessing {cell_id}: {raw_path.name}"
        )

        processed_df, tie_audit_df, summary = preprocess_cell(raw_path)

        processed_path = cells_dir / f"{cell_id}_processed.csv.gz"
        tie_audit_path = tie_audit_dir / f"{cell_id}_tie_audit.csv.gz"

        processed_df.to_csv(
            processed_path,
            index=False,
            compression="gzip",
        )
        tie_audit_df.to_csv(
            tie_audit_path,
            index=False,
            compression="gzip",
        )

        summary["processed_file"] = str(processed_path)
        summary["tie_audit_file"] = str(tie_audit_path)
        summaries.append(summary)

        if args.save_combined:
            combined_frames.append(processed_df)

    summary_df = (
        pd.DataFrame(summaries)
        .sort_values("cell_id")
        .reset_index(drop=True)
    )
    summary_path = args.output_dir / "wp5_1_preprocessing_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if args.save_combined:
        combined_df = pd.concat(combined_frames, ignore_index=True)
        combined_path = args.output_dir / "wp5_1_all_cells_long.csv.gz"
        combined_df.to_csv(
            combined_path,
            index=False,
            compression="gzip",
        )
        print(f"Combined dataset: {combined_path}")

    print("\nWP5.1 preprocessing completed.")
    print(f"Per-cell datasets: {cells_dir}")
    print(f"Tie audit files: {tie_audit_dir}")
    print(f"Summary table: {summary_path}")
    print(
        summary_df[
            [
                "cell_id",
                "raw_record_count",
                "merged_physical_cpg_count",
                "tie_count_retained_as_label_0",
                "final_labeled_cpg_count",
                "unmethylated_count",
                "methylated_count",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
