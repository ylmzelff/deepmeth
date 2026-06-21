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

DEFAULT_INPUT_DIR = Path("/content/deepmeth/data/raw/GSE65364/extracted")
DEFAULT_OUTPUT_DIR = Path(
    "/content/drive/MyDrive/1001_BioSeq_LLM/"
    "deepmeth_backup/GSE65364/processed_cells"
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


def preprocess_cell(
    raw_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    """
    Apply the selected primary preprocessing method.

    C/+ position p     -> canonical position p
    G/- position p + 1 -> canonical position p

    Methylated and unmethylated read counts are summed.
    Ties are removed before binary-label creation.
    """
    cell_id = infer_cell_id(raw_path)
    raw_df = load_raw_file(raw_path)

    raw_df["canonical_position"] = np.where(
        raw_df["representation"].eq("C/+"),
        raw_df["position"],
        raw_df["position"] - 1,
    ).astype("int64")

    raw_df["is_c_plus"] = (
        raw_df["representation"].eq("C/+").astype("int8")
    )
    raw_df["is_g_minus"] = (
        raw_df["representation"].eq("G/-").astype("int8")
    )

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

    processed_df = merged_df.loc[~tie_mask].copy().reset_index(drop=True)
    processed_df["label"] = (
        processed_df["count_m"].gt(processed_df["count_u"]).astype("int8")
    )
    processed_df["methylation_state"] = processed_df["label"].map(
        {0: "Unmethylated", 1: "Methylated"}
    )

    processed_df.insert(0, "cell_id", cell_id)
    tie_df.insert(0, "cell_id", cell_id)

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
        "label",
        "methylation_state",
    ]

    processed_df = (
        processed_df[processed_columns]
        .sort_values(["chrom", "canonical_position"])
        .reset_index(drop=True)
    )

    tie_df = (
        tie_df.sort_values(["chrom", "canonical_position"])
        .reset_index(drop=True)
    )

    # Validation: no read information may be lost during strand merging.
    if int(processed_df["count_m"].sum() + tie_df["count_m"].sum()) != int(
        raw_df["count_m"].sum()
    ):
        raise RuntimeError(f"{cell_id}: methylated-read total changed.")

    if int(processed_df["count_u"].sum() + tie_df["count_u"].sum()) != int(
        raw_df["count_u"].sum()
    ):
        raise RuntimeError(f"{cell_id}: unmethylated-read total changed.")

    if not processed_df["label"].isin([0, 1]).all():
        raise RuntimeError(f"{cell_id}: invalid labels were produced.")

    if processed_df["count_m"].eq(processed_df["count_u"]).any():
        raise RuntimeError(f"{cell_id}: tie records remain in processed data.")

    summary = {
        "cell_id": cell_id,
        "raw_record_count": len(raw_df),
        "c_plus_raw_count": int(raw_df["representation"].eq("C/+").sum()),
        "g_minus_raw_count": int(raw_df["representation"].eq("G/-").sum()),
        "merged_physical_cpg_count": len(merged_df),
        "both_strands_count": int(
            merged_df["strand_support"].eq("Both C/+ and G/-").sum()
        ),
        "c_plus_only_count": int(
            merged_df["strand_support"].eq("C/+ only").sum()
        ),
        "g_minus_only_count": int(
            merged_df["strand_support"].eq("G/- only").sum()
        ),
        "tie_count": len(tie_df),
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
        "mean_coverage": processed_df["coverage"].mean(),
        "median_coverage": processed_df["coverage"].median(),
        "maximum_coverage": processed_df["coverage"].max(),
    }

    return processed_df, tie_df, summary


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
            "Preprocess GSE65364 Ca_01-Ca_26 HCC RRBS files "
            "using canonical strand merging."
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
    parser.add_argument(
        "--save-combined",
        action="store_true",
        help="Also save all processed cells in one compressed CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cells_dir = args.output_dir / "cells"
    ties_dir = args.output_dir / "ties"
    cells_dir.mkdir(parents=True, exist_ok=True)
    ties_dir.mkdir(parents=True, exist_ok=True)

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

    summaries = []
    combined_frames = []

    for index, raw_path in enumerate(raw_files, start=1):
        cell_id = infer_cell_id(raw_path)
        print(
            f"[{index}/{len(raw_files)}] Preprocessing "
            f"{cell_id}: {raw_path.name}"
        )

        processed_df, tie_df, summary = preprocess_cell(raw_path)

        processed_path = cells_dir / f"{cell_id}_processed.csv.gz"
        tie_path = ties_dir / f"{cell_id}_ties.csv"

        processed_df.to_csv(
            processed_path,
            index=False,
            compression="gzip",
        )
        tie_df.to_csv(tie_path, index=False)

        summary["processed_file"] = str(processed_path)
        summary["tie_file"] = str(tie_path)
        summaries.append(summary)

        if args.save_combined:
            combined_frames.append(processed_df)

    summary_df = pd.DataFrame(summaries).sort_values("cell_id")
    summary_path = args.output_dir / "all_cells_preprocessing_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if args.save_combined:
        combined_df = pd.concat(combined_frames, ignore_index=True)
        combined_path = args.output_dir / "all_cells_processed_long.csv.gz"
        combined_df.to_csv(
            combined_path,
            index=False,
            compression="gzip",
        )
        print(f"Combined dataset: {combined_path}")

    print("\nPreprocessing completed.")
    print(f"Per-cell datasets: {cells_dir}")
    print(f"Removed tie sites: {ties_dir}")
    print(f"Summary table: {summary_path}")
    print(
        summary_df[
            [
                "cell_id",
                "raw_record_count",
                "merged_physical_cpg_count",
                "tie_count",
                "final_labeled_cpg_count",
                "unmethylated_count",
                "methylated_count",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
