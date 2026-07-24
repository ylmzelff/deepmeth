from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from twobitreader import TwoBitFile


# ============================================================
# Default project paths
# ============================================================

# This file is expected at:
# deepmeth/data_preprocessing.py
# ============================================================
# Default project paths
# ============================================================

# Current file:
# deepmeth/data_preprocess/data_preprocessing.py
SCRIPT_DIR = Path(__file__).resolve().parent

# Main project directory:
# deepmeth/
PROJECT_ROOT = SCRIPT_DIR.parent

# Existing dataset directory:
# deepmeth_backup/GSE65364/
BACKUP_ROOT = (
    PROJECT_ROOT.parent
    / "deepmeth_backup"
    / "GSE65364"
)

# Existing raw scRRBS files
DEFAULT_INPUT_DIR = (
    BACKUP_ROOT
    / "data"
    / "raw"
    / "extracted"
)
# Existing compact hg19 reference
DEFAULT_REFERENCE_PATH = (
    BACKUP_ROOT
    / "wp5_1"
    / "reference"
    / "hg19.2bit"
)

# New results will be written inside the current project
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

DEFAULT_PATTERN = "*_Ca_*_RRBS.single.CpG.txt.gz"


# ============================================================
# Input and output directories
# ============================================================



# ============================================================
# Raw file columns
# ============================================================

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


# ============================================================
# Work Package 5.1 settings
# ============================================================

FIRST_HCC_CELL = 1
LAST_HCC_CELL = 25

SEQUENCE_LENGTH = 501
CENTER_INDEX = 250
MAX_UNKNOWN_FRACTION = 0.10


# ============================================================
# Cell utilities
# ============================================================

def infer_cell_id(path: Path) -> str:
    """
    Extract a cell identifier such as Ca_01 from a filename.
    """
    match = re.search(
        r"(Ca_\d{2})",
        path.name,
    )

    if match is None:
        raise ValueError(
            f"Cell ID could not be inferred from: {path.name}"
        )

    return match.group(1)


def cell_number(cell_id: str) -> int:
    """
    Convert Ca_01 into integer 1.
    """
    return int(
        cell_id.split("_")[1]
    )


def is_wp51_hcc_cell(cell_id: str) -> bool:
    """
    Return True for Ca_01-Ca_25.
    """
    number = cell_number(cell_id)

    return (
        FIRST_HCC_CELL
        <= number
        <= LAST_HCC_CELL
    )


# ============================================================
# Chromosome utilities
# ============================================================

def normalize_chromosome_name(
    chromosome: str,
) -> str:
    """
    Convert chromosome names to UCSC hg19 format.

    Examples:
        1    -> chr1
        chr1 -> chr1
        X    -> chrX
    """
    chromosome = str(
        chromosome
    ).strip()

    if chromosome.lower().startswith("chr"):
        suffix = chromosome[3:]
    else:
        suffix = chromosome

    if suffix.lower() == "x":
        suffix = "X"
    elif suffix.lower() == "y":
        suffix = "Y"
    elif suffix.lower() in {"m", "mt"}:
        suffix = "M"

    return f"chr{suffix}"


# ============================================================
# Raw file loading and validation
# ============================================================

def load_raw_file(
    path: Path,
) -> pd.DataFrame:
    """
    Load and validate one GSE65364 scRRBS file.
    """
    dataframe = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=RAW_COLUMNS,
        compression="gzip",
    )

    if dataframe.empty:
        raise ValueError(
            f"{path.name}: file is empty."
        )

    dataframe[NUMERIC_COLUMNS] = (
        dataframe[NUMERIC_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    invalid_numeric_columns = (
        dataframe[NUMERIC_COLUMNS]
        .columns[
            dataframe[
                NUMERIC_COLUMNS
            ].isna().any()
        ]
        .tolist()
    )

    if invalid_numeric_columns:
        raise ValueError(
            f"{path.name}: missing or non-numeric values "
            f"in columns {invalid_numeric_columns}."
        )

    if (
        dataframe[
            [
                "position",
                "coverage",
                "count_m",
                "count_u",
            ]
        ]
        < 0
    ).any().any():
        raise ValueError(
            f"{path.name}: negative coordinates or read "
            "counts were found."
        )

    dataframe["position"] = (
        dataframe["position"].astype("int64")
    )

    dataframe["coverage"] = (
        dataframe["coverage"].astype("int64")
    )

    dataframe["count_m"] = (
        dataframe["count_m"].astype("int64")
    )

    dataframe["count_u"] = (
        dataframe["count_u"].astype("int64")
    )

    dataframe["chrom"] = (
        dataframe["chrom"]
        .map(normalize_chromosome_name)
    )

    dataframe["base"] = (
        dataframe["base"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    dataframe["strand"] = (
        dataframe["strand"]
        .astype(str)
        .str.strip()
    )

    dataframe["representation"] = (
        dataframe["base"]
        + "/"
        + dataframe["strand"]
    )

    unexpected_representations = sorted(
        set(dataframe["representation"])
        - {
            "C/+",
            "G/-",
        }
    )

    if unexpected_representations:
        raise ValueError(
            f"{path.name}: unexpected base/strand "
            f"combinations: {unexpected_representations}"
        )

    calculated_coverage = (
        dataframe["count_m"]
        + dataframe["count_u"]
    )

    coverage_mismatch_count = int(
        (
            dataframe["coverage"]
            != calculated_coverage
        ).sum()
    )

    if coverage_mismatch_count:
        raise ValueError(
            f"{path.name}: {coverage_mismatch_count} coverage "
            "values do not equal count_m + count_u."
        )

    return dataframe


# ============================================================
# Physical CpG strand merging
# ============================================================

def merge_cpg_strands(
    raw_dataframe: pd.DataFrame,
    cell_id: str,
) -> pd.DataFrame:
    """
    Merge C/+ and G/- representations of the same physical CpG.

    Coordinate rule:

        C/+ at position p     -> canonical position p
        G/- at position p + 1 -> canonical position p
    """
    dataframe = raw_dataframe.copy()

    dataframe["canonical_position"] = np.where(
        dataframe["representation"].eq("C/+"),
        dataframe["position"],
        dataframe["position"] - 1,
    ).astype("int64")

    if (
        dataframe["canonical_position"]
        <= 0
    ).any():
        raise ValueError(
            f"{cell_id}: invalid canonical CpG "
            "coordinates were generated."
        )

    dataframe["is_c_plus"] = (
        dataframe["representation"]
        .eq("C/+")
        .astype("int8")
    )

    dataframe["is_g_minus"] = (
        dataframe["representation"]
        .eq("G/-")
        .astype("int8")
    )

    merged_dataframe = (
        dataframe.groupby(
            [
                "chrom",
                "canonical_position",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            raw_record_count=(
                "position",
                "size",
            ),
            c_plus_record_count=(
                "is_c_plus",
                "sum",
            ),
            g_minus_record_count=(
                "is_g_minus",
                "sum",
            ),
            count_m=(
                "count_m",
                "sum",
            ),
            count_u=(
                "count_u",
                "sum",
            ),
        )
    )

    merged_dataframe["count_m"] = (
        merged_dataframe["count_m"]
        .astype("int64")
    )

    merged_dataframe["count_u"] = (
        merged_dataframe["count_u"]
        .astype("int64")
    )

    merged_dataframe["coverage"] = (
        merged_dataframe["count_m"]
        + merged_dataframe["count_u"]
    )

    merged_dataframe["methylation_ratio"] = (
        merged_dataframe["count_m"]
        / merged_dataframe["coverage"].replace(
            0,
            np.nan,
        )
    )

    merged_dataframe["strand_support"] = np.select(
        [
            (
                merged_dataframe[
                    "c_plus_record_count"
                ].gt(0)
                & merged_dataframe[
                    "g_minus_record_count"
                ].gt(0)
            ),
            merged_dataframe[
                "c_plus_record_count"
            ].gt(0),
            merged_dataframe[
                "g_minus_record_count"
            ].gt(0),
        ],
        [
            "Both C/+ and G/-",
            "C/+ only",
            "G/- only",
        ],
        default="Unexpected",
    )

    # Project label rule:
    #
    # count_m > count_u  -> Methylated, label 1
    # count_m <= count_u -> Unmethylated, label 0
    merged_dataframe["is_tie"] = (
        merged_dataframe["count_m"]
        .eq(
            merged_dataframe["count_u"]
        )
        .astype("int8")
    )

    merged_dataframe["label"] = (
        merged_dataframe["count_m"]
        .gt(
            merged_dataframe["count_u"]
        )
        .astype("int8")
    )

    merged_dataframe["methylation_state"] = (
        merged_dataframe["label"].map(
            {
                0: "Unmethylated",
                1: "Methylated",
            }
        )
    )

    merged_dataframe.insert(
        0,
        "cell_id",
        cell_id,
    )

    # Validate that strand merging preserves read totals.
    if int(
        merged_dataframe["count_m"].sum()
    ) != int(
        raw_dataframe["count_m"].sum()
    ):
        raise RuntimeError(
            f"{cell_id}: methylated-read total changed "
            "during strand merging."
        )

    if int(
        merged_dataframe["count_u"].sum()
    ) != int(
        raw_dataframe["count_u"].sum()
    ):
        raise RuntimeError(
            f"{cell_id}: unmethylated-read total changed "
            "during strand merging."
        )

    duplicate_count = int(
        merged_dataframe.duplicated(
            subset=[
                "chrom",
                "canonical_position",
            ]
        ).sum()
    )

    if duplicate_count:
        raise RuntimeError(
            f"{cell_id}: {duplicate_count} duplicated "
            "canonical CpGs remain."
        )

    if not (
        merged_dataframe["label"]
        .isin([0, 1])
        .all()
    ):
        raise RuntimeError(
            f"{cell_id}: invalid binary labels were produced."
        )

    tie_mask = (
        merged_dataframe["is_tie"].eq(1)
    )

    if not (
        merged_dataframe.loc[
            tie_mask,
            "label",
        ]
        .eq(0)
        .all()
    ):
        raise RuntimeError(
            f"{cell_id}: tie records must have label 0."
        )

    return merged_dataframe


# ============================================================
# hg19.2bit sequence extraction
# ============================================================

def extract_cpg_centered_sequence(
    genome: TwoBitFile,
    reference_chromosomes: set[str],
    chromosome: str,
    cpg_position: int,
    sequence_length: int = SEQUENCE_LENGTH,
) -> str | None:
    """
    Extract an odd-length sequence centered on the cytosine
    of a physical CpG.

    cpg_position is expected to be a 1-based coordinate.

    For a 501-bp sequence:

        sequence[250]     == "C"
        sequence[250:252] == "CG"
    """
    if (
        sequence_length <= 0
        or sequence_length % 2 == 0
    ):
        raise ValueError(
            "sequence_length must be a positive odd number."
        )

    chromosome = normalize_chromosome_name(
        chromosome
    )

    if chromosome not in reference_chromosomes:
        return None

    cpg_position = int(
        cpg_position
    )

    if cpg_position <= 0:
        return None

    center_index = (
        sequence_length // 2
    )

    # Convert the 1-based cytosine position to 0-based.
    c_index = (
        cpg_position - 1
    )

    start = (
        c_index - center_index
    )

    end = (
        start + sequence_length
    )

    if start < 0:
        return None

    chromosome_length = len(
        genome[chromosome]
    )

    if end > chromosome_length:
        return None

    sequence = str(
        genome[chromosome][start:end]
    ).upper()

    return sequence


def add_reference_sequences(
    dataframe: pd.DataFrame,
    genome: TwoBitFile,
    sequence_length: int = SEQUENCE_LENGTH,
    max_unknown_fraction: float = MAX_UNKNOWN_FRACTION,
) -> pd.DataFrame:
    """
    Add hg19 CpG-centered sequences and sequence QC columns.
    """
    if not (
        0.0
        <= max_unknown_fraction
        <= 1.0
    ):
        raise ValueError(
            "max_unknown_fraction must be between 0 and 1."
        )

    sequence_dataframe = (
        dataframe.copy()
    )

    reference_chromosomes = set(
        genome.keys()
    )

    sequences: list[str | None] = []

    for chromosome, position in zip(
        sequence_dataframe["chrom"],
        sequence_dataframe[
            "canonical_position"
        ],
    ):
        sequence = extract_cpg_centered_sequence(
            genome=genome,
            reference_chromosomes=(
                reference_chromosomes
            ),
            chromosome=str(chromosome),
            cpg_position=int(position),
            sequence_length=sequence_length,
        )

        sequences.append(
            sequence
        )

    sequence_dataframe["sequence"] = (
        sequences
    )

    sequence_dataframe["sequence_length"] = (
        sequence_dataframe["sequence"]
        .str.len()
    )

    center_index = (
        sequence_length // 2
    )

    sequence_dataframe[
        "center_dinucleotide"
    ] = (
        sequence_dataframe["sequence"]
        .str.slice(
            center_index,
            center_index + 2,
        )
    )

    # Any character other than A/C/G/T is unknown.
    sequence_dataframe[
        "unknown_base_count"
    ] = (
        sequence_dataframe["sequence"]
        .str.count(r"[^ACGT]")
    )

    # Characters beyond A/C/G/T/N are unsupported.
    sequence_dataframe[
        "unsupported_base_count"
    ] = (
        sequence_dataframe["sequence"]
        .str.count(r"[^ACGTN]")
    )

    sequence_dataframe[
        "unknown_fraction"
    ] = (
        sequence_dataframe[
            "unknown_base_count"
        ]
        / sequence_length
    )

    missing_sequence = (
        sequence_dataframe["sequence"]
        .isna()
    )

    incorrect_length = (
        sequence_dataframe["sequence"]
        .notna()
        & ~sequence_dataframe[
            "sequence_length"
        ].eq(sequence_length)
    )

    unsupported_bases = (
        sequence_dataframe["sequence"]
        .notna()
        & sequence_dataframe[
            "unsupported_base_count"
        ].gt(0)
    )

    incorrect_center = (
        sequence_dataframe["sequence"]
        .notna()
        & sequence_dataframe[
            "sequence_length"
        ].eq(sequence_length)
        & ~sequence_dataframe[
            "center_dinucleotide"
        ].eq("CG")
    )

    too_many_unknowns = (
        sequence_dataframe["sequence"]
        .notna()
        & sequence_dataframe[
            "unknown_fraction"
        ].gt(max_unknown_fraction)
    )

    sequence_dataframe[
        "sequence_status"
    ] = np.select(
        [
            missing_sequence,
            incorrect_length,
            unsupported_bases,
            incorrect_center,
            too_many_unknowns,
        ],
        [
            "Missing chromosome or boundary sequence",
            "Incorrect sequence length",
            "Unsupported reference base",
            "Center is not CG",
            "Unknown fraction above threshold",
        ],
        default="Valid",
    )

    return sequence_dataframe


# ============================================================
# Complete preprocessing for one cell
# ============================================================

def preprocess_cell(
    raw_path: Path,
    genome: TwoBitFile,
    sequence_length: int = SEQUENCE_LENGTH,
    max_unknown_fraction: float = MAX_UNKNOWN_FRACTION,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float | int | str],
]:
    """
    Preprocess one HCC cell.

    Returns:
        processed_dataframe
        tie_audit_dataframe
        invalid_sequence_dataframe
        summary
    """
    cell_id = infer_cell_id(
        raw_path
    )

    if not is_wp51_hcc_cell(
        cell_id
    ):
        raise ValueError(
            f"{cell_id} is outside Ca_01-Ca_25."
        )

    raw_dataframe = load_raw_file(
        raw_path
    )

    merged_dataframe = merge_cpg_strands(
        raw_dataframe=raw_dataframe,
        cell_id=cell_id,
    )

    merged_physical_cpg_count = len(
        merged_dataframe
    )

    sequenced_dataframe = add_reference_sequences(
        dataframe=merged_dataframe,
        genome=genome,
        sequence_length=sequence_length,
        max_unknown_fraction=max_unknown_fraction,
    )

    tie_audit_dataframe = (
        sequenced_dataframe.loc[
            sequenced_dataframe[
                "is_tie"
            ].eq(1)
        ]
        .copy()
        .reset_index(drop=True)
    )

    invalid_sequence_dataframe = (
        sequenced_dataframe.loc[
            ~sequenced_dataframe[
                "sequence_status"
            ].eq("Valid")
        ]
        .copy()
        .reset_index(drop=True)
    )

    valid_sequence_dataframe = (
        sequenced_dataframe.loc[
            sequenced_dataframe[
                "sequence_status"
            ].eq("Valid")
        ]
        .copy()
        .reset_index(drop=True)
    )

    processed_columns = [
        "cell_id",
        "chrom",
        "canonical_position",
        "sequence",
        "sequence_length",
        "center_dinucleotide",
        "unknown_base_count",
        "unknown_fraction",
        "sequence_status",
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

    processed_dataframe = (
        valid_sequence_dataframe[
            processed_columns
        ]
        .sort_values(
            [
                "chrom",
                "canonical_position",
            ]
        )
        .reset_index(drop=True)
    )

    if processed_dataframe.empty:
        raise RuntimeError(
            f"{cell_id}: no valid CpG sequences remain."
        )

    if processed_dataframe.duplicated(
        subset=[
            "chrom",
            "canonical_position",
        ]
    ).any():
        raise RuntimeError(
            f"{cell_id}: duplicated CpGs remain "
            "in the processed dataset."
        )

    if not (
        processed_dataframe[
            "sequence_length"
        ]
        .eq(sequence_length)
        .all()
    ):
        raise RuntimeError(
            f"{cell_id}: invalid sequence lengths remain."
        )

    if not (
        processed_dataframe[
            "center_dinucleotide"
        ]
        .eq("CG")
        .all()
    ):
        raise RuntimeError(
            f"{cell_id}: sequences without a central CG remain."
        )

    summary: dict[
        str,
        float | int | str
    ] = {
        "cell_id": cell_id,

        "raw_record_count": len(
            raw_dataframe
        ),

        "c_plus_raw_count": int(
            raw_dataframe[
                "representation"
            ].eq("C/+").sum()
        ),

        "g_minus_raw_count": int(
            raw_dataframe[
                "representation"
            ].eq("G/-").sum()
        ),

        "merged_physical_cpg_count": (
            merged_physical_cpg_count
        ),

        "both_strands_count": int(
            merged_dataframe[
                "strand_support"
            ]
            .eq("Both C/+ and G/-")
            .sum()
        ),

        "c_plus_only_count": int(
            merged_dataframe[
                "strand_support"
            ]
            .eq("C/+ only")
            .sum()
        ),

        "g_minus_only_count": int(
            merged_dataframe[
                "strand_support"
            ]
            .eq("G/- only")
            .sum()
        ),

        "tie_count_retained_as_label_0": int(
            merged_dataframe[
                "is_tie"
            ].sum()
        ),

        "invalid_sequence_count": len(
            invalid_sequence_dataframe
        ),

        "missing_or_boundary_sequence_count": int(
            invalid_sequence_dataframe[
                "sequence_status"
            ]
            .eq(
                "Missing chromosome or boundary sequence"
            )
            .sum()
        ),

        "incorrect_sequence_length_count": int(
            invalid_sequence_dataframe[
                "sequence_status"
            ]
            .eq(
                "Incorrect sequence length"
            )
            .sum()
        ),

        "unsupported_reference_base_count": int(
            invalid_sequence_dataframe[
                "sequence_status"
            ]
            .eq(
                "Unsupported reference base"
            )
            .sum()
        ),

        "wrong_center_count": int(
            invalid_sequence_dataframe[
                "sequence_status"
            ]
            .eq(
                "Center is not CG"
            )
            .sum()
        ),

        "high_unknown_fraction_count": int(
            invalid_sequence_dataframe[
                "sequence_status"
            ]
            .eq(
                "Unknown fraction above threshold"
            )
            .sum()
        ),

        "final_labeled_cpg_count": len(
            processed_dataframe
        ),

        "unmethylated_count": int(
            processed_dataframe[
                "label"
            ].eq(0).sum()
        ),

        "methylated_count": int(
            processed_dataframe[
                "label"
            ].eq(1).sum()
        ),

        "unmethylated_percentage": round(
            processed_dataframe[
                "label"
            ].eq(0).mean() * 100,
            4,
        ),

        "methylated_percentage": round(
            processed_dataframe[
                "label"
            ].eq(1).mean() * 100,
            4,
        ),

        "mean_coverage": float(
            processed_dataframe[
                "coverage"
            ].mean()
        ),

        "median_coverage": float(
            processed_dataframe[
                "coverage"
            ].median()
        ),

        "maximum_coverage": int(
            processed_dataframe[
                "coverage"
            ].max()
        ),
    }

    return (
        processed_dataframe,
        tie_audit_dataframe,
        invalid_sequence_dataframe,
        summary,
    )


# ============================================================
# File discovery
# ============================================================

def find_raw_files(
    input_dir: Path,
    pattern: str,
    selected_cells: list[str] | None,
) -> list[Path]:
    """
    Find the HCC raw files used in WP5.1.
    """
    files = sorted(
        input_dir.glob(pattern)
    )

    wp51_files: list[Path] = []

    for path in files:
        cell_id = infer_cell_id(
            path
        )

        if is_wp51_hcc_cell(
            cell_id
        ):
            wp51_files.append(
                path
            )

    if selected_cells:
        normalized_selected_cells = [
            str(cell).strip()
            for cell in selected_cells
        ]

        invalid_cells = sorted(
            cell
            for cell in normalized_selected_cells
            if not is_wp51_hcc_cell(cell)
        )

        if invalid_cells:
            raise ValueError(
                "Only Ca_01-Ca_25 are valid. "
                "Invalid selection: "
                + ", ".join(invalid_cells)
            )

        selected_cell_set = set(
            normalized_selected_cells
        )

        wp51_files = [
            path
            for path in wp51_files
            if infer_cell_id(path)
            in selected_cell_set
        ]

    return sorted(
        wp51_files,
        key=lambda path: cell_number(
            infer_cell_id(path)
        ),
    )


# ============================================================
# Command-line arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess the GSE65364 HCC scRRBS "
            "cells for the DeepMeth model."
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
        "--reference-path",
        type=Path,
        default=DEFAULT_REFERENCE_PATH,
    )

    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
    )

    parser.add_argument(
        "--cells",
        nargs="*",
        default=None,
        help=(
            "Optional cell subset. Example: "
            "--cells Ca_01 Ca_02"
        ),
    )

    parser.add_argument(
        "--expected-cells",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=SEQUENCE_LENGTH,
    )

    parser.add_argument(
        "--max-unknown-fraction",
        type=float,
        default=MAX_UNKNOWN_FRACTION,
    )

    parser.add_argument(
        "--save-combined",
        action="store_true",
        help=(
            "Save all processed cell datasets in "
            "one compressed CSV file."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(
            "Input directory does not exist: "
            f"{args.input_dir}"
        )

    if not args.reference_path.exists():
        raise FileNotFoundError(
            "Reference genome does not exist: "
            f"{args.reference_path}"
        )

    if (
        args.sequence_length <= 0
        or args.sequence_length % 2 == 0
    ):
        raise ValueError(
            "--sequence-length must be a positive odd number."
        )

    if not (
        0.0
        <= args.max_unknown_fraction
        <= 1.0
    ):
        raise ValueError(
            "--max-unknown-fraction must be between 0 and 1."
        )

    cells_dir = (
        args.output_dir
        / "cells"
    )

    tie_audit_dir = (
        args.output_dir
        / "tie_audit"
    )

    invalid_sequence_dir = (
        args.output_dir
        / "invalid_sequence_audit"
    )

    cells_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tie_audit_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    invalid_sequence_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_files = find_raw_files(
        input_dir=args.input_dir,
        pattern=args.pattern,
        selected_cells=args.cells,
    )

    if not raw_files:
        raise FileNotFoundError(
            f"No HCC files were found in {args.input_dir} "
            f"with pattern {args.pattern}"
        )

    if (
        args.cells is None
        and len(raw_files)
        != args.expected_cells
    ):
        raise RuntimeError(
            f"Expected {args.expected_cells} files, "
            f"found {len(raw_files)}."
        )

    print(
        "Input directory:",
        args.input_dir,
    )

    print(
        "Reference genome:",
        args.reference_path,
    )

    print(
        "Output directory:",
        args.output_dir,
    )

    print(
        "Loading hg19.2bit reference..."
    )

    genome = TwoBitFile(
        str(args.reference_path)
    )

    summaries: list[
        dict[str, float | int | str]
    ] = []

    combined_frames: list[
        pd.DataFrame
    ] = []

    for index, raw_path in enumerate(
        raw_files,
        start=1,
    ):
        cell_id = infer_cell_id(
            raw_path
        )

        print(
            f"[{index:02d}/{len(raw_files):02d}] "
            f"Preprocessing {cell_id}: "
            f"{raw_path.name}"
        )

        (
            processed_dataframe,
            tie_audit_dataframe,
            invalid_sequence_dataframe,
            summary,
        ) = preprocess_cell(
            raw_path=raw_path,
            genome=genome,
            sequence_length=(
                args.sequence_length
            ),
            max_unknown_fraction=(
                args.max_unknown_fraction
            ),
        )

        processed_path = (
            cells_dir
            / f"{cell_id}_processed.csv.gz"
        )

        tie_audit_path = (
            tie_audit_dir
            / f"{cell_id}_tie_audit.csv.gz"
        )

        invalid_sequence_path = (
            invalid_sequence_dir
            / (
                f"{cell_id}"
                "_invalid_sequences.csv.gz"
            )
        )

        processed_dataframe.to_csv(
            processed_path,
            index=False,
            compression="gzip",
        )

        tie_audit_dataframe.to_csv(
            tie_audit_path,
            index=False,
            compression="gzip",
        )

        invalid_sequence_dataframe.to_csv(
            invalid_sequence_path,
            index=False,
            compression="gzip",
        )

        summary["processed_file"] = str(
            processed_path
        )

        summary["tie_audit_file"] = str(
            tie_audit_path
        )

        summary[
            "invalid_sequence_file"
        ] = str(
            invalid_sequence_path
        )

        summaries.append(
            summary
        )

        if args.save_combined:
            combined_frames.append(
                processed_dataframe
            )

    summary_dataframe = (
        pd.DataFrame(summaries)
        .sort_values("cell_id")
        .reset_index(drop=True)
    )

    summary_path = (
        args.output_dir
        / "wp5_1_preprocessing_summary.csv"
    )

    summary_dataframe.to_csv(
        summary_path,
        index=False,
    )

    if args.save_combined:
        if not combined_frames:
            raise RuntimeError(
                "No processed datasets are available "
                "for the combined output."
            )

        combined_dataframe = pd.concat(
            combined_frames,
            ignore_index=True,
        )

        combined_path = (
            args.output_dir
            / "wp5_1_all_cells_long.csv.gz"
        )

        combined_dataframe.to_csv(
            combined_path,
            index=False,
            compression="gzip",
        )

        print(
            "Combined dataset:",
            combined_path,
        )

    print(
        "\nWP5.1 preprocessing completed."
    )

    print(
        "Per-cell datasets:",
        cells_dir,
    )

    print(
        "Tie audit files:",
        tie_audit_dir,
    )

    print(
        "Invalid-sequence audit files:",
        invalid_sequence_dir,
    )

    print(
        "Summary table:",
        summary_path,
    )

    print()

    print(
        summary_dataframe[
            [
                "cell_id",
                "raw_record_count",
                "merged_physical_cpg_count",
                "invalid_sequence_count",
                "final_labeled_cpg_count",
                "tie_count_retained_as_label_0",
                "unmethylated_count",
                "methylated_count",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()