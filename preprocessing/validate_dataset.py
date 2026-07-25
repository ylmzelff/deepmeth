"""
Validate the train/validation/test parquet files produced by preprocess.py.

Checks: required columns, duplicate CpGs, sequence integrity (length/center),
label-vs-counts consistency, coverage arithmetic, chromosome- and
coordinate-disjointness across splits, and reports the class balance per split.

Raises on the first failed check, so it can gate the pipeline (e.g. before
burning GPU time on feature extraction / training). Prints a full summary
either way.

Usage (no arguments needed):

    python preprocessing/validate_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.project_config import DATASET_DIR, SEQUENCE_LENGTH

REQUIRED_COLUMNS = {
    "chrom",
    "canonical_position",
    "count_m",
    "count_u",
    "coverage",
    "label",
    "consensus_methylation_ratio",
    "sequence",
    "sequence_length",
    "center_dinucleotide",
}

SPLIT_NAMES = ("train", "validation", "test")


def load_splits() -> dict[str, pd.DataFrame]:
    splits = {}

    for split_name in SPLIT_NAMES:
        path = DATASET_DIR / f"{split_name}.parquet"

        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist. Run preprocessing/preprocess.py first.")

        splits[split_name] = pd.read_parquet(path)

    return splits


def check_required_columns(splits: dict[str, pd.DataFrame]) -> None:
    for split_name, dataframe in splits.items():
        missing = REQUIRED_COLUMNS - set(dataframe.columns)

        if missing:
            raise ValueError(f"{split_name}: missing columns {sorted(missing)}")

    print("[PASS] Required columns present in all splits")


def check_no_duplicates(splits: dict[str, pd.DataFrame]) -> None:
    for split_name, dataframe in splits.items():
        duplicate_count = int(dataframe.duplicated(subset=["chrom", "canonical_position"]).sum())

        if duplicate_count:
            raise ValueError(f"{split_name}: {duplicate_count} duplicated (chrom, canonical_position) rows")

    print("[PASS] No duplicated CpG coordinates within any split")


def check_sequence_integrity(splits: dict[str, pd.DataFrame]) -> None:
    for split_name, dataframe in splits.items():
        wrong_stored_length = int(dataframe["sequence_length"].ne(SEQUENCE_LENGTH).sum())
        wrong_actual_length = int(dataframe["sequence"].str.len().ne(SEQUENCE_LENGTH).sum())

        center_index = SEQUENCE_LENGTH // 2
        actual_center = dataframe["sequence"].str.slice(center_index, center_index + 2)
        wrong_stored_center = int(dataframe["center_dinucleotide"].ne("CG").sum())
        wrong_actual_center = int(actual_center.ne("CG").sum())

        if wrong_stored_length or wrong_actual_length:
            raise ValueError(
                f"{split_name}: sequence length mismatch "
                f"(stored={wrong_stored_length}, actual={wrong_actual_length})"
            )

        if wrong_stored_center or wrong_actual_center:
            raise ValueError(
                f"{split_name}: center dinucleotide is not CG "
                f"(stored={wrong_stored_center}, actual={wrong_actual_center})"
            )

    print(f"[PASS] Every sequence is {SEQUENCE_LENGTH}bp with a CG center")


def check_label_consistency(splits: dict[str, pd.DataFrame]) -> None:
    for split_name, dataframe in splits.items():
        invalid_labels = int((~dataframe["label"].isin([0, 1])).sum())

        if invalid_labels:
            raise ValueError(f"{split_name}: {invalid_labels} rows with a label outside {{0, 1}}")

        coverage_mismatch = int(dataframe["coverage"].ne(dataframe["count_m"] + dataframe["count_u"]).sum())

        if coverage_mismatch:
            raise ValueError(f"{split_name}: {coverage_mismatch} rows where coverage != count_m + count_u")

        expected_label = (dataframe["count_m"] > dataframe["count_u"]).astype("int8")
        label_mismatch = int(dataframe["label"].ne(expected_label).sum())

        if label_mismatch:
            raise ValueError(
                f"{split_name}: {label_mismatch} rows where label does not match "
                "the count_m > count_u rule"
            )

        invalid_ratio = int(
            (~dataframe["consensus_methylation_ratio"].between(0.0, 1.0)).sum()
        )

        if invalid_ratio:
            raise ValueError(f"{split_name}: {invalid_ratio} rows with consensus_methylation_ratio outside [0, 1]")

    print("[PASS] label matches count_m > count_u for every row; coverage and ratio are consistent")


def check_disjoint_splits(splits: dict[str, pd.DataFrame]) -> None:
    chromosomes = {name: set(dataframe["chrom"].unique()) for name, dataframe in splits.items()}

    for name_1, name_2 in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = chromosomes[name_1] & chromosomes[name_2]

        if overlap:
            raise ValueError(f"{name_1} and {name_2} share chromosomes: {sorted(overlap)}")

    coordinates = {
        name: set(zip(dataframe["chrom"], dataframe["canonical_position"]))
        for name, dataframe in splits.items()
    }

    for name_1, name_2 in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = coordinates[name_1] & coordinates[name_2]

        if overlap:
            raise ValueError(f"{name_1} and {name_2} share {len(overlap)} CpG coordinates")

    print("[PASS] train/validation/test are chromosome- and coordinate-disjoint")


def report_class_balance(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for split_name, dataframe in splits.items():
        rows.append(
            {
                "split": split_name,
                "row_count": len(dataframe),
                "chromosome_count": dataframe["chrom"].nunique(),
                "unmethylated_count": int(dataframe["label"].eq(0).sum()),
                "methylated_count": int(dataframe["label"].eq(1).sum()),
                "methylated_fraction": float(dataframe["label"].mean()),
                "mean_coverage": float(dataframe["coverage"].mean()),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 70)
    print("Validating preprocessing output")
    print("=" * 70)

    splits = load_splits()

    for split_name, dataframe in splits.items():
        print(f"{split_name}: {len(dataframe):,} rows loaded from {DATASET_DIR / f'{split_name}.parquet'}")

    print()
    check_required_columns(splits)
    check_no_duplicates(splits)
    check_sequence_integrity(splits)
    check_label_consistency(splits)
    check_disjoint_splits(splits)

    print("\nAll checks passed.\n")
    print(report_class_balance(splits).to_string(index=False))


if __name__ == "__main__":
    main()
