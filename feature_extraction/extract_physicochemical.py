"""
Extract the [12, 500] dinucleotide physicochemical feature matrix for every
CpG in train/validation/test.parquet (physicochemical CNN branch input).

Reads each split in shards (parquet files are tens of millions of rows at
this dataset's scale, too large to convert in one array), writes compressed
.npz shards plus a manifest per split.

Usage (no arguments needed):

    python feature_extraction/extract_physicochemical.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow.parquet as pq
from openpyxl import load_workbook

from config.project_config import (
    DATASET_DIR,
    PHYSICOCHEMICAL_FEATURES_DIR,
    PHYSICOCHEMICAL_PROPERTY_FILE,
    PHYSICOCHEMICAL_SHARD_SIZE,
    SEQUENCE_LENGTH,
)

SPLIT_NAMES = ("train", "validation", "test")
COLUMNS_TO_READ = ["chrom", "canonical_position", "sequence", "label"]

# All possible DNA dinucleotides.
EXPECTED_DINUCLEOTIDES = {f"{first}{second}" for first in "ACGT" for second in "ACGT"}


def load_physicochemical_properties_di(file_path: str | Path) -> dict[str, np.ndarray]:
    """
    Load the normalized dinucleotide physicochemical property table.

    Expected Excel structure: column A = dinucleotide, columns B-M = 12
    physicochemical properties. Returns a dict mapping each dinucleotide to
    a float32 vector of 12 properties.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Physicochemical property file not found: {file_path}")

    workbook = load_workbook(filename=file_path, read_only=True, data_only=True)
    worksheet = workbook.active

    property_table: dict[str, np.ndarray] = {}

    try:
        for row_number, row in enumerate(worksheet.iter_rows(min_row=2, values_only=True), start=2):
            dinucleotide = str(row[0]).strip().upper()
            properties = np.asarray(row[1:13], dtype=np.float32)

            if dinucleotide not in EXPECTED_DINUCLEOTIDES:
                raise ValueError(f"Invalid dinucleotide '{dinucleotide}' at Excel row {row_number}.")

            if properties.shape != (12,):
                raise ValueError(f"Expected 12 properties for {dinucleotide}, received {properties.shape[0]}.")

            if not np.isfinite(properties).all():
                raise ValueError(f"Non-finite physicochemical value found for {dinucleotide}.")

            property_table[dinucleotide] = properties
    finally:
        workbook.close()

    missing_dinucleotides = EXPECTED_DINUCLEOTIDES.difference(property_table.keys())

    if missing_dinucleotides:
        raise ValueError("Missing dinucleotide properties: " + ", ".join(sorted(missing_dinucleotides)))

    return property_table


def convertSampleToPhyChemVector_Di(
    sampleSeq: Sequence[str],
    PhyChemPropTable_Di: Mapping[str, Sequence[float]],
    expected_length: int = SEQUENCE_LENGTH,
    unknown_strategy: str = "zero",
) -> np.ndarray:
    """
    Convert DNA sequences into dinucleotide physicochemical matrices.

    Follows the original Yeast Promoter implementation: every adjacent
    dinucleotide is represented using 12 physicochemical properties.

    Returns an array shaped [number_of_samples, 12, expected_length - 1]
    (i.e. [N, 12, 500] for 501bp sequences).

    unknown_strategy controls how dinucleotides containing "N" are handled:
    "zero" leaves them as the zero vector, "error" raises instead.
    """
    if len(sampleSeq) == 0:
        raise ValueError("sampleSeq must contain at least one DNA sequence.")

    sequences = [str(sequence).strip().upper() for sequence in sampleSeq]

    if unknown_strategy not in {"zero", "error"}:
        raise ValueError("unknown_strategy must be either 'zero' or 'error'.")

    for sample_index, sequence in enumerate(sequences):
        if len(sequence) != expected_length:
            raise ValueError(f"Sequence {sample_index} has length {len(sequence)}; expected {expected_length}.")

        invalid_bases = set(sequence).difference({"A", "C", "G", "T", "N"})

        if invalid_bases:
            raise ValueError(
                f"Sequence {sample_index} contains invalid bases: " + ", ".join(sorted(invalid_bases))
            )

    physicochemical_matrix = np.zeros((len(sequences), 12, expected_length - 1), dtype=np.float32)

    for sample_number, sequence in enumerate(sequences):
        for position in range(expected_length - 1):
            dinucleotide = sequence[position : position + 2]

            if dinucleotide in PhyChemPropTable_Di:
                physicochemical_matrix[sample_number, :, position] = np.asarray(
                    PhyChemPropTable_Di[dinucleotide], dtype=np.float32
                )
            elif "N" in dinucleotide:
                if unknown_strategy == "zero":
                    continue
                raise ValueError(
                    f"Unknown dinucleotide '{dinucleotide}' found in sequence {sample_number} "
                    f"at position {position}."
                )
            else:
                raise ValueError(
                    f"Dinucleotide '{dinucleotide}' does not exist in the physicochemical property table."
                )

    return physicochemical_matrix


def process_split(split_name: str, parquet_path: Path, property_table: dict) -> dict:
    output_dir = PHYSICOCHEMICAL_FEATURES_DIR / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = pq.ParquetFile(parquet_path)

    shard_records = []
    total_rows = 0
    shard_index = 0

    for record_batch in parquet_file.iter_batches(
        batch_size=PHYSICOCHEMICAL_SHARD_SIZE,
        columns=COLUMNS_TO_READ,
    ):
        batch_dataframe = record_batch.to_pandas()

        features = convertSampleToPhyChemVector_Di(
            sampleSeq=batch_dataframe["sequence"].tolist(),
            PhyChemPropTable_Di=property_table,
            expected_length=SEQUENCE_LENGTH,
            unknown_strategy="zero",
        )

        shard_path = output_dir / f"{split_name}_physicochemical_{shard_index:05d}.npz"

        np.savez_compressed(
            shard_path,
            features=features,
            labels=batch_dataframe["label"].to_numpy(dtype=np.int8),
            chromosomes=batch_dataframe["chrom"].astype(str).to_numpy(),
            positions=batch_dataframe["canonical_position"].to_numpy(dtype=np.int64),
        )

        shard_size = len(batch_dataframe)
        total_rows += shard_size

        shard_records.append(
            {
                "split": split_name,
                "shard_index": shard_index,
                "file": str(shard_path),
                "row_count": shard_size,
                "feature_shape": list(features.shape),
                "dtype": str(features.dtype),
            }
        )

        print(
            f"{split_name} | shard {shard_index:05d} | rows: {shard_size:,} | "
            f"total: {total_rows:,}"
        )

        shard_index += 1

    manifest_path = output_dir / "manifest.json"

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "split": split_name,
                "source_parquet": str(parquet_path),
                "shard_size": PHYSICOCHEMICAL_SHARD_SIZE,
                "total_rows": total_rows,
                "shard_count": shard_index,
                "feature_shape_per_sample": [12, SEQUENCE_LENGTH - 1],
                "dtype": "float32",
                "shards": shard_records,
            },
            file,
            indent=2,
        )

    return {
        "split": split_name,
        "total_rows": total_rows,
        "shard_count": shard_index,
        "manifest": str(manifest_path),
    }


def main() -> None:
    PHYSICOCHEMICAL_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    property_table = load_physicochemical_properties_di(PHYSICOCHEMICAL_PROPERTY_FILE)
    print(f"Loaded physicochemical properties for {len(property_table)} dinucleotides.")

    summaries = []

    for split_name in SPLIT_NAMES:
        parquet_path = DATASET_DIR / f"{split_name}.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"{parquet_path} does not exist. Run preprocessing/preprocess.py first."
            )

        print(f"\nProcessing split: {split_name}")
        summaries.append(process_split(split_name, parquet_path, property_table))

    summary_path = PHYSICOCHEMICAL_FEATURES_DIR / "extraction_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)

    print("\nPhysicochemical feature extraction completed.")
    print(f"Summary: {summary_path}")

    for summary in summaries:
        print(f"  {summary['split']}: {summary['total_rows']:,} rows in {summary['shard_count']} shards")


if __name__ == "__main__":
    main()
