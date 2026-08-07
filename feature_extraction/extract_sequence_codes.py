
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow.parquet as pq

from config.project_config import (
    BASE_ORDER,
    DATASET_DIR,
    PHYSICOCHEMICAL_FEATURES_DIR,
    PHYSICOCHEMICAL_SHARD_SIZE,
    SEQUENCE_CODES_DIR,
    SEQUENCE_LENGTH,
    UNKNOWN_BASE_CODE,
)

SPLIT_NAMES = ("train", "validation", "test")
COLUMNS_TO_READ = ["chrom", "canonical_position", "sequence", "label"]


def _build_base_code_lookup() -> np.ndarray:
    """(128,) uint8 ASCII-indexed lookup: base -> its 0-3 code, UNKNOWN_BASE_CODE otherwise."""
    lookup_table = np.full(128, UNKNOWN_BASE_CODE, dtype=np.uint8)

    for code, base in enumerate(BASE_ORDER):
        lookup_table[ord(base)] = code

    return lookup_table


def encode_sequences_to_base_codes(
    sequences: Sequence[str],
    expected_length: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """Vectorized ASCII lookup. Returns shape [number_of_samples, expected_length] uint8."""
    sequences = [str(sequence).strip().upper() for sequence in sequences]

    joined_sequences = "".join(sequences).encode("ascii")
    char_codes = np.frombuffer(joined_sequences, dtype=np.uint8).reshape(len(sequences), expected_length)

    base_code_lookup = _build_base_code_lookup()

    return base_code_lookup[char_codes]


def expand_codes_to_one_hot(codes: np.ndarray) -> np.ndarray:
   
    number_of_bases = len(BASE_ORDER)
    # Row `number_of_bases` (i.e. index == UNKNOWN_BASE_CODE) is the all-zero fallback.
    identity_with_zero_row = np.vstack(
        [np.eye(number_of_bases, dtype=np.float32), np.zeros((1, number_of_bases), dtype=np.float32)]
    )

    safe_codes = np.where(codes == UNKNOWN_BASE_CODE, number_of_bases, codes)

    return identity_with_zero_row[safe_codes].transpose(0, 2, 1)


def process_split(split_name: str, parquet_path: Path) -> dict:
    output_dir = SEQUENCE_CODES_DIR / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    physicochemical_manifest_path = PHYSICOCHEMICAL_FEATURES_DIR / split_name / "manifest.json"

    if not physicochemical_manifest_path.exists():
        raise FileNotFoundError(
            f"{physicochemical_manifest_path} does not exist. "
            "Run feature_extraction/extract_physicochemical.py first (shard boundaries must match)."
        )

    with physicochemical_manifest_path.open(encoding="utf-8") as file:
        physicochemical_manifest = json.load(file)

    expected_row_counts = [shard["row_count"] for shard in physicochemical_manifest["shards"]]

    parquet_file = pq.ParquetFile(parquet_path)

    shard_records = []
    total_rows = 0
    shard_index = 0

    for record_batch in parquet_file.iter_batches(
        batch_size=PHYSICOCHEMICAL_SHARD_SIZE,
        columns=COLUMNS_TO_READ,
    ):
        batch_dataframe = record_batch.to_pandas()

        if shard_index >= len(expected_row_counts):
            raise RuntimeError(
                f"{split_name}: produced more shards than the physicochemical manifest "
                f"({len(expected_row_counts)}) - shard boundaries no longer match."
            )

        if len(batch_dataframe) != expected_row_counts[shard_index]:
            raise RuntimeError(
                f"{split_name} shard {shard_index}: {len(batch_dataframe)} rows, expected "
                f"{expected_row_counts[shard_index]} (from the physicochemical manifest) - "
                "shard boundaries no longer match."
            )

        codes = encode_sequences_to_base_codes(
            sequences=batch_dataframe["sequence"].tolist(),
            expected_length=SEQUENCE_LENGTH,
        )

        shard_path = output_dir / f"{split_name}_sequence_codes_{shard_index:05d}.npz"

        np.savez_compressed(
            shard_path,
            codes=codes,
            labels=batch_dataframe["label"].to_numpy(dtype=np.int8),
            chromosomes=batch_dataframe["chrom"].astype(str).to_numpy(dtype="U32"),
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
                "codes_shape": list(codes.shape),
                "dtype": str(codes.dtype),
            }
        )

        print(
            f"{split_name} | shard {shard_index:05d} | rows: {shard_size:,} | "
            f"total: {total_rows:,}"
        )

        shard_index += 1

    if shard_index != len(expected_row_counts):
        raise RuntimeError(
            f"{split_name}: produced {shard_index} shards, expected {len(expected_row_counts)} "
            "(from the physicochemical manifest) - shard boundaries no longer match."
        )

    manifest_path = output_dir / "manifest.json"

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "split": split_name,
                "source_parquet": str(parquet_path),
                "shard_size": PHYSICOCHEMICAL_SHARD_SIZE,
                "total_rows": total_rows,
                "shard_count": shard_index,
                "codes_shape_per_sample": [SEQUENCE_LENGTH],
                "dtype": "uint8",
                "base_order": BASE_ORDER,
                "unknown_base_code": UNKNOWN_BASE_CODE,
                "note": "Use expand_codes_to_one_hot() to reconstruct the [4, 501] one-hot matrix at load time.",
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
    SEQUENCE_CODES_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []

    for split_name in SPLIT_NAMES:
        parquet_path = DATASET_DIR / f"{split_name}.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"{parquet_path} does not exist. Run preprocessing/preprocess_hepg2.py first."
            )

        print(f"\nProcessing split: {split_name}")
        summaries.append(process_split(split_name, parquet_path))

    summary_path = SEQUENCE_CODES_DIR / "extraction_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)

    print("\nSequence code extraction completed.")
    print(f"Summary: {summary_path}")

    for summary in summaries:
        print(f"  {summary['split']}: {summary['total_rows']:,} rows in {summary['shard_count']} shards")


if __name__ == "__main__":
    main()
