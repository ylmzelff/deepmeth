"""
Validate the physicochemical dinucleotide-code shards produced by
extract_physicochemical.py.

Checks, per split:
  1. manifest/shard bookkeeping matches the source parquet and the files on disk
  2. every shard has the right shape/dtype
  3. the center dinucleotide (index 250, always "CG" per preprocess.py's QC)
     has the "CG" code for every single row
  4. expanding the stored codes to the full [12, 500] matrix produces only
     finite values
  5. the strongest check: re-encode codes directly from the source parquet's
     real "sequence" column for a sample of rows and require an exact match
     against the stored shard - this catches misalignment or stale/mismatched
     shards, not just shape bugs

Usage (no arguments needed):

    python feature_extraction/validate_physicochemical.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow.parquet as pq

from config.project_config import PHYSICOCHEMICAL_FEATURES_DIR, PHYSICOCHEMICAL_PROPERTY_FILE, SEQUENCE_LENGTH
from feature_extraction.extract_physicochemical import (
    dinucleotide_codes,
    encode_sequences_to_codes,
    expand_codes_to_matrix,
    load_physicochemical_properties_di,
)

SPLIT_NAMES = ("train", "validation", "test")
SAMPLE_ROWS_FOR_RECOMPUTE_CHECK = 2000
CENTER_INDEX = SEQUENCE_LENGTH // 2  # 250, matches preprocess.py's center_dinucleotide slice


def load_manifest(split_name: str) -> dict:
    manifest_path = PHYSICOCHEMICAL_FEATURES_DIR / split_name / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} does not exist. Run feature_extraction/extract_physicochemical.py first."
        )

    return json.loads(manifest_path.read_text(encoding="utf-8"))


def check_manifest_consistency(split_name: str, manifest: dict) -> None:
    source_parquet = Path(manifest["source_parquet"])
    source_rows = pq.ParquetFile(source_parquet).metadata.num_rows

    if manifest["total_rows"] != source_rows:
        raise ValueError(
            f"{split_name}: manifest total_rows ({manifest['total_rows']:,}) != "
            f"source parquet rows ({source_rows:,})"
        )

    if len(manifest["shards"]) != manifest["shard_count"]:
        raise ValueError(f"{split_name}: manifest shard_count does not match the number of shard entries")

    row_sum = 0

    for shard_info in manifest["shards"]:
        shard_path = Path(shard_info["file"])

        if not shard_path.exists():
            raise FileNotFoundError(f"{split_name}: shard file listed in manifest is missing: {shard_path}")

        row_sum += shard_info["row_count"]

    if row_sum != manifest["total_rows"]:
        raise ValueError(
            f"{split_name}: shard row counts sum to {row_sum:,}, manifest total_rows is {manifest['total_rows']:,}"
        )

    print(f"[PASS] {split_name}: manifest matches source parquet ({source_rows:,} rows, {manifest['shard_count']} shards)")


def check_shard_shapes_and_center(split_name: str, manifest: dict, property_table: dict, cg_code: int) -> int:
    total_checked = 0

    for shard_info in manifest["shards"]:
        with np.load(shard_info["file"]) as shard:
            codes = shard["codes"]
            labels = shard["labels"]
            chromosomes = shard["chromosomes"]
            positions = shard["positions"]

            if codes.shape[1:] != (SEQUENCE_LENGTH - 1,):
                raise ValueError(f"{split_name}: shard {shard_info['shard_index']} has wrong codes shape {codes.shape}")

            if codes.dtype != np.uint8:
                raise ValueError(f"{split_name}: shard {shard_info['shard_index']} codes are {codes.dtype}, expected uint8")

            row_count = codes.shape[0]

            if len(labels) != row_count or len(chromosomes) != row_count or len(positions) != row_count:
                raise ValueError(f"{split_name}: shard {shard_info['shard_index']} array lengths are inconsistent")

            if not np.all(codes[:, CENTER_INDEX] == cg_code):
                raise ValueError(
                    f"{split_name}: shard {shard_info['shard_index']} has rows whose center dinucleotide "
                    "code is not 'CG' - possible misalignment"
                )

            expanded = expand_codes_to_matrix(codes, property_table)

            if not np.isfinite(expanded).all():
                raise ValueError(f"{split_name}: shard {shard_info['shard_index']} expands to NaN or infinite values")

            total_checked += row_count

    print(f"[PASS] {split_name}: {total_checked:,} rows have correct shape/dtype, center=='CG' everywhere, expand to finite values")

    return total_checked


def check_recompute_matches_source(split_name: str, manifest: dict, property_table: dict) -> None:
    first_shard_info = manifest["shards"][0]
    sample_size = min(SAMPLE_ROWS_FOR_RECOMPUTE_CHECK, first_shard_info["row_count"])

    with np.load(first_shard_info["file"]) as shard:
        stored_codes = shard["codes"][:sample_size]

    parquet_file = pq.ParquetFile(manifest["source_parquet"])
    first_batch = next(parquet_file.iter_batches(batch_size=sample_size, columns=["sequence"]))
    sequences = first_batch.to_pandas()["sequence"].tolist()

    recomputed_codes = encode_sequences_to_codes(
        sequences=sequences,
        property_table=property_table,
        expected_length=SEQUENCE_LENGTH,
    )

    if not np.array_equal(recomputed_codes, stored_codes):
        raise ValueError(
            f"{split_name}: re-encoding codes from the source parquet's own 'sequence' column does not "
            "exactly match the stored shard - the shard may be stale or misaligned"
        )

    print(
        f"[PASS] {split_name}: first {sample_size:,} rows re-encoded directly from source sequence "
        "exactly match the stored shard"
    )


def main() -> None:
    property_table = load_physicochemical_properties_di(PHYSICOCHEMICAL_PROPERTY_FILE)
    cg_code = dinucleotide_codes(property_table).index("CG")

    print("=" * 70)
    print("Validating physicochemical dinucleotide-code shards")
    print("=" * 70)

    for split_name in SPLIT_NAMES:
        print(f"\n{split_name}:")
        manifest = load_manifest(split_name)
        check_manifest_consistency(split_name, manifest)
        check_shard_shapes_and_center(split_name, manifest, property_table, cg_code)
        check_recompute_matches_source(split_name, manifest, property_table)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
