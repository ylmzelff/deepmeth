from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow.parquet as pq

from config.project_config import BASE_ORDER, SEQUENCE_LENGTH, UNKNOWN_BASE_CODE
from config.data_config.gm12878_config import GM12878_DATA_DIR
from feature_extraction.extract_sequence_codes import encode_sequences_to_base_codes

DATASET_DIR_GM12878 = GM12878_DATA_DIR / "proceed" / "disjoint_split"
SEQUENCE_CODES_DIR_GM12878 = GM12878_DATA_DIR / "sequence_codes"

SPLIT_NAMES = ("train", "validation", "test")
COLUMNS_TO_READ = ["chrom", "canonical_position", "sequence", "label"]
SHARD_SIZE = 50_000  # same order of magnitude as PHYSICOCHEMICAL_SHARD_SIZE, no real need to match exactly here


def process_split(split_name: str, parquet_path: Path) -> dict:
    output_dir = SEQUENCE_CODES_DIR_GM12878 / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = pq.ParquetFile(parquet_path)

    shard_records = []
    total_rows = 0
    shard_index = 0

    for record_batch in parquet_file.iter_batches(batch_size=SHARD_SIZE, columns=COLUMNS_TO_READ):
        batch_dataframe = record_batch.to_pandas()

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

        print(f"{split_name} | shard {shard_index:05d} | rows: {shard_size:,} | total: {total_rows:,}")
        shard_index += 1

    manifest_path = output_dir / "manifest.json"

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "split": split_name,
                "source_parquet": str(parquet_path),
                "shard_size": SHARD_SIZE,
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

    return {"split": split_name, "total_rows": total_rows, "shard_count": shard_index, "manifest": str(manifest_path)}


def main() -> None:
    if not DATASET_DIR_GM12878.exists():
        raise FileNotFoundError(f"{DATASET_DIR_GM12878} does not exist. Run preprocessing/preprocess_gm12878.py first.")

    SEQUENCE_CODES_DIR_GM12878.mkdir(parents=True, exist_ok=True)

    summaries = []

    for split_name in SPLIT_NAMES:
        parquet_path = DATASET_DIR_GM12878 / f"{split_name}.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(f"{parquet_path} does not exist. Run preprocessing/preprocess_gm12878.py first.")

        print(f"\nProcessing split: {split_name}")
        summaries.append(process_split(split_name, parquet_path))

    summary_path = SEQUENCE_CODES_DIR_GM12878 / "extraction_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)

    print("\nGM12878 sequence code extraction completed.")
    print(f"Summary: {summary_path}")

    for summary in summaries:
        print(f"  {summary['split']}: {summary['total_rows']:,} rows in {summary['shard_count']} shards")


if __name__ == "__main__":
    main()
