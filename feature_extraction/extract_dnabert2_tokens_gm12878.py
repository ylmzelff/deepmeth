from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow.parquet as pq
import torch

from config.data_config.gm12878_config import GM12878_DATA_DIR
from config.project_config import (
    DNABERT_BATCH_SIZE,
    DNABERT_HIDDEN_SIZE,
    DNABERT_MODEL_NAME,
    DNABERT_MODEL_REVISION,
    DNABERT_SAVE_DTYPE,
    FOUNDATION_TOKEN_MAX_LENGTH,
    SEQUENCE_LENGTH,
)
from feature_extraction.extract_dnabert2 import load_frozen_model

SPLIT_NAMES = ("train", "validation", "test")
DATASET_DIR_GM12878 = GM12878_DATA_DIR / "proceed" / "disjoint_split"
FOUNDATION_TOKEN_FEATURES_DIR_GM12878 = GM12878_DATA_DIR / "dnabert2_token_features"

COLUMNS_TO_READ = ["chrom", "canonical_position", "sequence", "label"]

# Smaller than the 50k rows/shard used for sequence/physchem: at 128 tokens x
# 768 dims x float16, a 50k-row shard here would be ~9.8GB per file. Keeping
# shards ~1GB makes individual writes on the Drive-synced data folder less
# risky (see [[gdrive-sync-hazards]]) and makes the per-shard skip-if-exists
# resume below cheap to restart after an interruption.
SHARD_SIZE = 5_000


def embed_tokens(
    sequences: list[str],
    tokenizer,
    model,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (token_embeddings [N, FOUNDATION_TOKEN_MAX_LENGTH, 768], attention_mask [N, FOUNDATION_TOKEN_MAX_LENGTH])."""
    all_embeddings = []
    all_masks = []

    for batch_start in range(0, len(sequences), DNABERT_BATCH_SIZE):
        batch = sequences[batch_start : batch_start + DNABERT_BATCH_SIZE]

        encoded = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=FOUNDATION_TOKEN_MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**encoded)
                token_embeddings = outputs[0]  # [b, FOUNDATION_TOKEN_MAX_LENGTH, 768]

        if token_embeddings.shape != (len(batch), FOUNDATION_TOKEN_MAX_LENGTH, DNABERT_HIDDEN_SIZE):
            raise RuntimeError(f"Unexpected token embedding shape: {tuple(token_embeddings.shape)}")

        if not torch.isfinite(token_embeddings).all():
            raise FloatingPointError("NaN or Inf detected in token embeddings.")

        all_embeddings.append(token_embeddings.detach().cpu().float().numpy())
        all_masks.append(encoded["attention_mask"].detach().cpu().numpy())

    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_masks, axis=0)


def process_split(
    split_name: str,
    parquet_path: Path,
    tokenizer,
    model,
    device: torch.device,
    save_dtype,
) -> dict:
    output_dir = FOUNDATION_TOKEN_FEATURES_DIR_GM12878 / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = pq.ParquetFile(parquet_path)

    shard_records = []
    total_rows = 0
    shard_index = 0
    started_at = time.time()

    for record_batch in parquet_file.iter_batches(batch_size=SHARD_SIZE, columns=COLUMNS_TO_READ):
        shard_path = output_dir / f"{split_name}_dnabert2_tokens_{shard_index:05d}.npz"

        if shard_path.exists():
            with np.load(shard_path) as data:
                row_count = int(len(data["labels"]))

            total_rows += row_count
            shard_records.append(
                {
                    "split": split_name,
                    "shard_index": shard_index,
                    "file": str(shard_path),
                    "row_count": row_count,
                    "skipped": True,
                }
            )
            print(f"{split_name} | shard {shard_index:05d} | already exists, skipping ({row_count:,} rows)")
            shard_index += 1
            continue

        batch_dataframe = record_batch.to_pandas()
        sequences = batch_dataframe["sequence"].astype(str).str.upper().tolist()

        invalid_length_count = sum(len(sequence) != SEQUENCE_LENGTH for sequence in sequences)
        if invalid_length_count:
            raise ValueError(f"{invalid_length_count} invalid sequence lengths were detected.")

        token_embeddings, attention_mask = embed_tokens(sequences, tokenizer, model, device)

        np.savez_compressed(
            shard_path,
            token_embeddings=token_embeddings.astype(save_dtype),
            attention_mask=attention_mask.astype(np.uint8),
            labels=batch_dataframe["label"].to_numpy(dtype=np.int8),
            chromosomes=batch_dataframe["chrom"].astype(str).to_numpy(dtype="U32"),
            positions=batch_dataframe["canonical_position"].to_numpy(dtype=np.int64),
        )

        row_count = len(sequences)
        total_rows += row_count

        shard_records.append(
            {
                "split": split_name,
                "shard_index": shard_index,
                "file": str(shard_path),
                "row_count": row_count,
                "skipped": False,
            }
        )

        elapsed = time.time() - started_at
        rate = total_rows / elapsed if elapsed > 0 else 0.0
        print(
            f"{split_name} | shard {shard_index:05d} | rows: {row_count:,} | total: {total_rows:,} | "
            f"{rate:.1f} rows/s | elapsed {elapsed / 60:.1f}min"
        )

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
                "token_max_length": FOUNDATION_TOKEN_MAX_LENGTH,
                "embedding_dim": DNABERT_HIDDEN_SIZE,
                "dtype": DNABERT_SAVE_DTYPE,
                "pooling": "none (token-level, padded/truncated to a fixed length; see attention_mask)",
                "model_name": DNABERT_MODEL_NAME,
                "model_revision": DNABERT_MODEL_REVISION,
                "note": (
                    "token_embeddings: [row_count, token_max_length, embedding_dim]. "
                    "attention_mask: [row_count, token_max_length], 1 = real token, 0 = padding."
                ),
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
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")

    device = torch.device("cuda")
    save_dtype = np.float16 if DNABERT_SAVE_DTYPE == "float16" else np.float32

    FOUNDATION_TOKEN_FEATURES_DIR_GM12878.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GM12878 DNABERT-2 per-CpG token-feature extraction (foundation branch)")
    print("=" * 70)
    print(f"Token max length: {FOUNDATION_TOKEN_MAX_LENGTH}")
    print(f"Shard size: {SHARD_SIZE:,}")

    tokenizer, model = load_frozen_model(device)
    print("[PASS] Frozen DNABERT-2 loaded")

    started_at = time.time()
    summaries = []

    for split_name in SPLIT_NAMES:
        parquet_path = DATASET_DIR_GM12878 / f"{split_name}.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(f"{parquet_path} does not exist. Run preprocessing/preprocess_gm12878.py first.")

        print(f"\nProcessing split: {split_name}")
        summaries.append(process_split(split_name, parquet_path, tokenizer, model, device, save_dtype))

    elapsed_seconds = time.time() - started_at

    summary = {
        "created_at": datetime.now().isoformat(),
        "model_name": DNABERT_MODEL_NAME,
        "model_revision": DNABERT_MODEL_REVISION,
        "frozen": True,
        "token_max_length": FOUNDATION_TOKEN_MAX_LENGTH,
        "elapsed_seconds": float(elapsed_seconds),
        "splits": summaries,
    }

    summary_path = FOUNDATION_TOKEN_FEATURES_DIR_GM12878 / "extraction_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nGM12878 DNABERT-2 token-feature extraction completed.")
    print(f"Elapsed: {elapsed_seconds / 60:.1f} minutes")
    print(f"Summary: {summary_path}")

    for split_summary in summaries:
        print(f"  {split_summary['split']}: {split_summary['total_rows']:,} rows in {split_summary['shard_count']} shards")


if __name__ == "__main__":
    main()
