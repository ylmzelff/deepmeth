"""
Extract frozen DNABERT-2 embeddings for every CpG in train/validation/test.parquet
(the raw material for the graph branch's node features, before per-node mean
pooling in prepare_graph_features.py).

Requires a CUDA GPU. Each split is skipped automatically if it was already
fully extracted in a previous run (its *_extraction_summary.json exists) -
safe to just re-run this script if a Colab session drops mid-way.

Usage (no arguments needed):

    python feature_extraction/extract_dnabert2.py
"""

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
from transformers import AutoModel, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig

from config.project_config import (
    DATASET_DIR,
    DNABERT_BATCH_SIZE,
    DNABERT_HIDDEN_SIZE,
    DNABERT_FEATURES_DIR,
    DNABERT_MODEL_NAME,
    DNABERT_MODEL_REVISION,
    DNABERT_SAVE_DTYPE,
    DNABERT_SHARD_SIZE,
    DNABERT_TOKENIZER_MAX_LENGTH,
    SEQUENCE_LENGTH,
)

SPLIT_NAMES = ("train", "validation", "test")
COLUMNS_TO_READ = ["sequence", "label", "chrom", "canonical_position"]


def load_frozen_model(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        DNABERT_MODEL_NAME,
        revision=DNABERT_MODEL_REVISION,
        trust_remote_code=True,
    )

    config = BertConfig.from_pretrained(DNABERT_MODEL_NAME, revision=DNABERT_MODEL_REVISION)
    config.pad_token_id = tokenizer.pad_token_id
    config.output_hidden_states = False
    config.output_attentions = False
    config.return_dict = True

    model = AutoModel.from_pretrained(
        DNABERT_MODEL_NAME,
        revision=DNABERT_MODEL_REVISION,
        config=config,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )

    model = model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_parameters == 0

    return tokenizer, model


def masked_mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    summed = (token_embeddings * mask).sum(dim=1)
    valid_counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / valid_counts


def save_shard(output_path: Path, embeddings, labels, chromosomes, positions) -> None:
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        labels=np.asarray(labels, dtype=np.int8),
        chromosomes=np.asarray(chromosomes, dtype="U32"),
        positions=np.asarray(positions, dtype=np.int64),
    )


def process_split(
    split_name: str,
    parquet_path: Path,
    tokenizer,
    model,
    device: torch.device,
    save_dtype,
) -> dict:
    output_dir = DNABERT_FEATURES_DIR / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"{split_name}_extraction_summary.json"

    if summary_path.exists():
        print(f"{split_name}: already extracted, skipping ({summary_path})")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    parquet_file = pq.ParquetFile(parquet_path)
    source_rows = parquet_file.metadata.num_rows

    print(f"\n{split_name}: {source_rows:,} rows to embed")

    embedding_buffer, label_buffer, chrom_buffer, position_buffer = [], [], [], []
    processed_rows = 0
    shard_index = 0
    start_time = time.time()

    def flush_buffer() -> None:
        nonlocal shard_index, embedding_buffer, label_buffer, chrom_buffer, position_buffer

        if not embedding_buffer:
            return

        embeddings = np.concatenate(embedding_buffer, axis=0).astype(save_dtype, copy=False)
        labels = np.concatenate(label_buffer, axis=0)
        chromosomes = np.concatenate(chrom_buffer, axis=0)
        positions = np.concatenate(position_buffer, axis=0)

        while len(embeddings) > 0:
            take = min(DNABERT_SHARD_SIZE, len(embeddings))
            shard_path = output_dir / f"{split_name}_dnabert2_{shard_index:05d}.npz"

            save_shard(
                shard_path,
                embeddings[:take],
                labels[:take],
                chromosomes[:take],
                positions[:take],
            )

            print(f"  [SAVED] {shard_path.name} | rows={take:,}")

            shard_index += 1
            embeddings, labels = embeddings[take:], labels[take:]
            chromosomes, positions = chromosomes[take:], positions[take:]

        embedding_buffer, label_buffer, chrom_buffer, position_buffer = [], [], [], []

    for record_batch in parquet_file.iter_batches(batch_size=DNABERT_BATCH_SIZE, columns=COLUMNS_TO_READ):
        dataframe = record_batch.to_pandas()

        sequences = dataframe["sequence"].astype(str).str.upper().tolist()

        invalid_lengths = [i for i, seq in enumerate(sequences) if len(seq) != SEQUENCE_LENGTH]
        if invalid_lengths:
            raise ValueError(f"Invalid sequence length in batch, first local index: {invalid_lengths[0]}")

        encoded = tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=DNABERT_TOKENIZER_MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**encoded)
                pooled = masked_mean_pooling(outputs[0], encoded["attention_mask"])

        assert pooled.shape == (len(sequences), DNABERT_HIDDEN_SIZE)

        if not torch.isfinite(pooled).all():
            raise FloatingPointError("NaN or Inf detected in DNABERT embeddings.")

        embedding_buffer.append(pooled.detach().cpu().float().numpy())
        label_buffer.append(dataframe["label"].to_numpy(dtype=np.int8))
        chrom_buffer.append(dataframe["chrom"].astype(str).to_numpy(dtype="U32"))
        position_buffer.append(dataframe["canonical_position"].to_numpy(dtype=np.int64))

        processed_rows += len(dataframe)

        if sum(len(a) for a in embedding_buffer) >= DNABERT_SHARD_SIZE:
            flush_buffer()

        if processed_rows % (DNABERT_BATCH_SIZE * 50) == 0 or processed_rows == source_rows:
            elapsed = time.time() - start_time
            rate = processed_rows / elapsed if elapsed > 0 else 0
            print(f"  [PROGRESS] {processed_rows:,}/{source_rows:,} | {rate:.1f} rows/s")

    flush_buffer()

    elapsed_seconds = time.time() - start_time

    summary = {
        "created_at": datetime.now().isoformat(),
        "split": split_name,
        "model_name": DNABERT_MODEL_NAME,
        "model_revision": DNABERT_MODEL_REVISION,
        "frozen": True,
        "pooling": "attention_masked_mean",
        "source_rows": int(source_rows),
        "processed_rows": int(processed_rows),
        "shard_count": int(shard_index),
        "embedding_dimension": DNABERT_HIDDEN_SIZE,
        "embedding_dtype": DNABERT_SAVE_DTYPE,
        "elapsed_seconds": float(elapsed_seconds),
        "rows_per_second": float(processed_rows / elapsed_seconds) if elapsed_seconds > 0 else 0.0,
    }

    assert processed_rows == source_rows

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{split_name}: completed in {elapsed_seconds / 60:.1f} minutes")

    return summary


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for DNABERT-2 embedding extraction.")

    device = torch.device("cuda")
    save_dtype = np.float16 if DNABERT_SAVE_DTYPE == "float16" else np.float32

    print("=" * 70)
    print("DNABERT-2 embedding extraction")
    print("=" * 70)
    print(f"Model: {DNABERT_MODEL_NAME} (revision {DNABERT_MODEL_REVISION})")
    print(f"Device: {device}")

    tokenizer, model = load_frozen_model(device)
    print("[PASS] Frozen DNABERT-2 loaded")

    for split_name in SPLIT_NAMES:
        parquet_path = DATASET_DIR / f"{split_name}.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(f"{parquet_path} does not exist. Run preprocessing/preprocess.py first.")

        process_split(split_name, parquet_path, tokenizer, model, device, save_dtype)

    print("\nDNABERT-2 extraction completed for all splits.")


if __name__ == "__main__":
    main()
