"""
Create DNABERT-2 node features for the Hi-C graph.

CpGs from train+validation+test are combined (chromosome-disjoint splits
already prevent leakage - a node's chromosome belongs to exactly one split,
and DNABERT-2 embeddings are frozen/unsupervised, so there is no label
information to leak). Using train CpGs only would leave every node on a
validation/test chromosome with zero contributing CpGs, since those
chromosomes never appear in the train split at all.

CpGs are mapped to 100-kb genomic nodes. At most a fixed number of CpGs is
selected deterministically from each node. Their frozen DNABERT-2
embeddings are mean-pooled to produce one 768-dimensional feature vector
per node. Output is one file per chromosome, so an interrupted run can
just be re-run - completed chromosomes are skipped.

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
import pandas as pd
import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig

from config.project_config import (
    DATASET_DIR,
    DNABERT_BATCH_SIZE,
    DNABERT_HIDDEN_SIZE,
    DNABERT_MAX_CPG_PER_NODE,
    DNABERT_MODEL_NAME,
    DNABERT_MODEL_REVISION,
    DNABERT_NODE_FEATURES_DIR,
    DNABERT_SAVE_DTYPE,
    DNABERT_TOKENIZER_MAX_LENGTH,
    GRAPH_RESOLUTION,
    SEQUENCE_LENGTH,
)

SPLIT_NAMES = ("train", "validation", "test")
SPLIT_PATHS = {split: DATASET_DIR / f"{split}.parquet" for split in SPLIT_NAMES}

COLUMNS_TO_READ = ["chrom", "canonical_position", "sequence"]


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

    # Avoid DNABERT-2's custom Triton flash-attention kernel, which needs
    # more shared memory than some GPUs (e.g. T4) provide. Forcing a
    # nonzero dropout routes the model through its plain PyTorch attention
    # path instead. Harmless: the model runs in eval()/inference_mode(),
    # where nn.Dropout is always a no-op regardless of its probability.
    config.attention_probs_dropout_prob = 0.1

    model = AutoModel.from_pretrained(
        DNABERT_MODEL_NAME,
        revision=DNABERT_MODEL_REVISION,
        config=config,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
        # Newer transformers versions decoupled "fast init" (constructs
        # parameters on a torch.device("meta") placeholder, then materializes
        # them from the checkpoint) from low_cpu_mem_usage - it now defaults
        # to True regardless. DNABERT-2's own bert_layers.py isn't meta-
        # device-aware: BertEncoder.__init__ eagerly computes a real ALiBi
        # bias tensor on CPU during construction, before the meta-initialized
        # parameters are materialized, so the two end up on different
        # devices ("Tensor on device meta is not on the expected device
        # cpu!"). _fast_init=False forces plain eager (CPU) construction,
        # side-stepping the mismatch - this repo's own model/*.py modules
        # aren't affected (they're not loaded via from_pretrained).
        _fast_init=False,
    )

    model = model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if trainable_parameters != 0:
        raise RuntimeError("DNABERT-2 must remain completely frozen.")

    return tokenizer, model


def masked_mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    summed = (token_embeddings * mask).sum(dim=1)
    valid_counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / valid_counts


def find_all_chromosomes() -> list[str]:
    """Union of chromosomes across train+validation+test (not just train)."""
    chromosomes: set[str] = set()

    for split_name, path in SPLIT_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist. Run preprocessing/preprocess.py first.")

        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=1_000_000, columns=["chrom"]):
            chromosomes.update(batch.column("chrom").to_pylist())

    def chromosome_key(chromosome: str) -> tuple[int, str]:
        suffix = chromosome.removeprefix("chr")
        return (int(suffix), "") if suffix.isdigit() else (10_000, suffix)

    return sorted(chromosomes, key=chromosome_key)


def load_chromosome(chromosome: str) -> pd.DataFrame:
    """Load one chromosome's CpGs from train+validation+test combined."""
    frames = []

    for split_name, path in SPLIT_PATHS.items():
        table = pq.read_table(path, columns=COLUMNS_TO_READ, filters=[("chrom", "=", chromosome)])
        frame = table.to_pandas()
        frames.append(frame)
        print(f"  {split_name}: {len(frame):,} CpGs on {chromosome}")

    dataframe = pd.concat(frames, ignore_index=True)
    dataframe = dataframe.sort_values("canonical_position").reset_index(drop=True)

    dataframe["bin_start"] = (
        (dataframe["canonical_position"] - 1) // GRAPH_RESOLUTION * GRAPH_RESOLUTION
    ).astype("int64")

    return dataframe


def select_cpgs_per_node(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Select CpGs deterministically and approximately evenly across every 100-kb node."""
    bin_starts = dataframe["bin_start"].to_numpy(dtype=np.int64)

    unique_bins, first_indices, counts = np.unique(bin_starts, return_index=True, return_counts=True)

    selected_indices: list[np.ndarray] = []

    for start, count in zip(first_indices, counts):
        if count <= DNABERT_MAX_CPG_PER_NODE:
            local_indices = np.arange(count, dtype=np.int64)
        else:
            local_indices = np.linspace(0, count - 1, num=DNABERT_MAX_CPG_PER_NODE, dtype=np.int64)
            local_indices = np.unique(local_indices)

        selected_indices.append(start + local_indices)

    selected = dataframe.iloc[np.concatenate(selected_indices)].copy().reset_index(drop=True)

    print(f"  Nodes: {len(unique_bins):,}")
    print(f"  Source CpGs: {len(dataframe):,}")
    print(f"  Selected CpGs: {len(selected):,}")

    return selected


def embed_and_pool_nodes(
    dataframe: pd.DataFrame,
    tokenizer,
    model,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_bins, inverse_indices = np.unique(dataframe["bin_start"].to_numpy(dtype=np.int64), return_inverse=True)

    node_sums = np.zeros((len(unique_bins), DNABERT_HIDDEN_SIZE), dtype=np.float32)
    node_counts = np.zeros(len(unique_bins), dtype=np.int32)

    total_rows = len(dataframe)
    started_at = time.time()

    for batch_start in range(0, total_rows, DNABERT_BATCH_SIZE):
        batch_end = min(batch_start + DNABERT_BATCH_SIZE, total_rows)
        batch = dataframe.iloc[batch_start:batch_end]

        sequences = batch["sequence"].astype(str).str.upper().tolist()

        invalid_length_count = sum(len(sequence) != SEQUENCE_LENGTH for sequence in sequences)
        if invalid_length_count:
            raise ValueError(f"{invalid_length_count} invalid sequence lengths were detected.")

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

        if pooled.shape != (len(sequences), DNABERT_HIDDEN_SIZE):
            raise RuntimeError(f"Unexpected embedding shape: {tuple(pooled.shape)}")

        if not torch.isfinite(pooled).all():
            raise FloatingPointError("NaN or Inf detected in embeddings.")

        batch_embeddings = pooled.detach().cpu().float().numpy()
        batch_node_indices = inverse_indices[batch_start:batch_end]

        np.add.at(node_sums, batch_node_indices, batch_embeddings)
        np.add.at(node_counts, batch_node_indices, 1)

        processed = batch_end

        if processed % (DNABERT_BATCH_SIZE * 50) == 0 or processed == total_rows:
            elapsed = time.time() - started_at
            rate = processed / elapsed if elapsed > 0 else 0.0
            print(f"  [PROGRESS] {processed:,}/{total_rows:,} | {rate:.1f} rows/s")

    if (node_counts == 0).any():
        raise RuntimeError("At least one selected node received no embeddings.")

    node_features = node_sums / node_counts[:, None]

    return unique_bins, node_features, node_counts


def process_chromosome(chromosome: str, tokenizer, model, device: torch.device, save_dtype) -> dict:
    output_path = DNABERT_NODE_FEATURES_DIR / f"{chromosome}_dnabert2_node_features.npz"

    if output_path.exists():
        print(f"{chromosome}: already completed, skipping")

        with np.load(output_path) as data:
            return {
                "chrom": chromosome,
                "node_count": int(len(data["bin_starts"])),
                "selected_cpg_count": int(data["sample_counts"].sum()),
                "output_path": str(output_path),
                "skipped": True,
            }

    print("\n" + "=" * 70)
    print(f"Processing chromosome: {chromosome}")
    print("=" * 70)

    dataframe = load_chromosome(chromosome)
    selected = select_cpgs_per_node(dataframe)
    bin_starts, node_features, sample_counts = embed_and_pool_nodes(selected, tokenizer, model, device)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        chromosomes=np.full(len(bin_starts), chromosome, dtype="U16"),
        bin_starts=bin_starts.astype(np.int64),
        embeddings=node_features.astype(save_dtype),
        sample_counts=sample_counts.astype(np.int16),
    )

    print(f"[SAVED] {output_path}")
    print(f"  Node features: {node_features.shape}")

    return {
        "chrom": chromosome,
        "source_cpg_count": int(len(dataframe)),
        "selected_cpg_count": int(len(selected)),
        "node_count": int(len(bin_starts)),
        "output_path": str(output_path),
        "skipped": False,
    }


def main() -> None:
    for path in SPLIT_PATHS.values():
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist. Run preprocessing/preprocess.py first.")

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")

    device = torch.device("cuda")
    save_dtype = np.float16 if DNABERT_SAVE_DTYPE == "float16" else np.float32

    DNABERT_NODE_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DNABERT-2 node-feature extraction (train + validation + test)")
    print("=" * 70)
    print(f"Datasets: {list(SPLIT_PATHS.values())}")
    print(f"Graph resolution: {GRAPH_RESOLUTION:,}")
    print(f"Maximum CpGs per node: {DNABERT_MAX_CPG_PER_NODE}")
    print(f"Device: {device}")

    tokenizer, model = load_frozen_model(device)
    print("[PASS] Frozen DNABERT-2 loaded")

    chromosomes = find_all_chromosomes()
    print(f"Chromosomes (train+validation+test): {chromosomes}")

    started_at = time.time()
    chromosome_summaries = [
        process_chromosome(chromosome, tokenizer, model, device, save_dtype) for chromosome in chromosomes
    ]

    elapsed_seconds = time.time() - started_at

    total_nodes = sum(item["node_count"] for item in chromosome_summaries)
    total_selected_cpgs = sum(item["selected_cpg_count"] for item in chromosome_summaries)

    summary = {
        "created_at": datetime.now().isoformat(),
        "model_name": DNABERT_MODEL_NAME,
        "model_revision": DNABERT_MODEL_REVISION,
        "frozen": True,
        "pooling": "masked_mean_per_sequence_then_mean_per_100kb_node",
        "source_splits": SPLIT_NAMES,
        "graph_resolution": GRAPH_RESOLUTION,
        "max_cpg_per_node": DNABERT_MAX_CPG_PER_NODE,
        "embedding_dimension": DNABERT_HIDDEN_SIZE,
        "embedding_dtype": DNABERT_SAVE_DTYPE,
        "chromosome_count": len(chromosomes),
        "node_count": int(total_nodes),
        "selected_cpg_count": int(total_selected_cpgs),
        "elapsed_seconds": float(elapsed_seconds),
        "chromosomes": chromosome_summaries,
    }

    summary_path = DNABERT_NODE_FEATURES_DIR / "node_feature_extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("DNABERT-2 NODE FEATURES COMPLETED")
    print("=" * 70)
    print(f"Selected CpGs: {total_selected_cpgs:,}")
    print(f"Nodes with features: {total_nodes:,}")
    print(f"Elapsed time: {elapsed_seconds / 60:.2f} minutes")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
