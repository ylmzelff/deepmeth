from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from twobitreader import TwoBitFile

from config.project_config import (
    DNABERT_HIDDEN_SIZE,
    DNABERT_MODEL_NAME,
    DNABERT_MODEL_REVISION,
    DNABERT_SAVE_DTYPE,
    GRAPH_RESOLUTION,
    MAX_UNKNOWN_FRACTION,
    SEQUENCE_LENGTH,
)
from config.data_config.gm12878_config import GM12878_DATA_DIR, HG19_2BIT_PATH
from feature_extraction.extract_dnabert2 import embed_and_pool_nodes, load_frozen_model
from preprocessing.preprocess_hepg2 import extract_cpg_centered_sequence


GM12878_MAX_CPG_PER_NODE = 128

GM12878_GRAPH_DIR = GM12878_DATA_DIR / "graph"
NODE_INDEX_PATH = GM12878_GRAPH_DIR / "node_index.parquet"
GM12878_DNABERT_NODE_FEATURES_DIR = GM12878_DATA_DIR / "dnabert2_node_features"

CG_REGEX = re.compile("CG")


def load_node_index() -> pd.DataFrame:
    if not NODE_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"{NODE_INDEX_PATH} does not exist. Run feature_extraction/prepare_hic_graph_gm12878.py first."
        )
    return pd.read_parquet(NODE_INDEX_PATH)


def find_genome_cpg_positions(genome: TwoBitFile, chrom: str) -> np.ndarray:
    """1-based canonical positions (position of the 'C' in 'CG') of every CG
    dinucleotide occurrence in this chromosome's reference sequence."""
    sequence = str(genome[chrom][:]).upper()
    return np.array([match.start() + 1 for match in CG_REGEX.finditer(sequence)], dtype=np.int64)


def select_genome_cpgs_per_node(positions: np.ndarray, bin_starts: np.ndarray) -> pd.DataFrame:
  
    order = np.argsort(bin_starts, kind="stable")
    positions = positions[order]
    bin_starts = bin_starts[order]

    unique_bins, first_indices, counts = np.unique(bin_starts, return_index=True, return_counts=True)

    selected_indices: list[np.ndarray] = []
    for start, count in zip(first_indices, counts):
        if count <= GM12878_MAX_CPG_PER_NODE:
            local_indices = np.arange(count, dtype=np.int64)
        else:
            local_indices = np.linspace(0, count - 1, num=GM12878_MAX_CPG_PER_NODE, dtype=np.int64)
            local_indices = np.unique(local_indices)
        selected_indices.append(start + local_indices)

    concatenated = np.concatenate(selected_indices)

    return pd.DataFrame({"canonical_position": positions[concatenated], "bin_start": bin_starts[concatenated]})


def process_chromosome(
    chromosome: str,
    chrom_node_index: pd.DataFrame,
    genome: TwoBitFile,
    reference_chromosomes: set[str],
    tokenizer,
    model,
    device: torch.device,
    save_dtype,
) -> dict:
    output_path = GM12878_DNABERT_NODE_FEATURES_DIR / f"{chromosome}_dnabert2_node_features.npz"

    if output_path.exists():
        print(f"{chromosome}: already completed, skipping")

        with np.load(output_path) as data:
            return {
                "chrom": chromosome,
                "node_count": int(len(data["bin_starts"])),
                "embedded_cpg_count": int(data["sample_counts"].sum()),
                "output_path": str(output_path),
                "skipped": True,
            }

    print("\n" + "=" * 70)
    print(f"Processing chromosome: {chromosome}")
    print("=" * 70)

    expected_bins = set(chrom_node_index["bin_start"].tolist())

    genome_positions = find_genome_cpg_positions(genome, chromosome)
    print(f"  CG dinucleotides in reference: {len(genome_positions):,}")

    genome_bin_starts = ((genome_positions - 1) // GRAPH_RESOLUTION * GRAPH_RESOLUTION).astype(np.int64)

    keep_mask = np.isin(genome_bin_starts, np.fromiter(expected_bins, dtype=np.int64))
    genome_positions = genome_positions[keep_mask]
    genome_bin_starts = genome_bin_starts[keep_mask]

    selected = select_genome_cpgs_per_node(genome_positions, genome_bin_starts)
    print(f"  Candidates selected (cap {GM12878_MAX_CPG_PER_NODE}/node): {len(selected):,}")

    selected["sequence"] = [
        extract_cpg_centered_sequence(genome, reference_chromosomes, chromosome, int(position))
        for position in selected["canonical_position"]
    ]

    unknown_fraction = selected["sequence"].str.count(r"[^ACGT]") / SEQUENCE_LENGTH
    valid_mask = selected["sequence"].notna() & (unknown_fraction <= MAX_UNKNOWN_FRACTION)
    dropped = len(selected) - int(valid_mask.sum())
    if dropped:
        print(f"  Dropped {dropped:,}/{len(selected):,} candidates (boundary or too many unknown bases)")
    selected = selected.loc[valid_mask].reset_index(drop=True)

    covered_bins = int(selected["bin_start"].nunique())
    uncovered_bins = len(expected_bins) - covered_bins
    if uncovered_bins:
        print(
            f"  WARNING: {uncovered_bins:,}/{len(expected_bins):,} nodes on {chromosome} have zero usable "
            "CG positions (likely assembly gaps) - will get zero-vector features."
        )

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
    print(f"  Node features: {node_features.shape}  (covered {covered_bins:,}/{len(expected_bins):,} expected nodes)")

    return {
        "chrom": chromosome,
        "genome_cg_count": int(len(genome_positions)),
        "embedded_cpg_count": int(len(selected)),
        "node_count": int(len(bin_starts)),
        "expected_node_count": int(len(expected_bins)),
        "output_path": str(output_path),
        "skipped": False,
    }


def chromosome_sort_key(chromosome: str) -> tuple[int, str]:
    suffix = chromosome.removeprefix("chr")
    return (int(suffix), "") if suffix.isdigit() else (10_000, suffix)


def main() -> None:
    if not HG19_2BIT_PATH.exists():
        raise FileNotFoundError(f"{HG19_2BIT_PATH} does not exist. Run preprocessing/download_data_gm12878.py first.")

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")

    device = torch.device("cuda")
    save_dtype = np.float16 if DNABERT_SAVE_DTYPE == "float16" else np.float32

    GM12878_DNABERT_NODE_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GM12878 DNABERT-2 node-feature extraction (genome-wide CG scan, hg19)")
    print("=" * 70)

    node_index = load_node_index()
    print(f"Total graph nodes: {len(node_index):,}")
    print(f"Maximum CpGs per node: {GM12878_MAX_CPG_PER_NODE}")

    genome = TwoBitFile(str(HG19_2BIT_PATH))
    reference_chromosomes = set(genome.keys())

    tokenizer, model = load_frozen_model(device)
    print("[PASS] Frozen DNABERT-2 loaded")

    chromosomes = sorted(node_index["chrom"].unique().tolist(), key=chromosome_sort_key)
    print(f"Chromosomes: {chromosomes}")

    started_at = time.time()
    chromosome_summaries = []
    for chromosome in chromosomes:
        chrom_node_index = node_index.loc[node_index["chrom"].eq(chromosome)]
        chromosome_summaries.append(
            process_chromosome(
                chromosome, chrom_node_index, genome, reference_chromosomes, tokenizer, model, device, save_dtype,
            )
        )

    elapsed_seconds = time.time() - started_at

    total_nodes = sum(item["node_count"] for item in chromosome_summaries)
    total_embedded_cpgs = sum(item["embedded_cpg_count"] for item in chromosome_summaries)

    summary = {
        "created_at": datetime.now().isoformat(),
        "model_name": DNABERT_MODEL_NAME,
        "model_revision": DNABERT_MODEL_REVISION,
        "frozen": True,
        "pooling": "masked_mean_per_sequence_then_mean_per_node",
        "source": "hg19 reference genome CG dinucleotide scan (NOT RRBS-observed CpGs)",
        "graph_resolution": GRAPH_RESOLUTION,
        "max_cpg_per_node": GM12878_MAX_CPG_PER_NODE,
        "embedding_dimension": DNABERT_HIDDEN_SIZE,
        "embedding_dtype": DNABERT_SAVE_DTYPE,
        "chromosome_count": len(chromosomes),
        "node_count": int(total_nodes),
        "expected_node_count": int(len(node_index)),
        "embedded_cpg_count": int(total_embedded_cpgs),
        "elapsed_seconds": float(elapsed_seconds),
        "chromosomes": chromosome_summaries,
    }

    summary_path = GM12878_DNABERT_NODE_FEATURES_DIR / "node_feature_extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("GM12878 DNABERT-2 NODE FEATURES COMPLETED")
    print("=" * 70)
    print(f"Embedded CpGs: {total_embedded_cpgs:,}")
    print(f"Nodes with features: {total_nodes:,}/{len(node_index):,}")
    print(f"Elapsed time: {elapsed_seconds / 60:.2f} minutes")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
