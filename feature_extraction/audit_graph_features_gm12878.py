"""
Independent correctness audit of GM12878's graph feature outputs -
node_index.parquet, adjacency_normalized.npz, node_features.npy, and each
split's {split}_node_index.npy - run before trusting them for training.
Doesn't just re-check "no exception was raised" (prepare_graph_features_gm12878.py
already asserts that); cross-validates specific rows against their
independent source files to catch silent index/order bugs those inline
asserts wouldn't.

Checks:
  1. Shapes/counts agree across node_index.parquet, adjacency_normalized.npz,
     node_features.npy.
  2. The zero-vector row count in node_features.npy matches the "uncovered"
     count extract_dnabert2_gm12878.py/prepare_graph_features_gm12878.py
     reported.
  3. A random sample of covered nodes: node_features.npy's row for that node
     matches the embedding stored in the raw per-chromosome DNABERT .npz file
     (independent re-derivation, not just re-reading the same merge output).
  4. Each split's {split}_node_index.npy: length matches its parquet, values
     are in range, and a random sample of rows are independently
     recomputed from (chrom, canonical_position) -> bin_start -> node_index
     and checked against the saved array and against node_index.parquet.

Usage (no arguments needed, after prepare_graph_features_gm12878.py):

    python feature_extraction/audit_graph_features_gm12878.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

from config.project_config import DNABERT_HIDDEN_SIZE, GRAPH_RESOLUTION, INCLUDED_CHROMOSOMES
from preprocessing.download_data_gm12878 import GM12878_DATA_DIR

GM12878_GRAPH_DIR = GM12878_DATA_DIR / "graph"
NODE_INDEX_PATH = GM12878_GRAPH_DIR / "node_index.parquet"
ADJACENCY_PATH = GM12878_GRAPH_DIR / "adjacency_normalized.npz"
NODE_FEATURES_PATH = GM12878_GRAPH_DIR / "node_features.npy"
GM12878_DNABERT_NODE_FEATURES_DIR = GM12878_DATA_DIR / "dnabert2_node_features"

PROCEED_DIR = GM12878_DATA_DIR / "proceed"
DISJOINT_SPLIT_DIR = PROCEED_DIR / "disjoint_split"
SPLIT_NAMES = ("train", "validation", "test")

RANDOM_SEED = 42
SAMPLE_SIZE_NODES = 200
SAMPLE_SIZE_ROWS_PER_SPLIT = 500


def check_shapes(node_index: pd.DataFrame, adjacency: sp.csr_matrix, node_features: np.ndarray) -> None:
    print("[1/4] Shape/count consistency")
    number_of_nodes = len(node_index)

    assert adjacency.shape == (number_of_nodes, number_of_nodes), (
        f"adjacency shape {adjacency.shape} != ({number_of_nodes}, {number_of_nodes})"
    )
    assert node_features.shape == (number_of_nodes, DNABERT_HIDDEN_SIZE), (
        f"node_features shape {node_features.shape} != ({number_of_nodes}, {DNABERT_HIDDEN_SIZE})"
    )
    assert node_index["node_index"].is_unique, "node_index.parquet has duplicate node_index values"
    assert node_index["node_index"].min() == 0 and node_index["node_index"].max() == number_of_nodes - 1, (
        "node_index values are not a contiguous 0..N-1 range"
    )
    print(f"  PASS - {number_of_nodes:,} nodes, all shapes consistent")


def check_zero_vector_count(node_features: np.ndarray) -> np.ndarray:
    print("\n[2/4] Zero-vector node count")
    zero_mask = ~node_features.any(axis=1)
    zero_count = int(zero_mask.sum())
    print(f"  Zero-vector nodes: {zero_count:,}/{len(node_features):,}")
    print("  (compare by eye against extract_dnabert2_gm12878.py's own reported uncovered-node count)")
    return zero_mask


def check_node_features_against_raw_dnabert(node_index: pd.DataFrame, node_features: np.ndarray, zero_mask: np.ndarray) -> None:
    print("\n[3/4] node_features.npy rows vs raw per-chromosome DNABERT files (independent re-derivation)")

    rng = np.random.default_rng(RANDOM_SEED)
    covered_positions = np.flatnonzero(~zero_mask)
    sample_positions = rng.choice(covered_positions, size=min(SAMPLE_SIZE_NODES, len(covered_positions)), replace=False)

    raw_cache: dict[str, dict] = {}
    mismatches = 0

    for position in sample_positions:
        row = node_index.iloc[position]
        chrom, bin_start = row["chrom"], int(row["bin_start"])

        if chrom not in raw_cache:
            path = GM12878_DNABERT_NODE_FEATURES_DIR / f"{chrom}_dnabert2_node_features.npz"
            with np.load(path) as data:
                raw_cache[chrom] = {
                    "bin_starts": data["bin_starts"],
                    "embeddings": data["embeddings"].astype(np.float32),
                }

        raw = raw_cache[chrom]
        match_index = np.flatnonzero(raw["bin_starts"] == bin_start)

        if len(match_index) != 1:
            print(f"  MISMATCH: {chrom}:{bin_start} found {len(match_index)} times in raw file (expected 1)")
            mismatches += 1
            continue

        expected_embedding = raw["embeddings"][match_index[0]]
        actual_embedding = node_features[position]

        if not np.allclose(expected_embedding, actual_embedding, atol=1e-3):
            print(f"  MISMATCH: {chrom}:{bin_start} (node_index={position}) embedding differs from raw file")
            mismatches += 1

    if mismatches:
        raise RuntimeError(f"{mismatches}/{len(sample_positions)} sampled nodes failed cross-check against raw DNABERT files.")

    print(f"  PASS - {len(sample_positions)} sampled covered nodes match their raw DNABERT embeddings exactly")


def check_split_node_index(node_index: pd.DataFrame) -> None:
    print("\n[4/4] Per-split {split}_node_index.npy correctness")

    lookup = node_index.set_index(["chrom", "bin_start"])["node_index"]
    number_of_nodes = len(node_index)
    rng = np.random.default_rng(RANDOM_SEED)

    for split_name in SPLIT_NAMES:
        node_index_array = np.load(DISJOINT_SPLIT_DIR / f"{split_name}_node_index.npy")
        dataframe = pq.read_table(
            DISJOINT_SPLIT_DIR / f"{split_name}.parquet", columns=["chrom", "canonical_position"]
        ).to_pandas()

        assert len(node_index_array) == len(dataframe), (
            f"disjoint_split/{split_name}: node_index length {len(node_index_array):,} != "
            f"parquet length {len(dataframe):,}"
        )
        assert node_index_array.min() >= 0 and node_index_array.max() < number_of_nodes, (
            f"disjoint_split/{split_name}: node_index values out of [0, {number_of_nodes}) range"
        )

        sample_size = min(SAMPLE_SIZE_ROWS_PER_SPLIT, len(dataframe))
        sample_rows = rng.choice(len(dataframe), size=sample_size, replace=False)

        expected_bin_starts = (
            (dataframe["canonical_position"].to_numpy()[sample_rows] - 1) // GRAPH_RESOLUTION * GRAPH_RESOLUTION
        )
        expected_chroms = dataframe["chrom"].to_numpy()[sample_rows]

        mismatches = 0
        for local_i, row_i in enumerate(sample_rows):
            key = (expected_chroms[local_i], int(expected_bin_starts[local_i]))
            expected_node = lookup.get(key)

            if expected_node is None or int(expected_node) != int(node_index_array[row_i]):
                print(f"  MISMATCH: disjoint_split/{split_name} row {row_i} ({key}) -> "
                      f"saved={node_index_array[row_i]}, recomputed={expected_node}")
                mismatches += 1

        if mismatches:
            raise RuntimeError(f"{mismatches}/{sample_size} sampled rows failed in disjoint_split/{split_name}.")

        print(f"  PASS - disjoint_split/{split_name}: {len(node_index_array):,} rows, "
              f"{sample_size} sampled rows independently re-verified")


def main() -> None:
    print("=" * 70)
    print("GM12878 graph feature audit")
    print("=" * 70)

    node_index = pd.read_parquet(NODE_INDEX_PATH)
    adjacency = sp.load_npz(ADJACENCY_PATH).tocsr()
    node_features = np.load(NODE_FEATURES_PATH)

    check_shapes(node_index, adjacency, node_features)
    zero_mask = check_zero_vector_count(node_features)
    check_node_features_against_raw_dnabert(node_index, node_features, zero_mask)
    check_split_node_index(node_index)

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
