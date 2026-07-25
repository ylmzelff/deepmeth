"""
Build the 100kb-resolution Hi-C graph for the HepG2 GRCh38 genome:
  1. a node index (one row per 100kb bin, chr1-22 + chrX)
  2. a normalized adjacency matrix (self-loops + symmetric degree
     normalization, same convention as the original ncVarPred model),
     intra-chromosomal contacts only (inter-chromosomal set to 0 - keeps
     the matrix tractable; a standard simplification for this kind of
     3D-genome-informed model)

Requires `pip install hic-straw` and the raw .hic file from
preprocessing/download_data.py.

Usage (no arguments needed):

    python feature_extraction/prepare_hic_graph.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hicstraw
import numpy as np
import pandas as pd
import scipy.sparse as sp

from config.project_config import (
    GENOME_ASSEMBLY,
    GRAPH_DIR,
    GRAPH_RESOLUTION,
    HIC_RAW_FILE_PATH,
    INCLUDED_CHROMOSOMES,
)

NODE_INDEX_PATH = GRAPH_DIR / "node_index.parquet"
ADJACENCY_PATH = GRAPH_DIR / "adjacency_normalized.npz"


def normalize_chromosome_name(name: str) -> str:
    name = str(name).strip()
    suffix = name[3:] if name.lower().startswith("chr") else name
    if suffix.lower() == "x":
        suffix = "X"
    elif suffix.lower() == "y":
        suffix = "Y"
    elif suffix.lower() in {"m", "mt"}:
        suffix = "M"
    return f"chr{suffix}"


def build_node_index() -> tuple[pd.DataFrame, "hicstraw.HiCFile"]:
    """One row per 100kb bin across chr1-22 + chrX, in a fixed global order."""
    hic_file = hicstraw.HiCFile(str(HIC_RAW_FILE_PATH))

    chrom_lengths = {
        normalize_chromosome_name(chrom.name): chrom.length
        for chrom in hic_file.getChromosomes()
        if normalize_chromosome_name(chrom.name) in INCLUDED_CHROMOSOMES
    }

    missing = set(INCLUDED_CHROMOSOMES) - set(chrom_lengths)
    if missing:
        raise RuntimeError(
            f"The .hic file is missing expected chromosomes: {sorted(missing)}. "
            "Check that it is really GRCh38 with chr1-22 + chrX."
        )

    print("Chromosome lengths from the .hic file:")
    for chrom in INCLUDED_CHROMOSOMES:
        print(f"  {chrom}: {chrom_lengths[chrom]:,} bp")

    rows = []
    for chrom in INCLUDED_CHROMOSOMES:
        length = chrom_lengths[chrom]
        for bin_start in range(0, length, GRAPH_RESOLUTION):
            bin_end = min(bin_start + GRAPH_RESOLUTION, length)
            rows.append({"chrom": chrom, "bin_start": bin_start, "bin_end": bin_end})

    node_index = pd.DataFrame(rows)
    node_index["node_index"] = np.arange(len(node_index), dtype=np.int64)

    duplicate_count = int(node_index.duplicated(subset=["chrom", "bin_start"]).sum())
    if duplicate_count:
        raise RuntimeError(f"{duplicate_count} duplicate (chrom, bin_start) bins in the node index.")

    return node_index, hic_file


def build_raw_adjacency(node_index: pd.DataFrame, hic_file) -> sp.coo_matrix:
    """Intra-chromosomal observed contact counts only, block-diagonal by chromosome."""
    number_of_nodes = len(node_index)

    chrom_to_node_offset = {
        chrom: int(group["node_index"].min())
        for chrom, group in node_index.groupby("chrom", sort=False)
    }

    row_indices: list[np.ndarray] = []
    col_indices: list[np.ndarray] = []
    values: list[np.ndarray] = []

    for chrom in INCLUDED_CHROMOSOMES:
        offset = chrom_to_node_offset[chrom]

        matrix_zoom_data = hic_file.getMatrixZoomData(
            chrom.removeprefix("chr"),
            chrom.removeprefix("chr"),
            "observed",
            "NONE",
            "BP",
            GRAPH_RESOLUTION,
        )

        chrom_length = int(node_index.loc[node_index["chrom"] == chrom, "bin_end"].max())
        records = matrix_zoom_data.getRecords(0, chrom_length, 0, chrom_length)

        print(f"  {chrom}: {len(records):,} raw contact records")

        if not records:
            continue

        bin_x = np.array([record.binX for record in records], dtype=np.int64) // GRAPH_RESOLUTION
        bin_y = np.array([record.binY for record in records], dtype=np.int64) // GRAPH_RESOLUTION
        counts = np.array([record.counts for record in records], dtype=np.float32)

        row_indices.append(bin_x + offset)
        col_indices.append(bin_y + offset)
        values.append(counts)

    all_rows = np.concatenate(row_indices)
    all_cols = np.concatenate(col_indices)
    all_values = np.concatenate(values)

    # Hi-C contact records are upper-triangular (binX <= binY); mirror to get a
    # symmetric matrix, without double-counting the diagonal.
    off_diagonal = all_rows != all_cols

    symmetric_rows = np.concatenate([all_rows, all_cols[off_diagonal]])
    symmetric_cols = np.concatenate([all_cols, all_rows[off_diagonal]])
    symmetric_values = np.concatenate([all_values, all_values[off_diagonal]])

    adjacency = sp.coo_matrix(
        (symmetric_values, (symmetric_rows, symmetric_cols)),
        shape=(number_of_nodes, number_of_nodes),
        dtype=np.float32,
    )

    return adjacency


def normalize_adjacency(adjacency: sp.coo_matrix) -> sp.csr_matrix:
    """Original ncVarPred normalization: A + I, then symmetric degree normalization."""
    adjacency = adjacency + sp.eye(adjacency.shape[0], dtype=np.float32, format="coo")

    row_sum = np.asarray(adjacency.sum(axis=1)).reshape(-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        degree_inverse_sqrt = np.power(row_sum, -0.5)

    degree_inverse_sqrt[~np.isfinite(degree_inverse_sqrt)] = 0.0

    degree_matrix = sp.diags(degree_inverse_sqrt)

    normalized = adjacency.dot(degree_matrix).transpose().dot(degree_matrix)

    return normalized.tocsr()


def main() -> None:
    print("=" * 70)
    print("Building the HepG2 100kb Hi-C graph (GRCh38)")
    print("=" * 70)
    print(f"Assembly: {GENOME_ASSEMBLY}")
    print(f"Source: {HIC_RAW_FILE_PATH}")
    print(f"Resolution: {GRAPH_RESOLUTION:,} bp")

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Building node index")
    node_index, hic_file = build_node_index()
    print(f"Total nodes: {len(node_index):,}")

    node_index.to_parquet(NODE_INDEX_PATH, index=False)
    print(f"Saved: {NODE_INDEX_PATH}")

    print("\n[2/3] Reading intra-chromosomal Hi-C contacts")
    raw_adjacency = build_raw_adjacency(node_index, hic_file)
    print(f"Raw adjacency non-zero entries: {raw_adjacency.nnz:,}")

    print("\n[3/3] Normalizing adjacency (self-loops + symmetric degree norm)")
    normalized_adjacency = normalize_adjacency(raw_adjacency)

    if not np.isfinite(normalized_adjacency.data).all():
        raise RuntimeError("Normalized adjacency contains NaN or infinite values.")

    sp.save_npz(ADJACENCY_PATH, normalized_adjacency)
    print(f"Saved: {ADJACENCY_PATH}")

    print("\nHi-C graph construction completed.")
    print(f"Nodes: {len(node_index):,}")
    print(f"Adjacency non-zero entries (post-normalization): {normalized_adjacency.nnz:,}")


if __name__ == "__main__":
    main()
