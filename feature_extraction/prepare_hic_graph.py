"""
Build the GRAPH_RESOLUTION-bp Hi-C graph for the HepG2 GRCh38 genome:
  1. a node index (one row per GRAPH_RESOLUTION-bp bin, chr1-22 + chrX)
  2. a 4-feature O/E edge representation for GATv2Structure's attention
     (see model/graph_branch_gat.py and compute_oe_edge_features below),
     intra-chromosomal contacts only (inter-chromosomal set to 0 - keeps
     the matrix tractable; a standard simplification for this kind of
     3D-genome-informed model)

Requires `pip install hic-straw` and the raw .hic file from
preprocessing/download_data_hepg2.py.

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

from config.data_config.hepg2_config import GENOME_ASSEMBLY, HIC_RAW_FILE_PATH
from config.project_config import (
    GRAPH_DIR,
    GRAPH_RESOLUTION,
    HIC_MAX_CONTACT_DISTANCE_BP,
    HIC_NORMALIZATION_TYPE,
    HIC_TOP_K_NEIGHBORS,
    INCLUDED_CHROMOSOMES,
)

NODE_INDEX_PATH = GRAPH_DIR / "node_index.parquet"
EDGE_FEATURES_PATH = GRAPH_DIR / "edge_features.npz"


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


def build_node_index() -> tuple[pd.DataFrame, "hicstraw.HiCFile", dict[str, str]]:
    """One row per GRAPH_RESOLUTION-bp bin across chr1-22 + chrX, in a fixed global order.

    Also returns a {normalized_name: raw_name} map, since .hic files vary in
    whether they store chromosomes as "chr1" or "1" - matrixZoomData lookups
    must use the file's own raw name, not a name we guessed.
    """
    hic_file = hicstraw.HiCFile(str(HIC_RAW_FILE_PATH))

    available_resolutions = hic_file.getResolutions()
    print(f"Resolutions available in the .hic file: {available_resolutions}")

    if GRAPH_RESOLUTION not in available_resolutions:
        raise RuntimeError(
            f"{GRAPH_RESOLUTION:,} bp is not one of the resolutions stored in this .hic file: "
            f"{available_resolutions}. Pick one of those instead (GRAPH_RESOLUTION in config)."
        )

    raw_names: dict[str, str] = {}
    chrom_lengths: dict[str, int] = {}

    for chrom in hic_file.getChromosomes():
        normalized = normalize_chromosome_name(chrom.name)
        if normalized in INCLUDED_CHROMOSOMES:
            raw_names[normalized] = chrom.name
            chrom_lengths[normalized] = chrom.length

    missing = set(INCLUDED_CHROMOSOMES) - set(chrom_lengths)
    if missing:
        raise RuntimeError(
            f"The .hic file is missing expected chromosomes: {sorted(missing)}. "
            "Check that it is really GRCh38 with chr1-22 + chrX."
        )

    print("Chromosome lengths from the .hic file (normalized name <- raw name in file):")
    for chrom in INCLUDED_CHROMOSOMES:
        print(f"  {chrom} <- {raw_names[chrom]!r}: {chrom_lengths[chrom]:,} bp")

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

    return node_index, hic_file, raw_names


def build_raw_adjacency(node_index: pd.DataFrame, hic_file, raw_names: dict[str, str]) -> sp.coo_matrix:
    """Intra-chromosomal observed contact counts only, block-diagonal by chromosome."""
    number_of_nodes = len(node_index)

    chrom_to_node_offset = {
        chrom: int(group["node_index"].min())
        for chrom, group in node_index.groupby("chrom", sort=False)
    }

    row_indices: list[np.ndarray] = []
    col_indices: list[np.ndarray] = []
    values: list[np.ndarray] = []
    chromosomes_with_no_records: list[str] = []

    for chrom in INCLUDED_CHROMOSOMES:
        offset = chrom_to_node_offset[chrom]
        raw_name = raw_names[chrom]

        matrix_zoom_data = hic_file.getMatrixZoomData(
            raw_name,
            raw_name,
            "observed",
            HIC_NORMALIZATION_TYPE,
            "BP",
            GRAPH_RESOLUTION,
        )

        chrom_length = int(node_index.loc[node_index["chrom"] == chrom, "bin_end"].max())
        records = matrix_zoom_data.getRecords(0, chrom_length, 0, chrom_length)

        print(f"  {chrom} (raw name {raw_name!r}): {len(records):,} raw contact records")

        if not records:
            chromosomes_with_no_records.append(chrom)
            continue

        bin_x = np.array([record.binX for record in records], dtype=np.int64) // GRAPH_RESOLUTION
        bin_y = np.array([record.binY for record in records], dtype=np.int64) // GRAPH_RESOLUTION
        counts = np.array([record.counts for record in records], dtype=np.float32)

        # KR (and other Hi-C balancing) normalization vectors don't always
        # converge for every bin (typically very low-coverage or
        # blacklisted/repetitive regions) - hic-straw returns NaN counts for
        # contacts touching those bins rather than raising. Drop them here
        # (silently keeping them would poison compute_oe_edge_features'
        # distance-pooled O/E averages downstream and trip its own
        # np.isfinite check anyway, just with a much less informative error).
        finite_mask = np.isfinite(counts)
        dropped = len(counts) - int(finite_mask.sum())
        if dropped:
            print(
                f"    dropped {dropped:,}/{len(counts):,} records with non-finite "
                f"{HIC_NORMALIZATION_TYPE}-normalized counts (balancing didn't converge "
                "for those bins)"
            )
        bin_x = bin_x[finite_mask]
        bin_y = bin_y[finite_mask]
        counts = counts[finite_mask]

        # Restrict to contacts within HIC_MAX_CONTACT_DISTANCE_BP - without
        # this, a chromosome's contact matrix is nearly fully dense (every
        # bin has some nonzero count with almost every other bin, mostly
        # background rather than real 3D structure - TADs/loops are almost
        # entirely within a few Mb) and downstream per-edge-attention
        # models (GATv2Structure) OOM trying to materialize a tensor sized
        # by edge count - see config.project_config's HIC_MAX_CONTACT_DISTANCE_BP.
        max_distance_bins = HIC_MAX_CONTACT_DISTANCE_BP // GRAPH_RESOLUTION
        distance_mask = np.abs(bin_x - bin_y) <= max_distance_bins
        distance_dropped = len(bin_x) - int(distance_mask.sum())
        if distance_dropped:
            print(
                f"    dropped {distance_dropped:,}/{len(bin_x):,} records beyond "
                f"{HIC_MAX_CONTACT_DISTANCE_BP:,} bp contact distance"
            )
        bin_x = bin_x[distance_mask]
        bin_y = bin_y[distance_mask]
        counts = counts[distance_mask]

        row_indices.append(bin_x + offset)
        col_indices.append(bin_y + offset)
        values.append(counts)

    if chromosomes_with_no_records:
        print(
            f"WARNING: no contact records for {len(chromosomes_with_no_records)} chromosome(s): "
            f"{chromosomes_with_no_records}"
        )

    if not row_indices:
        raise RuntimeError(
            "No Hi-C contact records were found for any chromosome. This usually means either "
            "the raw chromosome name or the resolution passed to getMatrixZoomData does not "
            "match what's actually stored in the .hic file - check the 'raw name' and "
            "'Resolutions available' lines printed above against what this call used."
        )

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

    before_top_k = adjacency.nnz
    adjacency = apply_top_k_sparsification(adjacency, HIC_TOP_K_NEIGHBORS)
    print(f"  Top-{HIC_TOP_K_NEIGHBORS} sparsification: {before_top_k:,} -> {adjacency.nnz:,} edges")

    return adjacency


def apply_top_k_sparsification(adjacency: sp.coo_matrix, top_k: int) -> sp.coo_matrix:
    """Keeps, for each node, only its top_k highest-weight edges (by
    distance-capped, KR/SCALE-normalized contact count) - bounds every
    node's degree by a fixed number regardless of local Hi-C density, so
    total edge count scales with node_count x top_k instead of with how
    densely populated the region happens to be (see config.project_config's
    HIC_TOP_K_NEIGHBORS for why this matters at finer resolutions).

    An edge (i, j) is kept if it's in EITHER node's top-K list (union via
    .maximum() with the transpose, not intersection) - otherwise a real
    but asymmetric-in-rank contact could vanish entirely just because node
    i has many strong neighbors crowding it out while node j doesn't.
    Self-loops aren't affected - they're added later in
    compute_oe_edge_features, after this function runs on the raw
    (self-loop-free) contacts.
    """
    csr = adjacency.tocsr()
    rows_out: list[np.ndarray] = []
    cols_out: list[np.ndarray] = []
    data_out: list[np.ndarray] = []

    for row in range(csr.shape[0]):
        start, end = csr.indptr[row], csr.indptr[row + 1]
        if end == start:
            continue

        if end - start <= top_k:
            selected = np.arange(start, end)
        else:
            row_values = csr.data[start:end]
            local_top_k = np.argpartition(row_values, -top_k)[-top_k:]
            selected = start + local_top_k

        rows_out.append(np.full(len(selected), row, dtype=np.int64))
        cols_out.append(csr.indices[selected])
        data_out.append(csr.data[selected])

    sparsified = sp.coo_matrix(
        (np.concatenate(data_out), (np.concatenate(rows_out), np.concatenate(cols_out))),
        shape=adjacency.shape,
    )

    return sparsified.maximum(sparsified.transpose()).tocoo()


def compute_oe_edge_features(adjacency: sp.coo_matrix, resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derives a 4-feature edge representation for GATv2Conv's edge_dim
    (see model/graph_branch_gat.py) - [log1p(KR/SCALE-normalized count),
    observed/expected ratio, log1p(genomic distance in bp), is_self_loop] -
    from the already top-k-sparsified adjacency `build_raw_adjacency`
    returns - a degree-normalized value (the old GCN_Structure's own
    convention, since removed - see project history) would conflate true
    contact strength with how connected a node is overall, the wrong signal
    to feed GATv2Conv's attention as an edge feature; the raw, bias-
    corrected-but-not-degree-normalized count here is the right one.
    Self-loops are added explicitly here (one per node, features
    [0, 1.0, 0, 1] - no real Hi-C signal, O/E=1.0 as a neutral/"as
    expected" placeholder, flagged via is_self_loop rather than faked with
    a made-up contact-strength value), since GATv2Structure is built with
    add_self_loops=False and expects to receive them pre-added.

    Distance is computed directly from row/col node-index difference - no
    extra lookup needed, since this adjacency only ever contains intra-
    chromosomal pairs (build_raw_adjacency never adds a cross-chromosome
    entry) and node indices are contiguous per chromosome, so |row - col|
    in node-index units already equals genomic distance in bins.

    Observed/expected: even after KR/SCALE balancing (which corrects for
    per-BIN sequencing/mappability/GC bias, not genomic distance - see
    config.project_config's HIC_NORMALIZATION_TYPE), contact counts still
    decay strongly with distance (closer bins simply touch more, a trivial
    polymer effect, not necessarily meaningful 3D structure). Dividing each
    contact by the mean contact strength observed at that exact distance
    (pooled across the whole graph) isolates contacts that are unusually
    strong FOR THEIR DISTANCE - the standard Hi-C convention (Lieberman-
    Aiden et al., 2009) for separating real loops/TADs from the "closer
    bins touch more" background trend GATv2Conv's attention had no way to
    account for when it only saw the raw/KR count (see project history).

    Returns (rows, cols, features[E, 4]) - rows/cols include both the real
    Hi-C edges and the added self-loops, ready to hand to GATv2Conv as
    edge_index/edge_attr directly (no further self-loop handling needed
    downstream).
    """
    distance_bins = np.abs(adjacency.row - adjacency.col).astype(np.int64)
    counts = adjacency.data.astype(np.float64)

    max_distance = int(distance_bins.max()) if len(distance_bins) else 0
    sum_per_distance = np.bincount(distance_bins, weights=counts, minlength=max_distance + 1)
    count_per_distance = np.bincount(distance_bins, minlength=max_distance + 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        expected_per_distance = sum_per_distance / count_per_distance
    expected_per_distance[count_per_distance == 0] = 1.0  # unreachable distances - never indexed below, placeholder only

    expected = expected_per_distance[distance_bins]
    observed_expected = counts / np.clip(expected, 1e-9, None)

    genomic_distance_bp = distance_bins.astype(np.float64) * resolution

    real_edge_features = np.stack(
        [
            np.log1p(counts),
            observed_expected,
            np.log1p(genomic_distance_bp),
            np.zeros(len(counts)),  # is_self_loop
        ],
        axis=1,
    )

    number_of_nodes = adjacency.shape[0]
    self_loop_index = np.arange(number_of_nodes, dtype=np.int64)
    self_loop_features = np.stack(
        [
            np.zeros(number_of_nodes),
            np.ones(number_of_nodes),  # O/E = 1.0: neutral "as expected" placeholder, not a real measurement
            np.zeros(number_of_nodes),
            np.ones(number_of_nodes),  # is_self_loop
        ],
        axis=1,
    )

    rows = np.concatenate([adjacency.row, self_loop_index])
    cols = np.concatenate([adjacency.col, self_loop_index])
    features = np.concatenate([real_edge_features, self_loop_features], axis=0).astype(np.float32)

    if not np.isfinite(features).all():
        raise RuntimeError("Non-finite values in computed O/E edge features.")

    return rows.astype(np.int64), cols.astype(np.int64), features


def main() -> None:
    print("=" * 70)
    print(f"Building the HepG2 {GRAPH_RESOLUTION:,}bp Hi-C graph (GRCh38)")
    print("=" * 70)
    print(f"Assembly: {GENOME_ASSEMBLY}")
    print(f"Source: {HIC_RAW_FILE_PATH}")
    print(f"Resolution: {GRAPH_RESOLUTION:,} bp")

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Building node index")
    node_index, hic_file, raw_names = build_node_index()
    print(f"Total nodes: {len(node_index):,}")

    node_index.to_parquet(NODE_INDEX_PATH, index=False)
    print(f"Saved: {NODE_INDEX_PATH}")

    print("\n[2/3] Reading intra-chromosomal Hi-C contacts")
    raw_adjacency = build_raw_adjacency(node_index, hic_file, raw_names)
    print(f"Raw adjacency non-zero entries: {raw_adjacency.nnz:,}")

    print("\n[3/3] Computing O/E edge features (log1p(count), O/E, log1p(distance), is_self_loop) - for GATv2Structure")
    edge_rows, edge_cols, edge_features = compute_oe_edge_features(raw_adjacency, GRAPH_RESOLUTION)
    np.savez_compressed(EDGE_FEATURES_PATH, rows=edge_rows, cols=edge_cols, features=edge_features)
    print(f"Saved: {EDGE_FEATURES_PATH}  ({len(edge_rows):,} edges incl. self-loops, {edge_features.shape[1]} features/edge)")

    print("\nHi-C graph construction completed.")
    print(f"Nodes: {len(node_index):,}")


if __name__ == "__main__":
    main()
