from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hicstraw
import numpy as np
import pandas as pd

from config.project_config import GRAPH_RESOLUTION, INCLUDED_CHROMOSOMES
from config.data_config.gm12878_config import GM12878_DATA_DIR, GM12878_HIC_RAW_FILE_PATH
from feature_extraction.prepare_hic_graph import build_raw_adjacency, compute_oe_edge_features

GM12878_GRAPH_DIR = GM12878_DATA_DIR / "graph"
NODE_INDEX_PATH = GM12878_GRAPH_DIR / "node_index.parquet"
# 4-feature GATv2 edge representation: log1p(KR count), O/E, log1p(distance),
# is_self_loop - see compute_oe_edge_features.
EDGE_FEATURES_PATH = GM12878_GRAPH_DIR / "edge_features.npz"


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
    
    hic_file = hicstraw.HiCFile(str(GM12878_HIC_RAW_FILE_PATH))

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
            "Check that it is really hg19 with chr1-22 + chrX."
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


def main() -> None:
    print("=" * 70)
    print(f"Building the GM12878 {GRAPH_RESOLUTION:,}bp Hi-C graph (hg19)")
    print("=" * 70)
    print(f"Source: {GM12878_HIC_RAW_FILE_PATH}")
    print(f"Resolution: {GRAPH_RESOLUTION:,} bp")

    GM12878_GRAPH_DIR.mkdir(parents=True, exist_ok=True)

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

    print("\nGM12878 Hi-C graph construction completed.")
    print(f"Nodes: {len(node_index):,}")


if __name__ == "__main__":
    main()
