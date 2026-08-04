"""
Combine the GM12878 Hi-C node index with the DNABERT-2 node features into
node_features.npy, and map every CpG in disjoint_split to its graph node -
the GM12878/hg19 counterpart to feature_extraction/prepare_graph_features.py
(HepG2/GRCh38).

(An earlier version also mapped a second, DeepMethyl-comparable
baseline_split - see preprocessing/preprocess_gm12878.py's docstring for
why that split was removed; disjoint_split is the only one any GM12878
experiment in this project actually uses.)

load_dnabert_node_features/build_node_features/compute_node_index_for_split
are local adaptations of prepare_graph_features.py's versions (same logic),
since the originals hardcode HepG2's DNABERT_NODE_FEATURES_DIR/DATASET_DIR
paths.

Requires prepare_hic_graph_gm12878.py and extract_dnabert2_gm12878.py to
have run first.

Usage (no arguments needed):

    python feature_extraction/prepare_graph_features_gm12878.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config.project_config import DNABERT_HIDDEN_SIZE, GRAPH_RESOLUTION, INCLUDED_CHROMOSOMES
from preprocessing.download_data_gm12878 import GM12878_DATA_DIR

GM12878_GRAPH_DIR = GM12878_DATA_DIR / "graph"
NODE_INDEX_PATH = GM12878_GRAPH_DIR / "node_index.parquet"
NODE_FEATURES_OUTPUT_PATH = GM12878_GRAPH_DIR / "node_features.npy"

# Must match model/graph_branch_gat.py's own EXTRA_NODE_FEATURE_DIM
# (independently derived from the same INCLUDED_CHROMOSOMES constant, not
# imported directly, to avoid a model/ -> feature_extraction/ dependency).
CHROMOSOME_TO_INDEX = {chrom: index for index, chrom in enumerate(INCLUDED_CHROMOSOMES)}
EXTRA_NODE_FEATURE_DIM = len(INCLUDED_CHROMOSOMES) + 2

GM12878_DNABERT_NODE_FEATURES_DIR = GM12878_DATA_DIR / "dnabert2_node_features"

PROCEED_DIR = GM12878_DATA_DIR / "proceed"
DISJOINT_SPLIT_DIR = PROCEED_DIR / "disjoint_split"
SPLIT_NAMES = ("train", "validation", "test")


def load_node_index() -> pd.DataFrame:
    if not NODE_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"{NODE_INDEX_PATH} does not exist. Run feature_extraction/prepare_hic_graph_gm12878.py first."
        )
    return pd.read_parquet(NODE_INDEX_PATH)


def load_dnabert_node_features() -> tuple[pd.DataFrame, np.ndarray]:
    """Concatenate every chromosome's DNABERT node-feature file into one flat table + embedding array."""
    chrom_chunks, bin_start_chunks, sample_count_chunks, embedding_chunks = [], [], [], []

    for chrom in INCLUDED_CHROMOSOMES:
        path = GM12878_DNABERT_NODE_FEATURES_DIR / f"{chrom}_dnabert2_node_features.npz"

        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist. Run feature_extraction/extract_dnabert2_gm12878.py first."
            )

        with np.load(path) as data:
            chrom_chunks.append(data["chromosomes"])
            bin_start_chunks.append(data["bin_starts"])
            sample_count_chunks.append(data["sample_counts"])
            embedding_chunks.append(data["embeddings"].astype(np.float32))

    index_dataframe = pd.DataFrame(
        {
            "chrom": np.concatenate(chrom_chunks),
            "bin_start": np.concatenate(bin_start_chunks),
            "sample_count": np.concatenate(sample_count_chunks),
        }
    )
    index_dataframe["flat_index"] = np.arange(len(index_dataframe), dtype=np.int64)

    embeddings = np.concatenate(embedding_chunks, axis=0)

    duplicate_count = int(index_dataframe.duplicated(subset=["chrom", "bin_start"]).sum())
    if duplicate_count:
        raise RuntimeError(f"{duplicate_count} duplicate (chrom, bin_start) entries across DNABERT node feature files.")

    return index_dataframe, embeddings


def build_node_features(
    node_index: pd.DataFrame,
    dnabert_index: pd.DataFrame,
    dnabert_embeddings: np.ndarray,
) -> np.ndarray:
    number_of_nodes = len(node_index)
    node_features = np.zeros((number_of_nodes, DNABERT_HIDDEN_SIZE), dtype=np.float32)

    merged = node_index.merge(
        dnabert_index[["chrom", "bin_start", "flat_index"]],
        on=["chrom", "bin_start"],
        how="left",
    )

    if len(merged) != number_of_nodes:
        raise RuntimeError(
            f"Merge produced {len(merged):,} rows, expected {number_of_nodes:,} "
            "- duplicate (chrom, bin_start) somewhere."
        )

    covered_mask = merged["flat_index"].notna()
    covered_node_positions = merged.loc[covered_mask, "node_index"].to_numpy(dtype=np.int64)
    covered_flat_indices = merged.loc[covered_mask, "flat_index"].to_numpy().astype(np.int64)

    node_features[covered_node_positions] = dnabert_embeddings[covered_flat_indices]

    covered_count = int(covered_mask.sum())
    uncovered_count = number_of_nodes - covered_count

    print(f"Nodes with DNABERT features: {covered_count:,}/{number_of_nodes:,}")

    if uncovered_count:
        print(
            f"  {uncovered_count:,} nodes have zero usable CG positions in the reference genome "
            "(assembly gaps, e.g. acrocentric/pericentromeric regions) - using zero vectors for them."
        )

    return node_features


def build_extra_node_features(node_index: pd.DataFrame, dnabert_index: pd.DataFrame) -> np.ndarray:
    """
    3 kinds of cheap, leak-free feature appended after the raw 768-d
    DNABERT-2 node embedding - none use methylation labels or any real
    epigenomic measurement, only genomic coordinates and the DNABERT
    extraction's own bookkeeping:

      - chromosome one-hot (len(INCLUDED_CHROMOSOMES) dims): GATv2Structure
        previously had no notion of which chromosome a node is on at all.
      - normalized position within chromosome (1 dim: bin_start / that
        chromosome's own max bin_end): previously no way to distinguish a
        telomere-proximal node from a centromere-proximal one.
      - normalized sample_count (1 dim): how many CG windows were actually
        averaged into this node's 768-d embedding (capped at
        GM12878_MAX_CPG_PER_NODE=128 in extract_dnabert2_gm12878.py) -
        this was already computed there but silently discarded when
        node_features.npy was first built (see project history). Lets the
        model distinguish a well-supported embedding (many CGs averaged,
        e.g. a CG-dense promoter/CpG-island region) from a noisy one (few
        CGs, e.g. mostly repeat/gap sequence) instead of treating both
        identically. Normalized by this dataset's own observed max
        (not hardcoding 128), so it stays correct if that cap ever changes.
    """
    number_of_nodes = len(node_index)

    chromosome_indices = node_index["chrom"].map(CHROMOSOME_TO_INDEX).to_numpy()
    if pd.isna(chromosome_indices).any():
        raise RuntimeError("node_index.parquet contains a chromosome not in INCLUDED_CHROMOSOMES.")
    chromosome_one_hot = np.zeros((number_of_nodes, len(INCLUDED_CHROMOSOMES)), dtype=np.float32)
    chromosome_one_hot[np.arange(number_of_nodes), chromosome_indices.astype(np.int64)] = 1.0

    chrom_max_bin_end = node_index.groupby("chrom")["bin_end"].transform("max").to_numpy()
    normalized_position = (node_index["bin_start"].to_numpy() / chrom_max_bin_end).astype(np.float32)

    merged = node_index.merge(
        dnabert_index[["chrom", "bin_start", "sample_count"]], on=["chrom", "bin_start"], how="left",
    )
    if len(merged) != number_of_nodes:
        raise RuntimeError(
            f"Merge produced {len(merged):,} rows, expected {number_of_nodes:,} - duplicate (chrom, bin_start) somewhere."
        )
    sample_count = merged["sample_count"].fillna(0).to_numpy(dtype=np.float32)
    max_sample_count = float(dnabert_index["sample_count"].max())
    normalized_sample_count = sample_count / max_sample_count if max_sample_count > 0 else sample_count

    extra_features = np.concatenate(
        [chromosome_one_hot, normalized_position[:, None], normalized_sample_count[:, None]], axis=1,
    ).astype(np.float32)

    if extra_features.shape != (number_of_nodes, EXTRA_NODE_FEATURE_DIM):
        raise RuntimeError(f"Unexpected extra feature shape: {extra_features.shape}")
    if not np.isfinite(extra_features).all():
        raise RuntimeError("Non-finite values in computed extra node features.")

    return extra_features


def compute_node_index_for_split(split_name: str, node_index: pd.DataFrame) -> None:
    dataset_path = DISJOINT_SPLIT_DIR / f"{split_name}.parquet"

    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} does not exist. Run preprocessing/preprocess_gm12878.py first.")

    dataframe = pd.read_parquet(dataset_path, columns=["chrom", "canonical_position"])

    dataframe["bin_start"] = (
        (dataframe["canonical_position"] - 1) // GRAPH_RESOLUTION * GRAPH_RESOLUTION
    ).astype("int64")

    lookup = node_index.set_index(["chrom", "bin_start"])["node_index"]
    mapped = dataframe.set_index(["chrom", "bin_start"]).index.map(lookup)

    missing_count = int(pd.isna(mapped).sum())
    if missing_count:
        raise RuntimeError(
            f"disjoint_split/{split_name}: {missing_count:,} CpGs could not be mapped to a graph node "
            "- the Hi-C node index may not cover this chromosome's full length."
        )

    node_index_array = np.asarray(mapped, dtype=np.int64)

    output_path = DISJOINT_SPLIT_DIR / f"{split_name}_node_index.npy"
    np.save(output_path, node_index_array)
    print(f"disjoint_split/{split_name}: {len(node_index_array):,} CpGs -> {output_path}")


def main() -> None:
    print("=" * 70)
    print("Preparing final GM12878 graph features")
    print("=" * 70)

    node_index = load_node_index()
    number_of_nodes = len(node_index)
    print(f"Total graph nodes: {number_of_nodes:,}")

    print("\n[1/2] Building node_features.npy from DNABERT-2 embeddings + chromosome/position/sample_count features")
    dnabert_index, dnabert_embeddings = load_dnabert_node_features()
    dnabert_node_features = build_node_features(node_index, dnabert_index, dnabert_embeddings)
    extra_node_features = build_extra_node_features(node_index, dnabert_index)
    node_features = np.concatenate([dnabert_node_features, extra_node_features], axis=1)

    if not np.isfinite(node_features).all():
        raise RuntimeError("node_features contains NaN or infinite values.")

    np.save(NODE_FEATURES_OUTPUT_PATH, node_features)
    print(f"Saved: {NODE_FEATURES_OUTPUT_PATH}  shape={node_features.shape}")

    print("\n[2/2] Mapping every CpG to its graph node (disjoint_split)")
    for split_name in SPLIT_NAMES:
        compute_node_index_for_split(split_name, node_index)

    print("\nGM12878 graph feature preparation completed.")


if __name__ == "__main__":
    main()
