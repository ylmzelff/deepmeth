from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.sparse as sp
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Project paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mini_test_ca01.csv.gz"
)

DEFAULT_GRAPH_INDEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "graph"
    / "if_matrix_matching_index_100000.txt"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "graph"
    / "mini_test_structure_matching.npy"
)


# ============================================================
# Original ncVarPred graph setting
# ============================================================

RESOLUTION = 100_000

def normalize_if_matrix(
    adjacency: np.ndarray,
) -> np.ndarray:
    """
    Normalize a Hi-C interaction-frequency matrix using
    the original ncVarPred-1D3D procedure.

    Steps:
        1. Convert the matrix to sparse COO format.
        2. Add self-loops: A + I.
        3. Apply symmetric degree normalization.
        4. Return the normalized dense matrix.
    """
    adjacency = sp.coo_matrix(
        adjacency,
        dtype=np.float32,
    )

    if (
        adjacency.shape[0]
        != adjacency.shape[1]
    ):
        raise ValueError(
            "The adjacency matrix must be square."
        )

    # Original ncVarPred step: A + I
    adjacency = (
        adjacency
        + sp.eye(
            adjacency.shape[0],
            dtype=np.float32,
            format="coo",
        )
    )

    row_sum = np.asarray(
        adjacency.sum(axis=1)
    ).reshape(-1)

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        degree_inverse_sqrt = np.power(
            row_sum,
            -0.5,
        )

    degree_inverse_sqrt[
        ~np.isfinite(
            degree_inverse_sqrt
        )
    ] = 0.0

    degree_matrix = sp.diags(
        degree_inverse_sqrt
    )

    # Same multiplication order used by ncVarPred.
    normalized_adjacency = (
        adjacency
        .dot(degree_matrix)
        .transpose()
        .dot(degree_matrix)
    )

    normalized_adjacency = (
        normalized_adjacency
        .toarray()
        .astype(np.float32)
    )

    return normalized_adjacency


def normalize_chromosome(
    chromosome: str,
) -> str:
    """
    Normalize chromosome names to the format used by
    ncVarPred's structure-matching index.
    """
    chromosome = str(chromosome).strip()

    if chromosome.lower().startswith("chr"):
        suffix = chromosome[3:]
    else:
        suffix = chromosome

    if suffix.lower() == "x":
        suffix = "X"
    elif suffix.lower() == "y":
        suffix = "Y"
    elif suffix.lower() in {"m", "mt"}:
        suffix = "M"

    return f"chr{suffix}"


def load_graph_index(
    graph_index_path: Path,
) -> pd.DataFrame:
    """
    Load the original ncVarPred 100-kb whole-genome
    structure-matching index.

    Columns:
        chromosome
        bin start
        bin end
    """
    graph_index = pd.read_csv(
        graph_index_path,
        sep="\t",
        header=None,
        names=[
            "chrom",
            "bin_start",
            "bin_end",
        ],
    )

    graph_index["chrom"] = (
        graph_index["chrom"]
        .map(normalize_chromosome)
    )

    graph_index["bin_start"] = pd.to_numeric(
        graph_index["bin_start"],
        errors="raise",
    ).astype("int64")

    graph_index["bin_end"] = pd.to_numeric(
        graph_index["bin_end"],
        errors="raise",
    ).astype("int64")

    graph_index["node_index"] = np.arange(
        len(graph_index),
        dtype=np.int64,
    )

    duplicated_nodes = graph_index.duplicated(
        subset=[
            "chrom",
            "bin_start",
        ]
    )

    if duplicated_nodes.any():
        raise RuntimeError(
            "Duplicate chromosome/bin-start pairs were "
            "found in the graph index."
        )

    return graph_index


def map_cpg_to_graph_nodes(
    data: pd.DataFrame,
    graph_index: pd.DataFrame,
    resolution: int = RESOLUTION,
) -> np.ndarray:
    """
    Map each 1-based CpG coordinate to the corresponding
    ncVarPred graph node.

    The original ncVarPred implementation maps coordinates
    to the beginning of their genomic resolution bin.

    Our CpG coordinates are 1-based, so position - 1 is used
    before calculating the zero-based bin start.
    """
    required_columns = {
        "chrom",
        "canonical_position",
    }

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing input columns: "
            f"{sorted(missing_columns)}"
        )

    mapping_data = data.copy()

    mapping_data["chrom"] = (
        mapping_data["chrom"]
        .map(normalize_chromosome)
    )

    mapping_data["canonical_position"] = (
        pd.to_numeric(
            mapping_data["canonical_position"],
            errors="raise",
        ).astype("int64")
    )

    if (
        mapping_data["canonical_position"]
        <= 0
    ).any():
        raise ValueError(
            "CpG coordinates must be positive 1-based values."
        )

    mapping_data["bin_start"] = (
        (
            mapping_data["canonical_position"]
            - 1
        )
        // resolution
        * resolution
    )

    node_lookup = {
        (
            row.chrom,
            int(row.bin_start),
        ): int(row.node_index)
        for row in graph_index.itertuples(
            index=False
        )
    }

    node_indices: list[int] = []
    missing_nodes: list[
        tuple[str, int, int]
    ] = []

    for row in mapping_data.itertuples(
        index=False
    ):
        key = (
            row.chrom,
            int(row.bin_start),
        )

        node_index = node_lookup.get(key)

        if node_index is None:
            missing_nodes.append(
                (
                    row.chrom,
                    int(row.canonical_position),
                    int(row.bin_start),
                )
            )
        else:
            node_indices.append(
                node_index
            )

    if missing_nodes:
        preview = missing_nodes[:10]

        raise RuntimeError(
            f"{len(missing_nodes)} CpGs could not be mapped "
            f"to graph nodes. First records: {preview}"
        )

    return np.asarray(
        node_indices,
        dtype=np.int64,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map DeepMeth CpG coordinates to the original "
            "ncVarPred 100-kb graph node index."
        )
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
    )

    parser.add_argument(
        "--graph-index-path",
        type=Path,
        default=DEFAULT_GRAPH_INDEX_PATH,
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(
            f"Input dataset does not exist: "
            f"{args.input_path}"
        )

    if not args.graph_index_path.exists():
        raise FileNotFoundError(
            f"Graph index does not exist: "
            f"{args.graph_index_path}"
        )

    data = pd.read_csv(
        args.input_path
    )

    graph_index = load_graph_index(
        args.graph_index_path
    )

    node_indices = map_cpg_to_graph_nodes(
        data=data,
        graph_index=graph_index,
        resolution=RESOLUTION,
    )

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        args.output_path,
        node_indices,
    )

    print(
        "Input dataset:",
        args.input_path,
    )

    print(
        "Graph index:",
        args.graph_index_path,
    )

    print(
        "Graph resolution:",
        RESOLUTION,
    )

    print(
        "Number of graph nodes:",
        len(graph_index),
    )

    print(
        "Number of input CpGs:",
        len(data),
    )

    print(
        "Mapped CpGs:",
        len(node_indices),
    )

    print(
        "Unique selected nodes:",
        len(np.unique(node_indices)),
    )

    print(
        "Output shape:",
        node_indices.shape,
    )

    print(
        "Output file:",
        args.output_path,
    )

    print(
        "First node indices:",
        node_indices[:10].tolist(),
    )


if __name__ == "__main__":
    main()