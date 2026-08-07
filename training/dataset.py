from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info

from config.project_config import (
    ACTIVE_GRAPH_DIR,
    ACTIVE_PHYSICOCHEMICAL_DIR,
    ACTIVE_SEQUENCE_CODES_DIR,
    ACTIVE_SPLIT_NODE_INDEX_DIR,
    GRAPH_RESOLUTION,
    PHYSICOCHEMICAL_PROPERTY_FILE,
    SHUFFLE_BUFFER_SIZE,
)
from feature_extraction.extract_physicochemical import (
    expand_codes_to_matrix,
    load_physicochemical_properties_di,
)
from feature_extraction.extract_sequence_codes import expand_codes_to_one_hot

NODE_INDEX_PATH = ACTIVE_GRAPH_DIR / "node_index.parquet"


class DeepMethShardDataset(IterableDataset):
    def __init__(self, split_name: str, shuffle: bool, seed: int = 0):
        super().__init__()
        self.split_name = split_name
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.physchem_dir = ACTIVE_PHYSICOCHEMICAL_DIR / split_name
        self.sequence_dir = ACTIVE_SEQUENCE_CODES_DIR / split_name

        with (self.physchem_dir / "manifest.json").open(encoding="utf-8") as file:
            self.physchem_manifest = json.load(file)

        with (self.sequence_dir / "manifest.json").open(encoding="utf-8") as file:
            self.sequence_manifest = json.load(file)

        if self.physchem_manifest["shard_count"] != self.sequence_manifest["shard_count"]:
            raise RuntimeError(
                f"{split_name}: physicochemical has {self.physchem_manifest['shard_count']} shards, "
                f"sequence codes has {self.sequence_manifest['shard_count']} - out of sync."
            )

        self.property_table = load_physicochemical_properties_di(PHYSICOCHEMICAL_PROPERTY_FILE)

        node_index_path = ACTIVE_SPLIT_NODE_INDEX_DIR / f"{split_name}_node_index.npy"
        if not node_index_path.exists():
            raise FileNotFoundError(
                f"{node_index_path} does not exist. Run the matching prepare_graph_features script first."
            )
        self.node_index_array = np.load(node_index_path)

        self.total_rows = self.physchem_manifest["total_rows"]
        if self.total_rows != len(self.node_index_array):
            raise RuntimeError(
                f"{split_name}: {self.total_rows:,} rows in physicochemical shards but "
                f"{len(self.node_index_array):,} rows in {node_index_path.name} - out of sync."
            )

        self.shard_row_counts = [shard["row_count"] for shard in self.physchem_manifest["shards"]]
        self.shard_offsets = np.concatenate([[0], np.cumsum(self.shard_row_counts)])[:-1]

        self.negative_count, self.positive_count = self._verify_alignment()

    def _physchem_shard_path(self, shard_record: dict) -> Path:
        return self.physchem_dir / Path(shard_record["file"]).name

    def _sequence_shard_path(self, shard_record: dict) -> Path:
        return self.sequence_dir / Path(shard_record["file"]).name

    def _verify_alignment(self) -> tuple[int, int]:
        graph_node_index = pd.read_parquet(NODE_INDEX_PATH, columns=["chrom", "bin_start", "node_index"])

        negative_count = 0
        positive_count = 0

        for shard_index, physchem_shard in enumerate(self.physchem_manifest["shards"]):
            sequence_shard = self.sequence_manifest["shards"][shard_index]

            if physchem_shard["row_count"] != sequence_shard["row_count"]:
                raise RuntimeError(
                    f"{self.split_name} shard {shard_index}: row count mismatch between "
                    "physicochemical and sequence-code shards."
                )

            with np.load(self._physchem_shard_path(physchem_shard)) as data:
                labels = data["labels"]
                chromosomes = data["chromosomes"]
                positions = data["positions"]

            with np.load(self._sequence_shard_path(sequence_shard)) as data:
                sequence_labels = data["labels"]
                sequence_chromosomes = data["chromosomes"]
                sequence_positions = data["positions"]

            if not (
                np.array_equal(labels, sequence_labels)
                and np.array_equal(chromosomes, sequence_chromosomes)
                and np.array_equal(positions, sequence_positions)
            ):
                raise RuntimeError(
                    f"{self.split_name} shard {shard_index}: physicochemical and sequence-code "
                    "shards disagree on (label, chrom, position) - shards are misaligned."
                )

            offset = int(self.shard_offsets[shard_index])
            row_count = physchem_shard["row_count"]
            precomputed_node_indices = self.node_index_array[offset : offset + row_count]

            bin_starts = (positions.astype(np.int64) - 1) // GRAPH_RESOLUTION * GRAPH_RESOLUTION
            lookup_frame = pd.DataFrame({"chrom": chromosomes, "bin_start": bin_starts})
            merged = lookup_frame.merge(graph_node_index, on=["chrom", "bin_start"], how="left")

            if merged["node_index"].isna().any():
                raise RuntimeError(
                    f"{self.split_name} shard {shard_index}: some (chrom, position) rows could not "
                    "be mapped to a graph node."
                )

            recomputed_node_indices = merged["node_index"].to_numpy(dtype=np.int64)

            if not np.array_equal(recomputed_node_indices, precomputed_node_indices):
                raise RuntimeError(
                    f"{self.split_name} shard {shard_index}: recomputed graph node indices don't match "
                    f"{self.split_name}_node_index.npy - out of sync, rerun the matching "
                    "prepare_graph_features script."
                )

            positive_count += int(labels.sum())
            negative_count += int(len(labels) - labels.sum())

        return negative_count, positive_count

    def __len__(self) -> int:
        return self.total_rows

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _shard_order(self) -> list[int]:
        shard_indices = list(range(len(self.shard_row_counts)))
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(shard_indices)
        return shard_indices

    @staticmethod
    def _worker_shard_slice(shard_indices: list[int]) -> list[int]:
        worker_info = get_worker_info()
        if worker_info is None:
            return shard_indices
        return shard_indices[worker_info.id :: worker_info.num_workers]

    def __iter__(self):
        shard_indices = self._worker_shard_slice(self._shard_order())
        rng = np.random.default_rng(self.seed + self.epoch + 1)
        buffer: list[tuple] = []

        for shard_index in shard_indices:
            physchem_shard = self.physchem_manifest["shards"][shard_index]
            sequence_shard = self.sequence_manifest["shards"][shard_index]

            with np.load(self._physchem_shard_path(physchem_shard)) as data:
                physchem_codes = data["codes"]
                labels = data["labels"]

            with np.load(self._sequence_shard_path(sequence_shard)) as data:
                sequence_codes = data["codes"]

            offset = int(self.shard_offsets[shard_index])
            row_count = physchem_shard["row_count"]
            node_indices = self.node_index_array[offset : offset + row_count]

            physchem_matrix = expand_codes_to_matrix(physchem_codes, self.property_table)
            one_hot = expand_codes_to_one_hot(sequence_codes)

            for row in range(row_count):
                sample = (one_hot[row].copy(), physchem_matrix[row].copy(), int(node_indices[row]), int(labels[row]))

                if not self.shuffle:
                    yield sample
                    continue

                if len(buffer) < SHUFFLE_BUFFER_SIZE:
                    buffer.append(sample)
                    continue

                swap_index = rng.integers(SHUFFLE_BUFFER_SIZE)
                yield buffer[swap_index]
                buffer[swap_index] = sample

        if self.shuffle:
            rng.shuffle(buffer)
            for sample in buffer:
                yield sample


def collate_batch(batch):
    one_hot, physchem, node_indices, labels = zip(*batch)

    return {
        "sequence": torch.from_numpy(np.stack(one_hot)).float(),
        "physicochemical": torch.from_numpy(np.stack(physchem)).float(),
        "node_index": torch.tensor(node_indices, dtype=torch.long),
        "label": torch.tensor(labels, dtype=torch.float32),
    }
