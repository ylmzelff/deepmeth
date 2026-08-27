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
    ACTIVE_FOUNDATION_TOKEN_DIR,
    # ACTIVE_GRAPH_DIR,  # graph branch retired for now - see model/deepmeth_model.py
    ACTIVE_PHYSICOCHEMICAL_DIR,
    ACTIVE_SEQUENCE_CODES_DIR,
    # ACTIVE_SPLIT_NODE_INDEX_DIR,  # graph branch retired for now
    # GRAPH_RESOLUTION,  # graph branch retired for now
    PHYSICOCHEMICAL_PROPERTY_FILE,
    SHUFFLE_BUFFER_SIZE,
)
from feature_extraction.extract_physicochemical import (
    expand_codes_to_matrix,
    load_physicochemical_properties_di,
)
from feature_extraction.extract_sequence_codes import expand_codes_to_one_hot

# NODE_INDEX_PATH = ACTIVE_GRAPH_DIR / "node_index.parquet"  # graph branch retired for now


class DeepMethShardDataset(IterableDataset):
    def __init__(self, split_name: str, shuffle: bool, seed: int = 0):
        super().__init__()
        self.split_name = split_name
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.physchem_dir = ACTIVE_PHYSICOCHEMICAL_DIR / split_name
        self.sequence_dir = ACTIVE_SEQUENCE_CODES_DIR / split_name
        self.foundation_dir = ACTIVE_FOUNDATION_TOKEN_DIR / split_name

        with (self.physchem_dir / "manifest.json").open(encoding="utf-8") as file:
            self.physchem_manifest = json.load(file)

        with (self.sequence_dir / "manifest.json").open(encoding="utf-8") as file:
            self.sequence_manifest = json.load(file)

        with (self.foundation_dir / "manifest.json").open(encoding="utf-8") as file:
            self.foundation_manifest = json.load(file)

        if self.physchem_manifest["shard_count"] != self.sequence_manifest["shard_count"]:
            raise RuntimeError(
                f"{split_name}: physicochemical has {self.physchem_manifest['shard_count']} shards, "
                f"sequence codes has {self.sequence_manifest['shard_count']} - out of sync."
            )

        if self.foundation_manifest["total_rows"] != self.physchem_manifest["total_rows"]:
            raise RuntimeError(
                f"{split_name}: physicochemical has {self.physchem_manifest['total_rows']:,} rows, "
                f"foundation token features have {self.foundation_manifest['total_rows']:,} - out of sync."
            )

        self.property_table = load_physicochemical_properties_di(PHYSICOCHEMICAL_PROPERTY_FILE)

        self.total_rows = self.physchem_manifest["total_rows"]

        self.shard_row_counts = [shard["row_count"] for shard in self.physchem_manifest["shards"]]
        self.shard_offsets = np.concatenate([[0], np.cumsum(self.shard_row_counts)])[:-1]

        # Foundation shards are a different (smaller) size than physchem/sequence
        # shards, so they can't be read by matching shard_index - instead we look
        # up whichever foundation shard(s) overlap a given absolute row range.
        self.foundation_shard_row_counts = [shard["row_count"] for shard in self.foundation_manifest["shards"]]
        self.foundation_shard_offsets = np.concatenate([[0], np.cumsum(self.foundation_shard_row_counts)])[:-1]

        self.negative_count, self.positive_count = self._verify_alignment()

    def _physchem_shard_path(self, shard_record: dict) -> Path:
        return self.physchem_dir / Path(shard_record["file"]).name

    def _sequence_shard_path(self, shard_record: dict) -> Path:
        return self.sequence_dir / Path(shard_record["file"]).name

    def _foundation_shard_path(self, shard_record: dict) -> Path:
        return self.foundation_dir / Path(shard_record["file"]).name

    def _foundation_shards_overlapping(self, start: int, count: int):
        """Yields (shard_record, local_start, local_end) for every foundation shard
        that overlaps the absolute row range [start, start + count)."""
        end = start + count

        for shard_index, shard_record in enumerate(self.foundation_manifest["shards"]):
            shard_start = int(self.foundation_shard_offsets[shard_index])
            shard_end = shard_start + self.foundation_shard_row_counts[shard_index]

            if shard_end <= start or shard_start >= end:
                continue

            local_start = max(start, shard_start) - shard_start
            local_end = min(end, shard_end) - shard_start
            yield shard_record, local_start, local_end

    def _foundation_metadata_for_range(self, start: int, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cheap: only labels/chromosomes/positions (for alignment verification) -
        never touches the large token_embeddings/attention_mask arrays."""
        label_chunks, chrom_chunks, position_chunks = [], [], []

        for shard_record, local_start, local_end in self._foundation_shards_overlapping(start, count):
            with np.load(self._foundation_shard_path(shard_record)) as data:
                label_chunks.append(data["labels"][local_start:local_end])
                chrom_chunks.append(data["chromosomes"][local_start:local_end])
                position_chunks.append(data["positions"][local_start:local_end])

        return (
            np.concatenate(label_chunks, axis=0),
            np.concatenate(chrom_chunks, axis=0),
            np.concatenate(position_chunks, axis=0),
        )

    def _foundation_tokens_for_range(self, start: int, count: int) -> tuple[np.ndarray, np.ndarray]:
        """token_embeddings [count, T, 768] + attention_mask [count, T] for the
        absolute row range [start, start + count)."""
        token_chunks, mask_chunks = [], []

        for shard_record, local_start, local_end in self._foundation_shards_overlapping(start, count):
            with np.load(self._foundation_shard_path(shard_record)) as data:
                token_chunks.append(data["token_embeddings"][local_start:local_end])
                mask_chunks.append(data["attention_mask"][local_start:local_end])

        return (
            np.concatenate(token_chunks, axis=0),
            np.concatenate(mask_chunks, axis=0),
        )

    def _verify_alignment(self) -> tuple[int, int]:
        # graph_node_index = pd.read_parquet(NODE_INDEX_PATH, columns=["chrom", "bin_start", "node_index"])

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
            foundation_labels, foundation_chromosomes, foundation_positions = self._foundation_metadata_for_range(
                offset, physchem_shard["row_count"]
            )

            if not (
                np.array_equal(labels, foundation_labels.astype(labels.dtype))
                and np.array_equal(chromosomes, foundation_chromosomes)
                and np.array_equal(positions, foundation_positions)
            ):
                raise RuntimeError(
                    f"{self.split_name} shard {shard_index}: physicochemical and foundation token-feature "
                    "shards disagree on (label, chrom, position) - out of sync, rerun "
                    "extract_dnabert2_tokens_gm12878.py."
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

            row_count = physchem_shard["row_count"]
            offset = int(self.shard_offsets[shard_index])
            foundation_tokens, foundation_mask = self._foundation_tokens_for_range(offset, row_count)

            physchem_matrix = expand_codes_to_matrix(physchem_codes, self.property_table)
            one_hot = expand_codes_to_one_hot(sequence_codes)

            for row in range(row_count):
                sample = (
                    one_hot[row].copy(),
                    physchem_matrix[row].copy(),
                    foundation_tokens[row].copy(),
                    foundation_mask[row].copy(),
                    int(labels[row]),
                )

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
    one_hot, physchem, foundation_tokens, foundation_mask, labels = zip(*batch)

    return {
        "sequence": torch.from_numpy(np.stack(one_hot)).float(),
        "physicochemical": torch.from_numpy(np.stack(physchem)).float(),
        # "node_index": ...,  # graph branch retired for now - see model/deepmeth_model.py
        "foundation_tokens": torch.from_numpy(np.stack(foundation_tokens)).float(),
        "foundation_attention_mask": torch.from_numpy(np.stack(foundation_mask)).long(),
        "label": torch.tensor(labels, dtype=torch.float32),
    }
