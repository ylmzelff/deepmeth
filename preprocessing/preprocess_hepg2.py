from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from twobitreader import TwoBitFile

from config.data_config.hepg2_config import (
    GRCH38_2BIT_PATH,
    MAX_REPLICATE_RATIO_DIFFERENCE,
    MIN_COVERAGE_PER_REPLICATE,
    MIN_TOTAL_COVERAGE,
    WGBS_REPLICATE_PATHS,
)
from config.project_config import (
    DATASET_DIR,
    INCLUDED_CHROMOSOMES,
    MAX_UNKNOWN_FRACTION,
    SEQUENCE_LENGTH,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
)
from scripts.analyze_data import load_bedmethyl

CENTER_INDEX = SEQUENCE_LENGTH // 2


# ============================================================
# Per-replicate: strand merge and coverage filter
# ============================================================

def add_read_counts(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Derive count_m / count_u from coverage and percent_methylated.

    count_m = round(coverage * percent_methylated / 100), count_u = coverage - count_m.
    """
    dataframe = dataframe.copy()

    dataframe["count_m"] = np.round(
        dataframe["coverage"] * dataframe["percent_methylated"] / 100.0
    ).astype("int64")

    dataframe["count_u"] = dataframe["coverage"] - dataframe["count_m"]

    return dataframe


def add_canonical_position(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert BED 0-based, strand-specific coordinates to a 1-based
    canonical CpG position (the cytosine of the "+" strand), exactly as
    the previous scRRBS pipeline did for C/+ and G/- pairs.
    """
    dataframe = dataframe.copy()

    reported_position = dataframe["chrom_start"] + 1  # 0-based -> 1-based

    dataframe["canonical_position"] = np.where(
        dataframe["strand"].eq("+"),
        reported_position,
        reported_position - 1,
    ).astype("int64")

    return dataframe


def merge_strands_per_replicate(dataframe: pd.DataFrame, label: str) -> pd.DataFrame:
    """Collapse +/- strand rows of the same physical CpG within one replicate."""
    merged = (
        dataframe.groupby(["chrom", "canonical_position"], as_index=False, sort=False)
        .agg(
            strand_row_count=("canonical_position", "size"),
            count_m=("count_m", "sum"),
            count_u=("count_u", "sum"),
        )
    )

    merged["coverage"] = merged["count_m"] + merged["count_u"]

    print(
        f"{label}: {len(dataframe):,} strand-specific rows -> "
        f"{len(merged):,} physical CpGs after strand merge"
    )

    
    strand_support_counts = merged["strand_row_count"].value_counts().sort_index()
    two_strand_fraction = float(merged["strand_row_count"].eq(2).mean())

    print(f"{label}: strand_row_count distribution: {strand_support_counts.to_dict()}")
    print(f"{label}: fraction of CpGs supported by both strands: {two_strand_fraction * 100:.2f}%")

    if two_strand_fraction < 0.5:
        raise RuntimeError(
            f"{label}: only {two_strand_fraction * 100:.2f}% of CpGs have both strands merged. "
            "This suggests the +/- strand canonical-position offset in add_canonical_position() "
            "does not match this file's convention - inspect a few raw rows before trusting the output."
        )

    return merged


def filter_min_coverage(dataframe: pd.DataFrame, label: str) -> pd.DataFrame:
    before = len(dataframe)

    dataframe = dataframe[dataframe["coverage"] >= MIN_COVERAGE_PER_REPLICATE].reset_index(drop=True)

    print(
        f"{label}: kept {len(dataframe):,}/{before:,} physical CpGs "
        f"with combined coverage >= {MIN_COVERAGE_PER_REPLICATE}"
    )

    return dataframe


def load_and_prepare_replicate(path: Path, label: str) -> pd.DataFrame:
    dataframe = load_bedmethyl(path)
    print(f"{label}: loaded {len(dataframe):,} raw bedMethyl rows")

    before = len(dataframe)
    dataframe = dataframe[dataframe["chrom"].isin(INCLUDED_CHROMOSOMES)].reset_index(drop=True)
    print(f"{label}: kept {len(dataframe):,}/{before:,} rows on standard chromosomes")

    dataframe = add_read_counts(dataframe)
    dataframe = add_canonical_position(dataframe)
    dataframe = merge_strands_per_replicate(dataframe, label)

    # Needed by merge_replicates' replicate-ratio-difference filter below -
    # same column GM12878's preprocess_gm12878.py computes at this point.
    dataframe["methylation_ratio"] = dataframe["count_m"] / dataframe["coverage"].replace(0, np.nan)

    dataframe = filter_min_coverage(dataframe, label)

    return dataframe


# ============================================================
# Cross-replicate merge and labeling
# ============================================================

def merge_replicates(replicate_1: pd.DataFrame, replicate_2: pd.DataFrame) -> pd.DataFrame:
    """Combine the two replicates, keeping only CpGs covered in both."""
    merged = replicate_1.merge(
        replicate_2,
        on=["chrom", "canonical_position"],
        how="inner",
        suffixes=("_rep1", "_rep2"),
    )

    merged["count_m"] = merged["count_m_rep1"] + merged["count_m_rep2"]
    merged["count_u"] = merged["count_u_rep1"] + merged["count_u_rep2"]
    merged["coverage"] = merged["count_m"] + merged["count_u"]

    before = len(merged)
    merged = merged[merged["coverage"] >= MIN_TOTAL_COVERAGE].reset_index(drop=True)
    print(f"Combined coverage >= {MIN_TOTAL_COVERAGE}: kept {len(merged):,}/{before:,}")

    merged["replicate_ratio_difference"] = (
        merged["methylation_ratio_rep1"] - merged["methylation_ratio_rep2"]
    ).abs()

    before = len(merged)
    merged = merged[merged["replicate_ratio_difference"] <= MAX_REPLICATE_RATIO_DIFFERENCE].reset_index(drop=True)
    print(
        f"Replicate ratio difference <= {MAX_REPLICATE_RATIO_DIFFERENCE}: "
        f"kept {len(merged):,}/{before:,}"
    )

    # Official grant-form label rule: methylated iff count_m > count_u, ties -> 0.
    merged["label"] = (merged["count_m"] > merged["count_u"]).astype("int8")

    merged["consensus_methylation_ratio"] = merged["count_m"] / merged["coverage"].replace(0, np.nan)

    merged = merged[
        [
            "chrom",
            "canonical_position",
            "count_m",
            "count_u",
            "coverage",
            "replicate_ratio_difference",
            "label",
            "consensus_methylation_ratio",
        ]
    ].sort_values(["chrom", "canonical_position"]).reset_index(drop=True)

    duplicate_count = int(merged.duplicated(subset=["chrom", "canonical_position"]).sum())

    if duplicate_count:
        raise RuntimeError(f"{duplicate_count} duplicated canonical CpGs after replicate merge.")

    print(f"Consensus: {len(merged):,} CpGs covered in both replicates")
    print(f"  Methylated:   {int(merged['label'].eq(1).sum()):,} ({merged['label'].mean() * 100:.2f}%)")
    print(f"  Unmethylated: {int(merged['label'].eq(0).sum()):,} ({(1 - merged['label'].mean()) * 100:.2f}%)")

    return merged


# ============================================================
# GRCh38 sequence extraction (same QC rules as the original pipeline)
# ============================================================

def extract_cpg_centered_sequence(
    genome: TwoBitFile,
    reference_chromosomes: set[str],
    chrom: str,
    canonical_position: int,
    sequence_length: int = SEQUENCE_LENGTH,
) -> str | None:
    if chrom not in reference_chromosomes:
        return None

    center_index = sequence_length // 2
    c_index = canonical_position - 1  # 1-based -> 0-based
    start = c_index - center_index
    end = start + sequence_length

    if start < 0:
        return None

    chrom_length = len(genome[chrom])

    if end > chrom_length:
        return None

    return str(genome[chrom][start:end]).upper()


def add_sequences_and_qc(dataframe: pd.DataFrame) -> pd.DataFrame:
    print(f"\nLoading GRCh38 reference: {GRCH38_2BIT_PATH}")
    genome = TwoBitFile(str(GRCH38_2BIT_PATH))
    reference_chromosomes = set(genome.keys())

    sequences = [
        extract_cpg_centered_sequence(genome, reference_chromosomes, chrom, int(position))
        for chrom, position in zip(dataframe["chrom"], dataframe["canonical_position"])
    ]

    dataframe = dataframe.copy()
    dataframe["sequence"] = sequences
    dataframe["sequence_length"] = dataframe["sequence"].str.len()

    center_index = SEQUENCE_LENGTH // 2
    dataframe["center_dinucleotide"] = dataframe["sequence"].str.slice(center_index, center_index + 2)
    dataframe["unknown_base_count"] = dataframe["sequence"].str.count(r"[^ACGT]")
    dataframe["unsupported_base_count"] = dataframe["sequence"].str.count(r"[^ACGTN]")
    dataframe["unknown_fraction"] = dataframe["unknown_base_count"] / SEQUENCE_LENGTH

    missing_sequence = dataframe["sequence"].isna()
    incorrect_length = dataframe["sequence"].notna() & dataframe["sequence_length"].ne(SEQUENCE_LENGTH)
    unsupported_bases = dataframe["sequence"].notna() & dataframe["unsupported_base_count"].gt(0)
    incorrect_center = (
        dataframe["sequence"].notna()
        & dataframe["sequence_length"].eq(SEQUENCE_LENGTH)
        & dataframe["center_dinucleotide"].ne("CG")
    )
    too_many_unknowns = dataframe["sequence"].notna() & dataframe["unknown_fraction"].gt(MAX_UNKNOWN_FRACTION)

    dataframe["sequence_status"] = np.select(
        [missing_sequence, incorrect_length, unsupported_bases, incorrect_center, too_many_unknowns],
        [
            "Missing chromosome or boundary sequence",
            "Incorrect sequence length",
            "Unsupported reference base",
            "Center is not CG",
            "Unknown fraction above threshold",
        ],
        default="Valid",
    )

    valid = dataframe[dataframe["sequence_status"].eq("Valid")].reset_index(drop=True)

    print(f"Sequence QC: {len(valid):,}/{len(dataframe):,} CpGs have a valid GRCh38 501bp sequence")

    invalid_counts = (
        dataframe.loc[~dataframe["sequence_status"].eq("Valid"), "sequence_status"]
        .value_counts()
    )

    for status, count in invalid_counts.items():
        print(f"  Dropped ({status}): {count:,}")

    return valid.drop(
        columns=[
            "unknown_base_count",
            "unsupported_base_count",
            "unknown_fraction",
            "sequence_status",
        ]
    )


# ============================================================
# Chromosome-disjoint train / validation / test split
# ============================================================

def assign_chromosomes_to_splits(chromosome_counts: dict[str, int]) -> dict[str, str]:
    """Greedily assign whole chromosomes to splits targeting the configured
    train/validation/test fractions, keeping every chromosome in exactly one split.
    """
    target_fractions = {
        "train": TRAIN_FRACTION,
        "validation": VALIDATION_FRACTION,
        "test": TEST_FRACTION,
    }

    total_count = sum(chromosome_counts.values())
    target_counts = {split: total_count * fraction for split, fraction in target_fractions.items()}
    current_counts = {split: 0 for split in target_fractions}

    assignment: dict[str, str] = {}

    for chrom, count in sorted(chromosome_counts.items(), key=lambda item: item[1], reverse=True):
        deficits = {split: target_counts[split] - current_counts[split] for split in target_fractions}
        chosen_split = max(deficits, key=deficits.get)
        assignment[chrom] = chosen_split
        current_counts[chosen_split] += count

    return assignment


def split_dataset(dataframe: pd.DataFrame) -> dict[str, pd.DataFrame]:
    chromosome_counts = dataframe["chrom"].value_counts().to_dict()
    chromosome_to_split = assign_chromosomes_to_splits(chromosome_counts)

    dataframe = dataframe.copy()
    dataframe["split"] = dataframe["chrom"].map(chromosome_to_split)

    splits = {
        split_name: dataframe[dataframe["split"].eq(split_name)].drop(columns=["split"]).reset_index(drop=True)
        for split_name in ("train", "validation", "test")
    }

    return splits


def save_splits(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    total_rows = sum(len(dataframe) for dataframe in splits.values())
    summary_rows = []

    for split_name, dataframe in splits.items():
        output_path = DATASET_DIR / f"{split_name}.parquet"
        dataframe.to_parquet(output_path, index=False)

        summary_rows.append(
            {
                "split": split_name,
                "row_count": len(dataframe),
                "chromosome_count": dataframe["chrom"].nunique(),
                "unmethylated_count": int(dataframe["label"].eq(0).sum()),
                "methylated_count": int(dataframe["label"].eq(1).sum()),
                "methylated_fraction": float(dataframe["label"].mean()),
                "mean_total_coverage": float(dataframe["coverage"].mean()),
                "fraction_of_consensus": len(dataframe) / total_rows,
            }
        )

        print(f"{split_name}: {len(dataframe):,} rows -> {output_path}")

    summary = pd.DataFrame(summary_rows)
    summary_path = DATASET_DIR / "split_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSplit summary: {summary_path}")

    return summary


def main() -> None:
    print("=" * 70)
    print("DeepMeth preprocessing: HepG2 WGBS consensus dataset (GRCh38)")
    print("=" * 70)

    labels = ["rep1", "rep2"]

    print("\n[1/4] Loading and preparing replicates")
    replicates = [
        load_and_prepare_replicate(path, label)
        for label, path in zip(labels, WGBS_REPLICATE_PATHS)
    ]

    print("\n[2/4] Merging replicates and assigning labels")
    consensus = merge_replicates(replicates[0], replicates[1])

    print("\n[3/4] Extracting GRCh38 sequence context")
    consensus = add_sequences_and_qc(consensus)

    print("\n[4/4] Chromosome-disjoint train/validation/test split")
    splits = split_dataset(consensus)
    summary = save_splits(splits)

    print("\nPreprocessing completed.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
