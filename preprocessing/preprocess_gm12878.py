from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


_ON_COLAB = Path("/content").exists()

if _ON_COLAB and not Path("/content/drive/MyDrive").exists():
    from google.colab import drive  # type: ignore[import-not-found]

    print("Mounting Google Drive (not yet mounted)...")
    drive.mount("/content/drive")

import numpy as np
import pandas as pd
from twobitreader import TwoBitFile

from config.data_config.gm12878_config import (
    GM12878_DATA_DIR,
    GM12878_RRBS_REPLICATE_PATHS,
    HG19_2BIT_PATH,
)
from config.project_config import MAX_UNKNOWN_FRACTION, SEQUENCE_LENGTH
from scripts.analyze_data import load_bedmethyl
from preprocessing.preprocess_hepg2 import (
    add_canonical_position,
    add_read_counts,
    assign_chromosomes_to_splits,
    extract_cpg_centered_sequence,
)

# ============================================================
# GM12878-specific settings
# ============================================================

OUTPUT_DIR = GM12878_DATA_DIR / "proceed"

MIN_COVERAGE_PER_REPLICATE = 5

INCLUDED_CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX"]


MIN_TWO_STRAND_FRACTION = 0.05


def merge_strands_per_replicate(dataframe: pd.DataFrame, label: str) -> pd.DataFrame:
    merged = (
        dataframe.groupby(["chrom", "canonical_position"], as_index=False, sort=False)
        .agg(
            strand_row_count=("canonical_position", "size"),
            count_m=("count_m", "sum"),
            count_u=("count_u", "sum"),
        )
    )

    merged["coverage"] = merged["count_m"] + merged["count_u"]

    support = merged["strand_row_count"].value_counts().sort_index().to_dict()
    two_strand_fraction = float(merged["strand_row_count"].eq(2).mean())

    print(f"{label}: strand support distribution: {support}")
    print(f"{label}: fraction of CpGs supported by both strands: {two_strand_fraction * 100:.2f}%")

    if two_strand_fraction < MIN_TWO_STRAND_FRACTION:
        raise RuntimeError(
            f"{label}: only {two_strand_fraction * 100:.2f}% of CpGs have both strands merged - "
            f"below even the RRBS-adjusted {MIN_TWO_STRAND_FRACTION * 100:.0f}% floor. Re-run "
            "preprocessing/analyze_gm12878.py's offset search before trusting this output."
        )

    return merged


def load_and_prepare_replicate(path: Path, label: str) -> pd.DataFrame:
    dataframe = load_bedmethyl(path)
    print(f"{label}: loaded {len(dataframe):,} raw bedMethyl rows")

    before = len(dataframe)
    dataframe = dataframe[dataframe["chrom"].isin(INCLUDED_CHROMOSOMES)].reset_index(drop=True)
    print(f"{label}: kept {len(dataframe):,}/{before:,} rows on standard chromosomes")

    dataframe = add_read_counts(dataframe)
    dataframe = add_canonical_position(dataframe)
    dataframe = merge_strands_per_replicate(dataframe, label)

    # preprocess.py's merge_strands_per_replicate doesn't compute a ratio
    # column (only preprocess_v2.py's local redefinition did, for HepG2) -
    # added here so merge_replicates can check replicate concordance below,
    # applying that lesson from the start instead of discovering it's
    # needed again after the fact.
    dataframe["methylation_ratio"] = dataframe["count_m"] / dataframe["coverage"].replace(0, np.nan)

    before = len(dataframe)
    dataframe = dataframe[dataframe["coverage"] >= MIN_COVERAGE_PER_REPLICATE].reset_index(drop=True)
    print(
        f"{label}: kept {len(dataframe):,}/{before:,} physical CpGs "
        f"with coverage >= {MIN_COVERAGE_PER_REPLICATE}"
    )

    return dataframe

MAX_REPLICATE_RATIO_DIFFERENCE = 0.20


def merge_replicates(replicate_1: pd.DataFrame, replicate_2: pd.DataFrame) -> pd.DataFrame:
    merged = replicate_1.merge(
        replicate_2,
        on=["chrom", "canonical_position"],
        how="inner",
        suffixes=("_rep1", "_rep2"),
    )
    print(f"CpGs covered in both replicates: {len(merged):,}")

    merged["count_m"] = merged["count_m_rep1"] + merged["count_m_rep2"]
    merged["count_u"] = merged["count_u_rep1"] + merged["count_u_rep2"]
    merged["coverage"] = merged["count_m"] + merged["count_u"]
    merged["replicate_ratio_difference"] = (
        merged["methylation_ratio_rep1"] - merged["methylation_ratio_rep2"]
    ).abs()

    before = len(merged)
    merged = merged[merged["replicate_ratio_difference"] <= MAX_REPLICATE_RATIO_DIFFERENCE].reset_index(drop=True)
    print(
        f"Replicate ratio difference <= {MAX_REPLICATE_RATIO_DIFFERENCE}: "
        f"kept {len(merged):,}/{before:,}"
    )

    # Official grant-form label rule - see [[methylation_label_definition]]:
    # applies regardless of data source (HepG2/WGBS or GM12878/RRBS).
    merged["label"] = (merged["count_m"] > merged["count_u"]).astype("int8")

    merged["consensus_methylation_ratio"] = merged["count_m"] / merged["coverage"].replace(0, np.nan)

    merged = merged[
        [
            "chrom", "canonical_position", "count_m", "count_u", "coverage",
            "replicate_ratio_difference", "label", "consensus_methylation_ratio",
        ]
    ].sort_values(["chrom", "canonical_position"]).reset_index(drop=True)

    duplicate_count = int(merged.duplicated(subset=["chrom", "canonical_position"]).sum())
    if duplicate_count:
        raise RuntimeError(f"{duplicate_count} duplicated canonical CpGs after replicate merge.")

    print(f"Consensus: {len(merged):,} CpGs covered in both replicates, concordant")
    print(f"  Methylated:   {int(merged['label'].eq(1).sum()):,} ({merged['label'].mean() * 100:.2f}%)")
    print(f"  Unmethylated: {int(merged['label'].eq(0).sum()):,} ({(1 - merged['label'].mean()) * 100:.2f}%)")

    return merged




def add_sequences_and_qc(dataframe: pd.DataFrame) -> pd.DataFrame:
    print(f"Loading hg19 reference: {HG19_2BIT_PATH}")
    genome = TwoBitFile(str(HG19_2BIT_PATH))
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

    print(f"Sequence QC: {len(valid):,}/{len(dataframe):,} CpGs have a valid hg19 501bp sequence")

    invalid_counts = dataframe.loc[~dataframe["sequence_status"].eq("Valid"), "sequence_status"].value_counts()
    for status, count in invalid_counts.items():
        print(f"  Dropped ({status}): {count:,}")

    return valid.drop(columns=["unknown_base_count", "unsupported_base_count", "unknown_fraction", "sequence_status"])


# ============================================================
# Splits
# ============================================================

def split_disjoint(dataframe: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every included chromosome, disjointly assigned - same approach as
    the main HepG2 pipeline (preprocessing/preprocess_hepg2.py).
    """
    chromosome_counts = dataframe["chrom"].value_counts().to_dict()
    chromosome_to_split = assign_chromosomes_to_splits(chromosome_counts)

    dataframe = dataframe.copy()
    dataframe["split"] = dataframe["chrom"].map(chromosome_to_split)

    return {
        split_name: dataframe[dataframe["split"].eq(split_name)].drop(columns=["split"]).reset_index(drop=True)
        for split_name in ("train", "validation", "test")
    }


def save_splits(splits: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_dataframe in splits.items():
        output_path = output_dir / f"{split_name}.parquet"
        split_dataframe.to_parquet(output_path, index=False)

        positive_fraction = split_dataframe["label"].mean() if len(split_dataframe) else float("nan")
        print(
            f"  {split_name}: {len(split_dataframe):,} rows, {split_dataframe['chrom'].nunique()} chromosome(s), "
            f"{positive_fraction * 100:.2f}% methylated -> {output_path}"
        )


def main() -> None:
    print("=" * 70)
    print("DeepMeth GM12878 preprocessing: RRBS consensus dataset (hg19)")
    print("=" * 70)

    labels = ["rep1", "rep2"]

    print("\n[1/4] Loading and preparing RRBS replicates")
    replicates = [
        load_and_prepare_replicate(path, label)
        for label, path in zip(labels, GM12878_RRBS_REPLICATE_PATHS)
    ]

    print("\n[2/4] Merging replicates and assigning official labels")
    consensus = merge_replicates(replicates[0], replicates[1])

    print("\n[3/4] Extracting hg19 sequence context")
    consensus = add_sequences_and_qc(consensus)

    print("\n[4/4] Building the disjoint split")

    print("\nDisjoint split (all chromosomes, 80/10/10):")
    disjoint_splits = split_disjoint(consensus)
    save_splits(disjoint_splits, OUTPUT_DIR / "disjoint_split")

    print("\nGM12878 preprocessing completed.")


if __name__ == "__main__":
    main()
