"""
Mandatory data-quality audit for the GM12878 consensus dataset, run
BEFORE feature extraction/training - not after, the way the HepG2
pipeline's redundancy/concordance problems were originally found (only
after several training runs, requiring preprocess_v2.py through
preprocess_v5.py to fix retroactively - see project history). Checks the
exact same things that turned out to matter for HepG2:

  1. Coverage distribution (min/mean/median, fraction below common
     thresholds)
  2. Class balance (methylated/unmethylated fraction, implied pos_weight)
  3. Genomic redundancy: how many CpGs share a SEQUENCE_LENGTH-sized
     genomic bin with another CpG (near-duplicate 501bp windows) - this
     was ~97% of rows on HepG2/WGBS; RRBS is expected to be less
     redundant (enrichment-based, not every CpG), but that's an
     expectation to verify, not assume.

Run against disjoint_split, the only split preprocess_gm12878.py produces
(an earlier baseline_split, chr1-train/chr21-test, was removed - see that
script's docstring).

Usage (no arguments needed, after preprocess_gm12878.py):

    python preprocessing/audit_gm12878.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from preprocessing.download_data_gm12878 import GM12878_DATA_DIR

OUTPUT_DIR = GM12878_DATA_DIR / "proceed"
GENOMIC_BIN_SIZE = 501  # matches SEQUENCE_LENGTH - see module docstring


def audit_split(path: Path, split_name: str) -> None:
    if not path.exists():
        print(f"  {split_name}: {path} does not exist, skipping.")
        return

    dataframe = pd.read_parquet(path)
    total = len(dataframe)

    print(f"\n--- {split_name} ({path}) ---")
    print(f"Total rows: {total:,}")

    print("\nCoverage:")
    print(dataframe["coverage"].describe().to_string())
    for threshold in (10, 20, 50):
        fraction_below = (dataframe["coverage"] < threshold).mean()
        print(f"  < {threshold}x: {fraction_below * 100:.2f}%")

    positive_fraction = dataframe["label"].mean()
    pos_weight = (1 - positive_fraction) / positive_fraction if positive_fraction > 0 else float("nan")
    print("\nClass balance:")
    print(f"  Methylated:   {positive_fraction * 100:.2f}%")
    print(f"  Unmethylated: {(1 - positive_fraction) * 100:.2f}%")
    print(f"  Implied pos_weight (neg/pos): {pos_weight:.4f}")

    print(f"\nGenomic redundancy ({GENOMIC_BIN_SIZE}bp bins):")
    dataframe = dataframe.copy()
    dataframe["_bin"] = (dataframe["canonical_position"] - 1) // GENOMIC_BIN_SIZE
    bin_sizes = dataframe.groupby(["chrom", "_bin"]).size()
    redundant_row_fraction = (dataframe.groupby(["chrom", "_bin"])["label"].transform("size") > 1).mean()

    print(f"  Distinct bins: {len(bin_sizes):,} ({len(bin_sizes) / total * 100:.2f}% of rows if deduplicated to 1/bin)")
    print(f"  Mean CpGs/bin: {bin_sizes.mean():.3f}  median: {bin_sizes.median()}  max: {bin_sizes.max()}")
    print(f"  Rows in a >1-CpG bin (redundant-with-something): {redundant_row_fraction * 100:.2f}%")
    print("  (HepG2/WGBS reference point: 96.91% - see project history)")

    if "replicate_ratio_difference" in dataframe.columns:
        print("\nReplicate concordance (already filtered to <= 0.20, distribution of survivors):")
        print(dataframe["replicate_ratio_difference"].describe().to_string())


def main() -> None:
    print("=" * 70)
    print("GM12878 data-quality audit - run this before feature extraction/training")
    print("=" * 70)

    print("\n### disjoint_split (used for actual training) ###")
    for split_name in ("train", "validation", "test"):
        audit_split(OUTPUT_DIR / "disjoint_split" / f"{split_name}.parquet", f"disjoint_split/{split_name}")

    print("\n\nAudit completed. Review the numbers above before running feature extraction.")


if __name__ == "__main__":
    main()
