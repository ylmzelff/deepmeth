"""
Exploratory analysis of the raw GM12878 RRBS bedMethyl replicates -
written and run BEFORE any preprocessing/transformation logic
(preprocess_gm12878.py), so every assumption that script makes (strand
pairing offset, coverage range, chromosome list, header lines, ...) is
based on what's actually in the file, not carried over unmodified from
the HepG2/WGBS pipeline. This is exactly what the first preprocess_gm12878.py
run needed but didn't have: it inherited WGBS's +/- strand canonical-
position offset assumption, which turned out not to match RRBS's
convention (only 15.29% of CpGs paired instead of the expected >50%) -
see project history for that failure.

Two things this reports, in order:

  1. A full raw-column profile per replicate (row count, dtypes,
     chromosome list, strand/coverage/percent_methylated distributions,
     sample rows) - the same kind of profiling analyze_data.py does for
     HepG2/WGBS, just run here first instead of skipped.
  2. Empirical +/- strand offset detection: rather than assuming a
     convention, this tests every plausible chrom_start offset between a
     '+' row and a '-' row against the actual data and reports which one
     actually pairs the most rows - preprocess_gm12878.py's
     add_canonical_position()-equivalent should then use whichever offset
     wins here, not WGBS's.

Usage (no arguments needed, after download_data_gm12878.py):

    python preprocessing/analyze_gm12878.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from preprocessing.analyze_data import load_bedmethyl
from preprocessing.download_data_gm12878 import GM12878_RRBS_REPLICATE_PATHS

REPLICATE_LABELS = ("rep1", "rep2")
OFFSET_SAMPLE_SIZE = 200_000
CANDIDATE_OFFSETS = (-2, -1, 0, 1, 2)


def profile_raw_columns(dataframe: pd.DataFrame, label: str) -> None:
    print(f"\n{'=' * 70}\n{label}: raw column profile\n{'=' * 70}")
    print(f"Rows: {len(dataframe):,}")
    print(f"Columns: {dataframe.columns.tolist()}")

    print("\nFirst 10 rows:")
    print(dataframe.head(10).to_string())

    print("\nDtypes:")
    print(dataframe.dtypes.to_string())

    print(f"\nChromosomes present ({dataframe['chrom'].nunique()}):")
    print(sorted(dataframe["chrom"].unique().tolist()))

    print("\nStrand value counts:")
    print(dataframe["strand"].value_counts().to_string())

    print("\nchrom_end - chrom_start value counts (feature width):")
    print((dataframe["chrom_end"] - dataframe["chrom_start"]).value_counts().to_string())

    print("\ncoverage describe:")
    print(dataframe["coverage"].describe().to_string())

    print("\npercent_methylated describe:")
    print(dataframe["percent_methylated"].describe().to_string())

    print(f"\n'name' field: {dataframe['name'].nunique()} unique values, first 5: "
          f"{dataframe['name'].unique()[:5].tolist()}")
    print(f"'score' field: {dataframe['score'].nunique()} unique values, first 10: "
          f"{sorted(dataframe['score'].unique().tolist())[:10]}")
    print(f"'item_rgb' field, first 5 unique: {dataframe['item_rgb'].unique()[:5].tolist()}")


def determine_strand_offset(dataframe: pd.DataFrame, label: str) -> int:
    """For a sample of '+' strand rows, test each candidate chrom_start
    offset against actual '-' strand rows on the same chromosome and
    report which one pairs the most - the empirical answer to "what
    canonical-position convention does this file use", instead of
    assuming WGBS's.
    """
    print(f"\n{'=' * 70}\n{label}: empirical +/- strand offset detection\n{'=' * 70}")

    plus_strand = dataframe[dataframe["strand"].eq("+")]
    minus_strand = dataframe[dataframe["strand"].eq("-")]
    print(f"'+' rows: {len(plus_strand):,}  '-' rows: {len(minus_strand):,}")

    minus_positions_by_chrom = {
        chrom: set(group["chrom_start"]) for chrom, group in minus_strand.groupby("chrom")
    }

    sample_size = min(OFFSET_SAMPLE_SIZE, len(plus_strand))
    sample = plus_strand.sample(n=sample_size, random_state=42)

    match_counts = {offset: 0 for offset in CANDIDATE_OFFSETS}

    for chrom, chrom_start in zip(sample["chrom"], sample["chrom_start"]):
        minus_positions = minus_positions_by_chrom.get(chrom)
        if not minus_positions:
            continue
        for offset in CANDIDATE_OFFSETS:
            if (chrom_start + offset) in minus_positions:
                match_counts[offset] += 1

    print(f"Sampled {sample_size:,} '+' rows, checked against '-' rows at chrom_start + offset:")
    for offset, count in sorted(match_counts.items(), key=lambda item: -item[1]):
        print(f"  offset {offset:+d}: {count:,} matches ({count / sample_size * 100:.2f}%)")

    best_offset = max(match_counts, key=match_counts.get)
    best_fraction = match_counts[best_offset] / sample_size
    print(f"\nBest-matching offset: {best_offset:+d} ({best_fraction * 100:.2f}% of sampled '+' rows paired)")

    if best_fraction < 0.5:
        print(
            "WARNING: even the best offset pairs less than half of sampled rows - "
            "this file may not have systematic +/- strand pairing at all (RRBS "
            "libraries can report only one strand per CpG in some cases). Inspect "
            "the raw rows above before assuming any offset."
        )

    return best_offset


def main() -> None:
    print("=" * 70)
    print("GM12878 RRBS raw data analysis - run BEFORE preprocess_gm12878.py")
    print("=" * 70)

    best_offsets = {}

    for label, path in zip(REPLICATE_LABELS, GM12878_RRBS_REPLICATE_PATHS):
        dataframe = load_bedmethyl(path)
        profile_raw_columns(dataframe, label)
        best_offsets[label] = determine_strand_offset(dataframe, label)

    print(f"\n\n{'=' * 70}\nSummary\n{'=' * 70}")
    for label, offset in best_offsets.items():
        print(f"  {label}: best +/- offset = {offset:+d}")

    if len(set(best_offsets.values())) > 1:
        print("\nWARNING: replicates disagree on the best offset - inspect both before proceeding.")


if __name__ == "__main__":
    main()
