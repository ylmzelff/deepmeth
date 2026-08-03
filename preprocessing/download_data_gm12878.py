"""
Download raw GM12878 RRBS methylation replicates, the GM12878 Hi-C contact
matrix, and the hg19 reference genome - a separate, architecture-validation
dataset alongside (not replacing) the main HepG2/GRCh38 pipeline.

Why this exists: the sequence-only diagnostic showed a real val_mcc
ceiling on HepG2 data, and the full 3-branch model has never been
validated against an easier, literature-precedented setting - so we can't
tell whether HepG2's low MCC reflects the task being genuinely hard or a
problem with our own architecture (GatedFusion, graph branch design, ...)
that HepG2's difficulty happens to mask. GM12878 RRBS+Hi-C (the same
biological data pairing DeepMethyl - Fang et al., Sci Rep 2016 - used to
reach MCC 0.886) is a way to test that: same three-branch architecture,
same feature-extraction approach (sequence/physicochemical from RRBS
CpGs, graph from Hi-C, DNABERT-2 node features), different, easier data.
If the architecture reaches a reasonable MCC here, that's evidence HepG2's
difficulty is about the data/task, not the model; if it doesn't, that's a
real architecture problem worth fixing here (fast iteration) before
returning to HepG2.

Why hg19 and not GRCh38: GM12878's RRBS processed files on ENCODE are
hg19-only (no GRCh38 version exists yet). Rather than lifting the RRBS
coordinates over to GRCh38 (extra complexity/coordinate-conversion risk),
everything for this dataset - RRBS, Hi-C, and the reference genome used
for sequence extraction/DNABERT-2 - stays in hg19, avoiding any
genome-build mismatch. This only affects this GM12878 sub-pipeline; the
main HepG2/GRCh38 pipeline (config/project_config.py's GENOME_ASSEMBLY
etc.) is untouched.

Files (verified against the ENCODE portal's JSON API before being hardcoded
here - see project history for how each was checked):
    RRBS replicate 1: ENCFF001TLQ (bed bedMethyl, "methylation state at CpG", hg19)
    RRBS replicate 2: ENCFF001TLR (bed bedMethyl, "methylation state at CpG", hg19)
    Hi-C:             ENCFF355OWW (hic, "mapping quality thresholded contact matrix", hg19)
    Reference genome: UCSC hg19.2bit

Reuses fetch_encode_file_metadata/stream_download/download_encode_file
from preprocessing/download_data.py (generic ENCODE-download logic, not
duplicated) - only download_encode_file's expected_assembly is overridden
to "hg19" here.

Usage (no arguments needed):

    python preprocessing/download_data_gm12878.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Must run BEFORE `from config.project_config import DATA_DIR` below:
# DATA_DIR resolves to the Drive path only if
# /content/drive/MyDrive/1001_BioSeq_LLM/data already exists at *import
# time* - if Drive isn't mounted yet when that import runs, it silently
# falls back to ephemeral local Colab disk (the same failure mode that
# previously lost HepG2 checkpoints across a Colab disconnect - see
# project history). Mounting here first, before config is ever imported,
# means GM12878 data can't land in the wrong place the way that did.
_ON_COLAB = Path("/content").exists()

if _ON_COLAB and not Path("/content/drive/MyDrive").exists():
    from google.colab import drive  # type: ignore[import-not-found]

    print("Mounting Google Drive (not yet mounted)...")
    drive.mount("/content/drive")

from config.project_config import DATA_DIR
from preprocessing.download_data import (
    download_encode_file,
    stream_download,
)

# Separate data root, sibling to the main data/ tree - never overlaps with
# HepG2/GRCh38 paths in config/project_config.py.
GM12878_DATA_DIR = DATA_DIR.parent / "data_gm12878"

_EXPECTED_DRIVE_PARENT = Path("/content/drive/MyDrive/1001_BioSeq_LLM")

if _ON_COLAB and DATA_DIR.parent != _EXPECTED_DRIVE_PARENT:
    print(
        f"WARNING: on Colab but DATA_DIR resolved to {DATA_DIR} (not under "
        f"{_EXPECTED_DRIVE_PARENT}) - GM12878 downloads would land on ephemeral "
        "local disk, not Drive, and will be lost on disconnect. Check that "
        "Google Drive actually mounted successfully above before continuing."
    )
GM12878_RAW_DIR = GM12878_DATA_DIR / "raw"
GM12878_REFERENCE_DIR = GM12878_DATA_DIR / "reference"

GM12878_ASSEMBLY = "hg19"

GM12878_RRBS_REPLICATE_1_ACCESSION = "ENCFF001TLQ"
GM12878_RRBS_REPLICATE_2_ACCESSION = "ENCFF001TLR"
GM12878_RRBS_REPLICATE_ACCESSIONS = (
    GM12878_RRBS_REPLICATE_1_ACCESSION,
    GM12878_RRBS_REPLICATE_2_ACCESSION,
)
GM12878_RRBS_RAW_DIR = GM12878_RAW_DIR / "rrbs"
GM12878_RRBS_REPLICATE_PATHS = tuple(
    GM12878_RRBS_RAW_DIR / f"{accession}.bed.gz" for accession in GM12878_RRBS_REPLICATE_ACCESSIONS
)

GM12878_HIC_FILE_ACCESSION = "ENCFF355OWW"
GM12878_HIC_RAW_DIR = GM12878_RAW_DIR / "hic"
GM12878_HIC_RAW_FILE_PATH = GM12878_HIC_RAW_DIR / f"{GM12878_HIC_FILE_ACCESSION}.hic"

HG19_2BIT_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/latest/hg19.2bit"
HG19_2BIT_PATH = GM12878_REFERENCE_DIR / "hg19.2bit"

EXPECTED_RRBS_OUTPUT_TYPE = "methylation state at CpG"
EXPECTED_HIC_OUTPUT_TYPE = "mapping quality thresholded contact matrix"


def download_reference_genome() -> None:
    stream_download(
        url=HG19_2BIT_URL,
        destination=HG19_2BIT_PATH,
        expected_size=None,
        expected_md5=None,
    )


def main() -> None:
    print("=" * 70)
    print("DeepMeth GM12878 architecture-validation data download (hg19)")
    print("=" * 70)
    print(f"Output root: {GM12878_DATA_DIR}")

    print(f"\n[1/3] GM12878 RRBS methylation replicates {GM12878_RRBS_REPLICATE_ACCESSIONS}")

    for accession, destination in zip(GM12878_RRBS_REPLICATE_ACCESSIONS, GM12878_RRBS_REPLICATE_PATHS):
        download_encode_file(
            accession=accession,
            destination=destination,
            expected_output_type=EXPECTED_RRBS_OUTPUT_TYPE,
            expected_assembly=GM12878_ASSEMBLY,
        )

    print(f"\n[2/3] GM12878 Hi-C contact matrix ({GM12878_HIC_FILE_ACCESSION})")

    download_encode_file(
        accession=GM12878_HIC_FILE_ACCESSION,
        destination=GM12878_HIC_RAW_FILE_PATH,
        expected_output_type=EXPECTED_HIC_OUTPUT_TYPE,
        expected_assembly=GM12878_ASSEMBLY,
    )

    print("\n[3/3] hg19 reference genome (UCSC hg19.2bit)")

    download_reference_genome()

    print("\nAll GM12878 downloads completed.")

    for path in (*GM12878_RRBS_REPLICATE_PATHS, GM12878_HIC_RAW_FILE_PATH, HG19_2BIT_PATH):
        size_gb = path.stat().st_size / 1024 ** 3
        print(f"  {path}  ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
