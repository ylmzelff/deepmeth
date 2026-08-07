"""
Download raw GM12878 RRBS methylation replicates, the GM12878 Hi-C contact
matrix, and the hg19 reference genome - a separate, architecture-validation
dataset alongside (not replacing) the main HepG2/GRCh38 pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_ON_COLAB = Path("/content").exists()

if _ON_COLAB and not Path("/content/drive/MyDrive").exists():
    from google.colab import drive  # type: ignore[import-not-found]

    print("Mounting Google Drive (not yet mounted)...")
    drive.mount("/content/drive")

from config.data_config.gm12878_config import (
    GM12878_ASSEMBLY,
    GM12878_DATA_DIR,
    GM12878_HIC_FILE_ACCESSION,
    GM12878_HIC_RAW_FILE_PATH,
    GM12878_RRBS_REPLICATE_ACCESSIONS,
    GM12878_RRBS_REPLICATE_PATHS,
    HG19_2BIT_PATH,
    HG19_2BIT_URL,
)
from config.project_config import DATA_DIR
from preprocessing.download_data_hepg2 import (
    EXPECTED_HIC_OUTPUT_TYPE,
    EXPECTED_METHYLATION_OUTPUT_TYPE,
    download_encode_file,
    print_downloaded_sizes,
    stream_download,
)

_EXPECTED_DRIVE_PARENT = Path("/content/drive/MyDrive/1001_BioSeq_LLM")

if _ON_COLAB and DATA_DIR.parent != _EXPECTED_DRIVE_PARENT:
    print(
        f"WARNING: on Colab but DATA_DIR resolved to {DATA_DIR} (not under "
        f"{_EXPECTED_DRIVE_PARENT}) - GM12878 downloads would land on ephemeral "
        "local disk, not Drive, and will be lost on disconnect. Check that "
        "Google Drive actually mounted successfully above before continuing."
    )

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
            expected_output_type=EXPECTED_METHYLATION_OUTPUT_TYPE,
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

    print_downloaded_sizes((*GM12878_RRBS_REPLICATE_PATHS, GM12878_HIC_RAW_FILE_PATH, HG19_2BIT_PATH))


if __name__ == "__main__":
    main()
