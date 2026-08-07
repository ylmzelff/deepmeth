"""
Download the raw ENCODE HepG2 WGBS methylation replicates, the HepG2 Hi-C
contact matrix, and the GRCh38 reference genome that the rest of the
pipeline consumes.

Every accession/path lives in config.data_config.hepg2_config.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from config.data_config.hepg2_config import (
    GENOME_ASSEMBLY,
    GRCH38_2BIT_PATH,
    GRCH38_2BIT_URL,
    HIC_FILE_ACCESSION,
    HIC_RAW_FILE_PATH,
    WGBS_REPLICATE_ACCESSIONS,
    WGBS_REPLICATE_PATHS,
)

ENCODE_API_ROOT = "https://www.encodeproject.org"
CHUNK_SIZE = 1024 * 1024  # 1 MB
PROGRESS_INTERVAL_BYTES = CHUNK_SIZE * 200  # print every ~200 MB


EXPECTED_METHYLATION_OUTPUT_TYPE = "methylation state at CpG"
EXPECTED_HIC_OUTPUT_TYPE = "mapping quality thresholded contact matrix"


def fetch_encode_file_metadata(accession: str) -> dict:
    """Fetch a file's ENCODE portal metadata as JSON."""
    url = f"{ENCODE_API_ROOT}/files/{accession}/?format=json"

    response = requests.get(
        url,
        headers={"Accept": "application/json"},
        timeout=60,
    )
    response.raise_for_status()

    return response.json()


def stream_download(
    url: str,
    destination: Path,
    expected_size: int | None,
    expected_md5: str | None,
) -> None:
    """Stream a URL to disk, skipping the download if a correctly-sized file already exists."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        if expected_size is None or destination.stat().st_size == expected_size:
            print(f"Already present, skipping: {destination}")
            return

        print(f"Existing file has the wrong size, re-downloading: {destination}")

    print(f"Downloading: {url}")
    print(f"  -> {destination}")

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    md5 = hashlib.md5()
    downloaded_bytes = 0
    next_progress_mark = PROGRESS_INTERVAL_BYTES

    with destination.open("wb") as file:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if not chunk:
                continue

            file.write(chunk)
            md5.update(chunk)
            downloaded_bytes += len(chunk)

            if downloaded_bytes >= next_progress_mark:
                print(f"  {downloaded_bytes / 1024 ** 3:.2f} GB downloaded...")
                next_progress_mark += PROGRESS_INTERVAL_BYTES

    if expected_size is not None and downloaded_bytes != expected_size:
        raise RuntimeError(
            f"{destination}: downloaded {downloaded_bytes} bytes, "
            f"expected {expected_size} bytes."
        )

    if expected_md5 is not None and md5.hexdigest() != expected_md5:
        raise RuntimeError(
            f"{destination}: md5 mismatch. Expected {expected_md5}, "
            f"got {md5.hexdigest()}."
        )

    print(f"  Done: {downloaded_bytes / 1024 ** 3:.2f} GB")


def download_encode_file(
    accession: str,
    destination: Path,
    expected_output_type: str,
    expected_assembly: str,
) -> None:
    """Download one ENCODE file after validating its assembly, output type and status."""
    metadata = fetch_encode_file_metadata(accession)

    assembly = metadata.get("assembly")

    if assembly != expected_assembly:
        raise RuntimeError(
            f"{accession}: expected assembly {expected_assembly}, found {assembly!r}. "
            "Refusing to download a mismatched genome build."
        )

    output_type = metadata.get("output_type")

    if output_type != expected_output_type:
        raise RuntimeError(
            f"{accession}: expected output_type {expected_output_type!r}, "
            f"found {output_type!r}."
        )

    status = metadata.get("status")

    if status != "released":
        raise RuntimeError(
            f"{accession}: file status is {status!r}, not 'released'."
        )

    href = metadata["href"]
    url = f"{ENCODE_API_ROOT}{href}"

    stream_download(
        url=url,
        destination=destination,
        expected_size=metadata.get("file_size"),
        expected_md5=metadata.get("md5sum"),
    )


def download_reference_genome() -> None:
    stream_download(
        url=GRCH38_2BIT_URL,
        destination=GRCH38_2BIT_PATH,
        expected_size=None,
        expected_md5=None,
    )


def print_downloaded_sizes(paths: Iterable[Path]) -> None:
    for path in paths:
        size_gb = path.stat().st_size / 1024 ** 3
        print(f"  {path}  ({size_gb:.2f} GB)")


def main() -> None:
    print("=" * 70)
    print("DeepMeth data download (HepG2, GRCh38)")
    print("=" * 70)

    print(f"\n[1/3] HepG2 WGBS methylation replicates {WGBS_REPLICATE_ACCESSIONS}")

    for accession, destination in zip(WGBS_REPLICATE_ACCESSIONS, WGBS_REPLICATE_PATHS):
        download_encode_file(
            accession=accession,
            destination=destination,
            expected_output_type=EXPECTED_METHYLATION_OUTPUT_TYPE,
            expected_assembly=GENOME_ASSEMBLY,
        )

    print(f"\n[2/3] HepG2 Hi-C contact matrix ({HIC_FILE_ACCESSION})")

    download_encode_file(
        accession=HIC_FILE_ACCESSION,
        destination=HIC_RAW_FILE_PATH,
        expected_output_type=EXPECTED_HIC_OUTPUT_TYPE,
        expected_assembly=GENOME_ASSEMBLY,
    )

    print("\n[3/3] GRCh38 reference genome (UCSC hg38.2bit)")

    download_reference_genome()

    print("\nAll downloads completed.")

    print_downloaded_sizes((*WGBS_REPLICATE_PATHS, HIC_RAW_FILE_PATH, GRCH38_2BIT_PATH))


if __name__ == "__main__":
    main()
