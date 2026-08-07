from __future__ import annotations

from config.project_config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATA_DIR,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    GRAPH_DIR,
    HIC_RAW_DIR,
    L1_LAMBDA,
    LEARNING_RATE,
    PHYSICOCHEMICAL_FEATURES_DIR,
    RAW_DIR,
    REFERENCE_DIR,
    RESULTS_DIR,
    SEQUENCE_CODES_DIR,
    WEIGHT_DECAY,
)

# ============================================================
# ENCODE data sources — HepG2, GRCh38 (verified live via the ENCODE API)
# ============================================================

GENOME_ASSEMBLY = "GRCh38"
CELL_LINE = "HepG2"

# WGBS methylation, experiment ENCSR881XOU: bed/bedMethyl, "methylation state at CpG"
WGBS_EXPERIMENT_ACCESSION = "ENCSR881XOU"
WGBS_REPLICATE_1_ACCESSION = "ENCFF847OWL"
WGBS_REPLICATE_2_ACCESSION = "ENCFF390OZB"
WGBS_REPLICATE_ACCESSIONS = (WGBS_REPLICATE_1_ACCESSION, WGBS_REPLICATE_2_ACCESSION)

# Hi-C, experiment ENCSR194SRI: .hic, "mapping quality thresholded contact matrix" (reps 1+2 combined)
HIC_EXPERIMENT_ACCESSION = "ENCSR194SRI"
HIC_FILE_ACCESSION = "ENCFF306VTV"

ENCODE_FILE_DOWNLOAD_URL_TEMPLATE = (
    "https://www.encodeproject.org/files/{accession}/@@download/{accession}.{extension}"
)

# GRCh38 reference genome, 2bit. UCSC's "hg38" build is GRCh38 (chr-prefixed contig names).
GRCH38_2BIT_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/latest/hg38.2bit"
GRCH38_2BIT_PATH = REFERENCE_DIR / "hg38.2bit"

# Local raw-download destinations (shared between download_data.py and later stages)
WGBS_RAW_DIR = RAW_DIR / "wgbs"
WGBS_REPLICATE_PATHS = tuple(
    WGBS_RAW_DIR / f"{accession}.bed.gz" for accession in WGBS_REPLICATE_ACCESSIONS
)
HIC_RAW_FILE_PATH = HIC_RAW_DIR / f"{HIC_FILE_ACCESSION}.hic"



MIN_COVERAGE_PER_REPLICATE = 10  # minimum reads at a CpG, in EACH replicate, to keep it
MIN_TOTAL_COVERAGE = 20          # minimum combined (both-replicate) coverage after merge

MAX_REPLICATE_RATIO_DIFFERENCE = 0.20

# ============================================================
# Active-dataset resolution for DATASET_CONFIG_PATH == this file
# ============================================================

ACTIVE_DATA_DIR = DATA_DIR
ACTIVE_SEQUENCE_CODES_DIR = SEQUENCE_CODES_DIR
ACTIVE_PHYSICOCHEMICAL_DIR = PHYSICOCHEMICAL_FEATURES_DIR
ACTIVE_GRAPH_DIR = GRAPH_DIR

ACTIVE_SPLIT_NODE_INDEX_DIR = GRAPH_DIR
ACTIVE_CHECKPOINT_DIR = CHECKPOINT_DIR
ACTIVE_RESULTS_DIR = RESULTS_DIR

ACTIVE_BATCH_SIZE = BATCH_SIZE
ACTIVE_LEARNING_RATE = LEARNING_RATE
ACTIVE_WEIGHT_DECAY = WEIGHT_DECAY
ACTIVE_L1_LAMBDA = L1_LAMBDA
ACTIVE_EPOCHS = EPOCHS
ACTIVE_EARLY_STOPPING_PATIENCE = EARLY_STOPPING_PATIENCE

ACTIVE_WARM_START = False
ACTIVE_FROZEN_LEARNING_RATE = LEARNING_RATE
ACTIVE_UNFREEZE_LEARNING_RATE = LEARNING_RATE
ACTIVE_WARMUP_FROZEN_EPOCHS = 0
ACTIVE_SEQUENCE_ONLY_CHECKPOINT_PATH = None
ACTIVE_GRAPH_ONLY_CHECKPOINT_PATH = None
ACTIVE_PHYSICOCHEMICAL_ONLY_CHECKPOINT_PATH = None

ACTIVE_HISTORY_FILENAME = "training_history.json"
