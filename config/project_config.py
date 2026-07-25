"""Central configuration for the DeepMeth pipeline (HepG2, GRCh38).

Every stage script imports its paths and hyperparameters from here instead of
defining its own defaults, so a Colab cell only needs to run:

    !python preprocessing/preprocess.py

with no arguments.
"""

from __future__ import annotations

from pathlib import Path

# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"
PROCESSED_DIR = DATA_DIR / "processed"
DATASET_DIR = PROCESSED_DIR / "dataset"  # train.parquet / validation.parquet / test.parquet

FEATURES_DIR = DATA_DIR / "features"
PHYSICOCHEMICAL_FEATURES_DIR = FEATURES_DIR / "physicochemical"
DNABERT_FEATURES_DIR = FEATURES_DIR / "dnabert2"

GRAPH_DIR = DATA_DIR / "graph"
HIC_RAW_DIR = DATA_DIR / "hic" / "raw"

RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

PHYSICOCHEMICAL_PROPERTY_FILE = DATA_DIR / "Physicochemical_properties_Di.xlsx"

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

# ============================================================
# Sequence extraction (unchanged from the previous project)
# ============================================================

SEQUENCE_LENGTH = 501
CENTER_INDEX = SEQUENCE_LENGTH // 2  # 250
MAX_UNKNOWN_FRACTION = 0.10

# ============================================================
# WGBS replicate preprocessing — NEW for the HepG2 pivot, open to discussion
# ============================================================

MIN_COVERAGE_PER_REPLICATE = 5      # minimum reads at a CpG, in EACH replicate, to keep it
MIN_REPLICATE_CORRELATION = 0.8     # minimum Spearman correlation between reps on shared, covered CpGs

# ============================================================
# Train / validation / test split (chromosome-disjoint, same strategy as before)
# ============================================================

TRAIN_FRACTION = 0.80
VALIDATION_FRACTION = 0.10
TEST_FRACTION = 0.10
SPLIT_RANDOM_SEED = 42

AUTOSOME_CHROMOSOMES = [f"chr{i}" for i in range(1, 23)]
INCLUDED_CHROMOSOMES = AUTOSOME_CHROMOSOMES + ["chrX"]  # chrY / chrM excluded

# ============================================================
# Physicochemical branch (unchanged)
# ============================================================

PHYSICOCHEMICAL_MATRIX_SHAPE = (12, SEQUENCE_LENGTH - 1)  # [12, 500]
PHYSICOCHEMICAL_CNN_OUTPUT_DIM = 480

# ============================================================
# Hi-C / graph branch
# ============================================================

GRAPH_RESOLUTION = 100_000  # 100 kb bins, unchanged from the previous project
GRAPH_INTRA_CHROMOSOME_ONLY = True  # inter-chromosomal contacts set to 0 (tractability)
NODE_FEATURE_DIM = 768  # DNABERT-2 hidden size

# ============================================================
# DNABERT-2 (unchanged)
# ============================================================

DNABERT_MODEL_NAME = "zhihan1996/DNABERT-2-117M"
DNABERT_MODEL_REVISION = "ec1f874253852eb3907081f57294991b4280ceb6"
DNABERT_HIDDEN_SIZE = 768
DNABERT_BATCH_SIZE = 64
DNABERT_SHARD_SIZE = 4096
DNABERT_SAVE_DTYPE = "float16"
DNABERT_TOKENIZER_MAX_LENGTH = 512

# ============================================================
# Model architecture dimensions (fixed — ported as-is, do not change)
# ============================================================

SEQUENCE_BRANCH_OUTPUT_DIM = 925
GRAPH_BRANCH_OUTPUT_DIM = 128
FUSION_INPUT_DIM = (
    SEQUENCE_BRANCH_OUTPUT_DIM
    + GRAPH_BRANCH_OUTPUT_DIM
    + PHYSICOCHEMICAL_CNN_OUTPUT_DIM
)  # 1533

# ============================================================
# Training hyperparameters (unchanged defaults from the previous project)
# ============================================================

EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6
L1_LAMBDA = 0.0
PHYSCHEM_DROPOUT = 0.5
DECISION_THRESHOLD = 0.5
EARLY_STOPPING_PATIENCE = 7
EARLY_STOPPING_MIN_DELTA = 1e-5
NUM_WORKERS = 4
TRAINING_SEED = 42

# "auto" computes pos_weight = n_negative / n_positive from the TRAIN split only;
# pass a float instead to override.
POS_WEIGHT_MODE = "auto"

DEVICE = "cuda:0"
