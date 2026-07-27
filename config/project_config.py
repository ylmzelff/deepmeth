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

# Locally (Windows, Drive desktop sync) the deepmeth repo lives inside
# 1001_BioSeq_LLM/, so PROJECT_ROOT.parent is the data root. In Colab the
# repo is instead git-cloned to /content/deepmeth - a separate location from
# where Drive gets mounted (/content/drive/MyDrive/1001_BioSeq_LLM) - so
# PROJECT_ROOT.parent is wrong there (resolves to /content). Prefer the
# Colab Drive mount path when it exists; otherwise fall back to the local
# sibling-of-repo layout. The large pipeline data (splits, extracted
# features, graph) lives flat under this data/ root (no intermediate
# "features" subfolder) - not inside the repo.
_COLAB_DRIVE_DATA_DIR = Path("/content/drive/MyDrive/1001_BioSeq_LLM/data")
DATA_DIR = _COLAB_DRIVE_DATA_DIR if _COLAB_DRIVE_DATA_DIR.exists() else PROJECT_ROOT.parent / "data"
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"
DATASET_DIR = DATA_DIR / "proceed"  # train.parquet / validation.parquet / test.parquet

PHYSICOCHEMICAL_FEATURES_DIR = DATA_DIR / "physicochemical"
SEQUENCE_CODES_DIR = DATA_DIR / "sequence_codes"
DNABERT_NODE_FEATURES_DIR = DATA_DIR / "dnabert2_node_features"

GRAPH_DIR = DATA_DIR / "graph"
HIC_RAW_DIR = DATA_DIR / "hic" / "raw"

RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Small static reference table (not a generated pipeline artifact) - stays
# inside the repo itself rather than the Drive-mounted data/ folder.
PHYSICOCHEMICAL_PROPERTY_FILE = PROJECT_ROOT / "data" / "Physicochemical_properties_Di.xlsx"

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

# One-hot base order for the sequence branch; index i is channel i of the
# [4, 501] one-hot input. "N" (and anything else) gets UNKNOWN_BASE_CODE,
# expanded to an all-zero one-hot column at load time.
BASE_ORDER = "ACGT"
UNKNOWN_BASE_CODE = 4

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

# Rows per shard file. At this scale (tens of millions of CpGs) a small
# shard size would create tens of thousands of tiny files; 50k keeps shard
# count reasonable (~430 shards for a 21M-row split) at ~500MB/shard.
PHYSICOCHEMICAL_SHARD_SIZE = 50_000

# ============================================================
# Hi-C / graph branch
# ============================================================

GRAPH_RESOLUTION = 100_000  # 100 kb bins, unchanged from the previous project
GRAPH_INTRA_CHROMOSOME_ONLY = True  # inter-chromosomal contacts set to 0 (tractability)
NODE_FEATURE_DIM = 768  # DNABERT-2 hidden size

# ============================================================
# DNABERT-2 (model/tokenizer settings unchanged). Embedding every one of
# the ~26.5M CpGs individually was projected to take ~15+ hours and isn't
# actually needed: the graph branch only ever consumes one pooled feature
# per 100kb node, not a per-CpG embedding. So we embed only a deterministic
# sample of up to DNABERT_MAX_CPG_PER_NODE CpGs per node (drawn from
# train+validation+test combined - see [[dnabert_node_feature_sampling]])
# and mean-pool those into the node feature.
# ============================================================

DNABERT_MODEL_NAME = "zhihan1996/DNABERT-2-117M"
DNABERT_MODEL_REVISION = "ec1f874253852eb3907081f57294991b4280ceb6"
DNABERT_HIDDEN_SIZE = 768
DNABERT_BATCH_SIZE = 256
DNABERT_SAVE_DTYPE = "float16"
DNABERT_TOKENIZER_MAX_LENGTH = 512
DNABERT_MAX_CPG_PER_NODE = 64

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
# Fusion head (gated, replaces plain concatenation + linear).
# Each branch is projected to FUSION_PROJECTED_DIM, a per-feature gate
# (softmax across the 3 modalities) is computed from the raw
# concatenated branch outputs, and the gated projections are summed
# before a small MLP head - see model/fusion.py.
# ============================================================

FUSION_PROJECTED_DIM = 256
FUSION_HIDDEN_DIM = 128
FUSION_DROPOUT = 0.3

# ============================================================
# Training hyperparameters (unchanged defaults from the previous project)
# ============================================================

EPOCHS = 50
BATCH_SIZE = 4096
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6
# ncVarPred's own training code (train_model_gcn.py) applies L1 on top of
# Adam's L2/weight_decay, targeted specifically at the fusion (FC) and GCN
# layers - not the conv/BiLSTM/physicochemical branches. 1e-5 (the first
# guess) turned out too aggressive in a synthetic test (~40% of
# prediction_loss); 1e-6 is the recalibrated, more conservative value being
# tried now.
L1_LAMBDA = 1e-6
PHYSCHEM_DROPOUT = 0.5
DECISION_THRESHOLD = 0.5
EARLY_STOPPING_PATIENCE = 7
EARLY_STOPPING_MIN_DELTA = 1e-5
# 0: single-process data loading, no DataLoader worker subprocesses at all.
# On TRUBA, worker processes restarting at the start of every epoch would
# hang indefinitely (SLURM/filesystem-specific - same code ran fine on
# Colab, so this isn't a bug in our multiprocessing usage, just something
# about that cluster's process/filesystem handling we couldn't pin down
# from here). num_workers=0 sidesteps the whole failure category since
# there's no worker process to restart. Costs some throughput (no loading
# parallelism) but that's far better than a job silently hanging for hours.
NUM_WORKERS = 0
TRAINING_SEED = 42

# "auto" computes pos_weight = n_negative / n_positive from the TRAIN split only;
# pass a float instead to override.
POS_WEIGHT_MODE = "auto"

# Shards are read in shuffled order and fed into a shuffle buffer (npz shards
# are compressed, so they aren't randomly-indexable - this is the standard
# shard-streaming shuffle pattern). Each DataLoader worker keeps its own
# buffer, so peak RAM is roughly NUM_WORKERS * SHUFFLE_BUFFER_SIZE * bytes
# per sample (~32KB with both sequence one-hot and physicochemical
# decoded) - at the previous 4 workers * 100,000 that was ~13GB, enough to
# OOM-kill a DataLoader worker on a standard Colab instance. ~1 shard worth
# of rows at the current NUM_WORKERS=2.
SHUFFLE_BUFFER_SIZE = 50_000

# How often (wall-clock seconds, not batch count - stays meaningful if
# BATCH_SIZE changes) to print in-epoch progress during training/validation.
LOG_INTERVAL_SECONDS = 30.0

DEVICE = "cuda:0"
