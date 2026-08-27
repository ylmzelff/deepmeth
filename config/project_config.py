from __future__ import annotations

from importlib import import_module
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


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

PHYSICOCHEMICAL_PROPERTY_FILE = REFERENCE_DIR / "Physicochemical_properties_Di.xlsx"

SEQUENCE_LENGTH = 501
CENTER_INDEX = SEQUENCE_LENGTH // 2  # 250
MAX_UNKNOWN_FRACTION = 0.10
BASE_ORDER = "ACGT"
UNKNOWN_BASE_CODE = 4

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
PHYSICOCHEMICAL_SHARD_SIZE = 50_000

# ============================================================
# Hi-C / graph branch
# ============================================================

GRAPH_RESOLUTION = 10_000
GRAPH_INTRA_CHROMOSOME_ONLY = True  # inter-chromosomal contacts set to 0 (tractability)
HIC_MAX_CONTACT_DISTANCE_BP = 5_000_000  # 5 Mb keeps the biologically relevant local neighborhood.
HIC_TOP_K_NEIGHBORS = 32
HIC_NORMALIZATION_TYPE = "KR"

# ============================================================
# DNABERT-2 (model/tokenizer settings unchanged)
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
FOUNDATION_BRANCH_OUTPUT_DIM = 256
FUSION_INPUT_DIM = (
    SEQUENCE_BRANCH_OUTPUT_DIM
    + GRAPH_BRANCH_OUTPUT_DIM
    + PHYSICOCHEMICAL_CNN_OUTPUT_DIM
)

FUSION_PROJECTED_DIM = 256
FUSION_HIDDEN_DIM = 128
FUSION_DROPOUT = 0.3

# ============================================================
# Training hyperparameters (unchanged defaults from the previous project)
# ============================================================

EPOCHS = 100
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
L1_LAMBDA = 1e-6
GRAD_CLIP_MAX_NORM = 1.0
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 1
LR_SCHEDULER_MIN_LR = 1e-6
PHYSCHEM_DROPOUT = 0.5
DECISION_THRESHOLD = 0.5
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 1e-5
NUM_WORKERS = 0
TRAINING_SEED = 42
POS_WEIGHT_MODE = "auto"
SHUFFLE_BUFFER_SIZE = 50_000
LOG_INTERVAL_SECONDS = 30.0
DEVICE = "cuda:0"

# ============================================================
# Active dataset 
# ============================================================

DATASET_CONFIG_PATH = PROJECT_ROOT / "config" / "data_config" / "gm12878_config.py"

_ACTIVE_NAMES = frozenset({
    "ACTIVE_DATA_DIR",
    "ACTIVE_SEQUENCE_CODES_DIR",
    "ACTIVE_PHYSICOCHEMICAL_DIR",
    "ACTIVE_GRAPH_DIR",
    "ACTIVE_SPLIT_NODE_INDEX_DIR",
    "ACTIVE_CHECKPOINT_DIR",
    "ACTIVE_RESULTS_DIR",
    "ACTIVE_BATCH_SIZE",
    "ACTIVE_LEARNING_RATE",
    "ACTIVE_WEIGHT_DECAY",
    "ACTIVE_L1_LAMBDA",
    "ACTIVE_EPOCHS",
    "ACTIVE_EARLY_STOPPING_PATIENCE",
    "ACTIVE_WARM_START",
    "ACTIVE_FROZEN_LEARNING_RATE",
    "ACTIVE_UNFREEZE_LEARNING_RATE",
    "ACTIVE_WARMUP_FROZEN_EPOCHS",
    "ACTIVE_SEQUENCE_ONLY_CHECKPOINT_PATH",
    "ACTIVE_GRAPH_ONLY_CHECKPOINT_PATH",
    "ACTIVE_PHYSICOCHEMICAL_ONLY_CHECKPOINT_PATH",
    "ACTIVE_HISTORY_FILENAME",
})

_active_module = None


def _resolve_active_module():
    global _active_module
    if _active_module is None:
        _active_module = import_module("config.data_config." + DATASET_CONFIG_PATH.stem)
    return _active_module


def __getattr__(name):
    if name in _ACTIVE_NAMES:
        return getattr(_resolve_active_module(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
