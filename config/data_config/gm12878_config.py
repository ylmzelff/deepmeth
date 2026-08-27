from __future__ import annotations

from config.project_config import DATA_DIR

# GM12878 lives in a sibling data root (see preprocessing/download_data_gm12878.py's
# docstring for why: a separate architecture-validation dataset, hg19, never
# overlapping HepG2/GRCh38's own data/ tree).
GM12878_DATA_DIR = DATA_DIR.parent / "data_gm12878"
GM12878_RAW_DIR = GM12878_DATA_DIR / "raw"
GM12878_REFERENCE_DIR = GM12878_DATA_DIR / "reference"

# ============================================================
# ENCODE data sources — GM12878, hg19
# ============================================================

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

# ============================================================
# Active-dataset resolution for DATASET_CONFIG_PATH == this file
# ============================================================

ACTIVE_DATA_DIR = GM12878_DATA_DIR
ACTIVE_SEQUENCE_CODES_DIR = GM12878_DATA_DIR / "sequence_codes"
ACTIVE_PHYSICOCHEMICAL_DIR = GM12878_DATA_DIR / "physicochemical"
# feature_extraction/extract_dnabert2_tokens_gm12878.py's output dir - per-CpG,
# token-level (not pooled) DNABERT-2 features for model/foundation_branch.py.
ACTIVE_FOUNDATION_TOKEN_DIR = GM12878_DATA_DIR / "dnabert2_token_features"
ACTIVE_GRAPH_DIR = GM12878_DATA_DIR / "graph"
ACTIVE_SPLIT_NODE_INDEX_DIR = GM12878_DATA_DIR / "proceed" / "disjoint_split"

ACTIVE_CHECKPOINT_DIR = GM12878_DATA_DIR / "checkpoints" / "full_model_gatv2_cpg_readout_10kb"
ACTIVE_RESULTS_DIR = GM12878_DATA_DIR / "results"

ACTIVE_BATCH_SIZE = 1024
ACTIVE_LEARNING_RATE = 1e-05  # only used if ACTIVE_WARM_START is False
ACTIVE_WEIGHT_DECAY = 1e-05
ACTIVE_L1_LAMBDA = 0.0
ACTIVE_EPOCHS = 200
ACTIVE_EARLY_STOPPING_PATIENCE = 20

ACTIVE_WARM_START = True
ACTIVE_FROZEN_LEARNING_RATE = 2.5e-05
ACTIVE_UNFREEZE_LEARNING_RATE = 1e-05
ACTIVE_WARMUP_FROZEN_EPOCHS = 5
ACTIVE_SEQUENCE_ONLY_CHECKPOINT_PATH = GM12878_DATA_DIR / "checkpoints" / "sequence_only" / "best_model.pt"

ACTIVE_GRAPH_ONLY_CHECKPOINT_PATH = (
    GM12878_DATA_DIR / "checkpoints" / "graph_gat_10kb" / "structure_branch_pretrained.pt"
)
ACTIVE_PHYSICOCHEMICAL_ONLY_CHECKPOINT_PATH = (
    GM12878_DATA_DIR / "checkpoints" / "physicochemical_only" / "best_model.pt"
)

ACTIVE_HISTORY_FILENAME = "training_history_full_gatv2_cpg_readout_10kb.json"
