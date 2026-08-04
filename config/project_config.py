from __future__ import annotations

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
PHYSICOCHEMICAL_SHARD_SIZE = 50_000

# ============================================================
# Hi-C / graph branch
# ============================================================

# 25kb, not the original 100kb: feature_extraction/analyze_graph_node_label_purity.py's
# majority-vote-per-node oracle (the real theoretical ceiling for a graph
# branch that gives every CpG in a node the SAME embedding) was only
# MCC~0.50 at 100kb, versus MCC~0.69 at 25kb (validation, GM12878) - 100kb
# bins were mixing ~40 CpGs of often-conflicting labels together. See
# feature_extraction/preview_graph_resolution_gm12878.py for the cheap
# (no Hi-C reading, no DNABERT) way this was checked before committing to
# the real, expensive rebuild at this resolution.
GRAPH_RESOLUTION = 25_000
GRAPH_INTRA_CHROMOSOME_ONLY = True  # inter-chromosomal contacts set to 0 (tractability)

# Maximum genomic distance (bp) between two bins for their Hi-C contact to be
# kept as a graph edge, applied before HIC_TOP_K_NEIGHBORS below. At 100kb
# resolution with no distance cutoff, a chromosome's contact matrix was
# nearly fully dense (every bin has *some* nonzero observed count with
# almost every other bin on the same chromosome, even far away - real
# signal decays with distance, but hic-straw still returns those low/
# background-level records) - GM12878's 100kb graph came out to 30,376
# nodes but 31.6M edges (~1,040 neighbors/node average), which is both
# (a) mostly background/noise rather than real 3D structure (TADs and
# loops - the actual biologically meaningful contacts - are almost
# entirely within a few Mb; Rao et al. 2014) and (b) large enough that a
# per-edge attention GNN (GATv2Structure) tries to materialize a
# multi-hundred-GB intermediate tensor and OOMs even on an 80GB+ GPU - see
# project history. 5 Mb keeps the biologically relevant local neighborhood.
HIC_MAX_CONTACT_DISTANCE_BP = 5_000_000

# Per-node cap on Hi-C edges, applied after the distance cap above (see
# feature_extraction/prepare_hic_graph.py's apply_top_k_sparsification).
# Needed because a fixed *distance* cap doesn't give a fixed *edge count*
# cap - at 25kb resolution the same 5 Mb window is 4x wider in bin terms
# than at 100kb, and GM12878 has ~4x more 25kb nodes than 100kb nodes, so
# distance-capped edge count alone would grow roughly 16x (~43M, re-
# triggering the same GATv2Conv OOM the distance cap was originally added
# to fix). Keeping only each node's top-K strongest contacts (by KR-
# normalized count) bounds edge count by node_count x K regardless of
# resolution or local Hi-C density, instead of by how dense the region
# happens to be.
HIC_TOP_K_NEIGHBORS = 32

# Hi-C matrix balancing applied by hic-straw before we read contact counts -
# NOT the same thing as the GCN-style symmetric degree normalization applied
# afterward in prepare_hic_graph.py's normalize_adjacency (that corrects for
# per-NODE degree in the graph; this corrects for per-BIN sequencing/mappability/
# GC bias in the raw Hi-C reads themselves - both apply, for different reasons).
# Raw ("NONE") counts were used originally, which conflates true 3D contact
# frequency with distance-decay and coverage artifacts.
#
# SCALE, not KR: Knight & Ruiz (2013) / Rao et al. 2014 introduced KR
# balancing as the original Hi-C convention, but Aiden Lab's own Juicer
# tooling has since moved to SCALE as its default for genome-wide/
# inter-chromosomal normalization (GW_SCALE/INTER_SCALE superseding
# GW_KR/INTER_KR) - same family of matrix-balancing method, but converges
# more reliably than KR on sparse/large matrices. hic-straw exposes both;
# if a given .hic file only has KR vectors precomputed (older files
# sometimes lack SCALE), fall back to "KR" here.
#
# GM12878's .hic file (ENCFF355OWW, an older ENCODE Hi-C release) only has
# KR vectors precomputed at 100kb - hic-straw raised "File did not contain
# SCALE normalization vectors for one or both chromosomes at 100000 BP"
# when this was set to "SCALE" - using the documented fallback.
HIC_NORMALIZATION_TYPE = "KR"
NODE_FEATURE_DIM = 768  # DNABERT-2 hidden size

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

# Must be the same value everywhere DeepMethModel(use_sequence_self_attention=...)
# is constructed, and everywhere a sequence-branch checkpoint is loaded into
# or out of that argument - the saved state_dict's shapes depend on which
# internal architecture (BiLSTM vs self-attention) was used. Living here
# (one place) instead of being duplicated as a local constant in each
# training script is what makes that "must match" guarantee actually hold.
# BiLSTM (False) won the GM12878 ablation - see project history.
USE_SEQUENCE_SELF_ATTENTION = False

SEQUENCE_BRANCH_OUTPUT_DIM = 925
GRAPH_BRANCH_OUTPUT_DIM = 128
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
# Active training dataset — single switch, everything below resolves off
# of it. training/dataset.py and training/train.py import ONLY the
# ACTIVE_* names, never the HepG2-specific ones above directly, so
# switching datasets is exactly this one line. The bare (non-ACTIVE_)
# constants above/below keep meaning "HepG2" unconditionally, for every
# other script (preprocessing/*, feature_extraction/*) that already
# hardcodes that assumption — changing DATASET here does not affect them.
#
# Set to "HEPG2" or "GM12878".
# ============================================================

DATASET = "GM12878"

# GM12878 lives in a sibling data root (see preprocessing/download_data_gm12878.py's
# docstring for why: a separate architecture-validation dataset, hg19, never
# overlapping HepG2/GRCh38's own data/ tree). Computed directly here (not
# imported from download_data_gm12878.py) to avoid a circular import — that
# module itself imports DATA_DIR from here.
_GM12878_DATA_DIR = DATA_DIR.parent / "data_gm12878"

if DATASET == "HEPG2":
    ACTIVE_DATA_DIR = DATA_DIR
    ACTIVE_SEQUENCE_CODES_DIR = SEQUENCE_CODES_DIR
    ACTIVE_PHYSICOCHEMICAL_DIR = PHYSICOCHEMICAL_FEATURES_DIR
    ACTIVE_GRAPH_DIR = GRAPH_DIR
    # HepG2 never had a second (baseline) split, so its per-split
    # {split}_node_index.npy files sit flat in GRAPH_DIR, not under a
    # named split subfolder.
    ACTIVE_SPLIT_NODE_INDEX_DIR = GRAPH_DIR
    ACTIVE_CHECKPOINT_DIR = CHECKPOINT_DIR
    ACTIVE_RESULTS_DIR = RESULTS_DIR

    ACTIVE_BATCH_SIZE = BATCH_SIZE
    ACTIVE_LEARNING_RATE = LEARNING_RATE
    ACTIVE_WEIGHT_DECAY = WEIGHT_DECAY
    ACTIVE_L1_LAMBDA = L1_LAMBDA
    ACTIVE_EPOCHS = EPOCHS
    ACTIVE_EARLY_STOPPING_PATIENCE = EARLY_STOPPING_PATIENCE

    # No standalone-branch checkpoints exist for HepG2 in this repo, so
    # warm-start is off: train.py falls back to plain single-phase
    # from-scratch training (this dataset's original, always-worked path).
    ACTIVE_WARM_START = False
    ACTIVE_FROZEN_LEARNING_RATE = LEARNING_RATE
    ACTIVE_UNFREEZE_LEARNING_RATE = LEARNING_RATE
    ACTIVE_WARMUP_FROZEN_EPOCHS = 0
    ACTIVE_SEQUENCE_ONLY_CHECKPOINT_PATH = None
    ACTIVE_GRAPH_ONLY_CHECKPOINT_PATH = None
    ACTIVE_PHYSICOCHEMICAL_ONLY_CHECKPOINT_PATH = None

    ACTIVE_HISTORY_FILENAME = "training_history.json"

elif DATASET == "GM12878":
    ACTIVE_DATA_DIR = _GM12878_DATA_DIR
    ACTIVE_SEQUENCE_CODES_DIR = _GM12878_DATA_DIR / "sequence_codes"
    ACTIVE_PHYSICOCHEMICAL_DIR = _GM12878_DATA_DIR / "physicochemical"
    ACTIVE_GRAPH_DIR = _GM12878_DATA_DIR / "graph"
    ACTIVE_SPLIT_NODE_INDEX_DIR = _GM12878_DATA_DIR / "proceed" / "disjoint_split"
    # Under the Drive-mounted data root (not PROJECT_ROOT-relative like
    # HepG2's CHECKPOINT_DIR/RESULTS_DIR above) - see
    # download_data_gm12878.py's docstring: that PROJECT_ROOT-relative
    # pattern is what lost HepG2 checkpoints across a Colab disconnect once.
    ACTIVE_CHECKPOINT_DIR = _GM12878_DATA_DIR / "checkpoints" / "full_model_warmstart_disjoint_split"
    ACTIVE_RESULTS_DIR = _GM12878_DATA_DIR / "results"

    # GM12878-tuned values (found via the sequence-branch hyperparameter
    # sweep + reused for the full model), not HepG2's config defaults above
    # - GM12878's train split is far smaller, and the HepG2-scale defaults
    # were repeatedly found to be a poor fit (see project history).
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
    ACTIVE_SEQUENCE_ONLY_CHECKPOINT_PATH = _GM12878_DATA_DIR / "checkpoints" / "sequence_only" / "best_model.pt"
    ACTIVE_GRAPH_ONLY_CHECKPOINT_PATH = _GM12878_DATA_DIR / "checkpoints" / "graph_only" / "best_model.pt"
    ACTIVE_PHYSICOCHEMICAL_ONLY_CHECKPOINT_PATH = (
        _GM12878_DATA_DIR / "checkpoints" / "physicochemical_only" / "best_model.pt"
    )

    ACTIVE_HISTORY_FILENAME = "training_history_full_warmstart_disjoint_split.json"

else:
    raise ValueError(f"Unknown DATASET={DATASET!r} - expected 'HEPG2' or 'GM12878'.")
