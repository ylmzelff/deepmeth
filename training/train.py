"""
Train the three-branch DeepMeth model (sequence + Hi-C graph + physicochemical).

Dataset-agnostic entry point: everything (paths, hyperparameters, whether
to warm-start from standalone-branch checkpoints) comes from
config.project_config's ACTIVE_* constants, which resolve off that
module's single DATASET switch ("HEPG2" or "GM12878"). To train the other
dataset, change that one line - nothing here needs to change.

Two training modes, selected by ACTIVE_WARM_START:
  - False (HepG2's original path): plain single-phase training, the whole
    model trainable from epoch 0, at ACTIVE_LEARNING_RATE.
  - True (GM12878's path, since standalone sequence/graph/physicochemical
    checkpoints exist for it): the three branches are loaded from those
    checkpoints and frozen for ACTIVE_WARMUP_FROZEN_EPOCHS epochs (only
    the fusion head trains, at ACTIVE_FROZEN_LEARNING_RATE), then
    unfrozen and fine-tuned jointly at ACTIVE_UNFREEZE_LEARNING_RATE.
    Empirically this beat from-scratch training on GM12878 (val_mcc
    0.66 vs 0.60) - see project history.

The graph branch runs a GCN over the *entire* static Hi-C graph on every
batch (this is the original ncVarPred design: full-graph message passing,
then a row-select pulls the batch's nodes out of the result) - so
node_features.npy and adjacency_normalized.npz are loaded once and kept
resident on the device for the whole run, not re-loaded per batch.

Resumable: if last_checkpoint.pt exists (under ACTIVE_CHECKPOINT_DIR),
training resumes from there (model/optimizer/scheduler state, epoch, best
validation loss, early-stopping counter, and - in warm-start mode - which
phase it was in) instead of starting over - Colab sessions can disconnect
mid-run.

Usage (no arguments needed - set DATASET in config/project_config.py):

    python training/train.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from config.project_config import (
    ACTIVE_BATCH_SIZE,
    ACTIVE_CHECKPOINT_DIR,
    ACTIVE_EARLY_STOPPING_PATIENCE,
    ACTIVE_EPOCHS,
    ACTIVE_FROZEN_LEARNING_RATE,
    ACTIVE_GRAPH_DIR,
    ACTIVE_GRAPH_ONLY_CHECKPOINT_PATH,
    ACTIVE_HISTORY_FILENAME,
    ACTIVE_L1_LAMBDA,
    ACTIVE_LEARNING_RATE,
    ACTIVE_PHYSICOCHEMICAL_ONLY_CHECKPOINT_PATH,
    ACTIVE_RESULTS_DIR,
    ACTIVE_SEQUENCE_ONLY_CHECKPOINT_PATH,
    ACTIVE_UNFREEZE_LEARNING_RATE,
    ACTIVE_WARM_START,
    ACTIVE_WARMUP_FROZEN_EPOCHS,
    ACTIVE_WEIGHT_DECAY,
    DEVICE,
    EARLY_STOPPING_MIN_DELTA,
    FUSION_DROPOUT,
    FUSION_HIDDEN_DIM,
    FUSION_PROJECTED_DIM,
    GRAD_CLIP_MAX_NORM,
    LOG_INTERVAL_SECONDS,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_MIN_LR,
    LR_SCHEDULER_PATIENCE,
    NUM_WORKERS,
    PHYSCHEM_DROPOUT,
    POS_WEIGHT_MODE,
    TRAINING_SEED,
    USE_SEQUENCE_SELF_ATTENTION,
)
from model.deepmeth_model import DeepMethModel
from training.dataset import DeepMethShardDataset, collate_batch

NODE_FEATURES_PATH = ACTIVE_GRAPH_DIR / "node_features.npy"
ADJACENCY_PATH = ACTIVE_GRAPH_DIR / "adjacency_normalized.npz"
LAST_CHECKPOINT_PATH = ACTIVE_CHECKPOINT_DIR / "last_checkpoint.pt"
BEST_CHECKPOINT_PATH = ACTIVE_CHECKPOINT_DIR / "best_model.pt"
HISTORY_PATH = ACTIVE_RESULTS_DIR / ACTIVE_HISTORY_FILENAME


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_graph_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    if not NODE_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{NODE_FEATURES_PATH} does not exist. Run the matching prepare_graph_features script first."
        )
    if not ADJACENCY_PATH.exists():
        raise FileNotFoundError(
            f"{ADJACENCY_PATH} does not exist. Run the matching prepare_hic_graph script first."
        )

    node_features = np.load(NODE_FEATURES_PATH)
    adjacency = sp.load_npz(ADJACENCY_PATH).tocsr()

    if node_features.shape[0] != adjacency.shape[0]:
        raise RuntimeError(
            f"node_features.npy has {node_features.shape[0]:,} nodes but adjacency_normalized.npz has "
            f"shape {adjacency.shape} - out of sync, rerun the feature_extraction graph scripts."
        )

    number_of_nodes = node_features.shape[0]
    node_features_tensor = torch.from_numpy(node_features).float().to(device)

    # CSR instead of COO for the sparse adjacency: torch.spmm (used inside
    # GraphConvolution) on a CSR tensor uses a much better-optimized CUDA
    # kernel than COO for repeated matmuls against a fixed sparse structure -
    # same math (bit-for-bit equivalent), just a faster storage format.
    adjacency_tensor = torch.sparse_csr_tensor(
        torch.tensor(adjacency.indptr, dtype=torch.int64),
        torch.tensor(adjacency.indices, dtype=torch.int64),
        torch.tensor(adjacency.data, dtype=torch.float32),
        size=adjacency.shape,
    ).to(device)

    return node_features_tensor, adjacency_tensor, number_of_nodes


def resolve_pos_weight(train_dataset: DeepMethShardDataset) -> float:
    if POS_WEIGHT_MODE != "auto":
        return float(POS_WEIGHT_MODE)

    negative_count = train_dataset.negative_count
    positive_count = train_dataset.positive_count

    if positive_count == 0:
        raise RuntimeError("Train split has zero positive labels - can't compute pos_weight.")

    pos_weight = negative_count / positive_count
    print(
        f"Train split label counts: negative={negative_count:,}, positive={positive_count:,} "
        f"-> pos_weight={pos_weight:.4f}"
    )
    return pos_weight


def build_loader(split_name: str, shuffle: bool) -> tuple[DeepMethShardDataset, DataLoader]:
    dataset = DeepMethShardDataset(split_name=split_name, shuffle=shuffle, seed=TRAINING_SEED)
    loader = DataLoader(
        dataset,
        batch_size=ACTIVE_BATCH_SIZE,
        collate_fn=collate_batch,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def load_branch_weights(submodule: nn.Module, checkpoint_path: Path, prefix: str, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} does not exist. Run the matching standalone branch training script first."
        )

    # weights_only=False: safe here (self-generated, trusted checkpoint) and
    # required on PyTorch 2.6+, whose new weights_only=True default rejects
    # the numpy scalar types (from sklearn metrics) these checkpoints contain.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    full_state_dict = checkpoint["model_state_dict"]

    branch_state_dict = {
        key[len(prefix):]: value for key, value in full_state_dict.items() if key.startswith(prefix)
    }
    if not branch_state_dict:
        raise RuntimeError(f"No keys starting with {prefix!r} found in {checkpoint_path} - wrong checkpoint?")

    submodule.load_state_dict(branch_state_dict, strict=True)
    print(f"  Loaded {len(branch_state_dict)} tensors from {checkpoint_path.name} -> {prefix.rstrip('.')}")


def set_branch_trainable(model: DeepMethModel, trainable: bool) -> None:
    for module in (model.sequence_branch, model.structure_branch, model.physchem_branch):
        for parameter in module.parameters():
            parameter.requires_grad = trainable


def build_optimizer_and_scheduler(model: DeepMethModel, learning_rate: float):
    # AdamW instead of Adam: Adam's weight_decay is coupled with its
    # adaptive per-parameter learning rate (the penalty gets scaled by the
    # same factor as the gradient), which makes the decay weaker/less
    # predictable than plain L2 - AdamW (Loshchilov & Hutter, 2017) decouples
    # it, applying weight decay directly to the weights instead.
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=ACTIVE_WEIGHT_DECAY)
    # Halves LR when val_loss hasn't improved for LR_SCHEDULER_PATIENCE
    # epochs, instead of training at a fixed LR regardless of where the loss
    # curve actually is.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE, min_lr=LR_SCHEDULER_MIN_LR,
    )
    return optimizer, scheduler


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    node_features: torch.Tensor,
    adjacency: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    phase_name: str,
) -> dict:
    is_training = optimizer is not None
    model.train(is_training)

    total_rows_in_split = len(loader.dataset)

    total_loss = 0.0
    total_samples = 0
    all_probabilities = []
    all_labels = []

    epoch_start_time = time.time()
    last_log_time = epoch_start_time

    with torch.set_grad_enabled(is_training):
        for batch in loader:
            sequence_input = batch["sequence"].to(device, non_blocking=True)
            physchem_input = batch["physicochemical"].to(device, non_blocking=True)
            node_index = batch["node_index"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(
                seq_input=sequence_input,
                node_input=node_features,
                adj_input=adjacency,
                node_index=node_index,
                physchem_input=physchem_input,
            ).squeeze(1)

            # This is what gets logged/compared across epochs and runs -
            # kept free of the L1 penalty term below so metrics stay
            # comparable regardless of ACTIVE_L1_LAMBDA.
            prediction_loss = criterion(logits, labels)

            if is_training:
                training_loss = prediction_loss

                if ACTIVE_L1_LAMBDA > 0:
                    # ncVarPred's own training code applies L1 specifically
                    # to the fusion (FC) and GCN layers, not the conv/
                    # BiLSTM/physicochemical branches - matched here.
                    l1_penalty = sum(
                        parameter.abs().sum()
                        for module in (model.fusion, model.structure_branch)
                        for parameter in module.parameters()
                    )
                    training_loss = training_loss + ACTIVE_L1_LAMBDA * l1_penalty

                optimizer.zero_grad()
                training_loss.backward()
                # Caps the rare large-gradient batch (BiLSTM/GCN stacks can
                # produce one) from knocking the model out of a good region
                # in a single step; a no-op on normal small-gradient steps.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
                optimizer.step()

            batch_size = labels.shape[0]
            total_loss += prediction_loss.item() * batch_size
            total_samples += batch_size

            all_probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

            now = time.time()
            if now - last_log_time >= LOG_INTERVAL_SECONDS:
                elapsed = now - epoch_start_time
                throughput = total_samples / elapsed
                remaining_rows = total_rows_in_split - total_samples
                eta_minutes = (remaining_rows / throughput) / 60 if throughput > 0 else float("nan")

                print(
                    f"  [{phase_name}] {total_samples:,}/{total_rows_in_split:,} rows | "
                    f"running_loss={total_loss / total_samples:.4f} | "
                    f"{throughput:.0f} rows/s | elapsed {elapsed / 60:.1f}min | ETA {eta_minutes:.1f}min"
                )
                last_log_time = now

    probabilities = np.concatenate(all_probabilities)
    labels_array = np.concatenate(all_labels)
    predictions = (probabilities >= 0.5).astype(np.int8)

    if not np.isfinite(probabilities).all():
        raise RuntimeError("Non-finite probabilities produced during this epoch.")

    metrics = {
        "loss": total_loss / total_samples,
        "balanced_accuracy": balanced_accuracy_score(labels_array, predictions),
        "f1": f1_score(labels_array, predictions, zero_division=0),
        "mcc": matthews_corrcoef(labels_array, predictions),
        "average_precision": average_precision_score(labels_array, probabilities),
    }
    metrics["roc_auc"] = (
        roc_auc_score(labels_array, probabilities) if len(np.unique(labels_array)) > 1 else float("nan")
    )

    return metrics


def main() -> None:
    set_seed(TRAINING_SEED)

    ACTIVE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"warm_start={ACTIVE_WARM_START}, batch_size={ACTIVE_BATCH_SIZE}, "
          f"weight_decay={ACTIVE_WEIGHT_DECAY:.0e}, l1_lambda={ACTIVE_L1_LAMBDA}, "
          f"epochs={ACTIVE_EPOCHS}, early_stopping_patience={ACTIVE_EARLY_STOPPING_PATIENCE}")

    print("\n[1/5] Loading graph tensors (node features + adjacency)")
    node_features, adjacency, number_of_nodes = load_graph_tensors(device)
    print(f"Graph nodes: {number_of_nodes:,}")

    print("\n[2/5] Building datasets")
    train_dataset, train_loader = build_loader("train", shuffle=True)
    validation_dataset, validation_loader = build_loader("validation", shuffle=False)
    print(f"Train rows: {len(train_dataset):,} | Validation rows: {len(validation_dataset):,}")

    pos_weight = resolve_pos_weight(train_dataset)

    print(f"\n[3/5] Building model (use_sequence_self_attention={USE_SEQUENCE_SELF_ATTENTION})")
    model = DeepMethModel(
        physchem_dropout_prob=PHYSCHEM_DROPOUT,
        fusion_projected_dim=FUSION_PROJECTED_DIM,
        fusion_hidden_dim=FUSION_HIDDEN_DIM,
        fusion_dropout_prob=FUSION_DROPOUT,
        use_sequence_self_attention=USE_SEQUENCE_SELF_ATTENTION,
    ).to(device)

    if ACTIVE_WARM_START:
        print("\n[4/5] Warm-starting branches from standalone checkpoints")
        load_branch_weights(model.sequence_branch, ACTIVE_SEQUENCE_ONLY_CHECKPOINT_PATH, "sequence_branch.", device)
        load_branch_weights(model.structure_branch, ACTIVE_GRAPH_ONLY_CHECKPOINT_PATH, "structure_branch.", device)
        load_branch_weights(model.physchem_branch, ACTIVE_PHYSICOCHEMICAL_ONLY_CHECKPOINT_PATH, "physchem_branch.", device)
    else:
        print("\n[4/5] No warm-start for this dataset - training from scratch")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    start_epoch = 0
    best_validation_loss = float("inf")
    patience_counter = 0
    history = []

    # With ACTIVE_WARM_START=False, ACTIVE_WARMUP_FROZEN_EPOCHS=0 so this
    # phase is skipped immediately (epoch 0 >= 0) - the whole model trains
    # from the first epoch, reproducing the pre-unification HepG2 path
    # exactly.
    is_frozen_phase = ACTIVE_WARM_START and ACTIVE_WARMUP_FROZEN_EPOCHS > 0
    if is_frozen_phase:
        set_branch_trainable(model, trainable=False)
        optimizer, scheduler = build_optimizer_and_scheduler(model, ACTIVE_FROZEN_LEARNING_RATE)
        print(f"  Phase: FROZEN branches, training fusion only, lr={ACTIVE_FROZEN_LEARNING_RATE:.0e}")
    else:
        optimizer, scheduler = build_optimizer_and_scheduler(model, ACTIVE_LEARNING_RATE)

    if LAST_CHECKPOINT_PATH.exists():
        print(f"\nResuming from {LAST_CHECKPOINT_PATH}")
        checkpoint = torch.load(LAST_CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_validation_loss = checkpoint["best_validation_loss"]
        patience_counter = checkpoint["patience_counter"]
        history = checkpoint["history"]
        is_frozen_phase = is_frozen_phase and start_epoch < ACTIVE_WARMUP_FROZEN_EPOCHS

        if ACTIVE_WARM_START and not is_frozen_phase:
            set_branch_trainable(model, trainable=True)
            optimizer, scheduler = build_optimizer_and_scheduler(model, ACTIVE_UNFREEZE_LEARNING_RATE)
            print(f"  Resumed into UNFROZEN phase, lr={ACTIVE_UNFREEZE_LEARNING_RATE:.0e}")

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print("\n[5/5] Training")
    for epoch in range(start_epoch, ACTIVE_EPOCHS):
        if is_frozen_phase and epoch >= ACTIVE_WARMUP_FROZEN_EPOCHS:
            print(f"\n  Unfreezing all branches at epoch {epoch + 1}, switching to lr={ACTIVE_UNFREEZE_LEARNING_RATE:.0e}")
            set_branch_trainable(model, trainable=True)
            optimizer, scheduler = build_optimizer_and_scheduler(model, ACTIVE_UNFREEZE_LEARNING_RATE)
            is_frozen_phase = False

        epoch_start = time.time()

        train_dataset.set_epoch(epoch)
        train_metrics = run_epoch(
            model, train_loader, node_features, adjacency, criterion, device, optimizer,
            phase_name="train",
        )

        validation_dataset.set_epoch(epoch)
        validation_metrics = run_epoch(
            model, validation_loader, node_features, adjacency, criterion, device, None,
            phase_name="validation",
        )

        epoch_duration = time.time() - epoch_start

        # Step the scheduler on validation loss before logging, so the
        # printed lr already reflects any drop triggered by this epoch's
        # result.
        scheduler.step(validation_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        improved = validation_metrics["loss"] < best_validation_loss - EARLY_STOPPING_MIN_DELTA
        phase_label = "frozen" if is_frozen_phase else "unfrozen"

        print(
            f"Epoch {epoch + 1}/{ACTIVE_EPOCHS} [{phase_label}] ({epoch_duration:.1f}s) | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={validation_metrics['loss']:.4f} | "
            f"val_balanced_acc={validation_metrics['balanced_accuracy']:.4f} | "
            f"val_f1={validation_metrics['f1']:.4f} | "
            f"val_mcc={validation_metrics['mcc']:.4f} | "
            f"val_ap={validation_metrics['average_precision']:.4f} | "
            f"val_auc={validation_metrics['roc_auc']:.4f} | "
            f"lr={current_lr:.2e}" + ("  <- best" if improved else "")
        )

        history.append({
            "epoch": epoch, "phase": phase_label, "train": train_metrics,
            "validation": validation_metrics, "epoch_seconds": epoch_duration,
        })

        if improved:
            best_validation_loss = validation_metrics["loss"]
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_metrics": validation_metrics,
                },
                BEST_CHECKPOINT_PATH,
            )
            print(f"  New best validation loss: {best_validation_loss:.4f} -> saved {BEST_CHECKPOINT_PATH}")
        else:
            patience_counter += 1

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_validation_loss": best_validation_loss,
                "patience_counter": patience_counter,
                "history": history,
            },
            LAST_CHECKPOINT_PATH,
        )

        with HISTORY_PATH.open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

        # Patience only counts once we're past any frozen warm-start phase -
        # during that phase the model hasn't started real joint fine-tuning
        # yet, so an early-stop check there would be premature. With
        # ACTIVE_WARM_START=False, is_frozen_phase is always False, so this
        # is unconditional early stopping, same as the original HepG2 loop.
        if not is_frozen_phase and patience_counter >= ACTIVE_EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping: no improvement for {ACTIVE_EARLY_STOPPING_PATIENCE} epochs.")
            break

    print("\nTraining completed.")
    print(f"Best validation loss: {best_validation_loss:.4f}")
    print(f"Best checkpoint: {BEST_CHECKPOINT_PATH}")
    print(f"History: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
