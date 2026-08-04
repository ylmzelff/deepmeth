"""
Full 3-branch DeepMeth model, using the GATv2-based Hi-C graph branch
(model/graph_branch_gat.py: GATv2Structure, 25kb resolution) instead of the
original GCN_Structure - with the ORIGINAL DanQ sequence branch kept
unchanged. Deliberately isolates the graph-branch upgrade's effect: this
is train.py's own baseline model (val_mcc 0.6636-0.6653, GCN_Structure)
with exactly one component swapped, so any change in the fused result is
attributable to the graph branch alone, not conflated with the separate
DNABERT-2 sequence branch experiment (training/train_full_fusion_dnabert.py).

Motivation for the graph-branch swap (see project history for the full
diagnosis): GCN_Structure's standalone val_mcc was only ~0.26 on GM12878 -
a graph branch that weak contributed almost nothing to train.py's fused
result. feature_extraction/analyze_graph_node_label_purity.py's majority-
vote-per-node oracle showed the real ceiling for a graph-only model (every
CpG in a node gets the same embedding) was ~0.50 at the original 100kb
resolution and ~0.69 at 25kb - GATv2Structure at 25kb (with top-k Hi-C edge
sparsification, KR-normalized contacts, residual connections, Jumping
Knowledge) reached standalone val_mcc ~0.57-0.58, versus GCN's 0.26.

Reuses training/dataset.py's DeepMethShardDataset/collate_batch and
training/train.py's build_loader/resolve_pos_weight/set_seed/
load_branch_weights/build_optimizer_and_scheduler UNCHANGED - the one-hot
sequence + physchem + node_index + label batch shape those already
produce is exactly what DanQ_Sequence + GATv2Structure + CNNNet_PhyChemDi
need; only the graph tensors (node_features/edge_index/edge_attr instead
of node_features/adjacency) and the model class itself differ. Also reuses
training/train_graph_gat.py's load_graph_tensors (identical graph-loading
logic, no need to duplicate it).

Warm-starts all three branches from their standalone checkpoints:
  - sequence_branch <- ACTIVE_SEQUENCE_ONLY_CHECKPOINT_PATH (the same DanQ
    checkpoint train.py itself warm-starts from - unchanged).
  - structure_branch <- training/train_graph_gat.py's GATv2Structure
    checkpoint (25kb, standalone val_mcc ~0.57-0.58).
  - physchem_branch <- ACTIVE_PHYSICOCHEMICAL_ONLY_CHECKPOINT_PATH
    (unchanged).

Usage (no arguments needed):

    python training/train_full_fusion_gatv2.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from config.project_config import (
    ACTIVE_CHECKPOINT_DIR,
    ACTIVE_EARLY_STOPPING_PATIENCE,
    ACTIVE_EPOCHS,
    ACTIVE_FROZEN_LEARNING_RATE,
    ACTIVE_PHYSICOCHEMICAL_ONLY_CHECKPOINT_PATH,
    ACTIVE_RESULTS_DIR,
    ACTIVE_SEQUENCE_ONLY_CHECKPOINT_PATH,
    ACTIVE_UNFREEZE_LEARNING_RATE,
    ACTIVE_WARMUP_FROZEN_EPOCHS,
    DEVICE,
    EARLY_STOPPING_MIN_DELTA,
    FUSION_DROPOUT,
    FUSION_HIDDEN_DIM,
    FUSION_PROJECTED_DIM,
    GRAD_CLIP_MAX_NORM,
    GRAPH_BRANCH_OUTPUT_DIM,
    LOG_INTERVAL_SECONDS,
    PHYSCHEM_DROPOUT,
    PHYSICOCHEMICAL_CNN_OUTPUT_DIM,
    SEQUENCE_BRANCH_OUTPUT_DIM,
    TRAINING_SEED,
    USE_SEQUENCE_SELF_ATTENTION,
)
from model.fusion import GatedFusion
from model.graph_branch_gat import GATv2Structure
from model.physicochemical_branch import CNNNet_PhyChemDi
from model.sequence_branch import DanQ_Sequence
from training.train import (
    build_loader,
    build_optimizer_and_scheduler,
    load_branch_weights,
    resolve_pos_weight,
    set_branch_trainable,
    set_seed,
)
from training.train_graph_gat import load_graph_tensors

# Sibling of train.py's own checkpoint dir - never collides with either the
# original GCN-based full model or train_full_fusion_dnabert.py's own run.
CHECKPOINT_DIR = ACTIVE_CHECKPOINT_DIR.parent / "full_model_gatv2_fusion"
LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / "last_checkpoint.pt"
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pt"
HISTORY_PATH = ACTIVE_RESULTS_DIR / "training_history_full_fusion_gatv2.json"

# training/train_graph_gat.py's own standalone checkpoint - a raw state_dict
# (torch.save(model.structure_branch.state_dict(), ...)), not a wrapped
# checkpoint dict, same convention as the DNABERT sequence branch's own
# SEQUENCE_BRANCH_WEIGHTS_PATH.
GATV2_STRUCTURE_BRANCH_CHECKPOINT_PATH = (
    ACTIVE_CHECKPOINT_DIR.parent / "graph_gat_only" / "structure_branch_pretrained.pt"
)


class DeepMethGATv2FusionModel(nn.Module):
    """Same 3-branch + gated fusion structure as model/deepmeth_model.py's
    DeepMethModel, with GCN_Structure swapped for GATv2Structure - the
    original DanQ sequence branch and physicochemical branch are
    unchanged, isolating the graph-branch upgrade's effect on the fused
    result. Attribute names (sequence_branch, structure_branch,
    physchem_branch, fusion) intentionally match DeepMethModel's so
    training.train's generic helpers work unmodified."""

    def __init__(
        self,
        physchem_dropout_prob: float,
        fusion_projected_dim: int,
        fusion_hidden_dim: int,
        fusion_dropout_prob: float,
        use_sequence_self_attention: bool = False,
    ):
        super().__init__()

        self.sequence_branch = DanQ_Sequence(use_self_attention=use_sequence_self_attention)
        self.structure_branch = GATv2Structure()
        self.physchem_branch = CNNNet_PhyChemDi(
            dropout_prob=physchem_dropout_prob, use_property_gate=False,
        )
        self.fusion = GatedFusion(
            sequence_dim=SEQUENCE_BRANCH_OUTPUT_DIM,
            graph_dim=GRAPH_BRANCH_OUTPUT_DIM,
            physchem_dim=PHYSICOCHEMICAL_CNN_OUTPUT_DIM,
            projected_dim=fusion_projected_dim,
            hidden_dim=fusion_hidden_dim,
            dropout_prob=fusion_dropout_prob,
        )

    def forward(
        self,
        seq_input: torch.Tensor,
        node_input: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_index: torch.Tensor,
        physchem_input: torch.Tensor,
    ) -> torch.Tensor:
        seq_output = self.sequence_branch(seq_input)  # [B, 4, 501] -> [B, 925]

        structure_output = self.structure_branch(
            node_input, edge_index, edge_attr, node_index,
        )  # [B, 128]

        physchem_output = self.physchem_branch(physchem_input)  # [B, 480]

        return self.fusion(seq_output, structure_output, physchem_output)  # [B, 1]


def load_structure_branch_weights(module: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} does not exist. Run training/train_graph_gat.py first."
        )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    module.load_state_dict(state_dict, strict=True)
    print(f"  Loaded GATv2 structure branch weights from {checkpoint_path.name}")


def run_epoch(
    model: nn.Module,
    loader,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
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
                edge_index=edge_index,
                edge_attr=edge_attr,
                node_index=node_index,
                physchem_input=physchem_input,
            ).squeeze(1)

            loss = criterion(logits, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
                optimizer.step()

            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
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

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"use_sequence_self_attention={USE_SEQUENCE_SELF_ATTENTION}")

    print("\n[1/5] Loading graph tensors (node features + Hi-C edge_index/edge_attr, 25kb)")
    node_features, edge_index, edge_attr = load_graph_tensors(device)
    print(f"Graph nodes: {node_features.shape[0]:,} | Edges (incl. self-loops): {edge_index.shape[1]:,}")

    print("\n[2/5] Building datasets")
    train_dataset, train_loader = build_loader("train", shuffle=True)
    validation_dataset, validation_loader = build_loader("validation", shuffle=False)
    print(f"Train rows: {len(train_dataset):,} | Validation rows: {len(validation_dataset):,}")

    pos_weight = resolve_pos_weight(train_dataset)

    print("\n[3/5] Building model (DanQ sequence branch + GATv2 graph branch)")
    model = DeepMethGATv2FusionModel(
        physchem_dropout_prob=PHYSCHEM_DROPOUT,
        fusion_projected_dim=FUSION_PROJECTED_DIM,
        fusion_hidden_dim=FUSION_HIDDEN_DIM,
        fusion_dropout_prob=FUSION_DROPOUT,
        use_sequence_self_attention=USE_SEQUENCE_SELF_ATTENTION,
    ).to(device)

    print("\n[4/5] Warm-starting branches from standalone checkpoints")
    load_branch_weights(model.sequence_branch, ACTIVE_SEQUENCE_ONLY_CHECKPOINT_PATH, "sequence_branch.", device)
    load_structure_branch_weights(model.structure_branch, GATV2_STRUCTURE_BRANCH_CHECKPOINT_PATH, device)
    load_branch_weights(model.physchem_branch, ACTIVE_PHYSICOCHEMICAL_ONLY_CHECKPOINT_PATH, "physchem_branch.", device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    start_epoch = 0
    best_validation_loss = float("inf")
    patience_counter = 0
    history = []

    is_frozen_phase = ACTIVE_WARMUP_FROZEN_EPOCHS > 0
    if is_frozen_phase:
        set_branch_trainable(model, trainable=False)
        optimizer, scheduler = build_optimizer_and_scheduler(model, ACTIVE_FROZEN_LEARNING_RATE)
        print(f"  Phase: FROZEN branches, training fusion only, lr={ACTIVE_FROZEN_LEARNING_RATE:.0e}")
    else:
        optimizer, scheduler = build_optimizer_and_scheduler(model, ACTIVE_UNFREEZE_LEARNING_RATE)

    if LAST_CHECKPOINT_PATH.exists():
        print(f"\nResuming from {LAST_CHECKPOINT_PATH}")
        checkpoint = torch.load(LAST_CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_validation_loss = checkpoint["best_validation_loss"]
        patience_counter = checkpoint["patience_counter"]
        history = checkpoint["history"]
        is_frozen_phase = is_frozen_phase and start_epoch < ACTIVE_WARMUP_FROZEN_EPOCHS

        if not is_frozen_phase:
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
            model, train_loader, node_features, edge_index, edge_attr, criterion, device, optimizer,
            phase_name="train",
        )

        validation_dataset.set_epoch(epoch)
        validation_metrics = run_epoch(
            model, validation_loader, node_features, edge_index, edge_attr, criterion, device, None,
            phase_name="validation",
        )

        epoch_duration = time.time() - epoch_start

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
                {"epoch": epoch, "model_state_dict": model.state_dict(), "validation_metrics": validation_metrics},
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

        if not is_frozen_phase and patience_counter >= ACTIVE_EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping: no improvement for {ACTIVE_EARLY_STOPPING_PATIENCE} epochs.")
            break

    print("\nFull fusion (GATv2 graph branch) training completed.")
    print(f"Best validation loss: {best_validation_loss:.4f}")
    print(f"Best checkpoint: {BEST_CHECKPOINT_PATH}")
    print(f"History: {HISTORY_PATH}")
    print("Compare val_mcc against train.py's original GCN-based full model result "
          "(val_mcc 0.6636-0.6653) to isolate the graph-branch upgrade's effect.")


if __name__ == "__main__":
    main()
