"""
Pretrain the new Transformer sequence branch (model/sequence_branch_transformer.py:
TransformerSequence - multi-scale conv stem + ALiBi relative-position
Transformer encoder) alone, on the active dataset's train/validation split,
and save just its weights - not a whole model. Same standalone-check-first
pattern as the original DanQ sequence branch (training/train_sequence.py):
verify this branch alone beats (or at least matches) DanQ_Sequence's own
standalone ceiling (val_mcc ~0.65 on GM12878) before deciding whether to
warm-start the full 3-branch model from it.

Dataset-agnostic: paths and dataset selection come from
config.project_config's ACTIVE_* constants (set DATASET there). Hyperparameters
below are set locally, not imported from config - this is a new architecture
being explored for the first time, not yet tuned, so it reuses the
sequence-branch hyperparameter sweep's winning scale (batch=1024, lr=5e-05,
weight_decay=1e-05) as a reasonable starting point rather than either
dataset's existing tuned-for-a-different-model defaults.

Usage (no arguments needed):

    python training/train_sequence_transformer.py
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
from torch.utils.data import DataLoader

from config.project_config import (
    ACTIVE_CHECKPOINT_DIR,
    ACTIVE_RESULTS_DIR,
    DEVICE,
    EARLY_STOPPING_MIN_DELTA,
    LOG_INTERVAL_SECONDS,
    TRAINING_SEED,
)
from model.sequence_branch_transformer import TransformerSequence
from training.train import build_loader, resolve_pos_weight, set_seed

BATCH_SIZE = 1024
LEARNING_RATE = 5e-05
WEIGHT_DECAY = 1e-05
EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20

HEAD_HIDDEN_DIM = 128
HEAD_DROPOUT_PROB = 0.3

CHECKPOINT_SUBDIR = ACTIVE_CHECKPOINT_DIR.parent / "sequence_transformer_only"
LAST_CHECKPOINT_PATH = CHECKPOINT_SUBDIR / "last_checkpoint.pt"
BEST_CHECKPOINT_PATH = CHECKPOINT_SUBDIR / "best_model.pt"

# What a future warm-start run would load - just the sequence_branch
# sub-module's weights, saved fresh every time a new best is found.
SEQUENCE_BRANCH_WEIGHTS_PATH = CHECKPOINT_SUBDIR / "sequence_branch_pretrained.pt"

HISTORY_PATH = ACTIVE_RESULTS_DIR / "training_history_sequence_transformer_only.json"


class SequenceOnlyModel(nn.Module):
    """TransformerSequence + a small linear head, for standalone pretraining only."""

    def __init__(self, head_hidden_dim: int, head_dropout_prob: float):
        super().__init__()
        self.sequence_branch = TransformerSequence()
        self.head = nn.Sequential(
            nn.Linear(925, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout_prob),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(self, seq_input: torch.Tensor) -> torch.Tensor:
        return self.head(self.sequence_branch(seq_input)).squeeze(1)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
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
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(sequence_input)
            loss = criterion(logits, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
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

    CHECKPOINT_SUBDIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Pretraining TransformerSequence branch alone "
          f"(batch_size={BATCH_SIZE}, lr={LEARNING_RATE:.0e}, weight_decay={WEIGHT_DECAY:.0e})")

    print("\n[1/3] Building datasets")
    train_dataset, train_loader_base = build_loader("train", shuffle=True)
    validation_dataset, validation_loader_base = build_loader("validation", shuffle=False)
    # build_loader (training/train.py) sizes batches off ACTIVE_BATCH_SIZE -
    # rebuild with this script's own BATCH_SIZE instead, everything else
    # (dataset, collate_fn, num_workers) unchanged.
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, collate_fn=train_loader_base.collate_fn,
        num_workers=train_loader_base.num_workers, pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, collate_fn=validation_loader_base.collate_fn,
        num_workers=validation_loader_base.num_workers, pin_memory=torch.cuda.is_available(),
    )
    print(f"Train rows: {len(train_dataset):,} | Validation rows: {len(validation_dataset):,}")

    pos_weight = resolve_pos_weight(train_dataset)

    print("\n[2/3] Building model")
    model = SequenceOnlyModel(
        head_hidden_dim=HEAD_HIDDEN_DIM,
        head_dropout_prob=HEAD_DROPOUT_PROB,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    start_epoch = 0
    best_validation_loss = float("inf")
    patience_counter = 0
    history = []

    if LAST_CHECKPOINT_PATH.exists():
        print(f"\nResuming from {LAST_CHECKPOINT_PATH}")
        checkpoint = torch.load(LAST_CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_validation_loss = checkpoint["best_validation_loss"]
        patience_counter = checkpoint["patience_counter"]
        history = checkpoint["history"]

    print("\n[3/3] Training")
    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()

        train_dataset.set_epoch(epoch)
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, phase_name="train")

        validation_dataset.set_epoch(epoch)
        validation_metrics = run_epoch(model, validation_loader, criterion, device, None, phase_name="validation")

        epoch_duration = time.time() - epoch_start

        improved = validation_metrics["loss"] < best_validation_loss - EARLY_STOPPING_MIN_DELTA
        print(
            f"Epoch {epoch + 1}/{EPOCHS} ({epoch_duration:.1f}s) | "
            f"train_loss={train_metrics['loss']:.4f} | val_loss={validation_metrics['loss']:.4f} | "
            f"val_balanced_acc={validation_metrics['balanced_accuracy']:.4f} | "
            f"val_mcc={validation_metrics['mcc']:.4f} | val_auc={validation_metrics['roc_auc']:.4f}"
            + ("  <- best" if improved else "")
        )

        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics, "epoch_seconds": epoch_duration})

        if improved:
            best_validation_loss = validation_metrics["loss"]
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "validation_metrics": validation_metrics},
                BEST_CHECKPOINT_PATH,
            )
            torch.save(model.sequence_branch.state_dict(), SEQUENCE_BRANCH_WEIGHTS_PATH)
            print(f"  New best validation loss: {best_validation_loss:.4f} -> saved {SEQUENCE_BRANCH_WEIGHTS_PATH}")
        else:
            patience_counter += 1

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_validation_loss": best_validation_loss,
                "patience_counter": patience_counter,
                "history": history,
            },
            LAST_CHECKPOINT_PATH,
        )

        with HISTORY_PATH.open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping: no improvement for {EARLY_STOPPING_PATIENCE} epochs.")
            break

    print("\nTransformerSequence pretraining completed.")
    print(f"Best validation loss: {best_validation_loss:.4f}")
    print(f"Sequence branch weights for warm-start: {SEQUENCE_BRANCH_WEIGHTS_PATH}")
    print("Compare this run's best val_mcc against DanQ_Sequence's own standalone "
          "ceiling (~0.65 on GM12878) before deciding whether to use this branch "
          "in the full model.")


if __name__ == "__main__":
    main()
