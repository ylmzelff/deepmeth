"""
Train the three-branch DeepMeth model (sequence + Hi-C graph + physicochemical).

The graph branch runs a GCN over the *entire* static 30k-node Hi-C graph on
every batch (this is the original ncVarPred design: full-graph message
passing, then a [B, N] one-hot matmul selects the batch's rows out of the
result) - so node_features.npy and adjacency_normalized.npz are loaded once
and kept resident on the device for the whole run, not re-loaded per batch.

Resumable: if checkpoints/last_checkpoint.pt exists, training resumes from
there (model/optimizer state, epoch, best validation loss, early-stopping
counter) instead of starting over - Colab sessions can disconnect mid-run.

Usage (no arguments needed):

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
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from config.project_config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DEVICE,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    GRAPH_DIR,
    LEARNING_RATE,
    NUM_WORKERS,
    PHYSCHEM_DROPOUT,
    POS_WEIGHT_MODE,
    RESULTS_DIR,
    TRAINING_SEED,
    WEIGHT_DECAY,
)
from model.deepmeth_model import DeepMethConcatenation
from training.dataset import DeepMethShardDataset, collate_batch

NODE_FEATURES_PATH = GRAPH_DIR / "node_features.npy"
ADJACENCY_PATH = GRAPH_DIR / "adjacency_normalized.npz"
LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / "last_checkpoint.pt"
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pt"
HISTORY_PATH = RESULTS_DIR / "training_history.json"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_graph_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    if not NODE_FEATURES_PATH.exists():
        raise FileNotFoundError(f"{NODE_FEATURES_PATH} does not exist. Run feature_extraction/prepare_graph_features.py first.")

    if not ADJACENCY_PATH.exists():
        raise FileNotFoundError(f"{ADJACENCY_PATH} does not exist. Run feature_extraction/prepare_hic_graph.py first.")

    node_features = np.load(NODE_FEATURES_PATH)
    adjacency = sp.load_npz(ADJACENCY_PATH).tocoo()

    if node_features.shape[0] != adjacency.shape[0]:
        raise RuntimeError(
            f"node_features.npy has {node_features.shape[0]:,} nodes but adjacency_normalized.npz has "
            f"shape {adjacency.shape} - out of sync, rerun the feature_extraction graph scripts."
        )

    number_of_nodes = node_features.shape[0]

    node_features_tensor = torch.from_numpy(node_features).float().to(device)

    indices = torch.tensor(np.vstack([adjacency.row, adjacency.col]), dtype=torch.long)
    values = torch.tensor(adjacency.data, dtype=torch.float32)
    adjacency_tensor = torch.sparse_coo_tensor(
        indices, values, size=adjacency.shape
    ).coalesce().to(device)

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
        batch_size=BATCH_SIZE,
        collate_fn=collate_batch,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    node_features: torch.Tensor,
    adjacency: torch.Tensor,
    number_of_nodes: int,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_samples = 0
    all_probabilities = []
    all_labels = []

    with torch.set_grad_enabled(is_training):
        for batch in loader:
            sequence_input = batch["sequence"].to(device, non_blocking=True)
            physchem_input = batch["physicochemical"].to(device, non_blocking=True)
            node_index = batch["node_index"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            index_input = F.one_hot(node_index, num_classes=number_of_nodes).float()

            logits = model(
                seq_input=sequence_input,
                node_input=node_features,
                adj_input=adjacency,
                index_input=index_input,
                physchem_input=physchem_input,
            ).squeeze(1)

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

    if len(np.unique(labels_array)) > 1:
        metrics["roc_auc"] = roc_auc_score(labels_array, probabilities)
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def main() -> None:
    set_seed(TRAINING_SEED)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n[1/4] Loading graph tensors (node features + adjacency)")
    node_features, adjacency, number_of_nodes = load_graph_tensors(device)
    print(f"Graph nodes: {number_of_nodes:,}")

    print("\n[2/4] Building datasets")
    train_dataset, train_loader = build_loader("train", shuffle=True)
    validation_dataset, validation_loader = build_loader("validation", shuffle=False)
    print(f"Train rows: {len(train_dataset):,} | Validation rows: {len(validation_dataset):,}")

    pos_weight = resolve_pos_weight(train_dataset)

    print("\n[3/4] Building model")
    model = DeepMethConcatenation(physchem_dropout_prob=PHYSCHEM_DROPOUT).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    start_epoch = 0
    best_validation_loss = float("inf")
    patience_counter = 0
    history = []

    if LAST_CHECKPOINT_PATH.exists():
        print(f"\nResuming from {LAST_CHECKPOINT_PATH}")
        checkpoint = torch.load(LAST_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_validation_loss = checkpoint["best_validation_loss"]
        patience_counter = checkpoint["patience_counter"]
        history = checkpoint["history"]

    print("\n[4/4] Training")
    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()

        train_dataset.set_epoch(epoch)
        train_metrics = run_epoch(
            model, train_loader, node_features, adjacency, number_of_nodes, criterion, device, optimizer
        )

        validation_dataset.set_epoch(epoch)
        validation_metrics = run_epoch(
            model, validation_loader, node_features, adjacency, number_of_nodes, criterion, device, None
        )

        epoch_duration = time.time() - epoch_start

        print(
            f"Epoch {epoch + 1}/{EPOCHS} ({epoch_duration:.1f}s) | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={validation_metrics['loss']:.4f} | "
            f"val_balanced_acc={validation_metrics['balanced_accuracy']:.4f} | "
            f"val_f1={validation_metrics['f1']:.4f} | "
            f"val_mcc={validation_metrics['mcc']:.4f} | "
            f"val_ap={validation_metrics['average_precision']:.4f} | "
            f"val_auc={validation_metrics['roc_auc']:.4f}"
        )

        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})

        improved = validation_metrics["loss"] < best_validation_loss - EARLY_STOPPING_MIN_DELTA

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

    print("\nTraining completed.")
    print(f"Best validation loss: {best_validation_loss:.4f}")
    print(f"Best checkpoint: {BEST_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
