from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from config.project_config import (
    ACTIVE_CHECKPOINT_DIR,
    ACTIVE_GRAPH_DIR,
    ACTIVE_RESULTS_DIR,
    ACTIVE_SPLIT_NODE_INDEX_DIR,
    DEVICE,
    EARLY_STOPPING_MIN_DELTA,
    GRAPH_BRANCH_OUTPUT_DIM,
    LOG_INTERVAL_SECONDS,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_MIN_LR,
    LR_SCHEDULER_PATIENCE,
    TRAINING_SEED,
)
from model.graph_branch_gat import GATv2Structure, load_oe_edge_index
from training.train import resolve_pos_weight, set_seed

BATCH_SIZE = 4096  # cheap per-sample cost (no sequence/physchem features), so a large batch is fine
LEARNING_RATE = 5e-04
WEIGHT_DECAY = 1e-05
EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20

HEAD_HIDDEN_DIM = 128
HEAD_DROPOUT_PROB = 0.3

NODE_FEATURES_PATH = ACTIVE_GRAPH_DIR / "node_features.npy"
EDGE_FEATURES_PATH = ACTIVE_GRAPH_DIR / "edge_features.npz"


CHECKPOINT_SUBDIR = ACTIVE_CHECKPOINT_DIR.parent / "graph_gat_10kb"
LAST_CHECKPOINT_PATH = CHECKPOINT_SUBDIR / "last_checkpoint.pt"
BEST_CHECKPOINT_PATH = CHECKPOINT_SUBDIR / "best_model.pt"
STRUCTURE_BRANCH_WEIGHTS_PATH = CHECKPOINT_SUBDIR / "structure_branch_pretrained.pt"

BEST_MCC_CHECKPOINT_PATH = CHECKPOINT_SUBDIR / "best_model_mcc.pt"
STRUCTURE_BRANCH_WEIGHTS_MCC_PATH = CHECKPOINT_SUBDIR / "structure_branch_pretrained_mcc.pt"


SELF_SUPERVISED_CHECKPOINT_PATH = (
    ACTIVE_CHECKPOINT_DIR.parent / "graph_gat_selfsupervised_10kb" / "structure_branch_selfsupervised.pt"
)

HISTORY_PATH = ACTIVE_RESULTS_DIR / "training_history_graph_gat_10kb.json"


class NodeLabelDataset(Dataset):
    """(node_index, label) pairs for one split - both already row-aligned
    with the split's parquet (see feature_extraction/audit_graph_features_gm12878.py's
    check_split_node_index, which verifies this same {split}_node_index.npy
    against independently recomputed values)."""

    def __init__(self, split_name: str):
        node_index_path = ACTIVE_SPLIT_NODE_INDEX_DIR / f"{split_name}_node_index.npy"
        parquet_path = ACTIVE_SPLIT_NODE_INDEX_DIR / f"{split_name}.parquet"

        if not node_index_path.exists():
            raise FileNotFoundError(
                f"{node_index_path} does not exist. Run the matching prepare_graph_features script first."
            )

        self.node_indices = np.load(node_index_path)
        labels = pq.read_table(parquet_path, columns=["label"]).to_pandas()["label"].to_numpy(dtype=np.int8)

        if len(labels) != len(self.node_indices):
            raise RuntimeError(
                f"{split_name}: {len(labels):,} rows in {parquet_path.name} but "
                f"{len(self.node_indices):,} rows in {node_index_path.name} - out of sync."
            )

        self.labels = labels
        self.positive_count = int(labels.sum())
        self.negative_count = int(len(labels) - labels.sum())

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[int, float]:
        return int(self.node_indices[index]), float(self.labels[index])


def collate_node_labels(batch):
    node_indices, labels = zip(*batch)
    return {
        "node_index": torch.tensor(node_indices, dtype=torch.long),
        "label": torch.tensor(labels, dtype=torch.float32),
    }


def build_loader(split_name: str, shuffle: bool) -> tuple[NodeLabelDataset, DataLoader]:
    dataset = NodeLabelDataset(split_name)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=shuffle, collate_fn=collate_node_labels,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def load_graph_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not NODE_FEATURES_PATH.exists():
        raise FileNotFoundError(f"{NODE_FEATURES_PATH} does not exist. Run the matching prepare_graph_features script first.")
    if not EDGE_FEATURES_PATH.exists():
        raise FileNotFoundError(f"{EDGE_FEATURES_PATH} does not exist. Run the matching prepare_hic_graph script first.")

    node_features = np.load(NODE_FEATURES_PATH)
    edge_index, edge_attr = load_oe_edge_index(EDGE_FEATURES_PATH)

    node_features_tensor = torch.from_numpy(node_features).float().to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)

    return node_features_tensor, edge_index, edge_attr


class GraphOnlyModel(nn.Module):
    """GATv2Structure + a small linear head, for standalone pretraining only."""

    def __init__(self, head_hidden_dim: int, head_dropout_prob: float):
        super().__init__()
        self.structure_branch = GATv2Structure()
        self.head = nn.Sequential(
            nn.Linear(GRAPH_BRANCH_OUTPUT_DIM, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout_prob),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_index: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.structure_branch(node_features, edge_index, edge_attr, node_index)
        return self.head(embedding).squeeze(1)


def collect_validation_probabilities(
    model: nn.Module,
    loader: DataLoader,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probabilities = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            node_index = batch["node_index"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(node_features, edge_index, edge_attr, node_index)
            all_probabilities.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_probabilities), np.concatenate(all_labels)


def sweep_best_threshold(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_mcc = matthews_corrcoef(labels, (probabilities >= 0.5).astype(np.int8))
    for threshold in np.arange(0.05, 0.951, 0.01):
        mcc = matthews_corrcoef(labels, (probabilities >= threshold).astype(np.int8))
        if mcc > best_mcc:
            best_mcc = float(mcc)
            best_threshold = float(threshold)
    return best_threshold, best_mcc


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
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
            node_index = batch["node_index"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(node_features, edge_index, edge_attr, node_index)
                loss = criterion(logits, labels)

            if is_training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

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

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required (full-graph GATv2 message passing every batch).")
    device = torch.device(DEVICE)
    print(f"Device: {device}")
    print(f"Pretraining GATv2Structure alone "
          f"(batch_size={BATCH_SIZE}, lr={LEARNING_RATE:.0e}, weight_decay={WEIGHT_DECAY:.0e})")

    print("\n[1/4] Loading graph tensors (node features + Hi-C edge_index/edge_attr)")
    node_features, edge_index, edge_attr = load_graph_tensors(device)
    print(f"Graph nodes: {node_features.shape[0]:,} | Edges (incl. self-loops): {edge_index.shape[1]:,}")

    print("\n[2/4] Building datasets")
    train_dataset, train_loader = build_loader("train", shuffle=True)
    validation_dataset, validation_loader = build_loader("validation", shuffle=False)
    print(f"Train rows: {len(train_dataset):,} | Validation rows: {len(validation_dataset):,}")

    pos_weight = resolve_pos_weight(train_dataset)

    print("\n[3/4] Building model")
    model = GraphOnlyModel(head_hidden_dim=HEAD_HIDDEN_DIM, head_dropout_prob=HEAD_DROPOUT_PROB).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    if SELF_SUPERVISED_CHECKPOINT_PATH.exists() and not LAST_CHECKPOINT_PATH.exists():
        structure_state_dict = torch.load(SELF_SUPERVISED_CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.structure_branch.load_state_dict(structure_state_dict)
        print(f"Warm-started structure_branch from {SELF_SUPERVISED_CHECKPOINT_PATH}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(device="cuda")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE, min_lr=LR_SCHEDULER_MIN_LR,
    )

    start_epoch = 0
    best_validation_loss = float("inf")
    best_validation_mcc = -float("inf")
    patience_counter = 0
    history = []

    if LAST_CHECKPOINT_PATH.exists():
        print(f"\nResuming from {LAST_CHECKPOINT_PATH}")
        checkpoint = torch.load(LAST_CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_validation_loss = checkpoint["best_validation_loss"]
        patience_counter = checkpoint["patience_counter"]
        history = checkpoint["history"]

        if "best_validation_mcc" in checkpoint:
            best_validation_mcc = checkpoint["best_validation_mcc"]
        elif history:
            best_validation_mcc = max(entry["validation"]["mcc"] for entry in history)
            print(f"  (older checkpoint format: reconstructed best_validation_mcc={best_validation_mcc:.4f} from history)")

    print("\n[4/4] Training")
    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()

        train_metrics = run_epoch(
            model, train_loader, node_features, edge_index, edge_attr, criterion, device, optimizer, scaler,
            phase_name="train",
        )
        validation_metrics = run_epoch(
            model, validation_loader, node_features, edge_index, edge_attr, criterion, device, None, scaler,
            phase_name="validation",
        )

        epoch_duration = time.time() - epoch_start

        scheduler.step(validation_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        improved = validation_metrics["loss"] < best_validation_loss - EARLY_STOPPING_MIN_DELTA
        mcc_improved = validation_metrics["mcc"] > best_validation_mcc + EARLY_STOPPING_MIN_DELTA
        print(
            f"Epoch {epoch + 1}/{EPOCHS} ({epoch_duration:.1f}s) | "
            f"train_loss={train_metrics['loss']:.4f} | val_loss={validation_metrics['loss']:.4f} | "
            f"val_balanced_acc={validation_metrics['balanced_accuracy']:.4f} | "
            f"val_mcc={validation_metrics['mcc']:.4f} | val_auc={validation_metrics['roc_auc']:.4f} | "
            f"lr={current_lr:.2e}"
            + ("  <- best loss" if improved else "")
            + ("  <- best mcc" if mcc_improved else "")
        )

        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics, "epoch_seconds": epoch_duration})

        if improved:
            best_validation_loss = validation_metrics["loss"]
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "validation_metrics": validation_metrics},
                BEST_CHECKPOINT_PATH,
            )
            torch.save(model.structure_branch.state_dict(), STRUCTURE_BRANCH_WEIGHTS_PATH)
            print(f"  New best validation loss: {best_validation_loss:.4f} -> saved {STRUCTURE_BRANCH_WEIGHTS_PATH}")
        else:
            patience_counter += 1

        if mcc_improved:
            best_validation_mcc = validation_metrics["mcc"]
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "validation_metrics": validation_metrics},
                BEST_MCC_CHECKPOINT_PATH,
            )
            torch.save(model.structure_branch.state_dict(), STRUCTURE_BRANCH_WEIGHTS_MCC_PATH)
            print(f"  New best validation MCC: {best_validation_mcc:.4f} -> saved {STRUCTURE_BRANCH_WEIGHTS_MCC_PATH}")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_validation_loss": best_validation_loss,
                "best_validation_mcc": best_validation_mcc,
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

    print("\nGATv2Structure pretraining completed.")
    print(f"Best validation loss: {best_validation_loss:.4f}")
    print(f"Best validation MCC (0.5 threshold): {best_validation_mcc:.4f}")
    print(f"Structure branch weights (val_loss-best): {STRUCTURE_BRANCH_WEIGHTS_PATH}")
    print(f"Structure branch weights (val_mcc-best):  {STRUCTURE_BRANCH_WEIGHTS_MCC_PATH}")

    print("\nSweeping validation decision threshold on the best-MCC checkpoint...")
    best_mcc_checkpoint = torch.load(BEST_MCC_CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(best_mcc_checkpoint["model_state_dict"])
    probabilities, labels_array = collect_validation_probabilities(
        model, validation_loader, node_features, edge_index, edge_attr, device,
    )
    best_threshold, best_threshold_mcc = sweep_best_threshold(probabilities, labels_array)
    print(
        f"Best threshold: {best_threshold:.2f} -> val_mcc={best_threshold_mcc:.4f} "
        f"(vs {best_validation_mcc:.4f} at the default 0.5 threshold)"
    )
    print("Compare this run's best val_mcc against GCN_Structure's own standalone "
          "ceiling (~0.26 on GM12878) before deciding whether to swap it into the full model.")


if __name__ == "__main__":
    main()
