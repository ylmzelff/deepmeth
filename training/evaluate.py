"""
Evaluate the best checkpoint from training/train.py on the held-out test split.

Usage (no arguments needed):

    python training/evaluate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config.project_config import CHECKPOINT_DIR, DEVICE, PHYSCHEM_DROPOUT, RESULTS_DIR, TRAINING_SEED
from model.deepmeth_model import DeepMethConcatenation
from training.dataset import collate_batch, DeepMethShardDataset
from training.train import build_loader, load_graph_tensors

BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pt"
TEST_METRICS_PATH = RESULTS_DIR / "test_metrics.json"


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    node_features: torch.Tensor,
    adjacency: torch.Tensor,
    number_of_nodes: int,
    device: torch.device,
) -> dict:
    model.eval()

    all_probabilities = []
    all_labels = []

    with torch.no_grad():
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

            all_probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    probabilities = np.concatenate(all_probabilities)
    labels_array = np.concatenate(all_labels)
    predictions = (probabilities >= 0.5).astype(np.int8)

    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels_array, predictions
    ).ravel()

    return {
        "sample_count": int(len(labels_array)),
        "positive_count": int(labels_array.sum()),
        "negative_count": int(len(labels_array) - labels_array.sum()),
        "balanced_accuracy": balanced_accuracy_score(labels_array, predictions),
        "precision": precision_score(labels_array, predictions, zero_division=0),
        "recall": recall_score(labels_array, predictions, zero_division=0),
        "f1": f1_score(labels_array, predictions, zero_division=0),
        "mcc": matthews_corrcoef(labels_array, predictions),
        "average_precision": average_precision_score(labels_array, probabilities),
        "roc_auc": roc_auc_score(labels_array, probabilities) if len(np.unique(labels_array)) > 1 else float("nan"),
        "confusion_matrix": {
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(true_positive),
        },
    }


def main() -> None:
    if not BEST_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"{BEST_CHECKPOINT_PATH} does not exist. Run training/train.py first.")

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n[1/3] Loading graph tensors")
    node_features, adjacency, number_of_nodes = load_graph_tensors(device)

    print("\n[2/3] Building test dataset")
    test_dataset, test_loader = build_loader("test", shuffle=False)
    print(f"Test rows: {len(test_dataset):,}")

    print("\n[3/3] Loading best checkpoint and evaluating")
    checkpoint = torch.load(BEST_CHECKPOINT_PATH, map_location=device)

    model = DeepMethConcatenation(physchem_dropout_prob=PHYSCHEM_DROPOUT).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Checkpoint from epoch {checkpoint['epoch']}, validation metrics: {checkpoint['validation_metrics']}")

    metrics = evaluate(model, test_loader, node_features, adjacency, number_of_nodes, device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with TEST_METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nTest set evaluation:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print(f"\nSaved: {TEST_METRICS_PATH}")


if __name__ == "__main__":
    main()
