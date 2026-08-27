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
    ACTIVE_BATCH_SIZE,
    ACTIVE_CHECKPOINT_DIR,
    ACTIVE_EARLY_STOPPING_PATIENCE,
    ACTIVE_EPOCHS,
    # ACTIVE_GRAPH_DIR,  # graph branch retired for now - see model/deepmeth_model.py
    ACTIVE_HISTORY_FILENAME,
    ACTIVE_L1_LAMBDA,
    ACTIVE_LEARNING_RATE,
    ACTIVE_RESULTS_DIR,
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
)
from model.deepmeth_model import DeepMethModel
# from model.graph_branch_gat import load_oe_edge_index  # graph branch retired for now
from training.dataset import DeepMethShardDataset, collate_batch

LAST_CHECKPOINT_PATH = ACTIVE_CHECKPOINT_DIR / "last_checkpoint.pt"
BEST_CHECKPOINT_PATH = ACTIVE_CHECKPOINT_DIR / "best_model.pt"
HISTORY_PATH = ACTIVE_RESULTS_DIR / ACTIVE_HISTORY_FILENAME


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)






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


def build_optimizer_and_scheduler(model: DeepMethModel, learning_rate: float):
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=ACTIVE_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE, min_lr=LR_SCHEDULER_MIN_LR,
    )
    return optimizer, scheduler


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    # node_features/edge_index/edge_attr removed with the graph branch - see model/deepmeth_model.py
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
            foundation_tokens = batch["foundation_tokens"].to(device, non_blocking=True)
            foundation_attention_mask = batch["foundation_attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(
                seq_input=sequence_input,
                foundation_token_embeddings=foundation_tokens,
                foundation_attention_mask=foundation_attention_mask,
                physchem_input=physchem_input,
            ).squeeze(1)

            prediction_loss = criterion(logits, labels)

            if is_training:
                training_loss = prediction_loss

                if ACTIVE_L1_LAMBDA > 0:
                    l1_penalty = sum(
                        parameter.abs().sum()
                        for module in (model.fusion, model.foundation_branch)
                        for parameter in module.parameters()
                    )
                    training_loss = training_loss + ACTIVE_L1_LAMBDA * l1_penalty

                optimizer.zero_grad()
                training_loss.backward()
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
    print(f"batch_size={ACTIVE_BATCH_SIZE}, "
          f"weight_decay={ACTIVE_WEIGHT_DECAY:.0e}, l1_lambda={ACTIVE_L1_LAMBDA}, "
          f"epochs={ACTIVE_EPOCHS}, early_stopping_patience={ACTIVE_EARLY_STOPPING_PATIENCE}")

  

    print("\n[1/4] Building datasets")
    train_dataset, train_loader = build_loader("train", shuffle=True)
    validation_dataset, validation_loader = build_loader("validation", shuffle=False)
    print(f"Train rows: {len(train_dataset):,} | Validation rows: {len(validation_dataset):,}")

    pos_weight = resolve_pos_weight(train_dataset)

    print("\n[2/4] Building model")
   
    model = DeepMethModel(
        physchem_dropout_prob=PHYSCHEM_DROPOUT,
        fusion_projected_dim=FUSION_PROJECTED_DIM,
        fusion_hidden_dim=FUSION_HIDDEN_DIM,
        fusion_dropout_prob=FUSION_DROPOUT,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    start_epoch = 0
    best_validation_loss = float("inf")
    patience_counter = 0
    history = []

    optimizer, scheduler = build_optimizer_and_scheduler(model, ACTIVE_LEARNING_RATE)

    if LAST_CHECKPOINT_PATH.exists():
        print(f"\nResuming from {LAST_CHECKPOINT_PATH}")
        checkpoint = torch.load(LAST_CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_validation_loss = checkpoint["best_validation_loss"]
        patience_counter = checkpoint["patience_counter"]
        history = checkpoint["history"]

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print("\n[4/4] Training")
    for epoch in range(start_epoch, ACTIVE_EPOCHS):
        epoch_start = time.time()

        train_dataset.set_epoch(epoch)
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer,
            phase_name="train",
        )

        validation_dataset.set_epoch(epoch)
        validation_metrics = run_epoch(
            model, validation_loader, criterion, device, None,
            phase_name="validation",
        )

        epoch_duration = time.time() - epoch_start

        scheduler.step(validation_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        improved = validation_metrics["loss"] < best_validation_loss - EARLY_STOPPING_MIN_DELTA

        print(
            f"Epoch {epoch + 1}/{ACTIVE_EPOCHS} ({epoch_duration:.1f}s) | "
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
            "epoch": epoch, "train": train_metrics,
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

        if patience_counter >= ACTIVE_EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping: no improvement for {ACTIVE_EARLY_STOPPING_PATIENCE} epochs.")
            break

    print("\nTraining completed.")
    print(f"Best validation loss: {best_validation_loss:.4f}")
    print(f"Best checkpoint: {BEST_CHECKPOINT_PATH}")
    print(f"History: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
