from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from model_training.model_deepmeth import DeepMethConcatenation


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(
    file_path: Path,
) -> TensorDataset:
    """
    Expected NPZ keys:

        sequence:
            [samples, 4, 501]

        physicochemical:
            [samples, 12, 500]

        node_index:
            [samples]

        label:
            [samples] or [samples, 1]
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file does not exist: {file_path}"
        )

    with np.load(
        file_path,
        allow_pickle=False,
    ) as data:
        required_keys = {
            "sequence",
            "physicochemical",
            "node_index",
            "label",
        }

        missing_keys = required_keys.difference(
            data.files
        )

        if missing_keys:
            raise KeyError(
                f"Missing dataset keys: {sorted(missing_keys)}"
            )

        sequence = torch.from_numpy(
            data["sequence"].astype(np.float32)
        )

        physicochemical = torch.from_numpy(
            data["physicochemical"].astype(np.float32)
        )

        node_index = torch.from_numpy(
            data["node_index"].astype(np.int64)
        ).reshape(-1)

        labels = torch.from_numpy(
            data["label"].astype(np.float32)
        ).reshape(-1, 1)

    number_of_samples = labels.shape[0]

    if sequence.shape != (
        number_of_samples,
        4,
        501,
    ):
        raise ValueError(
            "Sequence shape must be "
            f"[samples, 4, 501]. Received: {tuple(sequence.shape)}"
        )

    if physicochemical.shape != (
        number_of_samples,
        12,
        500,
    ):
        raise ValueError(
            "Physicochemical shape must be "
            f"[samples, 12, 500]. Received: "
            f"{tuple(physicochemical.shape)}"
        )

    if node_index.shape[0] != number_of_samples:
        raise ValueError(
            "Node-index count does not match sample count."
        )

    unique_labels = set(
        torch.unique(labels).tolist()
    )

    if not unique_labels.issubset(
        {0.0, 1.0}
    ):
        raise ValueError(
            f"Labels must contain only 0 and 1: {unique_labels}"
        )

    if not torch.isfinite(sequence).all():
        raise ValueError(
            "Sequence input contains NaN or infinity."
        )

    if not torch.isfinite(physicochemical).all():
        raise ValueError(
            "Physicochemical input contains NaN or infinity."
        )

    print(f"\nDataset loaded: {file_path}")
    print(f"Samples: {number_of_samples:,}")
    print(f"Sequence shape: {tuple(sequence.shape)}")
    print(
        "Physicochemical shape: "
        f"{tuple(physicochemical.shape)}"
    )
    print(f"Node-index shape: {tuple(node_index.shape)}")
    print(f"Label shape: {tuple(labels.shape)}")

    return TensorDataset(
        sequence,
        physicochemical,
        node_index,
        labels,
    )


def load_node_features(
    file_path: Path,
) -> torch.Tensor:
    """
    Expected shape:

        [number_of_nodes, 768]
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Node-feature file does not exist: {file_path}"
        )

    node_features = np.load(
        file_path,
        allow_pickle=False,
    ).astype(np.float32)

    node_features = torch.from_numpy(
        node_features
    )

    if node_features.ndim != 2:
        raise ValueError(
            "Node features must be a two-dimensional matrix."
        )

    if node_features.shape[1] != 768:
        raise ValueError(
            "Node-feature dimension must be 768. "
            f"Received: {tuple(node_features.shape)}"
        )

    if not torch.isfinite(node_features).all():
        raise ValueError(
            "Node features contain NaN or infinity."
        )

    print(f"\nNode features loaded: {file_path}")
    print(
        f"Node-feature shape: {tuple(node_features.shape)}"
    )

    return node_features


def load_normalized_adjacency(
    file_path: Path,
) -> torch.Tensor:
    """
    Load the normalized sparse Hi-C adjacency matrix.

    The adjacency matrix must already include self-loops and
    ncVarPred-style symmetric normalization.

    This function does not normalize the matrix again.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Adjacency file does not exist: {file_path}"
        )

    adjacency = sp.load_npz(
        file_path
    ).tocoo().astype(np.float32)

    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            "Adjacency matrix must be square. "
            f"Received: {adjacency.shape}"
        )

    indices = np.vstack(
        (
            adjacency.row,
            adjacency.col,
        )
    ).astype(np.int64)

    values = adjacency.data.astype(
        np.float32
    )

    adjacency_tensor = torch.sparse_coo_tensor(
        indices=torch.from_numpy(indices),
        values=torch.from_numpy(values),
        size=adjacency.shape,
        dtype=torch.float32,
    ).coalesce()

    if not torch.isfinite(
        adjacency_tensor.values()
    ).all():
        raise ValueError(
            "Adjacency matrix contains NaN or infinity."
        )

    print(f"\nAdjacency loaded: {file_path}")
    print(
        f"Adjacency shape: {tuple(adjacency_tensor.shape)}"
    )
    print(
        f"Non-zero values: {adjacency_tensor._nnz():,}"
    )

    return adjacency_tensor


def validate_node_indices(
    dataset: TensorDataset,
    number_of_nodes: int,
    split_name: str,
) -> None:
    node_indices = dataset.tensors[2]

    minimum_index = int(
        node_indices.min().item()
    )

    maximum_index = int(
        node_indices.max().item()
    )

    if minimum_index < 0:
        raise ValueError(
            f"{split_name} contains negative node indices."
        )

    if maximum_index >= number_of_nodes:
        raise ValueError(
            f"{split_name} contains an invalid node index. "
            f"Maximum index: {maximum_index}, "
            f"graph node count: {number_of_nodes}"
        )

    print(
        f"{split_name} node-index range: "
        f"{minimum_index}–{maximum_index}"
    )


def create_index_input(
    node_indices: torch.Tensor,
    number_of_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Original ncVarPred node-selection matrix.

    Input:
        node_indices [batch]

    Output:
        index_input [batch, number_of_nodes]
    """

    index_input = torch.zeros(
        (
            node_indices.shape[0],
            number_of_nodes,
        ),
        dtype=torch.float32,
        device=device,
    )

    index_input.scatter_(
        dim=1,
        index=node_indices.to(
            device=device,
            dtype=torch.long,
        ).reshape(-1, 1),
        value=1.0,
    )

    return index_input


def calculate_l1_penalty(
    model: nn.Module,
) -> torch.Tensor:
    penalty = torch.zeros(
        (),
        dtype=torch.float32,
        device=next(model.parameters()).device,
    )

    for parameter in model.parameters():
        if parameter.requires_grad:
            penalty = (
                penalty
                + torch.sum(
                    torch.abs(parameter)
                )
            )

    return penalty


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    labels = labels.reshape(-1).astype(
        np.int64
    )

    probabilities = probabilities.reshape(
        -1
    ).astype(np.float64)

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    results = {
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                labels,
                predictions,
            )
        ),
    }

    if len(np.unique(labels)) == 2:
        results["roc_auc"] = float(
            roc_auc_score(
                labels,
                probabilities,
            )
        )

        results["average_precision"] = float(
            average_precision_score(
                labels,
                probabilities,
            )
        )
    else:
        results["roc_auc"] = float("nan")
        results["average_precision"] = float(
            "nan"
        )

    return results


def forward_model(
    model: DeepMethConcatenation,
    sequence_input: torch.Tensor,
    physicochemical_input: torch.Tensor,
    node_features: torch.Tensor,
    adjacency: torch.Tensor,
    index_input: torch.Tensor,
) -> torch.Tensor:
    probabilities = model(
        seq_input=sequence_input,
        node_input=node_features,
        adj_input=adjacency,
        index_input=index_input,
        physchem_input=physicochemical_input,
    )

    probabilities = probabilities.reshape(
        -1,
        1,
    )

    if not torch.isfinite(
        probabilities
    ).all():
        raise FloatingPointError(
            "Model output contains NaN or infinity."
        )

    return probabilities


def run_epoch(
    model: DeepMethConcatenation,
    data_loader: DataLoader,
    node_features: torch.Tensor,
    adjacency: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    optimizer: Adam | None = None,
    l1_lambda: float = 0.0,
) -> dict[str, float]:
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    number_of_nodes = node_features.shape[0]

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:
        for (
            sequence_input,
            physicochemical_input,
            node_indices,
            labels,
        ) in data_loader:
            sequence_input = sequence_input.to(
                device,
                non_blocking=True,
            )

            physicochemical_input = (
                physicochemical_input.to(
                    device,
                    non_blocking=True,
                )
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            index_input = create_index_input(
                node_indices=node_indices,
                number_of_nodes=number_of_nodes,
                device=device,
            )

            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            probabilities = forward_model(
                model=model,
                sequence_input=sequence_input,
                physicochemical_input=(
                    physicochemical_input
                ),
                node_features=node_features,
                adjacency=adjacency,
                index_input=index_input,
            )

            binary_loss = criterion(
                probabilities,
                labels,
            )

            loss = binary_loss

            if training and l1_lambda > 0.0:
                loss = (
                    loss
                    + l1_lambda
                    * calculate_l1_penalty(model)
                )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Loss contains NaN or infinity."
                )

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.shape[0]

            total_loss += (
                float(loss.item())
                * batch_size
            )

            total_samples += batch_size

            all_labels.append(
                labels.detach()
                .cpu()
                .numpy()
            )

            all_probabilities.append(
                probabilities.detach()
                .cpu()
                .numpy()
            )

    labels_array = np.concatenate(
        all_labels
    )

    probabilities_array = np.concatenate(
        all_probabilities
    )

    results = calculate_metrics(
        labels=labels_array,
        probabilities=probabilities_array,
        threshold=threshold,
    )

    results["loss"] = (
        total_loss / total_samples
    )

    return results


def save_json(
    file_path: Path,
    data: object,
) -> None:
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            allow_nan=True,
        )


def train(
    arguments: argparse.Namespace,
) -> None:
    set_seed(arguments.seed)

    device = torch.device(
        arguments.device
        if torch.cuda.is_available()
        and arguments.device.startswith("cuda")
        else "cpu"
    )

    train_dataset = load_split(
        arguments.train_data
    )

    validation_dataset = load_split(
        arguments.validation_data
    )

    test_dataset = load_split(
        arguments.test_data
    )

    node_features = load_node_features(
        arguments.node_features
    )

    adjacency = load_normalized_adjacency(
        arguments.adjacency
    )

    if adjacency.shape[0] != node_features.shape[0]:
        raise ValueError(
            "Adjacency and node-feature node counts differ. "
            f"Adjacency: {adjacency.shape[0]}, "
            f"node features: {node_features.shape[0]}"
        )

    number_of_nodes = node_features.shape[0]

    validate_node_indices(
        dataset=train_dataset,
        number_of_nodes=number_of_nodes,
        split_name="Training",
    )

    validate_node_indices(
        dataset=validation_dataset,
        number_of_nodes=number_of_nodes,
        split_name="Validation",
    )

    validate_node_indices(
        dataset=test_dataset,
        number_of_nodes=number_of_nodes,
        split_name="Test",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=arguments.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    node_features = node_features.to(
        device
    )

    adjacency = adjacency.to(
        device
    )

    model = DeepMethConcatenation(
        physchem_dropout_prob=(
            arguments.physchem_dropout
        )
    ).to(device)

    criterion = nn.BCELoss()

    optimizer = Adam(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nDeepMeth training configuration")
    print(f"Device: {device}")
    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )
    print(
        f"Graph nodes: {number_of_nodes:,}"
    )
    print(f"Epochs: {arguments.epochs}")
    print(
        f"Batch size: {arguments.batch_size}"
    )
    print(
        f"Learning rate: "
        f"{arguments.learning_rate}"
    )
    print(
        f"L1 lambda: {arguments.l1_lambda}"
    )
    print(
        f"L2 weight decay: "
        f"{arguments.weight_decay}"
    )

    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    history: list[dict[str, float | int]] = []

    arguments.checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(
        1,
        arguments.epochs + 1,
    ):
        training_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            node_features=node_features,
            adjacency=adjacency,
            criterion=criterion,
            device=device,
            threshold=arguments.threshold,
            optimizer=optimizer,
            l1_lambda=arguments.l1_lambda,
        )

        validation_metrics = run_epoch(
            model=model,
            data_loader=validation_loader,
            node_features=node_features,
            adjacency=adjacency,
            criterion=criterion,
            device=device,
            threshold=arguments.threshold,
        )

        epoch_results = {
            "epoch": epoch,
            "train": training_metrics,
            "validation": validation_metrics,
        }

        history.append(
            epoch_results
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: "
            f"{training_metrics['loss']:.6f} | "
            f"Validation Loss: "
            f"{validation_metrics['loss']:.6f} | "
            f"Validation ROC-AUC: "
            f"{validation_metrics['roc_auc']:.4f} | "
            f"Validation AP: "
            f"{validation_metrics['average_precision']:.4f} | "
            f"Validation F1: "
            f"{validation_metrics['f1']:.4f} | "
            f"Validation MCC: "
            f"{validation_metrics['mcc']:.4f}"
        )

        improved = (
            validation_metrics["loss"]
            < best_validation_loss
            - arguments.min_delta
        )

        if improved:
            best_validation_loss = (
                validation_metrics["loss"]
            )

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "validation_metrics": (
                        validation_metrics
                    ),
                    "arguments": vars(arguments),
                },
                arguments.checkpoint,
            )

            print(
                f"Best checkpoint saved: "
                f"{arguments.checkpoint}"
            )
        else:
            epochs_without_improvement += 1

            print(
                "No validation improvement: "
                f"{epochs_without_improvement}/"
                f"{arguments.patience}"
            )

        if (
            epochs_without_improvement
            >= arguments.patience
        ):
            print("Early stopping activated.")
            break

    history_path = (
        arguments.output_dir
        / "training_history.json"
    )

    save_json(
        history_path,
        history,
    )

    checkpoint = torch.load(
        arguments.checkpoint,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_metrics = run_epoch(
        model=model,
        data_loader=test_loader,
        node_features=node_features,
        adjacency=adjacency,
        criterion=criterion,
        device=device,
        threshold=arguments.threshold,
    )

    test_results_path = (
        arguments.output_dir
        / "test_metrics.json"
    )

    save_json(
        test_results_path,
        test_metrics,
    )

    print("\nFinal test results")

    for metric_name, metric_value in (
        test_metrics.items()
    ):
        print(
            f"{metric_name}: "
            f"{metric_value:.6f}"
        )

    print(
        f"\nTraining history saved: "
        f"{history_path}"
    )

    print(
        f"Test metrics saved: "
        f"{test_results_path}"
    )

    print(
        f"Best model saved: "
        f"{arguments.checkpoint}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the real three-branch DeepMeth model."
        )
    )

    parser.add_argument(
        "--train-data",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--validation-data",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--test-data",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--node-features",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--adjacency",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/deepmeth_best.pt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--l1-lambda",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--physchem-dropout",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


