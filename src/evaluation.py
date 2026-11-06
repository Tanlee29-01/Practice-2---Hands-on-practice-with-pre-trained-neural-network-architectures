"""Evaluation utilities for CIFAR-10 classification."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


CIFAR10_CLASSES = ("airplane","automobile","bird","cat","deer","dog","frog","horse","ship", "truck")


def _validate_inputs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> None:
    if logits.ndim != 2:
        raise ValueError(
            f"Model output must have shape [N, C], received {tuple(logits.shape)}."
        )
    if logits.shape[1] != num_classes:
        raise ValueError(
            f"Expected {num_classes} output classes, received {logits.shape[1]}."
        )
    if labels.ndim != 1:
        raise ValueError(
            f"Labels must have shape [N], received {tuple(labels.shape)}."
        )
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("Logit and label batch sizes do not match.")
    if labels.numel() == 0:
        raise ValueError("Evaluation batch must not be empty.")
    if not torch.isfinite(logits).all():
        raise ValueError("Model logits contain NaN or Inf.")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError(f"Labels must have an integer dtype, received {labels.dtype}.")
    if torch.any(labels < 0) or torch.any(labels >= num_classes):
        raise ValueError(f"Labels must be in the range [0, {num_classes - 1}].")


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide with zero_division=0 semantics."""
    result = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator,denominator,out=result,where=denominator != 0)
    return result


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int,
    class_names: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]]]:
    # Rows are true labels; columns are predicted labels.
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)

    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    true_positive = np.diag(confusion).astype(np.float64)

    precision = _safe_divide(true_positive, predicted_count)
    recall = _safe_divide(true_positive, support)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    total = int(support.sum())
    if total == 0:
        raise ValueError("The evaluation dataset is empty.")

    # AUC-ROC is undefined if a class has no positive or no negative samples.
    missing_classes = [
        class_names[index]
        for index in range(num_classes)
        if support[index] == 0
    ]
    if missing_classes:
        raise ValueError(
            "AUC-ROC requires at least one positive sample for every class. "
            f"Missing true-label classes: {', '.join(missing_classes)}."
        )

    auc_by_class: dict[str, float] = {}
    roc_points: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for class_index, class_name in enumerate(class_names):
        binary_true = (y_true == class_index).astype(np.int64)
        # Sorting probability scores produces the one-vs-rest ROC curve.
        order = np.argsort(-probabilities[:, class_index], kind="mergesort")
        sorted_true = binary_true[order]
        sorted_scores = probabilities[order, class_index]

        # Evaluate only at the end of each tied-score group.
        distinct_end = np.r_[
            np.where(np.diff(sorted_scores))[0],
            len(sorted_scores) - 1,
        ]
        true_positive_cumulative = np.cumsum(sorted_true)[distinct_end]
        false_positive_cumulative = (
            1 + distinct_end - true_positive_cumulative
        )

        tpr = np.r_[0.0, true_positive_cumulative / support[class_index]]
        negatives = total - support[class_index]
        fpr = np.r_[0.0, false_positive_cumulative / negatives]

        # Include the endpoint (1, 1).
        if fpr[-1] != 1.0 or tpr[-1] != 1.0:
            fpr = np.r_[fpr, 1.0]
            tpr = np.r_[tpr, 1.0]

        auc_value = float(np.sum((tpr[1:] + tpr[:-1]) * np.diff(fpr) / 2.0))
        if not np.isfinite(auc_value):
            raise ValueError(f"AUC for class '{class_name}' is not finite.")

        auc_by_class[class_name] = auc_value
        roc_points[class_name] = (fpr, tpr)

    macro = {
        "precision": float(np.mean(precision)),
        "recall": float(np.mean(recall)),
        "f1_score": float(np.mean(f1)),
        "auc_roc": float(np.mean(list(auc_by_class.values()))),
    }

    weighted = {
        "precision": float(np.average(precision, weights=support)),
        "recall": float(np.average(recall, weights=support)),
        "f1_score": float(np.average(f1, weights=support)),
    }

    per_class = {}
    for index, class_name in enumerate(class_names):
        per_class[class_name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1_score": float(f1[index]),
            "auc_roc": auc_by_class[class_name],
            "support": int(support[index]),
        }

    metrics = {
        "accuracy": float(true_positive.sum() / total),
        "macro_avg": macro,
        "weighted_avg": weighted,
        "per_class": per_class,
        "zero_division": 0,
        "auc_method": "one-vs-rest; trapezoidal area under ROC using softmax scores",
    }

    numeric_values = [
        metrics["accuracy"],
        *macro.values(),
        *weighted.values(),
        *[
            value
            for class_metrics in per_class.values()
            for key, value in class_metrics.items()
            if key != "support"
        ],
    ]
    if not np.isfinite(numeric_values).all():
        raise ValueError("One or more evaluation metrics are NaN or Inf.")

    return confusion, metrics, roc_points


def _create_artifact_directory(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    artifact_dir = root / f"evaluation_{timestamp}"
    # exist_ok=False guarantees an existing evaluation is never overwritten.
    artifact_dir.mkdir(parents=False, exist_ok=False)
    return artifact_dir


def _save_confusion_matrix(
    confusion: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    normalized = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)

    figure, axes = plt.subplots(1, 2, figsize=(17, 7))
    matrices = (
        (confusion, "Confusion Matrix (counts)", "d"),
        (normalized, "Confusion Matrix (row-normalized)", ".2f"),
    )

    for axis, (matrix, title, value_format) in zip(axes, matrices):
        image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
        axis.set_title(title)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        axis.set_yticks(range(len(class_names)), class_names)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

        threshold = float(matrix.max()) / 2.0
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    format(matrix[row, column], value_format),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                    fontsize=7,
                )

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_roc_curve(
    roc_points: dict[str, tuple[np.ndarray, np.ndarray]],
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 7))

    for class_name, (false_positive_rate, true_positive_rate) in roc_points.items():
        auc_value = metrics["per_class"][class_name]["auc_roc"]
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{class_name} (AUC = {auc_value:.3f})",
            linewidth=1.5,
        )

    axis.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(
        f"Multiclass ROC Curve (OvR, macro AUC = "
        f"{metrics['macro_avg']['auc_roc']:.3f})"
    )
    axis.legend(loc="lower right", fontsize=8)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_metrics_csv(
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    fieldnames = [
        "class",
        "precision",
        "recall",
        "f1_score",
        "auc_roc",
        "support",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for class_name, values in metrics["per_class"].items():
            writer.writerow({"class": class_name, **values})

        for average_name in ("macro_avg", "weighted_avg"):
            values = metrics[average_name]
            writer.writerow({
                "class": average_name,
                **values,
                "support": sum(
                    item["support"] for item in metrics["per_class"].values()
                ),
            })


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str = "cpu",
    class_names: Sequence[str] = CIFAR10_CLASSES,
    output_dir: str | Path = "evaluation_results",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a locked model and save metrics, figures, and metadata.

    The model is not updated. A fresh timestamped directory is created for each
    call so existing evaluation artifacts are never overwritten.
    """
    if len(class_names) < 2 or len(set(class_names)) != len(class_names):
        raise ValueError("class_names must contain at least two unique names.")

    num_classes = len(class_names)
    device = torch.device(device)
    model = model.to(device)
    was_training = model.training

    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    try:
        model.eval()
        with torch.inference_mode():
            for batch in dataloader:
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    raise ValueError("Each dataloader batch must contain images and labels.")

                images, labels = batch[0].to(device), batch[1].to(device)
                logits = model(images)
                _validate_inputs(logits, labels, num_classes)

                probabilities = torch.softmax(logits, dim=1)
                if not torch.isfinite(probabilities).all():
                    raise ValueError("Softmax scores contain NaN or Inf.")

                all_labels.append(labels.detach().cpu().numpy().astype(np.int64))
                all_probabilities.append(probabilities.detach().cpu().numpy())
    finally:
        model.train(was_training)

    if not all_labels:
        raise ValueError("The evaluation dataloader produced no samples.")

    y_true = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probabilities, axis=0)
    if probabilities.shape != (len(y_true), num_classes):
        raise ValueError(
            f"Expected score shape {(len(y_true), num_classes)}, "
            f"received {probabilities.shape}."
        )

    y_pred = probabilities.argmax(axis=1)
    confusion, metrics, roc_points = _compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        num_classes=num_classes,
        class_names=class_names,
    )

    artifact_dir = _create_artifact_directory(output_dir)
    metadata = dict(metadata or {})
    report = {
        "metadata": metadata,
        "num_samples": int(len(y_true)),
        "num_classes": num_classes,
        "class_names": list(class_names),
        "confusion_matrix": confusion.tolist(),
        "metrics": metrics,
    }

    report_path = artifact_dir / "evaluation_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    _save_metrics_csv(metrics, artifact_dir / "metrics_per_class.csv")
    _save_confusion_matrix(
        confusion,
        class_names,
        artifact_dir / "confusion_matrix.png",
    )
    _save_roc_curve(
        roc_points,
        metrics,
        artifact_dir / "roc_curve.png",
    )

    report["artifact_dir"] = str(artifact_dir.resolve())
    report["report_path"] = str(report_path.resolve())
    return report


def _load_checkpoint_state(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Checkpoints produced by this project contain tensors and primitive values.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Invalid checkpoint: expected a dictionary containing model_state_dict."
        )
    return checkpoint


def main() -> None:
    """Run final test evaluation from a saved project checkpoint."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CIFAR-10 checkpoint on the test split."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_results"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    # Imports are local so metric utilities can be used without loading project data.
    import yaml
    from src.data_module import CIFAR10DataModule
    from src.model_builder import TransferModelBuilder

    config_path = args.config
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or "model_name" not in config:
        raise ValueError("Config must be a YAML mapping containing model_name.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    else:
        device = torch.device(args.device)

    checkpoint_path = args.checkpoint
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path
    checkpoint = _load_checkpoint_state(checkpoint_path, device)

    model = TransferModelBuilder(
        model_name=str(config["model_name"]),
        num_classes=len(CIFAR10_CLASSES),
        # The checkpoint already contains the trained weights; avoid a
        # redundant ImageNet download before loading it.
        pretrained=False,
    ).build()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    data_module = CIFAR10DataModule(
        data_dir=str(args.data_dir),
        batch_size=int(config.get("batch_size", 64)),
        num_workers=0,
    )
    data_module.setup()

    metadata = {
        "model_name": str(config["model_name"]),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metrics": checkpoint.get("metrics"),
        "seed": config.get("seed"),
        "device": str(device),
        "split": "CIFAR-10 official test split",
        "batch_size": int(config.get("batch_size", 64)),
    }
    report = evaluate_model(
        model=model,
        dataloader=data_module.test_dataloader(),
        device=device,
        class_names=CIFAR10_CLASSES,
        output_dir=args.output_dir,
        metadata=metadata,
    )

    metrics = report["metrics"]
    print(f"Evaluation samples: {report['num_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_avg']['precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_avg']['recall']:.4f}")
    print(f"Macro F1: {metrics['macro_avg']['f1_score']:.4f}")
    print(f"Macro AUC-ROC: {metrics['macro_avg']['auc_roc']:.4f}")
    print(f"Artifacts: {report['artifact_dir']}")


if __name__ == "__main__":
    main()
