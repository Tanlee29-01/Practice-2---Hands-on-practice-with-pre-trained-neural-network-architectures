"""Evaluate an existing CIFAR-10 checkpoint: python evaluate.py."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.checkpoint_utils import find_checkpoint, load_classifier
from src.data_module import CIFAR10DataModule
from src.evaluation_report import save_evaluation_report
from src.evaluator import ClassificationEvaluator


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "evaluation.yaml"


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_evaluation_config(path):
    with project_path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    required = {"checkpoint", "checkpoint_dir", "model_name", "data_dir", "output_dir",
                "batch_size", "num_workers", "image_size", "device", "download",
                "max_samples", "seed", "top_k", "plots"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f"Evaluation config must contain these keys: {', '.join(sorted(required))}")
    return config


def validate_config(config):
    for key in ("batch_size", "image_size", "top_k"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("num_workers", "max_samples", "seed"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a non-negative integer.")
    for key in ("download", "plots"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be a boolean.")
    for key in ("checkpoint_dir", "data_dir", "output_dir"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a non-empty path string.")
    for key in ("checkpoint", "model_name"):
        if config[key] is not None and (not isinstance(config[key], str) or not config[key].strip()):
            raise ValueError(f"{key} must be null or a non-empty string.")
    if not isinstance(config["device"], str) or config["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda.")


def run_evaluation(config):
    """Connect project data/model modules to the independent evaluation modules."""
    validate_config(config)
    device_name = config["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu or auto.")
    checkpoint_path = (project_path(config["checkpoint"]) if config["checkpoint"] else
                       find_checkpoint(project_path(config["checkpoint_dir"])))
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    model, metadata = load_classifier(checkpoint_path, model_name=config["model_name"])
    if metadata["num_classes"] != 10:
        raise ValueError("CIFAR-10 evaluation requires a checkpoint with exactly 10 output classes.")
    print(f"Model: {metadata['model_name']} | Device: {device_name}", flush=True)
    data = CIFAR10DataModule(
        data_dir=str(project_path(config["data_dir"])), batch_size=config["batch_size"],
        num_workers=config["num_workers"], image_size=config["image_size"],
    )
    try:
        data.setup(stage="test", download=config["download"])
    except RuntimeError as exc:
        raise RuntimeError(f"Cannot load CIFAR-10 at {data.data_dir}. "
                           "Use --download to download missing data. " + str(exc)) from exc
    class_names = data.test_dataset.classes
    if ("checkpoint_class_names" in metadata
            and metadata["checkpoint_class_names"] != class_names):
        raise ValueError("Checkpoint class order does not match the CIFAR-10 class order.")
    indices = np.arange(len(data.test_dataset))
    limit = config["max_samples"]
    if 0 < limit < len(indices):
        indices = np.sort(np.random.default_rng(config["seed"]).choice(indices, size=limit, replace=False))
    loader = DataLoader(
        Subset(data.test_dataset, indices.tolist()), batch_size=config["batch_size"],
        shuffle=False, num_workers=config["num_workers"], pin_memory=device_name == "cuda",
    )
    print(f"Evaluating {len(indices)}/{len(data.test_dataset)} test images...", flush=True)
    result = ClassificationEvaluator(model, class_names, device_name, config["top_k"]).evaluate(
        loader, progress=True,
    )
    timestamp = datetime.now().astimezone()
    metadata.update({
        "dataset": "CIFAR-10", "split": "test", "device": device_name,
        "evaluated_at": timestamp.isoformat(), "image_size": config["image_size"],
        "normalization_mean": [0.485, 0.456, 0.406], "normalization_std": [0.229, 0.224, 0.225],
        "batch_size": config["batch_size"], "seed": config["seed"],
        "full_test_set": len(indices) == len(data.test_dataset),
        "sample_selection": "full" if len(indices) == len(data.test_dataset) else "seeded random subset",
        "sample_indices_sha256": hashlib.sha256(json.dumps(indices.tolist()).encode("utf-8")).hexdigest(),
        "torch_version": str(torch.__version__),
        "evaluation_note": (
            "main.py selects best checkpoints using test accuracy. If this checkpoint was selected "
            "that way, this is a descriptive test report, not an independent generalization estimate. "
            "Select checkpoints on validation data for a final held-out test evaluation."
        ),
    })
    output_dir = project_path(config["output_dir"]) / (
        f"{checkpoint_path.stem}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
    )
    save_evaluation_report(result, output_dir, class_names, metadata, indices, plots=config["plots"])
    print(f"Loss: {result.metrics['loss']:.4f} | Accuracy: {result.metrics['accuracy']:.2%} | "
          f"Macro F1: {result.metrics['f1_macro']:.4f}", flush=True)
    print(f"Reports: {output_dir}", flush=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Evaluate a CIFAR-10 classification checkpoint.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    for name in ("checkpoint", "checkpoint-dir", "model-name", "data-dir", "output-dir"):
        parser.add_argument(f"--{name}", default=None)
    for name in ("batch-size", "num-workers", "image-size", "max-samples", "seed", "top-k"):
        parser.add_argument(f"--{name}", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--download", action="store_true", default=None)
    parser.add_argument("--no-plots", dest="plots", action="store_false", default=None)
    args = parser.parse_args()
    try:
        config = load_evaluation_config(args.config)
        config.update({key: value for key, value in vars(args).items() if key != "config" and value is not None})
        run_evaluation(config)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()
