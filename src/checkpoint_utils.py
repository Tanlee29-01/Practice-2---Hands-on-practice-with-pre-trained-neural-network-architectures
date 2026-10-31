"""Load existing project checkpoints without downloading ImageNet weights."""

from collections.abc import Mapping
from pathlib import Path
import pickle

import torch

from src.model_builder import TransferModelBuilder


def find_checkpoint(directory):
    """Prefer the newest *_best.pth, then the newest .pth or .pt file."""
    directory = Path(directory)
    best = [path for path in directory.glob("*_best.pth") if path.is_file()]
    candidates = best or [path for pattern in ("*.pth", "*.pt")
                          for path in directory.glob(pattern) if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint in {directory}. Run python main.py or pass --checkpoint PATH."
        )
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def load_classifier(path, model_name=None):
    """Support logger checkpoints, wrapped state_dicts and raw state_dicts.

    Architecture and class count come from classifier keys/shapes, so historical
    checkpoints lacking metadata still work. All weights must match strictly.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError) as exc:
        raise ValueError(f"Cannot read checkpoint weights from {path}: {exc}") from exc
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must contain a state_dict mapping.")
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint state_dict is empty or invalid.")
    if not all(isinstance(key, str) for key in state):
        raise ValueError("Checkpoint state_dict keys must be strings.")
    if all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}

    if "fc.weight" in state:
        detected, head_key = "resnet18", "fc.weight"
    elif "classifier.6.weight" in state:
        detected, head_key = "vgg16", "classifier.6.weight"
    elif "classifier.weight" in state:
        detected, head_key = "densenet121", "classifier.weight"
    else:
        raise ValueError("Cannot identify ResNet18/VGG16/DenseNet121 classifier weights.")
    if model_name is not None:
        normalized = model_name.lower().replace("-", "").replace("_", "").strip()
        if normalized != detected:
            raise ValueError(f"Requested {model_name}, but checkpoint contains {detected} weights.")
    head = state[head_key]
    if not isinstance(head, torch.Tensor) or head.ndim != 2 or head.shape[0] < 2:
        raise ValueError("Checkpoint classifier weight must have shape (classes, features).")
    num_classes = int(head.shape[0])
    model = TransferModelBuilder(
        model_name=detected, num_classes=num_classes, pretrained=False,
    ).build()
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Checkpoint weights do not match {detected}: {exc}") from exc
    metadata = {
        "checkpoint": str(path.resolve()), "model_name": detected,
        "num_classes": num_classes,
        "checkpoint_epoch": checkpoint.get("epoch") if state is not checkpoint else None,
    }
    if "class_names" in checkpoint:
        metadata["checkpoint_class_names"] = list(checkpoint["class_names"])
    return model, metadata
