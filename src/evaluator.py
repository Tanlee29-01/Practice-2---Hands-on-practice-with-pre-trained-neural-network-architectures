"""Reusable evaluation loop, independent of the optimizer and training logger."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from src.classification_metrics import classification_metrics


@dataclass
class EvaluationResult:
    metrics: dict
    targets: np.ndarray
    probabilities: np.ndarray

    @property
    def predictions(self):
        return self.probabilities.argmax(axis=1)


class ClassificationEvaluator:
    """Evaluate a classifier returning logits of shape (batch, classes)."""

    def __init__(self, model, class_names, device="cpu", top_k=5):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.class_names = list(class_names)
        self.top_k = top_k

    def evaluate(self, dataloader, progress=False):
        training_modes = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        labels_all, probabilities_all = [], []
        total_loss = 0.0
        total = 0
        try:
            with torch.inference_mode():
                for step, (images, labels) in enumerate(dataloader, start=1):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    if labels.ndim != 1 or labels.dtype != torch.long:
                        raise ValueError("Labels must be a 1D torch.long tensor.")
                    logits = self.model(images)
                    if (not isinstance(logits, torch.Tensor)
                            or logits.ndim != 2 or logits.shape[0] != labels.shape[0]
                            or logits.shape[1] != len(self.class_names)):
                        raise ValueError("Model output must have shape (batch, classes).")
                    if not torch.isfinite(logits).all():
                        raise ValueError("Model returned non-finite logits.")
                    if torch.any(labels < 0) or torch.any(labels >= len(self.class_names)):
                        raise ValueError("Label index is outside the configured classes.")
                    total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
                    total += labels.numel()
                    labels_all.append(labels.cpu().numpy())
                    probabilities_all.append(logits.softmax(dim=1).cpu().numpy())
                    if progress and (step == 1 or step % 25 == 0):
                        print(f"Evaluated {total} samples...", flush=True)
        finally:
            # Restore mixed modes too, e.g. a training head with frozen BatchNorm.
            for module, was_training in training_modes:
                module.training = was_training
        if total == 0:
            raise ValueError("Evaluation dataloader is empty.")

        targets = np.concatenate(labels_all)
        probabilities = np.concatenate(probabilities_all)
        metrics = classification_metrics(targets, probabilities, self.class_names, self.top_k)
        metrics["loss"] = total_loss / total
        return EvaluationResult(metrics, targets, probabilities)
