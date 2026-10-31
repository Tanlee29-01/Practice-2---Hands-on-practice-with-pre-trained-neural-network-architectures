"""Metrics for single-label classification, using the existing NumPy dependency.

All scores use the [0, 1] scale. Undefined precision/recall/F1 are zero;
ROC AUC and average precision are None when a class cannot be evaluated.
"""

import numpy as np


def binary_curves(targets, scores):
    """Return tie-aware ROC/PR points, ROC AUC and non-interpolated AP.

    Each distinct score is one threshold: tied scores must enter together.
    PR points are ordered by increasing recall, suitable for plotting.
    """
    targets = np.asarray(targets, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if targets.ndim != 1 or scores.shape != targets.shape:
        raise ValueError("Binary targets and scores must be matching 1D arrays.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite.")
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(-scores, kind="stable")
    sorted_targets = targets[order]
    sorted_scores = scores[order]
    last_at_threshold = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    tp = np.cumsum(sorted_targets)[last_at_threshold].astype(float)
    fp = 1 + last_at_threshold - tp
    tpr = np.r_[0.0, tp / positives]
    fpr = np.r_[0.0, fp / negatives]
    precision = np.r_[1.0, tp / (tp + fp)]
    recall = tpr.copy()
    auc = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)
    ap = np.sum(np.diff(recall) * precision[1:])
    return {
        "fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall,
        "roc_auc": float(auc), "average_precision": float(ap),
    }


def classification_metrics(targets, probabilities, class_names, top_k=5):
    """Compute aggregate/per-class metrics and a confusion matrix.

    Rows of the confusion matrix are true classes; columns are predictions.
    Macro averages include every configured class, including absent classes.
    Macro AUC/AP include only classes with both positive and negative samples.
    """
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities, dtype=float)
    class_names = list(class_names)
    count = len(class_names)
    if count < 2 or len(set(class_names)) != count:
        raise ValueError("Provide at least two unique class names.")
    if targets.ndim != 1 or len(targets) == 0 or targets.dtype.kind not in "iu":
        raise ValueError("Targets must be a non-empty 1D integer array.")
    if probabilities.shape != (len(targets), count):
        raise ValueError("Probabilities must have shape (samples, classes).")
    if np.any(targets < 0) or np.any(targets >= count):
        raise ValueError("Target index is outside the configured classes.")
    if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)):
        raise ValueError("Probabilities must be finite, in [0, 1], and sum to 1.")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    predictions = probabilities.argmax(axis=1)
    matrix = np.bincount(
        count * targets + predictions, minlength=count * count,
    ).reshape(count, count)
    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    tp = matrix.diagonal()
    fp = predicted_count - tp
    fn = support - tp
    tn = len(targets) - tp - fp - fn

    def divide(numerator, denominator):
        return np.divide(numerator, denominator, out=np.zeros(count), where=denominator != 0)

    precision = divide(tp, predicted_count)
    recall = divide(tp, support)
    f1 = divide(2 * precision * recall, precision + recall)
    specificity = divide(tn, tn + fp)
    curves = [binary_curves(targets == i, probabilities[:, i]) for i in range(count)]
    valid_curves = [curve for curve in curves if curve is not None]
    k = min(top_k, count)
    top_indices = np.argsort(-probabilities, axis=1, kind="stable")[:, :k]

    metrics = {
        "num_samples": len(targets),
        "num_classes": count,
        "accuracy": float(tp.sum() / len(targets)),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "top_k": k,
        "top_k_accuracy": float(np.any(top_indices == targets[:, None], axis=1).mean()),
        "roc_auc_macro_ovr": (float(np.mean([c["roc_auc"] for c in valid_curves]))
                              if valid_curves else None),
        "average_precision_macro": (float(np.mean([c["average_precision"] for c in valid_curves]))
                                    if valid_curves else None),
        "auc_valid_classes": len(valid_curves),
        "confusion_matrix": matrix.tolist(),
        "per_class": [],
        "warnings": [],
    }
    for name, values in (("precision", precision), ("recall", recall), ("f1", f1)):
        metrics[f"{name}_macro"] = float(values.mean())
        metrics[f"{name}_weighted"] = float(np.average(values, weights=support))
    for i, name in enumerate(class_names):
        curve = curves[i]
        metrics["per_class"].append({
            "class_index": i, "class_name": name, "support": int(support[i]),
            "precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]), "specificity": float(specificity[i]),
            "tp": int(tp[i]), "fp": int(fp[i]), "fn": int(fn[i]), "tn": int(tn[i]),
            "roc_auc_ovr": curve["roc_auc"] if curve else None,
            "average_precision": curve["average_precision"] if curve else None,
        })
        if curve is None:
            metrics["warnings"].append(
                f"{name}: ROC AUC/AP unavailable (requires positive and negative samples)."
            )
    return metrics
