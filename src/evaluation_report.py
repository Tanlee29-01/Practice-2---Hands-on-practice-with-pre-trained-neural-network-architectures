"""Export classification metrics, per-image predictions and publication files."""

import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from src.classification_metrics import binary_curves


def _figure():
    figure = Figure(figsize=(10, 8), layout="constrained")
    FigureCanvasAgg(figure)
    return figure, figure.subplots()


def _save_plots(result, class_names, output_dir):
    matrix = np.asarray(result.metrics["confusion_matrix"])
    row_total = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_total, out=np.zeros_like(matrix, dtype=float), where=row_total != 0)
    for name, values in (("confusion_matrix", matrix), ("confusion_matrix_normalized", normalized)):
        figure, ax = _figure()
        heatmap = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if name.endswith("normalized") else None)
        figure.colorbar(heatmap, ax=ax)
        ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
               xticklabels=class_names, yticklabels=class_names,
               xlabel="Predicted class", ylabel="True class", title=name.replace("_", " ").title())
        ax.tick_params(axis="x", labelrotation=45)
        for row in range(len(class_names)):
            for col in range(len(class_names)):
                value = values[row, col]
                text = f"{value:.2f}" if name.endswith("normalized") else str(int(value))
                ax.text(col, row, text, ha="center", va="center",
                        color="white" if value > values.max() / 2 else "black", fontsize=8)
        figure.savefig(output_dir / f"{name}.png", dpi=160)
        figure.clear()

    roc_figure, roc_ax = _figure()
    pr_figure, pr_ax = _figure()
    plotted = 0
    for index, class_name in enumerate(class_names):
        curve = binary_curves(result.targets == index, result.probabilities[:, index])
        if curve is None:
            continue
        plotted += 1
        roc_ax.plot(curve["fpr"], curve["tpr"], label=f"{class_name} (AUC={curve['roc_auc']:.3f})")
        pr_ax.step(curve["recall"], curve["precision"], where="pre",
                   label=f"{class_name} (AP={curve['average_precision']:.3f})")
    roc_ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    roc_ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC: one vs rest",
               xlim=(0, 1), ylim=(0, 1.02))
    pr_ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall: one vs rest",
              xlim=(0, 1), ylim=(0, 1.02))
    if plotted:
        roc_ax.legend(loc="lower right", fontsize=8)
        pr_ax.legend(loc="best", fontsize=8)
    else:
        for ax in (roc_ax, pr_ax):
            ax.text(0.5, 0.5, "Curves unavailable: only one target class present", ha="center", wrap=True)
    for figure, name in ((roc_figure, "roc_curves"), (pr_figure, "precision_recall_curves")):
        figure.savefig(output_dir / f"{name}.png", dpi=160)
        figure.clear()


def save_evaluation_report(result, output_dir, class_names, metadata=None, sample_indices=None, plots=True):
    """Write JSON, TXT, CSV and optional PNGs; return the output directory."""
    output_dir = Path(output_dir)
    class_names = list(class_names)
    if sample_indices is None:
        sample_indices = np.arange(len(result.targets))
    if len(sample_indices) != len(result.targets):
        raise ValueError("sample_indices must match the number of evaluated samples.")
    output_dir.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1, "score_scale": "0..1",
        "class_names": class_names, "metadata": metadata or {}, "metrics": result.metrics,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(document, file, ensure_ascii=False, indent=2, allow_nan=False)

    rows = result.metrics["per_class"]
    with (output_dir / "per_class_metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "confusion_matrix.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted", *class_names])
        for name, row in zip(class_names, result.metrics["confusion_matrix"]):
            writer.writerow([name, *row])
    with (output_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_index", "true_index", "true_class", "predicted_index",
                         "predicted_class", "confidence", "correct",
                         *[f"probability_{name}" for name in class_names]])
        for index, target, predicted, probabilities in zip(
                sample_indices, result.targets, result.predictions, result.probabilities):
            writer.writerow([int(index), int(target), class_names[target], int(predicted),
                             class_names[predicted], float(probabilities[predicted]),
                             int(target == predicted), *map(float, probabilities)])

    metrics = result.metrics
    lines = ["CLASSIFICATION EVALUATION", "Scores use 0..1; accuracy below is displayed as percent.",
             f"Samples: {metrics['num_samples']}", f"Cross-entropy loss: {metrics['loss']:.6f}",
             f"Accuracy: {metrics['accuracy']:.2%}",
             f"Balanced accuracy: {metrics['balanced_accuracy']:.2%}",
             f"Top-{metrics['top_k']} accuracy: {metrics['top_k_accuracy']:.2%}", "",
             f"{'Class':<18} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>9}"]
    for row in rows:
        lines.append(f"{row['class_name']:<18} {row['precision']:>10.4f} {row['recall']:>10.4f} "
                     f"{row['f1']:>10.4f} {row['support']:>9}")
    for average in ("macro", "weighted"):
        lines.append(f"{average + ' avg':<18} {metrics['precision_' + average]:>10.4f} "
                     f"{metrics['recall_' + average]:>10.4f} {metrics['f1_' + average]:>10.4f} "
                     f"{metrics['num_samples']:>9}")
    lines.extend(["", f"Macro ROC AUC (OvR): {metrics['roc_auc_macro_ovr']}",
                  f"Macro average precision: {metrics['average_precision_macro']}",
                  f"Classes with defined AUC/AP: {metrics['auc_valid_classes']}/{len(class_names)}",
                  "", "Conventions: undefined precision/recall/F1 = 0; undefined AUC/AP = null.",
                  "Macro P/R/F1 includes all configured classes; balanced accuracy uses present classes.",
                  "Macro AUC/AP averages only classes with both positive and negative samples.",
                  "Top-k ties use ascending class index. Confusion rows=true, columns=predicted.",
                  "", "Metadata:", json.dumps(metadata or {}, ensure_ascii=False, indent=2)])
    if metrics["warnings"]:
        lines.extend(["", "Warnings:", *metrics["warnings"]])
    (output_dir / "classification_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if plots:
        _save_plots(result, class_names, output_dir)
    return output_dir
