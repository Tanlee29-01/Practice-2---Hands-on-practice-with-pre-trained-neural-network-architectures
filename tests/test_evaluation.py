import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation import evaluate_model


class FixedClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.linear = nn.Linear(2, num_classes)

    def forward(self, images):
        return self.linear(images)


def make_three_class_loader():
    images = torch.tensor([
        [3.0, 0.0],
        [0.0, 3.0],
        [-3.0, 0.0],
        [2.0, 0.0],
        [0.0, 2.0],
        [-2.0, 0.0],
    ])
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    return DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)


def make_model():
    model = FixedClassifier(num_classes=3)
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([
            [2.0, 0.0],
            [0.0, 2.0],
            [-2.0, 0.0],
        ]))
        model.linear.bias.zero_()
    return model


def test_evaluation_returns_metrics_and_creates_artifacts(tmp_path):
    model = make_model()
    model.train()

    report = evaluate_model(
        model=model,
        dataloader=make_three_class_loader(),
        device="cpu",
        class_names=("class_a", "class_b", "class_c"),
        output_dir=tmp_path,
        metadata={"checkpoint": "tiny-test.pth", "seed": 42},
    )

    assert report["num_samples"] == 6
    assert len(report["confusion_matrix"]) == 3
    assert all(len(row) == 3 for row in report["confusion_matrix"])
    assert sum(map(sum, report["confusion_matrix"])) == 6
    assert 0.0 <= report["metrics"]["accuracy"] <= 1.0
    assert set(report["metrics"]["per_class"]) == {
        "class_a", "class_b", "class_c"
    }
    assert set(report["metrics"]["macro_avg"]) == {
        "precision", "recall", "f1_score", "auc_roc"
    }
    assert set(report["metrics"]["weighted_avg"]) == {
        "precision", "recall", "f1_score"
    }
    assert all(
        0.0 <= values["auc_roc"] <= 1.0
        for values in report["metrics"]["per_class"].values()
    )

    artifact_dir = Path(report["artifact_dir"])
    assert (artifact_dir / "evaluation_report.json").is_file()
    assert (artifact_dir / "metrics_per_class.csv").is_file()
    assert (artifact_dir / "confusion_matrix.png").is_file()
    assert (artifact_dir / "roc_curve.png").is_file()

    saved_report = json.loads(
        (artifact_dir / "evaluation_report.json").read_text(encoding="utf-8")
    )
    assert saved_report["metadata"]["seed"] == 42
    assert model.training  # Original train/eval mode is restored.


def test_evaluation_does_not_overwrite_previous_artifacts(tmp_path):
    kwargs = {
        "model": make_model(),
        "dataloader": make_three_class_loader(),
        "device": "cpu",
        "class_names": ("class_a", "class_b", "class_c"),
        "output_dir": tmp_path,
    }

    first = evaluate_model(**kwargs)
    second = evaluate_model(**kwargs)

    assert first["artifact_dir"] != second["artifact_dir"]
    assert Path(first["artifact_dir"]).exists()
    assert Path(second["artifact_dir"]).exists()


def test_evaluation_rejects_missing_class_for_auc(tmp_path):
    images = torch.randn(4, 2)
    labels = torch.tensor([0, 0, 1, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    with pytest.raises(ValueError, match="Missing true-label classes"):
        evaluate_model(
            make_model(),
            loader,
            class_names=("class_a", "class_b", "class_c"),
            output_dir=tmp_path,
        )


def test_evaluation_rejects_out_of_range_labels(tmp_path):
    images = torch.randn(2, 2)
    labels = torch.tensor([0, 3])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    with pytest.raises(ValueError, match="Labels must be in the range"):
        evaluate_model(
            make_model(),
            loader,
            class_names=("class_a", "class_b", "class_c"),
            output_dir=tmp_path,
        )


def test_evaluation_rejects_wrong_model_output_shape(tmp_path):
    class WrongOutputModel(nn.Module):
        def forward(self, images):
            return torch.randn(images.shape[0], 2)

    with pytest.raises(ValueError, match="Expected 3 output classes"):
        evaluate_model(
            WrongOutputModel(),
            make_three_class_loader(),
            class_names=("class_a", "class_b", "class_c"),
            output_dir=tmp_path,
        )


def test_evaluation_rejects_non_finite_logits(tmp_path):
    class NonFiniteModel(nn.Module):
        def forward(self, images):
            output = torch.zeros(images.shape[0], 3)
            output[0, 0] = float("nan")
            return output

    with pytest.raises(ValueError, match="NaN or Inf"):
        evaluate_model(
            NonFiniteModel(),
            make_three_class_loader(),
            class_names=("class_a", "class_b", "class_c"),
            output_dir=tmp_path,
        )


def test_evaluation_rejects_empty_loader(tmp_path):
    empty_loader = DataLoader(
        TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)),
        batch_size=2,
    )

    with pytest.raises(ValueError, match="produced no samples"):
        evaluate_model(
            make_model(),
            empty_loader,
            class_names=("class_a", "class_b", "class_c"),
            output_dir=tmp_path,
        )
