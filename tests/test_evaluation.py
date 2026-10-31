"""Offline tests: python -m unittest discover -s tests -p test_evaluation.py."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from evaluate import DEFAULT_CONFIG, load_evaluation_config, run_evaluation, validate_config
from src.checkpoint_utils import find_checkpoint, load_classifier
from src.classification_metrics import binary_curves, classification_metrics
from src.data_module import CIFAR10DataModule
from src.evaluation_report import save_evaluation_report
from src.evaluator import ClassificationEvaluator


class TinyClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(3 * 4 * 4, num_classes)

    def forward(self, images):
        return self.fc(images.flatten(1))


class MetricsTests(unittest.TestCase):
    def test_known_confusion_and_averages(self):
        probabilities = [[.8, .1, .1], [.2, .7, .1], [.1, .8, .1], [.1, .2, .7], [.1, .2, .7]]
        metrics = classification_metrics([0, 0, 1, 1, 2], probabilities, ["a", "b", "c"], top_k=2)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1, 0], [0, 1, 1], [0, 0, 1]])
        self.assertAlmostEqual(metrics["accuracy"], .6)
        self.assertAlmostEqual(metrics["precision_macro"], 2 / 3)
        self.assertAlmostEqual(metrics["recall_macro"], 2 / 3)
        self.assertAlmostEqual(metrics["f1_macro"], 11 / 18)
        self.assertAlmostEqual(metrics["precision_weighted"], .7)
        self.assertAlmostEqual(metrics["f1_weighted"], .6)
        self.assertEqual(metrics["per_class"][0]["support"], 2)
        self.assertAlmostEqual(metrics["per_class"][1]["specificity"], 2 / 3)
        self.assertEqual(metrics["top_k_accuracy"], 1.0)

    def test_binary_curves_handle_ties_and_average_precision(self):
        tied = binary_curves([False, True, False, True], [.5, .5, .5, .5])
        self.assertAlmostEqual(tied["roc_auc"], .5)
        self.assertAlmostEqual(tied["average_precision"], .5)
        ranked = binary_curves([False, False, True, True], [.1, .4, .35, .8])
        self.assertAlmostEqual(ranked["roc_auc"], .75)
        self.assertAlmostEqual(ranked["average_precision"], 5 / 6)
        np.testing.assert_array_equal(ranked["fpr"][[0, -1]], [0, 1])
        np.testing.assert_array_equal(ranked["tpr"][[0, -1]], [0, 1])

    def test_missing_classes_are_explicit_and_json_safe(self):
        metrics = classification_metrics([0, 0], [[1., 0., 0.], [1., 0., 0.]], ["a", "b", "c"])
        self.assertEqual(metrics["f1_macro"], 1 / 3)
        self.assertEqual(metrics["balanced_accuracy"], 1)
        self.assertIsNone(metrics["roc_auc_macro_ovr"])
        self.assertEqual(metrics["auc_valid_classes"], 0)
        self.assertEqual(metrics["per_class"][1]["precision"], 0)
        self.assertEqual(len(metrics["warnings"]), 3)
        json.dumps(metrics, allow_nan=False)

    def test_invalid_inputs_fail_clearly(self):
        cases = [([], [], ["a", "b"]), ([2], [[.5, .5]], ["a", "b"]),
                 ([0], [[float("nan"), .5]], ["a", "b"]),
                 ([0], [[.4, .4]], ["a", "b"]), ([0], [[.5, .5]], ["a", "a"])]
        for targets, probabilities, classes in cases:
            with self.subTest(targets=targets, probabilities=probabilities):
                with self.assertRaises(ValueError):
                    classification_metrics(targets, probabilities, classes)


class EvaluatorTests(unittest.TestCase):
    def test_loss_weights_unequal_batches_and_disables_gradients(self):
        logits = torch.tensor([[4., 0.], [0., 2.], [1., 2.], [0., 1.], [3., 0.]])
        labels = torch.tensor([0, 1, 0, 1, 1])
        model = nn.Identity()
        model.train()
        result = ClassificationEvaluator(model, ["a", "b"]).evaluate(
            DataLoader(TensorDataset(logits, labels), batch_size=2),
        )
        self.assertAlmostEqual(result.metrics["loss"], F.cross_entropy(logits, labels).item(), places=6)
        self.assertAlmostEqual(result.metrics["accuracy"], .6)
        self.assertTrue(model.training)
        np.testing.assert_allclose(result.probabilities, logits.softmax(1).numpy(), atol=1e-7)

    def test_evaluation_does_not_update_weights_or_batchnorm(self):
        model = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 2))
        model[0].eval()
        original = {key: value.clone() for key, value in model.state_dict().items()}
        loader = DataLoader(TensorDataset(torch.randn(5, 2), torch.tensor([0, 1, 0, 1, 0])), batch_size=2)
        ClassificationEvaluator(model, ["a", "b"]).evaluate(loader)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(original[key], value))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertTrue(model.training)
        self.assertFalse(model[0].training)
        self.assertTrue(model[1].training)

    def test_empty_loader_and_bad_logits(self):
        evaluator = ClassificationEvaluator(nn.Identity(), ["a", "b"])
        with self.assertRaisesRegex(ValueError, "empty"):
            evaluator.evaluate(DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long))))
        for logits in (torch.ones(1, 3), torch.tensor([[float("nan"), 0.]])):
            with self.subTest(logits=logits):
                with self.assertRaises(ValueError):
                    evaluator.evaluate(DataLoader(TensorDataset(logits, torch.tensor([0]))))
        self.assertTrue(evaluator.model.training)


class CheckpointTests(unittest.TestCase):
    def test_formats_and_parallel_prefix(self):
        model = TinyClassifier()
        state = model.state_dict()
        variants = [state, {"model_state_dict": state, "epoch": 3},
                    {"state_dict": {f"module.{key}": value for key, value in state.items()}}]
        with tempfile.TemporaryDirectory() as temporary:
            for variant in variants:
                path = Path(temporary) / "model.pth"
                torch.save(variant, path)
                with patch("src.checkpoint_utils.TransferModelBuilder") as builder:
                    builder.return_value.build.return_value = TinyClassifier()
                    loaded, metadata = load_classifier(path)
                    builder.assert_called_once_with(model_name="resnet18", num_classes=10, pretrained=False)
                self.assertEqual(metadata["model_name"], "resnet18")
                for key, value in loaded.state_dict().items():
                    self.assertTrue(torch.equal(value, state[key]))
                with self.assertRaisesRegex(ValueError, "contains resnet18"):
                    load_classifier(path, model_name="vgg16")

    def test_all_project_architectures_dispatch_without_pretrained_download(self):
        for name in ("vgg16", "densenet121"):
            model = nn.Module()
            model.classifier = (nn.Sequential(*[nn.Identity() for _ in range(6)], nn.Linear(2, 10))
                                if name == "vgg16" else nn.Linear(2, 10))
            with self.subTest(model=name), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "model.pth"
                torch.save(model.state_dict(), path)
                with patch("src.checkpoint_utils.TransferModelBuilder") as builder:
                    builder.return_value.build.return_value = model
                    _, metadata = load_classifier(path)
                    builder.assert_called_once_with(model_name=name, num_classes=10, pretrained=False)
                    self.assertEqual(metadata["num_classes"], 10)

    def test_checkpoint_discovery_and_invalid_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaisesRegex(FileNotFoundError, "No checkpoint"):
                find_checkpoint(directory)
            best = directory / "resnet18_best.pth"
            best.touch()
            (directory / "latest.pth").touch()
            self.assertEqual(find_checkpoint(directory), best)
            torch.save({"not_a_classifier": torch.ones(2)}, best)
            with self.assertRaisesRegex(ValueError, "Cannot identify"):
                load_classifier(best)


class PipelineTests(unittest.TestCase):
    def test_data_setup_test_stage_does_not_load_train(self):
        with patch("src.data_module.datasets.CIFAR10") as dataset:
            data = CIFAR10DataModule(num_workers=0)
            data.setup(stage="test", download=False)
            dataset.assert_called_once()
            self.assertFalse(dataset.call_args.kwargs["train"])
            self.assertFalse(dataset.call_args.kwargs["download"])
            self.assertIsNone(data.train_dataset)
        with self.assertRaises(ValueError):
            data.setup(stage="invalid")

    def test_reports_and_end_to_end_pipeline(self):
        dataset = TensorDataset(torch.randn(13, 3, 4, 4), torch.arange(13) % 10)
        dataset.classes = [f"class_{i}" for i in range(10)]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = directory / "model.pth"
            torch.save({"model_state_dict": TinyClassifier().state_dict(), "epoch": 2}, checkpoint)
            config = load_evaluation_config(DEFAULT_CONFIG)
            config.update(checkpoint=str(checkpoint), output_dir=str(directory / "reports"),
                          image_size=4, max_samples=11, batch_size=4, plots=False, device="cpu")
            with patch("src.data_module.datasets.CIFAR10", return_value=dataset), \
                    patch("src.checkpoint_utils.TransferModelBuilder") as builder:
                builder.return_value.build.return_value = TinyClassifier()
                output = run_evaluation(config)
            document = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(document["metrics"]["num_samples"], 11)
            self.assertEqual(document["metadata"]["checkpoint_epoch"], 2)
            self.assertFalse(document["metadata"]["full_test_set"])
            expected = np.sort(np.random.default_rng(42).choice(np.arange(13), size=11, replace=False))
            with (output / "predictions.csv").open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([int(row["sample_index"]) for row in rows], expected.tolist())
            self.assertEqual(len(rows), 11)
            self.assertTrue((output / "classification_report.txt").is_file())
            self.assertTrue((output / "per_class_metrics.csv").is_file())
            self.assertTrue((output / "confusion_matrix.csv").is_file())

    def test_report_plots_with_absent_class(self):
        result = ClassificationEvaluator(nn.Identity(), ["a", "b", "c"]).evaluate(
            DataLoader(TensorDataset(torch.tensor([[2., 1., 0.], [0., 2., 1.]]), torch.tensor([0, 1]))),
        )
        with tempfile.TemporaryDirectory() as temporary:
            save_evaluation_report(result, temporary, ["a", "b", "c"], plots=True)
            self.assertEqual(len(list(Path(temporary).glob("*.png"))), 4)
            for path in Path(temporary).glob("*.png"):
                self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_config_rejects_invalid_sizes_and_types(self):
        for key, value in (("batch_size", 0), ("max_samples", -1), ("download", "false"),
                           ("image_size", 2.5), ("device", "invalid"), ("top_k", True)):
            config = load_evaluation_config(DEFAULT_CONFIG)
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(config)


if __name__ == "__main__":
    unittest.main()
