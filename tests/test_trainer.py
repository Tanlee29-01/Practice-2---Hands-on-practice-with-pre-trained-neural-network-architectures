"""Tests for the training loop and CIFAR-10 data module."""

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_module import CIFAR10DataModule
from src.trainer import ModelTrainer


class TinyClassifier(nn.Module):
    """Small model that makes the trainer tests fast and deterministic."""

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(3 * 4 * 4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.flatten(inputs))


def make_loader() -> DataLoader:
    images = torch.randn(8, 3, 4, 4)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    return DataLoader(TensorDataset(images, labels), batch_size=4)


def test_training_loop_returns_metrics_and_updates_model():
    torch.manual_seed(42)
    model = TinyClassifier()
    initial_weights = model.classifier.weight.detach().clone()
    loader = make_loader()

    trainer = ModelTrainer(
        model=model,
        train_dataloader=loader,
        test_dataloader=loader,
        learning_rate=0.01,
        device=torch.device("cpu"),
    )

    loss, accuracy = trainer.training_loop()

    assert isinstance(loss, float)
    assert loss >= 0
    assert 0 <= accuracy <= 100
    assert not torch.equal(initial_weights, model.classifier.weight)


def test_evaluate_returns_metrics_without_gradients():
    model = TinyClassifier()
    loader = make_loader()
    trainer = ModelTrainer(
        model=model,
        train_dataloader=loader,
        test_dataloader=loader,
        device=torch.device("cpu"),
    )

    loss, accuracy = trainer.evaluate()

    assert isinstance(loss, float)
    assert loss >= 0
    assert 0 <= accuracy <= 100
    assert not model.training


def test_fit_returns_one_metric_per_epoch():
    trainer = ModelTrainer(
        model=TinyClassifier(),
        train_dataloader=make_loader(),
        test_dataloader=make_loader(),
        device=torch.device("cpu"),
    )

    history = trainer.fit(num_epochs=2)

    assert set(history) == {
        "train_loss",
        "train_accuracy",
        "test_losses",
        "test_accuracy",
    }
    assert all(len(values) == 2 for values in history.values())


def test_invalid_optimizer_name_is_rejected():
    with pytest.raises(ValueError, match="Optimizer_name"):
        ModelTrainer(
            model=TinyClassifier(),
            train_dataloader=make_loader(),
            test_dataloader=make_loader(),
            optimizer_name="RMSprop",
            device=torch.device("cpu"),
        )


def test_data_module_setup_uses_local_cifar10_files():
    data_module = CIFAR10DataModule(
        data_dir=str(Path(__file__).parents[1] / "data"),
        batch_size=4,
        num_workers=0,
    )
    data_module.setup()

    assert len(data_module.train_dataset) == 50000
    assert len(data_module.test_dataset) == 10000
    assert data_module.train_dataset[0][0].shape == (3, 224, 224)
    assert data_module.test_dataset[0][0].shape == (3, 224, 224)

    train_batch = next(iter(data_module.train_dataloader()))
    test_batch = next(iter(data_module.test_dataloader()))
    assert train_batch[0].shape == (4, 3, 224, 224)
    assert test_batch[0].shape == (4, 3, 224, 224)


def test_data_module_requires_setup_before_loaders():
    data_module = CIFAR10DataModule(num_workers=0)

    with pytest.raises(RuntimeError, match="Call setup"):
        data_module.train_dataloader()

    with pytest.raises(RuntimeError, match="Call setup"):
        data_module.test_dataloader()
