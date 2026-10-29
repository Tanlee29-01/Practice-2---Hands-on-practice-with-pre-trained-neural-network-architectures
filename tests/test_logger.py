"""Unit tests for TensorBoardLogger in src/logger.py (Chương 6)."""
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.logger import TensorBoardLogger


class SimpleTestModel(nn.Module):
    """Mô hình đơn giản phục vụ test graph và checkpointing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def temp_dirs():
    """Tạo thư mục tạm thời cho log và checkpoint."""
    temp_dir = tempfile.mkdtemp()
    log_dir = Path(temp_dir) / "runs"
    checkpoint_dir = Path(temp_dir) / "checkpoints"
    yield log_dir, checkpoint_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_logger_initialization(temp_dirs):
    log_dir, checkpoint_dir = temp_dirs
    logger = TensorBoardLogger(
        log_dir=log_dir,
        experiment_name="test_exp",
        checkpoint_dir=checkpoint_dir,
    )
    assert logger.log_dir.exists()
    assert logger.checkpoint_dir.exists()
    assert logger.experiment_name == "test_exp"
    logger.close()


def test_add_scalar_and_scalars(temp_dirs):
    log_dir, checkpoint_dir = temp_dirs
    with TensorBoardLogger(log_dir=log_dir, experiment_name="test_scalar", checkpoint_dir=checkpoint_dir) as logger:
        logger.add_scalar("Loss/train", 0.5, step=1)
        logger.add_scalars("Accuracy", {"train": 80.0, "test": 75.0}, step=1)
        logger.flush()


def test_add_graph(temp_dirs):
    log_dir, checkpoint_dir = temp_dirs
    model = SimpleTestModel()
    dummy_input = torch.randn(2, 10)
    with TensorBoardLogger(log_dir=log_dir, experiment_name="test_graph", checkpoint_dir=checkpoint_dir) as logger:
        logger.add_graph(model, dummy_input)
        logger.flush()


def test_add_image_and_grid(temp_dirs):
    log_dir, checkpoint_dir = temp_dirs
    dummy_images = torch.rand(4, 3, 32, 32)
    with TensorBoardLogger(log_dir=log_dir, experiment_name="test_image", checkpoint_dir=checkpoint_dir) as logger:
        logger.add_image("single_image", dummy_images[0], step=0)
        logger.add_images("batch_images", dummy_images, step=0)
        logger.add_image_grid("grid_images", dummy_images, nrow=2, step=0, unnormalize=True)
        logger.flush()


def test_log_epoch_and_history(temp_dirs):
    log_dir, checkpoint_dir = temp_dirs
    history = {
        "train_loss": [0.8, 0.4],
        "train_accuracy": [70.0, 85.0],
        "test_losses": [0.7, 0.35],
        "test_accuracy": [72.0, 87.0],
    }
    with TensorBoardLogger(log_dir=log_dir, experiment_name="test_epoch", checkpoint_dir=checkpoint_dir) as logger:
        logger.log_epoch(1, 0.8, 70.0, 0.7, 72.0)
        logger.log_history(history)
        logger.flush()


def test_save_and_load_checkpoint(temp_dirs):
    log_dir, checkpoint_dir = temp_dirs
    model = SimpleTestModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    logger = TensorBoardLogger(
        log_dir=log_dir,
        experiment_name="test_ckpt",
        checkpoint_dir=checkpoint_dir,
    )

    # Lưu checkpoint
    ckpt_path = logger.save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=1,
        metrics={"acc": 85.5},
        filename="model_test.pth",
    )
    assert ckpt_path.exists()

    # Load vào model mới
    new_model = SimpleTestModel()
    new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)
    loaded_data = logger.load_checkpoint(new_model, ckpt_path, optimizer=new_optimizer)

    assert loaded_data["epoch"] == 1
    assert loaded_data["metrics"]["acc"] == 85.5

    # Kiểm tra weights khớp nhau
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)

    logger.close()


def test_save_best_model(temp_dirs):
    log_dir, checkpoint_dir = temp_dirs
    model = SimpleTestModel()
    logger = TensorBoardLogger(
        log_dir=log_dir,
        experiment_name="test_best",
        checkpoint_dir=checkpoint_dir,
    )

    # Lần 1: accuracy 75.0 -> lưu
    saved_1 = logger.save_best_model(model, metric=75.0, epoch=1)
    assert saved_1 is True

    # Lần 2: accuracy 70.0 -> không lưu
    saved_2 = logger.save_best_model(model, metric=70.0, epoch=2)
    assert saved_2 is False

    # Lần 3: accuracy 82.0 -> lưu model mới
    saved_3 = logger.save_best_model(model, metric=82.0, epoch=3)
    assert saved_3 is True
    assert logger.best_metric == 82.0

    best_file = checkpoint_dir / "test_best_best.pth"
    assert best_file.exists()

    logger.close()

