"""Entry point for the pretrained-model training pipeline."""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from src.data_module import CIFAR10DataModule
from src.logger import TensorBoardLogger
from src.model_builder import TransferModelBuilder
from src.trainer import ModelTrainer


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the YAML training configuration."""
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    required = (
        "batch_size",
        "learning_rate",
        "epochs",
        "model_name",
        "device",
        "weight_decay",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(
            f"Missing configuration keys: {', '.join(missing)}"
        )

    if int(config["batch_size"]) <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if float(config["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be greater than 0.")
    if int(config["epochs"]) <= 0:
        raise ValueError("epochs must be greater than 0.")
    if float(config["weight_decay"]) < 0:
        raise ValueError("weight_decay must not be negative.")

    return config


def resolve_device(device_name: str) -> torch.device:
    """Resolve the configured device and fail clearly for unavailable CUDA."""
    normalized = device_name.lower().strip()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda.")
    return torch.device(normalized)


def run(config: dict[str, Any], checkpoint_path: Path | None = None) -> Path:
    """Run training and return the path of the best checkpoint."""
    batch_size = int(config["batch_size"])
    learning_rate = float(config["learning_rate"])
    epochs = int(config["epochs"])
    model_name = str(config["model_name"])
    weight_decay = float(config["weight_decay"])
    device = resolve_device(str(config["device"]))

    print(f"Device: {device}", flush=True)
    print("Loading CIFAR-10 data...", flush=True)

    data_module = CIFAR10DataModule(
        data_dir=str(ROOT / "data"),
        batch_size=batch_size,
        num_workers=0,
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()

    print(f"Building pretrained model: {model_name}", flush=True)
    model = TransferModelBuilder(
        model_name=model_name,
        num_classes=10,
        pretrained=True,
    ).build()

    trainer = ModelTrainer(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        learning_rate=learning_rate,
        optimizer_name="Adam",
        device=device,
        weight_decay=weight_decay,
    )

    experiment_name = (
        f"{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    logger = TensorBoardLogger(
        log_dir=ROOT / "runs",
        experiment_name=experiment_name,
        checkpoint_dir=ROOT / "checkpoints",
    )

    try:
        start_epoch = 1
        if checkpoint_path is not None:
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Checkpoint file not found: {checkpoint_path}"
                )

            checkpoint = logger.load_checkpoint(
                model=model,
                path=checkpoint_path,
                optimizer=trainer.optimizer,
                device=device,
            )
            checkpoint_epoch = checkpoint.get("epoch")
            if isinstance(checkpoint_epoch, int):
                start_epoch = checkpoint_epoch + 1

            print(
                f"Loaded checkpoint: {checkpoint_path} "
                f"(epoch={checkpoint_epoch})",
                flush=True,
            )

        sample_images, _ = next(iter(train_loader))
        logger.add_graph(model, sample_images[: min(2, len(sample_images))])
        logger.add_image_grid(
            "Samples/train",
            sample_images[: min(16, len(sample_images))],
            step=0,
        )
        logger.add_text("Configuration", yaml.safe_dump(config), step=0)

        for epoch in range(start_epoch, epochs + 1):
            train_loss, train_accuracy = trainer.training_loop()
            test_loss, test_accuracy = trainer.evaluate()

            logger.log_epoch(
                epoch,
                train_loss,
                train_accuracy,
                test_loss,
                test_accuracy,
            )
            logger.save_best_model(
                model,
                metric=test_accuracy,
                optimizer=trainer.optimizer,
                epoch=epoch,
            )
            logger.flush()

            print(
                f"Epoch [{epoch}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.2f}% | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Acc: {test_accuracy:.2f}%",
                flush=True,
            )

        best_checkpoint = (
            ROOT / "checkpoints" / f"{experiment_name}_best.pth"
        )
        print(f"Best checkpoint: {best_checkpoint}", flush=True)
        print(f"TensorBoard logs: {ROOT / 'runs' / experiment_name}", flush=True)
        return best_checkpoint
    finally:
        logger.close()


def main() -> None:
    """Parse command-line arguments and run the training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train a pretrained image classifier on CIFAR-10."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Local checkpoint to load before training. "
            "Relative paths are resolved from the project root."
        ),
    )
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    checkpoint_path = args.checkpoint
    if checkpoint_path is not None and not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path

    config = load_config(config_path)
    run(config, checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    main()
