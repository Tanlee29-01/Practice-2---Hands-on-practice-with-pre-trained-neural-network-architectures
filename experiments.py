"""Thu nghiem learning rate, batch size va weight decay."""

import argparse
import copy
import csv
import random
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.data_module import CIFAR10DataModule
from src.model_builder import TransferModelBuilder_ResNet18
from src.trainer import ModelTrainer


ROOT = Path(__file__).resolve().parent
SEED = 42

# Moi lan chi thay doi mot thong so so voi baseline.
CONFIGS = [
    ("baseline", 0.001, 64, 0.0001),
    ("lr_low", 0.0001, 64, 0.0001),
    ("lr_high", 0.01, 64, 0.0001),
    ("batch_small", 0.001, 32, 0.0001),
    ("batch_large", 0.001, 128, 0.0001),
    ("no_l2", 0.001, 64, 0.0),
    ("l2_high", 0.001, 64, 0.001),
]


def keep_batchnorm_eval(layer, inputs):
    """Giu BatchNorm o che do eval, khong thay doi dau vao."""
    layer.eval()
    return None


class FrozenBackboneTrainer(ModelTrainer):
    """Giu thong ke BatchNorm cua backbone co dinh."""

    def training_loop(self):
        handles = []

        try:
            for module in self.model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    handle = module.register_forward_pre_hook(
                        keep_batchnorm_eval
                    )
                    handles.append(handle)

            return super().training_loop()
        finally:
            for handle in handles:
                handle.remove()


def run(args):
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("Dang kiem tra va nap du lieu CIFAR-10...", flush=True)

    data = CIFAR10DataModule(
        data_dir=str(ROOT / "data"),
        num_workers=0,
    )
    data.setup()

    # Tach validation tu tap train.
    # Khong dung tap test de lua chon sieu tham so.
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(
        len(data.train_dataset),
        generator=generator,
    ).tolist()

    val_indices = indices[:5000]
    train_indices = indices[5000:]

    # Validation khong su dung phep lat anh ngau nhien.
    validation_dataset = copy.copy(data.train_dataset)
    validation_dataset.transform = data.test_transform

    if args.train_samples > 0:
        train_indices = train_indices[:args.train_samples]

    if args.val_samples > 0:
        val_indices = val_indices[:args.val_samples]

    train_set = Subset(data.train_dataset, train_indices)
    val_set = Subset(validation_dataset, val_indices)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    configs = CONFIGS if args.sweep else CONFIGS[:1]

    output_dir = ROOT / "results" / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "experiments.csv"

    fields = [
        "experiment",
        "epoch",
        "learning_rate",
        "batch_size",
        "weight_decay",
        "train_samples",
        "val_samples",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
    ]

    print(f"Device: {device}", flush=True)
    print(
        f"Train: {len(train_set)} anh | "
        f"Validation: {len(val_set)} anh",
        flush=True,
    )
    print(f"So cau hinh: {len(configs)}", flush=True)

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for name, lr, batch_size, weight_decay in configs:
            # Khoi tao lai model va thu tu du lieu.
            random.seed(SEED)
            torch.manual_seed(SEED)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)

            train_loader = DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                generator=torch.Generator().manual_seed(SEED),
            )

            val_loader = DataLoader(
                val_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )

            print(
                f"\n{name}: lr={lr}, batch={batch_size}, "
                f"weight_decay={weight_decay}",
                flush=True,
            )
            print(
                "Dang tao ResNet18; lan dau co the can tai trong so.",
                flush=True,
            )

            model = TransferModelBuilder_ResNet18(
                num_classes=10,
                pretrained=True,
            ).build()

            trainer = FrozenBackboneTrainer(
                model=model,
                train_dataloader=train_loader,
                test_dataloader=val_loader,
                learning_rate=lr,
                optimizer_name="Adam",
                device=device,
                weight_decay=weight_decay,
            )

            for epoch in range(1, args.epochs + 1):
                print(
                    f"Dang chay epoch {epoch}/{args.epochs}...",
                    flush=True,
                )

                train_loss, train_acc = trainer.training_loop()
                val_loss, val_acc = trainer.evaluate()

                writer.writerow({
                    "experiment": name,
                    "epoch": epoch,
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "weight_decay": weight_decay,
                    "train_samples": len(train_set),
                    "val_samples": len(val_set),
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                })
                file.flush()

                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"Train acc: {train_acc:.2f}% | "
                    f"Val acc: {val_acc:.2f}% | "
                    f"Val loss: {val_loss:.4f}",
                    flush=True,
                )

            del trainer, model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nDa luu ket qua: {output_file}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--val-samples", type=int, default=128)
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs phai >= 1")

    if args.train_samples < 0 or args.val_samples < 0:
        parser.error("So anh phai >= 0; dung 0 de lay toan bo")

    run(args)