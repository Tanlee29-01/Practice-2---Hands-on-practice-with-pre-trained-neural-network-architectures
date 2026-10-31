"""Check CIFAR-10 loading: python -m scripts.check_data [--download]."""

import argparse
from pathlib import Path

from src.data_module import CIFAR10DataModule


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Check CIFAR-10 data and batch shapes.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.image_size <= 0 or args.num_workers < 0:
        parser.error("batch-size/image-size must be positive; num-workers must be non-negative.")
    data_dir = args.data_dir if args.data_dir.is_absolute() else ROOT / args.data_dir
    data = CIFAR10DataModule(
        data_dir=str(data_dir), batch_size=args.batch_size,
        image_size=args.image_size, num_workers=args.num_workers,
    )
    try:
        data.setup(stage="test" if args.test_only else "fit", download=args.download)
        print(f"Data directory: {data_dir}")
        if not args.test_only:
            print(f"Train samples: {len(data.train_dataset)}")
            images, labels = next(iter(data.train_dataloader()))
            print(f"Train batch: images={tuple(images.shape)}, labels={tuple(labels.shape)}")
        print(f"Test samples: {len(data.test_dataset)}")
        images, labels = next(iter(data.test_dataloader()))
        print(f"Test batch: images={tuple(images.shape)}, labels={tuple(labels.shape)}")
        print(f"Classes: {', '.join(data.test_dataset.classes)}")
    except (RuntimeError, OSError) as exc:
        parser.exit(1, f"Data check failed: {exc}\nUse --download if CIFAR-10 is missing.\n")


if __name__ == "__main__":
    main()
