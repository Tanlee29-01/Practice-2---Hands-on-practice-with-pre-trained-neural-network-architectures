"""Khám phá dữ liệu ảnh CIFAR-10 trước khi huấn luyện."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def analyze_pixels(images, indices):
    """Thống kê pixel RGB gốc theo từng khối nhỏ."""
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(
            f"Cần ảnh dạng [N, H, W, 3], nhận được {images.shape}"
        )

    if len(indices) == 0:
        raise ValueError("Tập ảnh cần phân tích không được rỗng.")

    total = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    minimum = np.full(3, np.inf)
    maximum = np.full(3, -np.inf)
    pixel_count = 0

    for start in range(0, len(indices), 128):
        selected = indices[start:start + 128]
        block = images[selected].astype(np.float64)

        nan_count = int(np.isnan(block).sum())
        inf_count = int(np.isinf(block).sum())
        outside = int(((block < 0) | (block > 255)).sum())

        if nan_count or inf_count or outside:
            raise ValueError(
                f"Pixel không hợp lệ trong các ảnh chỉ số "
                f"{selected.tolist()}: "
                f"NaN={nan_count}, Inf={inf_count}, "
                f"ngoài [0,255]={outside}"
            )

        total += block.sum(axis=(0, 1, 2))
        squares += (block * block).sum(axis=(0, 1, 2))
        minimum = np.minimum(
            minimum, block.min(axis=(0, 1, 2))
        )
        maximum = np.maximum(
            maximum, block.max(axis=(0, 1, 2))
        )

        pixel_count += (
            block.shape[0] * block.shape[1] * block.shape[2]
        )

    mean = total / pixel_count
    variance = squares / pixel_count - mean * mean
    std = np.sqrt(np.maximum(variance, 0))

    return {
        "samples_analyzed": len(indices),
        "image_shape_HWC": list(images.shape[1:]),
        "dtype": str(images.dtype),
        "channel_order": ["R", "G", "B"],
        "pixel_space": "Original [0,255], before transforms",
        "min": minimum.tolist(),
        "max": maximum.tolist(),
        "mean": mean.tolist(),
        "std_population": std.tolist(),
        "mean_0_1": (mean / 255).tolist(),
        "std_0_1": (std / 255).tolist(),
        "nan_count": 0,
        "inf_count": 0,
        "out_of_range_count": 0,
    }


def run(args):
    import torch
    from torchvision.datasets import CIFAR10
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Dang nap anh goc CIFAR-10...", flush=True)

    train_data = CIFAR10(
        root=str(args.data_dir),
        train=True,
        download=args.download,
        transform=None,
    )
    test_data = CIFAR10(
        root=str(args.data_dir),
        train=False,
        download=args.download,
        transform=None,
    )

    splits = {
        "train": (train_data, np.arange(len(train_data))),
        "test": (test_data, np.arange(len(test_data))),
    }

    output = (
        args.output_dir
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)

    summary = {
        "source": "torchvision.datasets.CIFAR10",
        "data_directory": str(args.data_dir.resolve()),
        "seed": args.seed,
        "torch_version": str(torch.__version__),
        "mode": "smoke_subset" if args.max_samples else "full",
        "split_policy": (
            "Standard CIFAR-10 splits: 50000 train, 10000 test."
        ),
        "training_transforms_from_source": {
            "resize": [224, 224],
            "random_horizontal_flip": "train only",
            "ToTensor": "[0,255] to [0,1]",
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
        },
        "note": (
            "Statistics describe original images before transforms. "
            "ImageNet normalization constants are reported from source. "
            "Test statistics are descriptive only, not used for tuning."
        ),
        "splits": {},
    }

    rows = []
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 8), constrained_layout=True
    )

    for axis, (name, (dataset, full_indices)) in zip(
        axes, splits.items()
    ):
        indices = (
            full_indices[:args.max_samples]
            if args.max_samples
            else full_indices
        )

        labels = np.asarray(dataset.targets)

        if np.any((labels < 0) | (labels >= 10)):
            raise ValueError("Nhãn nằm ngoài khoảng 0 đến 9.")

        counts = np.bincount(
            labels[indices], minlength=10
        )

        stats = analyze_pixels(dataset.data, indices)
        stats["full_split_size"] = len(full_indices)
        stats["class_names"] = list(dataset.classes)
        stats["class_counts"] = counts.tolist()
        stats["exactly_balanced"] = bool(
            np.all(counts == counts[0])
        )
        summary["splits"][name] = stats

        for class_name, count in zip(dataset.classes, counts):
            rows.append({
                "split": name,
                "class": class_name,
                "count": int(count),
                "percent": float(count / len(indices) * 100),
            })

        axis.bar(dataset.classes, counts, color="#008397")
        axis.set_title(
            f"{name}: {len(indices)} analyzed "
            f"/ {len(full_indices)} total"
        )
        axis.set_ylabel("Images")
        axis.tick_params(axis="x", labelrotation=30)

        print(
            f"{name}: {len(indices)} anh | "
            f"shape={stats['image_shape_HWC']} | "
            f"dtype={stats['dtype']} | "
            "NaN=0, Inf=0, ngoai khoang=0",
            flush=True,
        )

    with (output / "class_distribution.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["split", "class", "count", "percent"],
        )
        writer.writeheader()
        writer.writerows(rows)

    (output / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    np.savez(
        output / "split_indices.npz",
        **{name: pair[1] for name, pair in splits.items()},
    )

    fig.savefig(
        output / "class_distribution.png", dpi=160
    )
    plt.close(fig)

    print(f"Che do: {summary['mode']}", flush=True)
    print(f"Da luu ket qua: {output.resolve()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EDA original CIFAR-10 images."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "eda",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Images per split. 0 means all, 100 for smoke test.",
    )
    parser.add_argument("--download", action="store_true")

    args = parser.parse_args()

    if args.max_samples < 0:
        parser.error("--max-samples must be >= 0")

    run(args)