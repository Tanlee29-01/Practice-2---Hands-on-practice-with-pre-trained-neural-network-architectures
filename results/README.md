# Cách đọc kết quả của dự án

## Cấu trúc

```text
results/
├── experiments/
│   └── <thoi_gian>/experiments.csv
└── evaluation/
    └── <checkpoint>_<thoi_gian>/
        ├── metrics.json
        ├── classification_report.txt
        ├── per_class_metrics.csv
        ├── predictions.csv
        ├── confusion_matrix.csv
        └── *.png
```

## CSV thí nghiệm

`experiments.py` ghi kết quả train/validation vào `experiments/<thời_gian>/experiments.csv`.
Mỗi dòng là một cấu hình ở một epoch. Accuracy trong CSV này có đơn vị phần trăm (0–100).

Hai lần chạy đã có được chuyển vào:

- [Sweep 3 epoch, 1.000 ảnh train / 500 ảnh validation](experiments/20261028_222705_829922/experiments.csv).
- [Chạy baseline 1 epoch, 256 ảnh train / 128 ảnh validation](experiments/20261030_192943_838095/experiments.csv).

Các CSV này không chứa trọng số mô hình; không thể dùng CSV làm checkpoint cho `evaluate.py`.

## Báo cáo đánh giá

`evaluate.py` ghi kết quả trên CIFAR-10 **test**, một thư mục riêng cho mỗi lần chạy.
Trong `metrics.json`, xem `metadata.full_test_set`, `metrics.num_samples` và đường dẫn checkpoint.
Chỉ số accuracy/precision/recall/F1 trong JSON và CSV đánh giá dùng thang 0–1.

Hai báo cáo được tạo ở lần đánh giá trước:

| Thư mục | Số ảnh | Mục đích |
| --- | ---: | --- |
| `resnet18_20261030-222811_best_20261031_075526_374326` | 64 | Kiểm tra nhanh các file và biểu đồ |
| `resnet18_20261030-222811_best_20261031_080111_789486` | 10.000 | Báo cáo đầy đủ của checkpoint hiện có |

Khi viết báo cáo lab, dùng kết quả toàn bộ test: accuracy 79,13%, F1 macro 0,7912.
Kết quả trên 64 ảnh chỉ là chạy thử, có thể khác đáng kể với toàn bộ test.

Các file đánh giá được sinh cục bộ và bị Git bỏ qua, nên có thể chưa tồn tại trên máy vừa clone.
Tạo lại bằng `python evaluate.py` sau khi đã có checkpoint và dữ liệu.

Xem [hướng dẫn từng bước](../docs/README_LAB.md), [cách tính chỉ số](../docs/EVALUATION.md)
và [phân tích, hạn chế của kết quả](../docs/PROJECT_REVIEW.md).
