# Data plan

## Define

Dataset mục tiêu là CIFAR-10 gồm 10 lớp, ảnh màu 32x32. Cần trả lời trước khi
code: dữ liệu lấy ở đâu, split nào dùng cho mục đích nào, transform nào chỉ
được áp dụng cho train, và batch đầu ra có shape gì.

## Design

### Split

- `train`: dùng để cập nhật trọng số, có augmentation nhẹ.
- `validation`: tách từ train, dùng để chọn model/hyperparameter.
- `test`: giữ nguyên đến đánh giá cuối cùng.
- Không dùng test accuracy để chọn learning rate hoặc batch size.
- Ghi seed và số lượng mẫu của từng split.

### Transform

Train có thể dùng `Resize`, chuyển tensor, normalize theo ImageNet và
augmentation phù hợp như random crop/flip. Validation/test dùng transform
deterministic, không random. Mean/std phải nhất quán với pretrained weights
được chọn.

### Data module contract

`CIFAR10DataModule` cần nhận `data_dir`, `batch_size`, `num_workers` và expose
dataset/loader rõ ràng. `setup()` phải tải hoặc đọc dữ liệu, kiểm tra dataset
không rỗng và chuẩn bị các split. Loader train shuffle; validation/test không
shuffle. Batch size phải là số dương.

### Missing implementation checklist

- [ ] Tải CIFAR-10 bằng `torchvision.datasets.CIFAR10`.
- [ ] Tạo transform train và evaluation riêng.
- [ ] Tách validation có seed cố định.
- [ ] Tạo train/validation/test DataLoader.
- [ ] Kiểm tra batch image/label shape.
- [ ] Không tải lại dataset trong mỗi epoch.
- [ ] Báo lỗi rõ ràng khi thư mục không ghi được hoặc dữ liệu hỏng.

## Build prompt

```text
Implement only the data module for CIFAR-10.
Use the existing project conventions and do not touch model/trainer files.
Add deterministic train/validation split, train/eval transforms, and loaders.
Validate positive batch_size and non-empty datasets.
After editing, run the smallest import and one-batch smoke test.
Report changed files, commands, and observed shapes.
```

## Verify

Chạy một batch và xác nhận `images` là tensor 4 chiều, `labels` là tensor 1
chiều, batch dimension khớp, pixel đã được normalize và label thuộc 10 lớp.
Kiểm tra lặp lại cùng seed cho ra cùng split.