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

## EDA dữ liệu hình ảnh

Trước khi training, phải thực hiện EDA trên CIFAR-10 và ghi lại bằng chứng.
EDA không được làm thay đổi dataset gốc, split hoặc nhãn.

### 1. Kiểm tra kích thước (Resolution)

- Xác nhận số chiều, chiều cao, chiều rộng và số kênh của ảnh.
- Kiểm tra resolution thực tế của một mẫu và toàn bộ dataset nếu cần.
- Phân biệt resolution gốc `32x32x3` với kích thước sau `Resize` trong
  transform.
- Báo cáo các resolution khác nhau nếu phát hiện dữ liệu không đồng nhất.

### 2. Cân bằng nhãn (Class Distribution)

- Đếm số mẫu của từng lớp trong train, validation và test.
- Hiển thị tên lớp, số lượng và tỷ lệ phần trăm.
- Nêu rõ dataset có cân bằng hay không và cách xử lý nếu có mất cân bằng.
- Không dùng test distribution để điều chỉnh training hoặc hyperparameter.

### 3. Thống kê giá trị Pixel

- Báo cáo min, max, mean và standard deviation theo kênh trước normalize.
- Nêu rõ không gian giá trị (`[0, 255]` hoặc `[0, 1]`) và dtype.
- Nếu dùng ImageNet normalization, ghi mean/std được sử dụng và thời điểm
  áp dụng.
- Kiểm tra NaN, Inf và giá trị ngoài khoảng hợp lệ.

## EDA prompt

```text
Implement exactly one task: thực hiện EDA cho dữ liệu ảnh CIFAR-10.

Chỉ sửa các file EDA đã được nêu rõ trong task hiện tại.
Không sửa model, trainer, logger, main hoặc dataset gốc.
Không thay đổi split, label, transform dùng cho training hay hyperparameter.

EDA bắt buộc:
1. Kiểm tra resolution, số kênh, dtype và shape.
2. Thống kê class distribution riêng cho train/validation/test.
3. Tính pixel min, max, mean và standard deviation theo kênh trước normalize.
4. Kiểm tra NaN, Inf và giá trị ngoài khoảng hợp lệ.

Báo rõ nguồn dữ liệu, số mẫu, không gian pixel và sự khác nhau giữa ảnh gốc
với ảnh sau transform. Nếu tạo artifact, ghi đường dẫn cụ thể.
Chạy smoke test trên subset nhỏ trước, sau đó mới chạy EDA đầy đủ.
Báo command, output, exit code và không kết luận PASS nếu thiếu bằng chứng.
```

## EDA acceptance criteria

- Resolution và số kênh được xác nhận bằng giá trị thực tế.
- Class distribution có đủ số lượng và tỷ lệ của từng lớp ở mỗi split.
- Pixel statistics có min, max, mean, standard deviation theo kênh.
- Không phát hiện NaN/Inf hoặc phải báo rõ mẫu và nguyên nhân.
- Kết quả EDA tái lập được với cùng dataset và seed.