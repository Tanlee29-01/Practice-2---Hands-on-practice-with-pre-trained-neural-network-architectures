# Data plan

## Define

Dataset mục tiêu là CIFAR-10 gồm 10 lớp, ảnh màu 32x32. Cần trả lời trước khi
code: dữ liệu lấy ở đâu, split nào dùng cho mục đích nào, transform nào chỉ
được áp dụng cho train, và batch đầu ra có shape gì.

## Design

### Split

- `train`: 50.000 ảnh màu (5.000 ảnh/lớp), dùng để cập nhật trọng số mô hình, có augmentation nhẹ.
- `test`: 10.000 ảnh màu (1.000 ảnh/lớp), dùng để theo dõi loss/accuracy sau mỗi epoch và đánh giá cuối cùng.
- CIFAR-10 được chia sẵn thành 2 tập chuẩn (50.000 train / 10.000 test), không cần chia thêm tập validation.

### Transform

Train dùng `Resize`, chuyển tensor, normalize theo ImageNet và
augmentation phù hợp như `RandomHorizontalFlip`. Test dùng transform
deterministic, không random. Mean/std phải nhất quán với pretrained weights
được chọn (ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

### Data module contract

`CIFAR10DataModule` nhận `data_dir`, `batch_size`, `num_workers` và `image_size`.
`setup()` phải tải CIFAR-10, kiểm tra dataset không rỗng, chuẩn bị `train_dataset`
(50.000 mẫu) và `test_dataset` (10.000 mẫu). Expose đầy đủ 2 dataloaders:
- `train_dataloader()`: shuffle=True, áp dụng train transform (có augmentation).
- `test_dataloader()`: shuffle=False, áp dụng evaluation transform (deterministic).
Batch size phải là số dương (> 0).

### Missing implementation checklist

- [ ] Tải CIFAR-10 bằng `torchvision.datasets.CIFAR10`.
- [ ] Tạo transform train (có augmentation) và test riêng (deterministic).
- [ ] Tạo đủ `train_dataloader` (50.000 mẫu) và `test_dataloader` (10.000 mẫu).
- [ ] Kiểm tra batch image [B, 3, H, W] và label [B] shape, batch_size > 0.
- [ ] Không tải lại dataset trong mỗi epoch.
- [ ] Báo lỗi rõ ràng khi thư mục không ghi được hoặc dữ liệu hỏng.

## Build prompt

```text
Implement exactly one task: hoàn thiện data module cho CIFAR-10.

Làm việc từ repository root và tuân thủ pattern OOP hiện có.
Chỉ được sửa:
- src/data_module.py

Yêu cầu:
- Tải CIFAR-10 bằng torchvision.datasets.CIFAR10 gồm 2 split chuẩn:
  + train_dataset: 50.000 ảnh màu
  + test_dataset: 10.000 ảnh màu
- Tạo transform riêng biệt:
  + Train transform: Resize((image_size, image_size)), RandomHorizontalFlip(), ToTensor(), Normalize.
  + Evaluation transform: Resize((image_size, image_size)), ToTensor(), Normalize.
  + Mean/std chuẩn hóa theo ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
- Cung cấp đủ 2 loader:
  + train_dataloader(): shuffle=True, pin_memory phù hợp.
  + test_dataloader(): shuffle=False, pin_memory phù hợp.
- Kiểm tra batch_size > 0 và dataset không rỗng; báo lỗi ValueError / RuntimeError rõ ràng.
- Hỗ trợ num_workers và pin_memory phù hợp.

Sau khi sửa:
1. Chạy import test:
   .\venv\Scripts\python.exe -c "from src.data_module import CIFAR10DataModule; print('Import PASS')"
2. Chạy one-batch smoke test:
   .\venv\Scripts\python.exe -c "from src.data_module import CIFAR10DataModule; dm = CIFAR10DataModule(batch_size=4); dm.setup(); x, y = next(iter(dm.train_dataloader())); print('Train batch:', x.shape, y.shape); xt, yt = next(iter(dm.test_dataloader())); print('Test batch:', xt.shape, yt.shape)"
3. Báo command, output, exit code và shape tensor quan sát được.
4. Không sửa file ngoài phạm vi.
```

## Verify

Chạy một batch và xác nhận `images` là tensor 4 chiều `[batch_size, 3, 224, 224]`,
`labels` là tensor 1 chiều `[batch_size]`, batch dimension khớp, pixel đã được
normalize và label thuộc 10 lớp `[0, 9]`. Xác nhận có đủ train_dataloader (50.000 mẫu)
và test_dataloader (10.000 mẫu).

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

- Đếm số mẫu của từng lớp trong 2 split: train (50.000 ảnh) và test (10.000 ảnh).
- Hiển thị tên lớp, số lượng và tỷ lệ phần trăm (mỗi lớp có 5.000 ảnh train và 1.000 ảnh test, hoàn toàn cân bằng).
- Nêu rõ dataset có cân bằng hay không và kết luận.

### 3. Thống kê giá trị Pixel

- Báo cáo min, max, mean và standard deviation theo kênh trước normalize.
- Nêu rõ không gian giá trị (`[0, 255]` hoặc `[0, 1]`) và dtype.
- Nếu dùng ImageNet normalization, ghi mean/std được sử dụng và thời điểm
  áp dụng.
- Kiểm tra NaN, Inf và giá trị ngoài khoảng hợp lệ.

## EDA prompt

```text
Implement exactly one task: thực hiện EDA cho dữ liệu ảnh CIFAR-10.

Làm việc từ repository root.
Chỉ được sửa các file EDA:
- src/eda.py
- notebooks/cifar10_eda.ipynb (nếu cập nhật notebook phân tích)

Không sửa model, trainer, logger, main hoặc dataset gốc.
Không thay đổi split, label, transform dùng cho training hay hyperparameter.

EDA bắt buộc:
1. Kiểm tra resolution (gốc 32x32x3 vs sau Resize 224x224x3), số kênh, dtype và shape tensor.
2. Thống kê class distribution chi tiết cho 2 split: train (50.000) và test (10.000).
3. Tính pixel min, max, mean và standard deviation theo kênh RGB trước normalize (trong thang đo [0, 255]).
4. Kiểm tra NaN, Inf và giá trị ngoài khoảng hợp lệ.
5. Xuất kết quả ra file JSON/CSV hoặc hiển thị trực quan trong artifact path / notebook.

Sau khi sửa:
1. Chạy smoke test:
   .\venv\Scripts\python.exe -m src.eda --max-samples 100
2. Chạy full EDA:
   .\venv\Scripts\python.exe -m src.eda
3. Báo command, output, exit code và artifact tạo ra. Không kết luận PASS nếu thiếu bằng chứng.
```

## EDA acceptance criteria

- Resolution và số kênh được xác nhận bằng giá trị thực tế.
- Class distribution có đủ số lượng và tỷ lệ của từng lớp ở mỗi split.
- Pixel statistics có min, max, mean, standard deviation theo kênh.
- Không phát hiện NaN/Inf hoặc phải báo rõ mẫu và nguyên nhân.
- Kết quả EDA tái lập được với cùng dataset và seed.