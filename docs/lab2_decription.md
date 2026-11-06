# Lab 2: Pre-trained Neural Network Architectures

## 1. Define

### What

Xây dựng một pipeline transfer learning cho bài toán phân loại ảnh CIFAR-10
với các mô hình đã được huấn luyện trên ImageNet bằng PyTorch và torchvision.
Pipeline phải hỗ trợ ResNet-18, VGG-16 và DenseNet-121, thay lớp phân loại cuối,
đóng băng backbone, huấn luyện classifier, fine-tune có kiểm soát và ghi log
bằng TensorBoard.

### Why

- Thực hành cách tải và đọc kiến trúc từ `torchvision.models`.
- Hiểu sự khác nhau giữa feature extractor, classifier và fine-tuning.
- Đo ảnh hưởng của learning rate, batch size và weight decay.
- Có bằng chứng kiểm thử, metric và log thay vì chỉ chạy được một lần.

### When

Thực hiện theo thứ tự: hiểu yêu cầu, thiết kế, chia nhỏ, xây dựng, triển khai,
debug/test, rồi mới commit. Mỗi thay đổi phải được chạy và kiểm tra trước khi
chuyển sang bước kế tiếp.

### Where

- Mã nguồn dự kiến: `src/data_module.py`, `src/model_builder.py`,
  `src/trainer.py`, `src/logger.py`.
- Entry point: `main.py`.
- Cấu hình: `configs/config.yaml`.
- Dữ liệu: `data/`; log: `runs/`; checkpoint: `checkpoints/`.
- Tài liệu kế hoạch: sáu file trong thư mục `docs/`.

### Which

Thử tối thiểu ba kiến trúc: ResNet-18, VGG-16 và DenseNet-121. Baseline dùng
CIFAR-10 với 10 lớp, CrossEntropyLoss, Adam, một giá trị learning rate và
batch size rõ ràng. Mọi thí nghiệm phải ghi model, seed, device, hyperparameter,
train metric và validation/test metric.

### How

1. Kiểm tra môi trường, dependency, GPU/CPU và cấu trúc repository.
2. Chốt contract đầu vào/đầu ra của từng module trước khi viết code.
3. Chia công việc thành các prompt nhỏ, mỗi prompt chỉ có một mục tiêu.
4. Sau mỗi thay đổi chạy test tối thiểu và kiểm tra output thực tế.
5. Kiểm duyệt output của AI bằng code review, metric, log và tài liệu.

## 2. Design

### Cấu trúc thư mục

```text
Practice-2---Hands-on-practice-with-pre-trained-neural-network-architectures/
│
├── configs/
│   └── config.yaml             # Cấu hình siêu tham số (batch_size, lr, epochs, model_name, device)
│
├── data/                       # Thư mục lưu dữ liệu CIFAR-10 (tự động tải về qua torchvision)
│
├── src/                        # Mã nguồn chính (Được thiết kế theo chuẩn OOP)
│   ├── __init__.py             # Biến src thành Python Package
│   ├── data_module.py          # Class CIFAR10DataModule (Transforms, Dataset, DataLoader)
│   ├── model_builder.py        # Class TransferModelBuilder (ResNet18, VGG16, DenseNet121)
│   ├── trainer.py              # Class ModelTrainer (Vòng lặp Training 5 bước & Validation)
│   └── logger.py               # Class TensorBoardLogger (add_scalar, add_graph, add_image)
│   ├── eda.py                  # EDA resolution, class distribution, pixel statistics
│   └── evalution.py            # Confusion Matrix, Precision, Recall, F1-score, AUC-ROC
|
├── runs/                       # Thư mục lưu log TensorBoard (SummaryWriter)
│
├── checkpoints/                # Thư mục lưu trọng số mô hình đã huấn luyện (.pth)
│
├── notebooks/                  # Jupyter Notebooks phục vụ thử nghiệm & phân tích kết quả
│   └── experiment_demo.ipynb
│
├── main.py                     # Entry point chính chạy toàn bộ Pipeline
├── requirements.txt            # Thư viện phụ thuộc (PyTorch, Torchvision, Tensorboard,...)
└── README.md                   # Tài liệu hướng dẫn dự án
```
### Luồng dữ liệu

`CIFAR-10 -> transforms -> Dataset -> DataLoader -> model -> loss ->
optimizer -> train/evaluate -> TensorBoard/checkpoint`.

Tách validation từ train để chọn hyperparameter; không dùng test set để chọn
cấu hình. Test chỉ được dùng cho đánh giá cuối cùng.

### Ranh giới module

- `data_module`: transforms, tải dataset, train/validation/test loader.
- `model_builder`: tải weights, thay classifier, freeze/unfreeze, kiểm tra số
  lớp và số parameter trainable.
- `trainer`: training loop, evaluation loop, loss, accuracy, device.
- `logger`: scalar, graph, ảnh mẫu, text config, checkpoint và close writer.
- `main`: đọc config, lắp pipeline, chạy epoch và xử lý checkpoint.
- `eda`: kiểm tra resolution, số kênh, dtype, class distribution và thống kê pixel.
- `evaluation`: tính confusion matrix, precision, recall, F1-score và AUC-ROC.

### Phạm vi EDA và evaluation

- EDA dữ liệu phải kiểm tra resolution, số kênh, dtype, class distribution và
  pixel statistics trước normalize.
- Evaluation cuối phải tạo confusion matrix, precision, recall, F1-score và
  AUC-ROC cho bài toán 10 lớp.
- Confusion matrix dùng nhãn thật theo hàng và nhãn dự đoán theo cột; AUC-ROC
  dùng score/xác suất trước `argmax` theo one-vs-rest.
- Chỉ dùng validation để tuning; chỉ dùng test sau khi khóa checkpoint và
  hyperparameter.
- Mọi metric phải có kết quả theo lớp, aggregate phù hợp, seed, device,
  checkpoint và đường dẫn artifact.

### Hợp đồng tối thiểu

- Input: config hợp lệ, dataset có thể tải, model name được hỗ trợ.
- Output: EDA report, metric dạng số hữu hạn, confusion matrix, precision,
  recall, F1-score, AUC-ROC, log TensorBoard, best checkpoint và thông báo lỗi
  rõ ràng khi input không hợp lệ.
- Không nuốt exception, không trả về kết quả thành công giả.

## 3. Break Down

Mỗi task được thực hiện theo quy trình:

`Description -> Generate -> Smoke Test -> Run (Normal Mode) -> Check with Description`

Các task chính:

1. Data: xác định transform, split và DataLoader.
2. Model: tải weights và thay classifier cho 10 lớp.
3. Training: freeze backbone, loss, optimizer, loop và validation.
4. Tuning: thay một hyperparameter mỗi thí nghiệm.
5. Logging: ghi loss/accuracy, graph, config và checkpoint.
6. EDA: resolution, class distribution và pixel statistics.
7. Evaluation: confusion matrix, precision, recall, F1-score, AUC-ROC và
   phân tích lỗi.
8. Deployment: lệnh chạy, TensorBoard và resume checkpoint.
9. Git: review diff và commit sau khi xác minh.

## 4. Build và triển khai

Xây dựng từng module theo thứ tự Data -> Model -> Trainer -> Logger -> Main.
Chỉ mở rộng sang model tiếp theo sau khi baseline ResNet-18 chạy được. Dùng
config để thay hyperparameter, không hard-code giá trị trong loop.

## 4.1 Quy trình thực hiện bắt buộc

Quy trình cho mỗi task hoặc mỗi thay đổi nhỏ:

### 1. Description

Đọc và mô tả yêu cầu trước khi code:

- What: cần xây dựng hoặc sửa gì?
- Why: tại sao cần thay đổi?
- Where: file/module nào liên quan?
- Input/Output: dữ liệu vào và kết quả mong đợi?
- Constraints: giới hạn về file, dependency và behavior?
- Test: kiểm tra thành công bằng cách nào?

### 2. Generate

Tạo code hoặc tài liệu cho đúng một nhiệm vụ nhỏ.

- Chỉ chỉnh sửa các file được cho phép.
- Tuân theo design và pattern hiện có.
- Không thêm tính năng ngoài yêu cầu.
- Không che giấu lỗi bằng fallback im lặng.

### 3. Smoke Test

Chạy kiểm tra nhanh ngay sau khi generate:

- Import module.
- Kiểm tra syntax.
- Kiểm tra một batch dữ liệu.
- Kiểm tra model forward.
- Kiểm tra output shape và metric cơ bản.

Smoke test phải nhanh và dùng subset nhỏ, không thay thế test đầy đủ.

### 4. Run - Normal Mode

Sau khi smoke test thành công, chạy pipeline ở chế độ bình thường:

- Chạy training với config thật.
- Chạy evaluation trên validation/test đúng mục đích.
- Kiểm tra TensorBoard log.
- Kiểm tra checkpoint và output metric.
- Ghi lại command, device, seed và kết quả.

### 5. Check with Description

Đối chiếu kết quả thực tế với yêu cầu ban đầu:

- Đã hoàn thành đúng mục tiêu chưa?
- Có thay đổi ngoài phạm vi không?
- Test có passed không?
- Metric có hợp lệ, finite và tái lập được không?
- Có đủ log, checkpoint và tài liệu giải thích không?
- Definition of Done đã đạt chưa?

Nếu chưa đạt, quay lại bước Description hoặc Generate; không đánh dấu
hoàn thành chỉ vì command không báo lỗi.

## 5. Debug and Test

### Debug prompt

Đưa lỗi, command đã chạy, traceback đầy đủ, file liên quan và expected behavior.
Yêu cầu AI chỉ ra nguyên nhân có bằng chứng, đề xuất một sửa đổi nhỏ, rồi nêu
test tái hiện và test xác nhận.

### Minimum test

- Import được toàn bộ module.
- DataLoader trả batch đúng shape và label trong `[0, 9]`.
- Model trả logits shape `[batch_size, 10]`.
- Có parameter trainable ở classifier và backbone bị freeze đúng policy.
- Một smoke run 1 epoch trên subset nhỏ chạy được CPU.
- Metric không phải NaN/Inf, TensorBoard event và checkpoint được tạo.
- EDA xác nhận resolution, phân phối nhãn và thống kê pixel.
- Evaluation tạo confusion matrix `10x10`, precision, recall, F1-score và
  AUC-ROC từ score/xác suất đa lớp.

## 6. Git

Trước commit: `git diff`, kiểm tra chỉ có thay đổi đúng phạm vi, chạy minimum
test và ghi lại command/output. Commit nhỏ theo một mục tiêu; không commit
dataset, checkpoint, log lớn, secret hay file tạm. Không rewrite lịch sử khi
chưa được yêu cầu.

## 7. Definition of Done

- Đúng: đáp ứng mục tiêu transfer learning, tuning và TensorBoard.
- Code chạy được: setup mới có thể chạy bằng lệnh được ghi trong docs.
- Test passed: minimum test và smoke run đã chạy thành công.
- Kết quả tái lập được với seed/config đã ghi.
- Có metric và artifact để kiểm tra, không kết luận từ cảm tính.
- Có giải thích ngắn cho các quyết định freeze, classifier và hyperparameter.

## 8. Additional files

- `src/eda.py`: EDA resolution, class distribution, pixel statistics.
- `src/evaluation.py`: Confusion Matrix, Precision, Recall, F1-score, AUC-ROC.
