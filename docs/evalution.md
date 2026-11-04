# Evaluation, debug and test plan

## Define

Đánh giá phải trả lời: model nào tốt hơn, hyperparameter nào phù hợp, kết quả
có tái lập không, và lỗi tập trung ở lớp nào. Ngoài loss và accuracy, báo cáo
định lượng bắt buộc gồm confusion matrix, precision, recall, F1-score và
AUC-ROC curve.

## Metrics bắt buộc

Chỉ tính các metric cuối trên test set sau khi đã khóa model, weights,
hyperparameters và threshold. Validation chỉ dùng để chọn cấu hình.

### 1. Confusion Matrix

- Ma trận có kích thước `10x10`, hàng là nhãn thật và cột là nhãn dự đoán.
- Hiển thị cả số lượng tuyệt đối; có thể bổ sung bản normalize theo hàng.
- Ghi tên đầy đủ của 10 lớp CIFAR-10 và chỉ ra các cặp lớp dễ nhầm.

### 2. Precision

- Báo precision theo từng lớp và macro/weighted average.
- Nêu công thức hoặc quy ước zero division nếu một lớp không có prediction.

### 3. Recall

- Báo recall theo từng lớp và macro/weighted average.
- Dùng recall để nhận diện lớp có nhiều mẫu bị bỏ sót.

### 4. F1-score

- Báo F1-score theo từng lớp và macro/weighted average.
- Không chỉ báo một giá trị trung bình nếu các lớp có hành vi khác nhau.

### 5. AUC-ROC Curve

- Dùng xác suất hoặc score trước argmax, không dùng nhãn dự đoán rời rạc.
- Với bài toán đa lớp, dùng one-vs-rest và báo macro AUC cùng AUC theo lớp
  nếu thư viện hỗ trợ.
- Ghi rõ cách tính, số lớp, label binarization và trường hợp một lớp thiếu
  mẫu trong split.
- Lưu curve/figure và đường dẫn artifact; không kết luận từ accuracy thay cho
  AUC.

## Evaluation prompt

```text
Implement exactly one task: bổ sung evaluation đầy đủ cho mô hình CIFAR-10.

Chỉ sửa các file evaluation đã được nêu rõ trong task hiện tại.
Không sửa data split, model architecture, training loop hoặc hyperparameter.
Không dùng test set để tuning hoặc chọn checkpoint.

Sau khi model đã được khóa, tính trên test set:
1. Confusion matrix 10x10, có tên lớp và số lượng tuyệt đối.
2. Precision theo lớp, macro average và weighted average.
3. Recall theo lớp, macro average và weighted average.
4. F1-score theo lớp, macro average và weighted average.
5. AUC-ROC one-vs-rest cho bài toán 10 lớp, dùng score/xác suất trước argmax.

Yêu cầu:
- Kiểm tra shape, label range và số lớp trước khi tính.
- Xử lý zero division theo quy ước rõ ràng, không nuốt exception.
- Metric phải hữu hạn; báo lỗi rõ nếu thiếu lớp hoặc score không hợp lệ.
- Lưu figure/report vào artifact path đã cấu hình, không ghi đè artifact cũ.
- Báo model, checkpoint, seed, device, split, threshold/quy ước và command.

Chạy smoke test trên subset nhỏ, sau đó chạy evaluation đầy đủ.
Báo output, exit code, metric và đường dẫn artifact.
```

## Evaluation protocol

1. Chọn cấu hình bằng validation.
2. Khóa cấu hình và chỉ chạy test một lần cho báo cáo cuối.
3. Báo model, weights, freeze policy, seed, device, epochs và hyperparameters.
4. Báo train/validation/test loss và accuracy cùng đường dẫn artifact.
5. Không kết luận chỉ từ một batch hoặc training accuracy.

## Minimum tests

### Static/import

- Import `src` và các module không lỗi.
- Config thiếu key, số âm hoặc model không hỗ trợ phải fail rõ ràng.

### Data

- Loader trả đúng image/label shape.
- Split không chồng lấn và seed cố định cho kết quả lặp lại.

### Model

- Ba model đều forward được.
- Logits có 10 lớp.
- Freeze policy và trainable count đúng.

### Training

- Một batch backward được.
- Một epoch subset nhỏ chạy CPU.
- Loss/accuracy không NaN hoặc Inf.

### Artifact

- TensorBoard event được tạo và có scalar tags.
- Checkpoint tồn tại, load được và giữ được metric/config.

### Evaluation

- Confusion matrix đúng shape `10x10`, tổng số phần tử bằng số mẫu test.
- Precision, recall và F1-score có kết quả theo lớp và macro/weighted average.
- AUC-ROC dùng score/xác suất đa lớp, không dùng argmax.
- Figure/report có thể mở lại và gắn với đúng checkpoint/test split.
- Metric không phải NaN/Inf và có seed, device, model, config đi kèm.

## Debug prompt

```text
Debug this exact failure without changing unrelated files.
Context: <file/module and expected behavior>
Command: <exact command>
Observed traceback/output: <full output>
First identify the root cause from evidence, then propose one surgical fix.
Add or run the smallest regression test and report before/after behavior.
Do not hide the exception or replace it with a silent default.
```

## Review prompt

```text
Review the current diff against the lab requirements.
Check correctness, data leakage, model adaptation, device handling,
TensorBoard/checkpoint artifacts, reproducibility, and error reporting.
List only actionable findings with file, evidence, severity, and a test.
Do not rewrite code while reviewing.
```

## Acceptance evidence

Lưu command, exit code, metric output, confusion matrix, precision, recall,
F1-score, AUC-ROC, TensorBoard tag list và checkpoint path trong kết quả
thực nghiệm/report. Nếu test không chạy được do môi trường, ghi rõ blocker và
không đánh dấu Definition of Done là passed.