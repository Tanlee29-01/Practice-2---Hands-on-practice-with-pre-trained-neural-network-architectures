# Evaluation, debug and test plan

## Define

Đánh giá phải trả lời: model nào tốt hơn, hyperparameter nào phù hợp, kết quả
có tái lập không, và lỗi tập trung ở lớp nào. Ngoài loss và accuracy, báo cáo
định lượng bắt buộc gồm confusion matrix, precision, recall, F1-score và
AUC-ROC curve.

## Metrics bắt buộc

Dataset sử dụng 2 split chuẩn: `train` (50.000 mẫu) và `test` (10.000 mẫu).
Trong quá trình huấn luyện, mô hình được theo dõi qua loss và accuracy trên test set.
Sau khi huấn luyện hoàn tất, checkpoint tốt nhất được đưa vào module evaluation để
tính toán các metric chuyên sâu và báo cáo định lượng: confusion matrix,
precision, recall, F1-score và AUC-ROC curve.

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

## Build prompt: Evaluation module (src/evaluation.py)

```text
Implement exactly one task: hoàn thiện module evaluation độc lập trong src/evaluation.py.

Làm việc từ repository root và tuân thủ pattern OOP hiện có.
Chỉ được sửa:
- src/evaluation.py

Yêu cầu:
- Xây dựng hàm evaluate_model() và CLI entry point (python -m src.evaluation).
- Nhận checkpoint path, config, test dataloader và device.
- Chạy forward trên test set với torch.no_grad() và eval mode.
- Thu thập raw logits, tính xác suất softmax và predicted labels.
- Tính toán đầy đủ các metrics:
  1. Confusion Matrix 10x10 (số lượng tuyệt đối và normalized theo hàng).
  2. Precision theo từng lớp, macro average và weighted average (zero_division=0).
  3. Recall theo từng lớp, macro average và weighted average.
  4. F1-score theo từng lớp, macro average và weighted average.
  5. AUC-ROC đa lớp theo chiến lược one-vs-rest (OvR) tính từ xác suất softmax trước argmax,
     cùng macro AUC.
- Xuất artifacts vào output_dir:
  + evaluation_report.json (metadata, confusion matrix, metrics).
  + metrics_per_class.csv (bảng chi tiết theo lớp).
  + confusion_matrix.png (biểu đồ nhiệt trực quan).
  + roc_curve.png (đường cong ROC cho 10 lớp).
- Kiểm tra tính hợp lệ: label range [0, 9], tensor không chứa NaN/Inf. Báo lỗi rõ ràng nếu thiếu lớp hoặc input hỏng.

Sau khi sửa:
1. Chạy unit test evaluation:
   .\venv\Scripts\python.exe -m pytest tests/test_evaluation.py
2. Báo command, output và exit code.
3. Không sửa file ngoài phạm vi.
```

## Execution prompt: Evaluation on test set

```text
Implement exactly one task: chạy đánh giá mô hình CIFAR-10 trên test set bằng checkpoint đã khóa.

Chỉ được thực hiện sau khi model đã hoàn tất huấn luyện và checkpoint tốt nhất đã được lưu.
Không sửa data split (50.000 train / 10.000 test), model architecture, training loop hoặc hyperparameter.

Lệnh thực hiện từ repository root:
.\venv\Scripts\python.exe -m src.evaluation --checkpoint <đường_dẫn_checkpoint_best.pth> --config configs/config.yaml --device auto

Yêu cầu báo cáo:
- Model name, checkpoint path, seed, device.
- Test accuracy, Macro F1, Weighted F1, Macro AUC-ROC.
- Top các cặp lớp bị nhầm lẫn nhiều nhất trong confusion matrix.
- Đường dẫn các artifacts: evaluation_report.json, metrics_per_class.csv, confusion_matrix.png, roc_curve.png.
- Báo command đã chạy, exit code và xác nhận không có NaN/Inf.
```

## Evaluation protocol

1. Huấn luyện mô hình và lưu checkpoint tốt nhất dựa trên test metric.
2. Dùng checkpoint tốt nhất đã lưu để chạy đánh giá toàn diện cho báo cáo cuối.
3. Báo model, weights, freeze policy, seed, device, epochs và hyperparameters.
4. Báo train loss/accuracy và test loss/accuracy cùng đường dẫn artifact.
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
Debug đúng một lỗi dưới đây.

File/module:
<điền file>

Command:
<điền command>

Traceback/output:
<dán toàn bộ lỗi>

Expected behavior:
<mô tả kết quả mong muốn>

Hãy:
1. Xác định nguyên nhân dựa trên bằng chứng
2. Chỉ ra file cần sửa
3. Đề xuất một thay đổi nhỏ nhất
4. Thực hiện sửa đổi
5. Chạy regression test
6. Báo kết quả trước và sau

Không sửa file không liên quan.
Không dùng fallback im lặng hoặc broad exception.
```

## Review prompt

```text
Review toàn bộ diff hiện tại theo yêu cầu Practice 2.

Kiểm tra:
- Đúng cấu hình 2 split: train (50.000) và test (10.000)
- Sai classifier shape (phải đúng 10 classes)
- Sai freeze/fine-tune policy
- Sai device handling (CPU/CUDA)
- Sai loss/accuracy calculation
- Sai confusion matrix hoặc đảo true/predicted axis (hàng là true, cột là predicted)
- Precision/recall/F1 chỉ có aggregate, thiếu kết quả theo lớp
- AUC-ROC tính từ argmax thay vì score/xác suất
- EDA làm thay đổi dữ liệu gốc
- Sai train/eval mode
- Thiếu TensorBoard artifact hoặc checkpoint không đầy đủ
- Không tái lập được
- Thay đổi ngoài phạm vi

Chỉ trả về các finding có:
- File
- Vấn đề
- Mức độ nghiêm trọng
- Bằng chứng
- Test đề xuất

Không tự sửa code.
```

## Acceptance evidence

Lưu command, exit code, metric output, confusion matrix, precision, recall,
F1-score, AUC-ROC, TensorBoard tag list và checkpoint path trong kết quả
thực nghiệm/report. Nếu test không chạy được do môi trường, ghi rõ blocker và
không đánh dấu Definition of Done là passed.

## Implementation usage

Module: `src/evaluation.py`

Run the final test evaluation from the repository root after selecting and
locking the checkpoint:

```powershell
python -m src.evaluation --checkpoint checkpoints\cifar10_checkpoints\<experiment>_best.pth --config configs\config.yaml --device auto
```

Optional arguments:

- `--data-dir`: CIFAR-10 data directory (default: `data`).
- `--output-dir`: root directory for evaluation artifacts (default:
  `evaluation_results`).
- `--device`: `auto`, `cpu`, or `cuda`.

Each run creates a new timestamped folder containing:

- `evaluation_report.json`: model metadata, confusion matrix and all metrics.
- `metrics_per_class.csv`: per-class, macro and weighted metrics.
- `confusion_matrix.png`: count and row-normalized confusion matrices.
- `roc_curve.png`: one-vs-rest ROC curves and per-class AUC.

`zero_division=0` is used for precision/recall/F1 when a class has no predicted
samples. AUC-ROC is calculated from softmax scores, one-vs-rest, using the
trapezoidal area under each class ROC curve; macro AUC is the unweighted mean
of the ten class AUC values. Evaluation fails clearly if a true-label class is
absent, because its ROC AUC is undefined.

The module can also be called from Python with `evaluate_model(model,
dataloader, device, class_names, output_dir, metadata)` for a small smoke test
or another compatible classification dataset.

### Pipeline split design

Pipeline được thiết kế nhất quán với 2 tập dữ liệu chuẩn của CIFAR-10: tập huấn luyện (`train`: 50.000 mẫu) và tập kiểm thử (`test`: 10.000 mẫu). Trong mỗi epoch, mô hình được cập nhật trên train set và theo dõi hiệu năng trên test set để lưu best checkpoint. Sau khi kết thúc huấn luyện, module `src.evaluation` đọc best checkpoint để thực hiện đánh giá toàn diện (Confusion Matrix, Precision, Recall, F1, ROC-AUC) trên test set.