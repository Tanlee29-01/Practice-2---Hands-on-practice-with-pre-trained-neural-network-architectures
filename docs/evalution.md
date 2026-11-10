# Evaluation Specification Plan (Version 1.0)

> **Document Metadata**
> - **Module Name:** `src/evaluation.py` (`evaluate_model`, CLI `python -m src.evaluation`), kiểm thử bởi `tests/test_evaluation.py`
> - **Created Date:** 10/11/2026
> - **Last Updated:** 10/11/2026
> - **Target Dataset:** CIFAR-10 (`test` split chuẩn: 10.000 ảnh, 10 lớp: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`)
> - **Config & Checkpoint:** `configs/config.yaml`, `checkpoints/cifar10_checkpoints/*_best.pth`

---

## 1. Purpose (Mục đích)

Mục đích chính của module `src/evaluation.py` là thực hiện **đánh giá định lượng toàn diện (post-training evaluation)** cho mô hình phân loại ảnh CIFAR-10 sau khi đã hoàn tất huấn luyện và khóa trọng số tốt nhất (*locked best checkpoint*). Quá trình đánh giá phải trả lời chính xác 4 câu hỏi cốt lõi của thực nghiệm học sâu: **mô hình nào tốt hơn, bộ siêu tham số nào phù hợp, kết quả có tái lập được không, và lỗi phân loại tập trung ở những cặp lớp nào**.

### Lý do & Vai trò kỹ thuật:
1. **Đánh giá độc lập trên Locked Checkpoint (`eval()` + `torch.inference_mode()`):**
   - Chỉ thực hiện sau khi quá trình huấn luyện kết thúc và file `*_best.pth` đã được chốt.
   - Chạy suy diễn trong chế độ `model.eval()` kết hợp `torch.inference_mode()` để khóa hoàn toàn việc cập nhật trọng số, tắt Dropout (ở VGG-16) và giữ cố định thống kê BatchNorm (ở ResNet-18, DenseNet-121).
   - Bảo toàn trạng thái `model.training` ban đầu thông qua khối `try ... finally`.
2. **Vượt qua giới hạn của Accuracy đơn thuần (Multi-metric Assessment):**
   - Trong quá trình huấn luyện (`ModelTrainer`), mô hình chỉ được giám sát nhanh bằng `Loss` và `Accuracy`. Tuy nhiên, một giá trị Accuracy tổng thể không thể cho biết lớp nào đang bị dự đoán sai nhiều nhất.
   - Module tính toán đầy đủ 5 nhóm chỉ số định lượng bắt buộc: **Confusion Matrix (`10x10`), Precision, Recall, F1-score** (chi tiết theo từng lớp kèm `macro_avg` và `weighted_avg`) và **Multiclass AUC-ROC Curve**.
3. **Tính AUC-ROC đa lớp chuẩn xác từ xác suất Softmax (One-vs-Rest):**
   - Tuyệt đối không tính AUC-ROC từ nhãn dự đoán rời rạc (`argmax`).
   - Module chuyển đổi raw logits `[N, 10]` thành phân phối xác suất liên tục qua `torch.softmax(logits, dim=1)`, sau đó áp dụng chiến lược **One-vs-Rest (OvR)** kết hợp công thức diện tích hình thang (*trapezoidal rule*) để tính AUC cho từng lớp và trung bình `macro AUC`.
4. **Phân tích lỗi trực quan qua Confusion Matrix kép:**
   - Xuất đồng thời ma trận số lượng tuyệt đối (*counts*, tổng 10.000 mẫu test) và ma trận chuẩn hóa theo hàng (*row-normalized*, tỷ lệ dự đoán theo từng nhãn thật).
   - Giúp nhận diện ngay các cặp lớp có đặc trưng thị giác tương đồng dễ bị nhầm lẫn (ví dụ: `cat` $\leftrightarrow$ `dog`, `automobile` $\leftrightarrow$ `truck`, `airplane` $\leftrightarrow$ `ship`).
5. **Kiểm tra hợp lệ nghiêm ngặt (Fail-fast Validation & Zero-division Semantics):**
   - Kiểm tra chặt chẽ kích thước logits `[N, C]`, nhãn `[N]` kiểu số nguyên trong miền `[0, C-1]`, và từ chối ngay nếu xuất hiện `NaN`/`Inf`.
   - Áp dụng quy ước `zero_division = 0` an toàn khi một lớp không có mẫu dự đoán nào (`predicted_count == 0`), đồng thời báo lỗi rõ ràng nếu tập đánh giá thiếu mẫu dương của bất kỳ lớp nào khiến AUC-ROC không xác định.
6. **Lưu trữ Artifact bất biến có đóng dấu thời gian (Immutable Timestamped Artifacts):**
   - Mỗi lần chạy tạo một thư mục độc lập `evaluation_results/evaluation_<YYYYMMDD_HHMMSS_ffffff>/` với `exist_ok=False` để không bao giờ ghi đè kết quả cũ.
   - Lưu đầy đủ 4 tệp kết quả (`evaluation_report.json`, `metrics_per_class.csv`, `confusion_matrix.png`, `roc_curve.png`) gắn kèm metadata (`model_name`, `checkpoint`, `epoch`, `seed`, `device`, `batch_size`).

**Ngoài phạm vi tài liệu này:** tiền xử lý dữ liệu và EDA (`data.md`), kiến trúc mạng Transfer Learning (`model_v1.md`), vòng lặp huấn luyện và TensorBoard logger (`training.md`).

---

## 2. Input/Output & Diagram (Thiết kế luồng và hợp đồng dữ liệu)

### 2.1 Sơ đồ luồng tổng thể (Evaluation Pipeline Diagram)

```text
configs/config.yaml ──────────────► Đọc model_name, batch_size, seed
checkpoints/.../*_best.pth ───────► _load_checkpoint_state(weights_only=True)
CIFAR10DataModule(data_dir) ──────► test_dataloader (10.000 ảnh test, shuffle=False)
                                          │
                                          ▼
                   TransferModelBuilder(model_name, num_classes=10, pretrained=False).build()
                   + model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                                          │
                                          ▼
         evaluate_model(model, dataloader, device, class_names, output_dir, metadata)
                                          │
         ┌────────────────────────────────┴────────────────────────────────┐
         ▼                                                                 ▼
 [1. Forward & Validate]                                          [2. Compute Metrics]
  • model.eval() + torch.inference_mode()                          • Confusion Matrix 10x10 (counts & row-norm)
  • logits = model(images)  ──► _validate_inputs()                 • Precision, Recall, F1 (per-class, macro, weighted)
  • probs  = softmax(logits, dim=1)                                • Multiclass ROC points & OvR AUC (trapezoidal)
  • y_pred = probs.argmax(axis=1)                                  • Kiểm tra toàn bộ metric hữu hạn (no NaN/Inf)
  • finally: model.train(was_training)                                     │
         └────────────────────────────────┬────────────────────────────────┘
                                          ▼
              _create_artifact_directory("evaluation_results/evaluation_<timestamp>")
                         ├─► evaluation_report.json   (metadata + confusion_matrix + metrics)
                         ├─► metrics_per_class.csv    (10 lớp + macro_avg + weighted_avg)
                         ├─► confusion_matrix.png     (heatmap tuyệt đối & chuẩn hóa theo hàng)
                         └─► roc_curve.png            (đường cong ROC 10 lớp + macro AUC)
```

---

### 2.2 Đặc tả toán học của 5 nhóm Metrics bắt buộc (Mathematical Specification)

Pipeline sử dụng nhất quán 2 split chuẩn của CIFAR-10: `train` (`50.000` mẫu) và `test` (`10.000` mẫu). Trên tập `test` gồm $N = 10.000$ mẫu và $K = 10$ lớp:

| STT | Nhóm Metric | Công thức & Quy tắc tính toán trong `src/evaluation.py` | Ý nghĩa phân tích |
| :---: | :--- | :--- | :--- |
| **1** | **Confusion Matrix**<br>($C \in \mathbb{N}^{K \times K}$) | - $C_{i, j} = \sum_{n=1}^{N} \mathbb{I}(y_{\text{true}}^{(n)} = i \;\wedge\; y_{\text{pred}}^{(n)} = j)$<br>- Hàng $i$: nhãn thật (`True label`); Cột $j$: nhãn dự đoán (`Predicted label`).<br>- $\text{support}_i = \sum_{j=0}^{K-1} C_{i,j}$, $\quad \text{pred\_count}_j = \sum_{i=0}^{K-1} C_{i,j}$, $\quad \text{TP}_i = C_{i,i}$.<br>- Chuẩn hóa theo hàng: $\tilde{C}_{i,j} = \frac{C_{i,j}}{\max(\text{support}_i, 1)}$. | Xác định chính xác những cặp lớp bị mô hình nhầm lẫn nhiều nhất và tỷ lệ nhầm theo từng lớp thật. |
| **2** | **Precision**<br>(Độ chuẩn xác) | - Theo từng lớp: $\text{Precision}_i = \frac{\text{TP}_i}{\text{pred\_count}_i}$ (nếu $\text{pred\_count}_i = 0 \Rightarrow 0.0$ theo quy ước `zero_division=0`).<br>- $\text{Macro Precision} = \frac{1}{K}\sum_{i=0}^{K-1} \text{Precision}_i$.<br>- $\text{Weighted Precision} = \frac{1}{N}\sum_{i=0}^{K-1} \text{support}_i \times \text{Precision}_i$. | Trong các mẫu được mô hình gán nhãn là lớp $i$, có bao nhiêu tỷ lệ là đúng lớp $i$ thật (phát hiện báo động giả - *False Positives*). |
| **3** | **Recall**<br>(Độ phủ / Độ nhạy) | - Theo từng lớp: $\text{Recall}_i = \frac{\text{TP}_i}{\text{support}_i}$ (với `zero_division=0`).<br>- $\text{Macro Recall} = \frac{1}{K}\sum_{i=0}^{K-1} \text{Recall}_i$.<br>- $\text{Weighted Recall} = \frac{1}{N}\sum_{i=0}^{K-1} \text{support}_i \times \text{Recall}_i$. | Trong toàn bộ mẫu thật của lớp $i$, mô hình nhận diện đúng được bao nhiêu phần trăm (phát hiện các lớp bị bỏ sót - *False Negatives*). |
| **4** | **F1-score** | - Theo từng lớp: $\text{F1}_i = \frac{2 \times \text{Precision}_i \times \text{Recall}_i}{\text{Precision}_i + \text{Recall}_i}$ (với `zero_division=0`).<br>- Báo đầy đủ theo từng lớp, `macro_avg` và `weighted_avg`. | Trung bình điều hòa cân bằng giữa Precision và Recall trên từng lớp; không bị che lấp bởi các lớp dễ. |
| **5** | **Multiclass AUC-ROC**<br>(One-vs-Rest) | - Tính xác suất trước `argmax`: $P_{n, c} = \text{Softmax}(\text{logits}_n)_c$.<br>- Với mỗi lớp $c \in \{0..9\}$: nhị phân hóa nhãn $b_n = \mathbb{I}(y_{\text{true}}^{(n)} = c)$, sắp xếp giảm dần theo $P_{n, c}$, gom nhóm các điểm có xác suất bằng nhau (`distinct_end`) để tính $(\text{FPR}_m, \text{TPR}_m)$ từ $(0,0)$ đến $(1,1)$.<br>- Diện tích hình thang: $\text{AUC}_c = \sum_{m} \frac{\text{TPR}_m + \text{TPR}_{m-1}}{2} (\text{FPR}_m - \text{FPR}_{m-1})$.<br>- $\text{Macro AUC} = \frac{1}{K}\sum_{c=0}^{K-1} \text{AUC}_c$. | Đánh giá khả năng phân tách xác suất của mô hình trên mọi ngưỡng quyết định (threshold-independent), độc lập với việc chọn `argmax`. |

---

### 2.3 Cấu trúc Code AI sinh ra và Vị trí lưu trữ (What & Where)

* **Vị trí file mã nguồn:** [`src/evaluation.py`](file:///d:/deepln/Practice-2---Hands-on-practice-with-pre-trained-neural-network-architectures/src/evaluation.py)
* **Vị trí file kiểm thử:** [`tests/test_evaluation.py`](file:///d:/deepln/Practice-2---Hands-on-practice-with-pre-trained-neural-network-architectures/tests/test_evaluation.py)
* **Kiến trúc Module và Artifacts:**
  ```text
  src/evaluation.py
  ├── CIFAR10_CLASSES = ("airplane", "automobile", ..., "truck")   # 10 lớp chuẩn CIFAR-10
  ├── _validate_inputs(logits, labels, num_classes) -> None        # Bẫy lỗi shape, dtype, range, NaN/Inf
  ├── _safe_divide(numerator, denominator) -> np.ndarray           # Chia an toàn với zero_division=0
  ├── _compute_metrics(y_true, y_pred, probabilities, ...)         # Tính Confusion Matrix, P/R/F1, OvR ROC-AUC
  ├── _create_artifact_directory(output_dir) -> Path               # Tạo thư mục evaluation_<timestamp> (exist_ok=False)
  ├── _save_confusion_matrix(confusion, class_names, output_path)  # Vẽ 2 heatmap (counts & row-normalized)
  ├── _save_roc_curve(roc_points, metrics, output_path)            # Vẽ 10 đường cong ROC OvR + đường chéo ngẫu nhiên
  ├── _save_metrics_csv(metrics, output_path)                      # Xuất bảng CSV (10 lớp + macro_avg + weighted_avg)
  ├── evaluate_model(model, dataloader, device, class_names,       # Hàm API chính trả về report dict
  │                  output_dir, metadata) -> dict[str, Any]
  ├── _load_checkpoint_state(checkpoint_path, device) -> dict      # Nạp file .pth an toàn (weights_only=True)
  └── main() -> None                                               # CLI entry point (python -m src.evaluation)

  evaluation_results/
  └── evaluation_<YYYYMMDD_HHMMSS_ffffff>/                         # Tự động tạo mới cho mỗi lần chạy
        ├── evaluation_report.json                                 # Toàn bộ metadata, confusion matrix, metrics
        ├── metrics_per_class.csv                                  # Bảng chỉ số theo từng lớp & trung bình
        ├── confusion_matrix.png                                   # Biểu đồ nhiệt kép (17x7 inch, 180 DPI)
        └── roc_curve.png                                          # Biểu đồ ROC đa lớp (9x7 inch, 180 DPI)
  ```

---

### 2.4 Chi tiết Input / Output Contract

#### 2.4.1 Hàm `evaluate_model()` và CLI `python -m src.evaluation`

| Thành phần | Đầu vào (Input) | Đầu ra (Output) / Hành vi |
| :--- | :--- | :--- |
| **`evaluate_model()`** | - `model: nn.Module` (trả raw logits `[B, C]`)<br>- `dataloader: DataLoader` (trả `(images, labels)`)<br>- `device: torch.device \| str = "cpu"`<br>- `class_names: Sequence[str] = CIFAR10_CLASSES`<br>- `output_dir: str \| Path = "evaluation_results"`<br>- `metadata: dict[str, Any] \| None = None` | Trả về `report: dict` gồm:<br>`metadata`, `num_samples`, `num_classes`, `class_names`, `confusion_matrix` (`list[list[int]]`), `metrics` (`accuracy`, `macro_avg`, `weighted_avg`, `per_class`, `zero_division`, `auc_method`), `artifact_dir`, `report_path`. |
| **CLI (`main`)** | `--checkpoint <path>` *(bắt buộc)*<br>`--config configs/config.yaml`<br>`--data-dir data`<br>`--output-dir evaluation_results`<br>`--device auto \| cpu \| cuda` | Khởi tạo model với `pretrained=False` (tránh tải lại ImageNet không cần thiết), nạp `model_state_dict` (`strict=True`), chạy `evaluate_model()` trên `test_dataloader()` và in tóm tắt ra console. |
| **Bẫy lỗi (Exceptions)** | - `logits.ndim != 2` hoặc `logits.shape[1] != C`<br>- `labels` không phải kiểu nguyên hoặc ngoài `[0, C-1]`<br>- `logits` / `probabilities` / `metrics` có `NaN` hoặc `Inf`<br>- Dataloader rỗng hoặc thiếu lớp dương cho AUC-ROC<br>- Checkpoint không tồn tại / thiếu `model_state_dict` | Ném `ValueError`, `TypeError`, `FileNotFoundError` hoặc `RuntimeError` kèm thông báo nguyên nhân cụ thể; tuyệt đối không nuốt lỗi im lặng. |

#### 2.4.2 Hợp đồng dữ liệu của 4 Artifacts xuất ra

| Tên file Artifact | Định dạng | Nội dung bắt buộc bên trong |
| :--- | :--- | :--- |
| **`evaluation_report.json`** | JSON (`utf-8`, indent 2) | Chứa `metadata` (`model_name`, `checkpoint`, `checkpoint_epoch`, `checkpoint_metrics`, `seed`, `device`, `split`, `batch_size`), `num_samples` (`10000`), `num_classes` (`10`), `class_names`, `confusion_matrix` (`10x10`), và `metrics` đầy đủ. |
| **`metrics_per_class.csv`** | CSV (`utf-8-sig`) | Cột: `class,precision,recall,f1_score,auc_roc,support`. Gồm 10 dòng cho 10 lớp CIFAR-10 và 2 dòng tổng hợp `macro_avg`, `weighted_avg`. |
| **`confusion_matrix.png`** | PNG (`dpi=180`) | Gồm 2 biểu đồ cạnh nhau: **Confusion Matrix (counts)** định dạng số nguyên `d` và **Confusion Matrix (row-normalized)** định dạng `.2f`, trục Y là `True label`, trục X là `Predicted label`. |
| **`roc_curve.png`** | PNG (`dpi=180`) | Vẽ 10 đường cong ROC One-vs-Rest kèm chú thích `class_name (AUC = 0.xxx)`, đường chéo đứt nét `Random classifier` và tiêu đề hiển thị `macro AUC`. |

---

## 3. Success Criteria (Tiêu chí thành công & Nghiệm thu)

Module `src/evaluation.py` và kết quả đánh giá chỉ được nghiệm thu đạt chuẩn (**PASS**) khi thỏa mãn toàn bộ các tiêu chí định lượng sau:

| STT | Tiêu chí (Criteria) | Điều kiện chấp thuận (Acceptance Condition) | Cách kiểm tra |
| :---: | :--- | :--- | :--- |
| **1** | **Chế độ suy diễn & Bảo toàn trạng thái** | Chạy forward với `model.eval()` và `torch.inference_mode()`; không làm thay đổi trọng số model; khôi phục đúng `model.training` ban đầu sau khi hàm kết thúc (kể cả khi có lỗi). | Đặt `model.train()` trước khi gọi `evaluate_model()`, kiểm tra `assert model.training` sau khi chạy xong. |
| **2** | **Confusion Matrix chuẩn `K x K`** | Ma trận có kích thước `10x10` (hoặc `K x K`), hàng là `True label`, cột là `Predicted label`, kiểu số nguyên không âm, tổng toàn bộ phần tử bằng đúng tổng số mẫu test (`10.000`). | Kiểm tra `len(cm) == K`, `all(len(r) == K for r in cm)` và `sum(map(sum, cm)) == num_samples`. |
| **3** | **Precision, Recall, F1 đầy đủ** | Có kết quả cho từng lớp (`per_class`), `macro_avg` và `weighted_avg` nằm trong đoạn `[0.0, 1.0]`; áp dụng `zero_division=0` khi một lớp không có mẫu dự đoán. | Kiểm tra các khóa trong `report["metrics"]` và miền giá trị `0.0 <= val <= 1.0`. |
| **4** | **AUC-ROC đa lớp từ xác suất Softmax** | Tính từ xác suất `torch.softmax(logits, dim=1)` theo chiến lược One-vs-Rest (tuyệt đối không dùng `argmax`); có `auc_roc` cho từng lớp và `macro_avg["auc_roc"]` trong `[0.0, 1.0]`. | Kiểm tra `metrics["auc_method"]`, giá trị AUC từng lớp và `macro_avg["auc_roc"]`. |
| **5** | **Tính hữu hạn (No NaN/Inf)** | Toàn bộ logits, xác suất softmax và mọi chỉ số trong `metrics` đều hữu hạn (`isfinite`). Nếu mô hình sinh `NaN` hoặc `Inf`, phải ném `ValueError` ngay lập tức. | Test với mô hình giả lập trả `float("nan")` (`test_evaluation_rejects_non_finite_logits`). |
| **6** | **Bẫy lỗi đầu vào & Thiếu lớp** | Ném `ValueError`/`TypeError` rõ ràng khi: dataloader rỗng, sai số chiều/số lớp logits, nhãn ngoài `[0, K-1]`, hoặc thiếu bất kỳ lớp thật nào khiến AUC không xác định. | Chạy 5 bài test bẫy lỗi trong `tests/test_evaluation.py`. |
| **7** | **Tạo đủ 4 Artifacts & Không ghi đè** | Mỗi lần gọi tạo một thư mục `evaluation_<timestamp>` riêng biệt (`exist_ok=False`), xuất đủ 4 file: `evaluation_report.json`, `metrics_per_class.csv`, `confusion_matrix.png`, `roc_curve.png`. | Gọi `evaluate_model()` 2 lần liên tiếp và kiểm tra 2 đường dẫn khác nhau, cả 4 file đều tồn tại. |
| **8** | **Truy vết Metadata & Tái lập** | File `evaluation_report.json` lưu đầy đủ `metadata` (`model_name`, `checkpoint`, `checkpoint_epoch`, `seed`, `device`, `split`, `batch_size`) khớp với checkpoint đã khóa. | Đọc lại `evaluation_report.json` bằng `json.loads()` và đối chiếu khóa `metadata`. |
| **9** | **Test Suite tự động** | `pytest tests/test_evaluation.py -v` vượt qua toàn bộ 7/7 bài kiểm thử với `exit code 0`. | Chạy Lệnh 1 ở mục 4.4. |

---

## 4. How to do (Cách thực hiện, Prompting & Smoke Test)

### 4.1 Quy trình kỹ thuật từng bước (Step-by-step Implementation)

1. **Bước 1: Thu thập dự đoán an toàn trong `evaluate_model()`:**
   - Lưu trạng thái `was_training = model.training`, chuyển `model.eval()` và mở ngữ cảnh `with torch.inference_mode():`.
   - Với mỗi batch từ `dataloader`, đưa `images, labels` lên `device`, tính `logits = model(images)`, gọi `_validate_inputs(logits, labels, num_classes)`.
   - Tính xác suất `probabilities = torch.softmax(logits, dim=1)` và gom về CPU numpy array. Khôi phục `model.train(was_training)` trong khối `finally`.
2. **Bước 2: Tính Confusion Matrix, Precision, Recall, F1 và OvR AUC-ROC trong `_compute_metrics()`:**
   ```python
   confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
   np.add.at(confusion, (y_true, y_pred), 1)

   support = confusion.sum(axis=1)
   predicted_count = confusion.sum(axis=0)
   true_positive = np.diag(confusion).astype(np.float64)

   precision = _safe_divide(true_positive, predicted_count)
   recall = _safe_divide(true_positive, support)
   f1 = _safe_divide(2 * precision * recall, precision + recall)
   ```
   - Với mỗi lớp `class_index`, nhị phân hóa `binary_true = (y_true == class_index)`, sắp xếp giảm dần theo `probabilities[:, class_index]`, lấy các điểm cuối của nhóm đồng hạng (`distinct_end`), tính `tpr`, `fpr` và tích phân hình thang `auc_value = float(np.sum((tpr[1:] + tpr[:-1]) * np.diff(fpr) / 2.0))`.
3. **Bước 3: Xuất 4 Artifacts vào thư mục đóng dấu thời gian:**
   - Tạo thư mục `evaluation_results/evaluation_<YYYYMMDD_HHMMSS_ffffff>` bằng `_create_artifact_directory()`.
   - Ghi `evaluation_report.json`, `metrics_per_class.csv`, vẽ biểu đồ nhiệt kép `confusion_matrix.png` và đường cong `roc_curve.png` (sử dụng backend `matplotlib.use("Agg")` để chạy ổn định trên mọi môi trường không cần cửa sổ GUI).
4. **Bước 4: Xây dựng CLI `main()` để đánh giá từ Checkpoint đã khóa:**
   - Đọc `configs/config.yaml`, nạp checkpoint bằng `torch.load(..., map_location=device, weights_only=True)`.
   - Khởi tạo mô hình qua `TransferModelBuilder(model_name, num_classes=10, pretrained=False).build()` và nạp `model.load_state_dict(checkpoint["model_state_dict"], strict=True)`.
   - Chạy `evaluate_model()` trên `data_module.test_dataloader()` và in các chỉ số tổng hợp cùng đường dẫn thư mục artifact.

---

### 4.2 Build Prompt 1: Evaluation Module (`src/evaluation.py` - Independent Prompt)

Dán đoạn prompt độc lập sau vào AI khi cần tạo mới hoặc cập nhật module `src/evaluation.py`:

```text
Repository root:
d:\deepln\Practice-2---Hands-on-practice-with-pre-trained-neural-network-architectures

Implement exactly one task: hoàn thiện module evaluation độc lập trong src/evaluation.py.

Làm việc từ repository root và tuân thủ cấu trúc hiện có của dự án.
Chỉ được sửa:
- src/evaluation.py

Yêu cầu:
1. Xây dựng hàm evaluate_model(model, dataloader, device="cpu", class_names=CIFAR10_CLASSES,
   output_dir="evaluation_results", metadata=None) và CLI entry point (python -m src.evaluation).
2. Chạy forward trên test dataloader với model.eval() và torch.inference_mode(); luôn khôi phục
   trạng thái model.training ban đầu trong khối finally.
3. Kiểm tra hợp lệ đầu vào (_validate_inputs): logits phải có shape [N, C] hữu hạn (không NaN/Inf),
   labels có shape [N] kiểu số nguyên nằm trong đoạn [0, C-1], dataloader không được rỗng.
4. Thu thập raw logits, tính xác suất bằng torch.softmax(logits, dim=1) và nhãn dự đoán argmax.
5. Tính toán đầy đủ 5 nhóm metrics định lượng:
   - Confusion Matrix KxK (hàng là true label, cột là predicted label; có cả bản đếm tuyệt đối
     và bản chuẩn hóa theo hàng khi vẽ hình).
   - Precision theo từng lớp, macro_avg và weighted_avg (quy ước zero_division=0).
   - Recall theo từng lớp, macro_avg và weighted_avg (quy ước zero_division=0).
   - F1-score theo từng lớp, macro_avg và weighted_avg (quy ước zero_division=0).
   - AUC-ROC đa lớp theo chiến lược One-vs-Rest (OvR) tính bằng quy tắc hình thang từ xác suất
     softmax trước argmax (tuyệt đối không dùng nhãn rời rạc), gồm AUC từng lớp và macro AUC.
     Nếu thiếu bất kỳ lớp dương nào trong y_true thì ném ValueError rõ ràng.
6. Tạo thư mục timestamped mới trong output_dir (evaluation_<YYYYMMDD_HHMMSS_ffffff>, exist_ok=False)
   để không bao giờ ghi đè kết quả cũ, và xuất đủ 4 artifacts:
   - evaluation_report.json (metadata, num_samples, num_classes, class_names, confusion_matrix, metrics).
   - metrics_per_class.csv (chi tiết 10 lớp + macro_avg + weighted_avg).
   - confusion_matrix.png (2 heatmap: counts và row-normalized).
   - roc_curve.png (đường cong ROC cho 10 lớp kèm macro AUC).
7. Trong hàm CLI main(): nhận --checkpoint (bắt buộc), --config, --data-dir, --output-dir, --device;
   khởi tạo model qua TransferModelBuilder(..., pretrained=False).build(), nạp model_state_dict(strict=True)
   và đánh giá trên CIFAR10DataModule.test_dataloader().

Sau khi sửa:
1. Chạy import test:
   .\venv\Scripts\python.exe -c "from src.evaluation import evaluate_model; print('Import PASS')"
2. Chạy unit test evaluation:
   .\venv\Scripts\python.exe -m pytest tests/test_evaluation.py -v
3. Báo command, output và exit code.
4. Không sửa file ngoài phạm vi.
```

---

### 4.3 Build Prompt 2: Execution, Debug & Review Prompts (Vận hành & Kiểm định)

#### 4.3.1 Execution Prompt (Chạy đánh giá cuối cùng trên tập Test CIFAR-10)
```text
Implement exactly one task: chạy đánh giá mô hình CIFAR-10 trên test set bằng checkpoint tốt nhất đã khóa.

Chỉ được thực hiện sau khi model đã hoàn tất huấn luyện và checkpoint tốt nhất (*_best.pth) đã được lưu.
Không sửa data split (50.000 train / 10.000 test), model architecture, training loop hoặc hyperparameter.

Lệnh thực hiện từ repository root:
.\venv\Scripts\python.exe -m src.evaluation --checkpoint <đường_dẫn_checkpoint_best.pth> --config configs/config.yaml --device auto

Yêu cầu báo cáo:
1. Model name, checkpoint path, checkpoint epoch, seed, device.
2. Test Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1, Macro AUC-ROC.
3. Bảng kết quả theo từng lớp (10 lớp CIFAR-10) và Top các cặp lớp bị nhầm lẫn nhiều nhất từ Confusion Matrix.
4. Đường dẫn đầy đủ của 4 artifacts: evaluation_report.json, metrics_per_class.csv, confusion_matrix.png, roc_curve.png.
5. Báo command đã chạy, exit code và xác nhận toàn bộ metric hữu hạn (không có NaN/Inf).
```

#### 4.3.2 Debug Prompt (Khắc phục lỗi phát sinh trong quá trình đánh giá)
```text
Debug đúng một lỗi dưới đây.

File/module:
<điền file, ví dụ: src/evaluation.py>

Command:
<điền command đã chạy>

Traceback/output:
<dán toàn bộ lỗi>

Expected behavior:
<mô tả kết quả mong muốn>

Hãy:
1. Xác định nguyên nhân gốc rễ dựa trên bằng chứng trong code và traceback.
2. Chỉ ra chính xác file và dòng cần sửa.
3. Đề xuất một thay đổi nhỏ nhất (minimal fix).
4. Thực hiện sửa đổi.
5. Chạy lại regression test (pytest tests/test_evaluation.py -v).
6. Báo kết quả trước và sau khi sửa.

Không sửa file không liên quan. Không dùng fallback im lặng hoặc broad exception.
```

#### 4.3.3 Review Prompt (Kiểm tra chất lượng trước khi Merge / Nộp báo cáo)
```text
Review toàn bộ diff hiện tại theo yêu cầu Practice 2 (Module Evaluation & Toàn hệ thống).

Kiểm tra:
- Đúng cấu hình 2 split chuẩn: train (50.000) và test (10.000).
- Classifier head đúng shape 10 classes và đúng freeze policy.
- Đúng chế độ model.eval() + torch.inference_mode() và khôi phục model.training.
- Confusion Matrix đúng kích thước 10x10 và đúng chiều (hàng là true label, cột là predicted label).
- Precision/Recall/F1 có đủ kết quả theo từng lớp (per-class) lẫn macro_avg và weighted_avg.
- AUC-ROC tính từ xác suất softmax (One-vs-Rest), tuyệt đối không tính từ argmax.
- Artifacts được lưu vào thư mục timestamped mới, không ghi đè lần chạy trước.
- Metric không chứa NaN/Inf và lưu kèm metadata (model, checkpoint, seed, device, config).

Chỉ trả về các finding có: File | Vấn đề | Mức độ nghiêm trọng | Bằng chứng | Test đề xuất.
Không tự sửa code.
```

---

### 4.4 Smoke Test và Lệnh kiểm tra (Verification Commands)

Tất cả các lệnh dưới đây được chạy từ thư mục gốc của dự án (`Practice-2---Hands-on-practice-with-pre-trained-neural-network-architectures`) bằng PowerShell. Smoke test chạy nhanh trên CPU với dữ liệu giả lập để xác minh toàn bộ hợp đồng I/O và tiêu chí nghiệm thu trước khi chạy trên 10.000 ảnh thật.

#### Lệnh 1: Chạy toàn bộ Unit Tests của Module Evaluation
```powershell
.\venv\Scripts\python.exe -m pytest tests/test_evaluation.py -v
```
*(Kết quả mong đợi: Toàn bộ `7 passed` với `exit code 0` trong < 5 giây).*

#### Lệnh 2: Smoke Test `evaluate_model()` trên CPU (Kiểm tra Tiêu chí 1, 2, 3, 4, 7, 8)
```powershell
@'
import json
import math
import tempfile
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.evaluation import evaluate_model

torch.manual_seed(42)
class_names = ("airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck")
# Tạo 40 mẫu giả lập đảm bảo đủ cả 10 lớp (mỗi lớp 4 mẫu)
x = torch.randn(40, 3, 16, 16)
y = torch.arange(10).repeat(4)
loader = DataLoader(TensorDataset(x, y), batch_size=12, shuffle=False)

model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 10))
model.train()  # Đặt ở train mode để kiểm tra evaluate_model có khôi phục lại không
before_weights = model[1].weight.detach().clone()

tmp_dir = Path(tempfile.mkdtemp())
report1 = evaluate_model(
    model=model,
    dataloader=loader,
    device="cpu",
    class_names=class_names,
    output_dir=tmp_dir,
    metadata={"model_name": "smoke_net", "checkpoint": "smoke.pth", "seed": 42},
)
report2 = evaluate_model(
    model=model,
    dataloader=loader,
    device="cpu",
    class_names=class_names,
    output_dir=tmp_dir,
    metadata={"model_name": "smoke_net", "checkpoint": "smoke.pth", "seed": 42},
)

# Kiểm tra bảo toàn trạng thái và trọng số (Tiêu chí 1)
assert model.training, "model.training phải được khôi phục về True"
assert torch.equal(before_weights, model[1].weight), "Trọng số model không được thay đổi khi evaluate"

# Kiểm tra Confusion Matrix 10x10 (Tiêu chí 2)
cm = report1["confusion_matrix"]
assert len(cm) == 10 and all(len(row) == 10 for row in cm)
assert sum(map(sum, cm)) == 40

# Kiểm tra Precision, Recall, F1, AUC-ROC (Tiêu chí 3, 4, 5)
m = report1["metrics"]
assert 0.0 <= m["accuracy"] <= 1.0
for avg_key in ("macro_avg", "weighted_avg"):
    for k, v in m[avg_key].items():
        assert math.isfinite(v) and 0.0 <= v <= 1.0, f"{avg_key}.{k} không hợp lệ: {v}"
assert len(m["per_class"]) == 10
for cls_name, cls_vals in m["per_class"].items():
    assert cls_vals["support"] == 4
    assert 0.0 <= cls_vals["auc_roc"] <= 1.0

# Kiểm tra 4 Artifacts & Không ghi đè (Tiêu chí 7, 8)
assert report1["artifact_dir"] != report2["artifact_dir"], "Không được ghi đè thư mục artifact cũ"
art_dir = Path(report1["artifact_dir"])
for fname in ("evaluation_report.json", "metrics_per_class.csv", "confusion_matrix.png", "roc_curve.png"):
    assert (art_dir / fname).is_file(), f"Thiếu artifact: {fname}"

saved_json = json.loads((art_dir / "evaluation_report.json").read_text(encoding="utf-8"))
assert saved_json["metadata"]["seed"] == 42

print("Evaluation smoke PASS | Accuracy:", round(m["accuracy"], 4), "| Macro AUC:", round(m["macro_avg"]["auc_roc"], 4))
'@ | .\venv\Scripts\python.exe -
```
*(Kết quả mong đợi: In ra `Evaluation smoke PASS | Accuracy: ... | Macro AUC: ...` với `exit code 0`).*

#### Lệnh 3: Smoke Test kiểm tra Bẫy lỗi (Kiểm tra Tiêu chí 5 & 6)
```powershell
@'
import tempfile
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.evaluation import evaluate_model

tmp_dir = tempfile.mkdtemp()
model = nn.Linear(4, 3)
classes = ("a", "b", "c")

# 1. Thiếu lớp thứ 3 (chỉ có nhãn 0 và 1) -> phải báo lỗi ValueError
loader_missing = DataLoader(TensorDataset(torch.randn(4, 4), torch.tensor([0, 0, 1, 1])), batch_size=2)
try:
    evaluate_model(model, loader_missing, class_names=classes, output_dir=tmp_dir)
    raise AssertionError("Phải ném ValueError khi thiếu lớp cho AUC-ROC")
except ValueError as e:
    assert "Missing true-label classes" in str(e)

# 2. Nhãn vượt quá phạm vi [0, 2] -> phải báo lỗi ValueError
loader_oob = DataLoader(TensorDataset(torch.randn(3, 4), torch.tensor([0, 1, 5])), batch_size=3)
try:
    evaluate_model(model, loader_oob, class_names=classes, output_dir=tmp_dir)
    raise AssertionError("Phải ném ValueError khi nhãn ngoài phạm vi")
except ValueError as e:
    assert "Labels must be in the range" in str(e)

print("Validation error-handling smoke PASS")
'@ | .\venv\Scripts\python.exe -
```
*(Kết quả mong đợi: In ra `Validation error-handling smoke PASS`).*

#### Lệnh 4: Chạy đánh giá thật trên tập Test CIFAR-10 (Normal Mode từ Checkpoint đã khóa)
```powershell
.\venv\Scripts\python.exe -m src.evaluation --checkpoint checkpoints\cifar10_resnet18_experiment_best.pth --config configs\config.yaml --device auto
Get-ChildItem -Recurse evaluation_results
```
*(Lưu ý: Thay đường dẫn `--checkpoint` bằng file `*_best.pth` thực tế trong `checkpoints/` hoặc `checkpoints/cifar10_checkpoints/`. Sau khi chạy xong, mở thư mục mới nhất trong `evaluation_results/` để lấy 2 biểu đồ `confusion_matrix.png`, `roc_curve.png` và bảng `metrics_per_class.csv` đưa vào báo cáo).*
