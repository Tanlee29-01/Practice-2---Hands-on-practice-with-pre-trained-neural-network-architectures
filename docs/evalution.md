# Evaluation, debug and test plan

## Define

Đánh giá phải trả lời: model nào tốt hơn, hyperparameter nào phù hợp, kết quả
có tái lập không, và lỗi tập trung ở lớp nào. Accuracy là metric bắt buộc;
loss và confusion matrix là metric/phân tích bổ sung khi có thể.

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

Lưu command, exit code, metric output, TensorBoard tag list và checkpoint path
trong kết quả thực nghiệm/report. Nếu test không chạy được do môi trường,
ghi rõ blocker và không đánh dấu Definition of Done là passed.