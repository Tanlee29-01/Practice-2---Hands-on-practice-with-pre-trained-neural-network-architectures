# Training and TensorBoard plan

## Define

Training phải có training loop và evaluation loop riêng, dùng CrossEntropyLoss,
optimizer cấu hình được, device rõ ràng, checkpoint tốt nhất và TensorBoard.
Hyperparameter tối thiểu: `learning_rate`, `batch_size`, `epochs`, `weight_decay`,
`model_name`, `device`, `seed`.

## Design

### One epoch

1. Set model train mode.
2. Đưa batch lên device.
3. Zero gradient.
4. Forward, tính loss, backward và optimizer step.
5. Cộng loss/correct samples và trả mean loss/accuracy.

Evaluation dùng `eval()` và `no_grad()`, không cập nhật trọng số. Metric phải
được tính trên toàn bộ loader, không phải trung bình không trọng số của batch
cuối.

### Hyperparameter experiments

Baseline đề xuất: Adam, `lr=0.001`, `batch_size=64`, `weight_decay=0.0001`.
Thay từng yếu tố một: lr thấp/cao, batch nhỏ/lớn, weight decay bằng 0/cao.
Giữ seed, split, epochs và model cố định; lưu kết quả dạng bảng. Nếu so sánh
model, dùng cùng protocol và validation split.

### TensorBoard

Mỗi run có tên chứa model và timestamp/seed. Ghi:

- `Loss/train`, `Loss/validation`
- `Accuracy/train`, `Accuracy/validation`
- model graph từ một input mẫu
- ảnh mẫu và config/hyperparameter
- best metric và checkpoint path

Flush sau mỗi epoch và close writer trong cleanup. Không ghi secret hoặc dữ
liệu nhạy cảm vào log.

### Checkpoint

Checkpoint tối thiểu gồm model state, optimizer state, epoch, metric tốt nhất,
model name và config. Chỉ cập nhật best checkpoint khi validation metric tốt
hơn. Resume phải kiểm tra file tồn tại và báo lỗi rõ nếu state không tương thích.

## Build prompt

```text
Implement only the trainer and logger integration for one epoch at a time.
Use train/eval mode correctly, CrossEntropyLoss, configurable Adam parameters,
finite loss/accuracy, TensorBoard scalars, and best checkpoints.
Do not add silent fallbacks or broad exception handling.
Run a one-epoch CPU smoke test on a tiny deterministic subset.
Report metrics, event file, and checkpoint path.
```

## Verify

Smoke run phải hoàn tất, loss hữu hạn, accuracy nằm trong `[0, 100]`, model
parameters thay đổi khi classifier trainable, event file đọc được bằng
TensorBoard và checkpoint load lại được vào model tương thích.

## Explain

Nêu rõ sự khác nhau giữa training và evaluation mode, vai trò của optimizer,
weight decay, checkpoint tốt nhất và lý do không dùng test set để tuning.