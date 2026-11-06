# Training and TensorBoard plan

## Define

Training phải có training loop và evaluation loop riêng, dùng CrossEntropyLoss,
optimizer cấu hình được, device rõ ràng, checkpoint tốt nhất và TensorBoard.
Hyperparameter tối thiểu: `learning_rate`, `batch_size`, `epochs`, `weight_decay`,
`model_name`, `device`, `seed`.

## Design

### One epoch

1. Set model train mode: `model.train()`.
2. Đưa batch lên device (CPU/CUDA).
3. Zero gradient: `optimizer.zero_grad()`.
4. Forward, tính loss với CrossEntropyLoss, backward và optimizer step.
5. Cộng dồn loss/correct samples và trả mean loss/accuracy trên toàn bộ tập train.

Evaluation loop thực hiện trên tập `test` (qua `test_dataloader`), dùng
`model.eval()` và `torch.no_grad()`, không cập nhật trọng số. Metric phải được tính
trên toàn bộ loader, không phải trung bình không trọng số của batch cuối.
Chỉ truyền parameter cần học (`requires_grad=True`) vào optimizer.

### Hyperparameter experiments

Baseline đề xuất: Adam, `lr=0.001`, `batch_size=64`, `weight_decay=0.0001`.
Thay từng yếu tố một: lr thấp/cao, batch nhỏ/lớn, weight decay bằng 0/cao.
Giữ seed, split (50.000 train / 10.000 test), epochs và model cố định; lưu kết quả dạng bảng.

### TensorBoard

Mỗi run có tên chứa model và timestamp/seed. Ghi:

- `Loss/train`, `Loss/test`
- `Accuracy/train`, `Accuracy/test`
- model graph từ một input mẫu
- ảnh mẫu và config/hyperparameter
- best metric và checkpoint path

Flush sau mỗi epoch và close writer trong cleanup. Không ghi secret hoặc dữ
liệu nhạy cảm vào log.

### Checkpoint

Checkpoint tối thiểu gồm model state_dict, optimizer state_dict, epoch, metric tốt nhất,
model name và config. Chỉ cập nhật best checkpoint khi test metric tốt hơn
(ví dụ: test accuracy cao hơn hoặc test loss thấp hơn). Resume phải kiểm tra
file tồn tại và báo lỗi rõ nếu state không tương thích.

## Build prompt 1: Trainer (src/trainer.py)

```text
Implement exactly one task: hoàn thiện training và evaluation loop trong ModelTrainer.

Làm việc từ repository root và tuân thủ pattern OOP hiện có.
Chỉ được sửa:
- src/trainer.py

Yêu cầu:
- Khởi tạo ModelTrainer nhận model, train_dataloader, test_dataloader,
  learning_rate, optimizer_name ("Adam" hoặc "SGD"), device, weight_decay.
- Hàm mất mát nn.CrossEntropyLoss().
- Optimizer chỉ nhận trainable parameters: filter(lambda p: p.requires_grad, self.model.parameters()).
- Phương thức training_loop(): đặt model.train(), zero_grad, forward, backward, optimizer.step,
  tính chính xác trung bình loss và accuracy (%) trên toàn bộ train_dataloader.
- Phương thức validation() (hoặc evaluate()): đặt model.eval() với torch.no_grad(), forward trên test_dataloader,
  tính chính xác trung bình loss và accuracy (%) trên toàn bộ test_dataloader.
- Cung cấp alias evaluate = validation để tương thích với test suite.
- Hỗ trợ cả CPU và CUDA linh hoạt; kiểm tra tensor finite (không trả NaN/Inf).
- Báo lỗi ValueError rõ ràng nếu optimizer_name không hợp lệ; không dùng broad exception.

Sau khi sửa:
1. Chạy import test:
   .\venv\Scripts\python.exe -c "from src.trainer import ModelTrainer; print('Import PASS')"
2. Chạy one-epoch CPU smoke test trên subset nhỏ:
   .\venv\Scripts\python.exe -m pytest tests/test_trainer.py
3. Báo train loss, train accuracy, test loss, test accuracy và exit code.
4. Không sửa file ngoài phạm vi.
```

## Build prompt 2: TensorBoard Logger & Checkpoint (src/logger.py)

```text
Implement exactly one task: hoàn thiện TensorBoard logging và checkpoint trong TensorBoardLogger.

Làm việc từ repository root và tuân thủ pattern OOP hiện có.
Chỉ được sửa:
- src/logger.py

Yêu cầu:
- Khởi tạo TensorBoardLogger với log_dir và checkpoint_dir.
- Ghi scalar: Loss/train, Loss/test, Accuracy/train, Accuracy/test qua phương thức log_epoch().
- Ghi model graph từ sample input qua add_graph().
- Ghi grid ảnh mẫu qua add_image_grid().
- Ghi config dưới dạng text qua add_text().
- Quản lý checkpoint:
  + save_best_model(): lưu checkpoint khi test accuracy cao hơn best hiện tại.
  + Checkpoint dict gồm: epoch, model_state_dict, optimizer_state_dict, metric, model_name, config.
  + load_checkpoint(): khôi phục trọng số an toàn, báo lỗi FileNotFoundError / RuntimeError rõ ràng nếu file hỏng.
- Có flush() sau mỗi epoch và close() để đóng SummaryWriter.

Sau khi sửa:
1. Chạy unit/smoke test cho logger:
   .\venv\Scripts\python.exe -m pytest tests/test_logger.py
2. Xác nhận event file TensorBoard và file checkpoint .pth được tạo thành công.
3. Báo command, output và exit code.
4. Không sửa file ngoài phạm vi.
```

## Verify

Smoke run phải hoàn tất, loss hữu hạn, accuracy nằm trong `[0, 100]`, model
parameters thay đổi khi classifier trainable, event file đọc được bằng
TensorBoard và checkpoint load lại được vào model tương thích.

## Explain

Nêu rõ sự khác nhau giữa training và evaluation mode, vai trò của optimizer,
weight decay, checkpoint tốt nhất và ý nghĩa của việc theo dõi metrics trên test set.