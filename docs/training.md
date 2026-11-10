# Training Specification Plan (Version 1.0)

> **Document Metadata**
> - **Module Name:** `src/trainer.py` (`ModelTrainer`), `src/logger.py` (`TensorBoardLogger`), được điều phối bởi `main.py`
> - **Created Date:** 09/11/2026
> - **Last Updated:** 09/11/2026
> - **Target Dataset:** CIFAR-10 (50.000 ảnh train / 10.000 ảnh test, 10 lớp)
> - **Config:** `configs/config.yaml`

---

## 1. Purpose (Mục đích)

Mục đích chính của pipeline training là huấn luyện **classifier mới** của mô hình Transfer Learning (do `TransferModelBuilder` tạo ra, backbone đã đóng băng) trên CIFAR-10, đánh giá trên tập test sau mỗi epoch, ghi toàn bộ quá trình lên **TensorBoard** và lưu **checkpoint tốt nhất** để resume hoặc đưa vào module evaluation.

### Lý do & Vai trò kỹ thuật:
1. **Tách Training Loop và Evaluation Loop:**
   - `training_loop()` chạy ở `model.train()` và cập nhật trọng số.
   - `validation()` chạy ở `model.eval()` kèm `torch.no_grad()`, chỉ đo metric, không cập nhật trọng số.
   - Hai chế độ khác nhau vì Dropout (classifier của VGG-16) và BatchNorm (ResNet-18, DenseNet-121) hành xử khác nhau giữa train và eval.
2. **Hàm mất mát và Optimizer cấu hình được:**
   - `nn.CrossEntropyLoss()` nhận trực tiếp raw logits `[B, 10]` (model không có Softmax).
   - Optimizer là `Adam` hoặc `SGD` (momentum 0.9), nhận `learning_rate` và `weight_decay` từ config.
   - Chỉ tham số `requires_grad=True` (head mới, vài nghìn tham số) cần được đưa vào optimizer.
3. **Weight Decay (L2 Regularization):** phạt trọng số lớn để giảm overfitting; được thử ở 3 mức `0`, `1e-4`, `1e-3`.
4. **Theo dõi bằng TensorBoard:** so sánh `Loss/Accuracy` của train và test theo epoch để phát hiện overfitting (train accuracy tăng nhưng test accuracy đứng yên), đồng thời lưu model graph, ảnh mẫu và config của mỗi run.
5. **Checkpoint tốt nhất và Resume:** chỉ ghi đè file `*_best.pth` khi test accuracy cao hơn; file này là đầu vào của `src/evaluation.py` và của lệnh `--checkpoint` để huấn luyện tiếp.
6. **Thí nghiệm hyperparameter có kiểm soát:** mỗi thí nghiệm chỉ đổi **một** yếu tố (learning rate, batch size hoặc weight decay) so với baseline; seed, subset dữ liệu và số epoch giữ cố định.

**Ngoài phạm vi tài liệu này:** pipeline dữ liệu (`data.md`), kiến trúc mô hình (`model_v1.md`), evaluation chuyên sâu như confusion matrix, precision, recall, F1, AUC-ROC (`evalution.md`).

---

## 2. Input/Output & Diagram (Thiết kế luồng và hợp đồng dữ liệu)

### 2.1 Sơ đồ luồng tổng thể (Pipeline Diagram)

```text
configs/config.yaml ──► load_config() ──► resolve_device()   (auto | cpu | cuda)
                              │
CIFAR10DataModule ───────────►├── train_loader (50.000 ảnh, shuffle=True)
                              └── test_loader  (10.000 ảnh, shuffle=False)
TransferModelBuilder ────────► model  (backbone frozen, head mới trainable)
                              │
                              ▼
        ModelTrainer(model, train_loader, test_loader,
                     learning_rate, optimizer_name, device, weight_decay)
                              │
                 ┌────────────┴─────────── lặp qua từng epoch ───────────┐
                 ▼                                                       ▼
   training_loop()  [model.train()]                  validation()  [model.eval() + no_grad]
   ──► (train_loss, train_acc)                       ──► (test_loss, test_acc)
                 └────────────┬──────────────────────────────────────────┘
                              ▼
                   TensorBoardLogger
                     ├─ log_epoch()        ──► runs/<experiment>/events.out.tfevents.*
                     ├─ save_best_model()  ──► checkpoints/cifar10_checkpoints/<experiment>_best.pth
                     ├─ flush()            (sau mỗi epoch)
                     └─ close()            (trong finally)
```

### 2.2 Năm bước của một batch trong `training_loop()` (5-Step Loop)

| Bước | Lệnh | Mục đích |
| :---: | :--- | :--- |
| **0** | `images, labels = images.to(device), labels.to(device)` | Đưa batch lên cùng device với model (CPU/CUDA). |
| **1** | `optimizer.zero_grad()` | Xóa gradient của batch trước, tránh cộng dồn. |
| **2** | `outputs = model(images)` | Forward pass, sinh logits `[B, 10]`. |
| **3** | `loss = criterion(outputs, labels)` | Tính CrossEntropyLoss trên batch. |
| **4** | `loss.backward()` | Lan truyền ngược, chỉ tham số `requires_grad=True` nhận gradient. |
| **5** | `optimizer.step()` | Cập nhật trọng số classifier. |

Sau mỗi batch, hai biến được cộng dồn: `total_loss += loss.item() * images.size(0)` và `correct += (predicted == labels).sum().item()`. Cuối epoch:
`avg_loss = total_loss / N`, `accuracy = 100 * correct / N` (với `N` là tổng số mẫu của loader). Cách cộng dồn có trọng số theo kích thước batch giúp batch cuối nhỏ hơn không làm lệch kết quả.

### 2.3 Cấu trúc Code và Vị trí lưu trữ (What & Where)

```text
src/trainer.py
└── class ModelTrainer
      ├── __init__(model, train_dataloader, test_dataloader, learning_rate=0.001,
      │            optimizer_name="Adam", device=None, weight_decay=0.0)
      ├── training_loop() -> (avg_loss: float, accuracy: float)   # model.train(), 5 bước
      ├── validation()    -> (avg_loss: float, accuracy: float)   # model.eval(), no_grad
      ├── evaluate = validation                                   # alias cho test suite
      └── fit(num_epochs) -> history: dict                        # train_loss, train_accuracy,
                                                                  # test_losses, test_accuracy
src/logger.py
└── class TensorBoardLogger  (hỗ trợ `with ... as logger`)
      ├── __init__(log_dir=None, experiment_name=None, checkpoint_dir=None)
      ├── log_epoch(epoch, train_loss, train_acc, test_loss, test_acc)
      ├── add_graph(model, input_to_model) / add_image_grid(tag, img_batch, ...) / add_text(tag, text)
      ├── save_checkpoint(model, optimizer, epoch, metrics, filename) -> Path
      ├── save_best_model(model, metric, optimizer, epoch, higher_is_better=True) -> bool
      ├── load_checkpoint(model, path, optimizer=None, device=None) -> dict
      └── flush() / close()

main.py
├── load_config(path)        # kiểm tra đủ khóa và giá trị hợp lệ
├── resolve_device(name)     # auto | cpu | cuda, báo lỗi rõ nếu không có CUDA
└── run(config, checkpoint_path=None) -> Path   # lắp data + model + trainer + logger, chạy epoch

tests/experiments.py          # quét hyperparameter (ResNet-18), ghi experiments.csv
tests/test_trainer.py         # 6 test cho ModelTrainer và CIFAR10DataModule
tests/test_logger.py          # 7 test cho TensorBoardLogger
```

### 2.4 Chi tiết Input / Output Contract

#### 2.4.1 `ModelTrainer`

| Thuộc tính | Đầu vào | Đầu ra |
| :--- | :--- | :--- |
| **Khởi tạo** | `model: nn.Module` (logits `[B, 10]`), hai `DataLoader` trả `(images [B,3,224,224] float32, labels [B] int64)`, `learning_rate: float`, `optimizer_name: "Adam" \| "SGD"`, `device`, `weight_decay: float` | Đối tượng `ModelTrainer`; model đã được `.to(device)`. |
| **`training_loop()`** | Không có tham số (dùng `train_dataloader`) | `(avg_loss, accuracy)`: `float`, accuracy tính theo **%** trong `[0, 100]`. |
| **`validation()`** | Không có tham số (dùng `test_dataloader`) | `(avg_loss, accuracy)`: cùng định dạng; không đổi trọng số. |
| **Lỗi** | `optimizer_name` ngoài `{"Adam", "SGD"}` | `ValueError` nêu rõ giá trị hợp lệ. |

#### 2.4.2 `configs/config.yaml`

| Khóa | Kiểu | Giá trị hiện tại | Ràng buộc (kiểm tra trong `load_config`) |
| :--- | :--- | :--- | :--- |
| `batch_size` | int | `64` | `> 0` |
| `learning_rate` | float | `0.001` | `> 0` |
| `epochs` | int | `5` | `> 0` |
| `model_name` | str | `resnet18` | `resnet18 \| vgg16 \| densenet121` (do `TransferModelBuilder` kiểm tra) |
| `device` | str | `cuda` | `auto \| cpu \| cuda` |
| `weight_decay` | float | `0.0001` | `>= 0` |
| `seed` | int | *(chưa có trong file)* | Spec yêu cầu có (xem tiêu chí 12). |

#### 2.4.3 Tag TensorBoard

| Tag | Loại | Ghi bởi | Step |
| :--- | :--- | :--- | :--- |
| `Loss/train`, `Loss/test` | scalar | `log_epoch()` | epoch (bắt đầu từ 1) |
| `Accuracy/train`, `Accuracy/test` | scalar | `log_epoch()` | epoch |
| `Loss`, `Accuracy` (train và test cùng biểu đồ) | scalars | `log_epoch()` | epoch |
| Model graph | graph | `add_graph()` với 2 ảnh mẫu | 0 |
| `Samples/train` | image grid (đã un-normalize) | `add_image_grid()` với tối đa 16 ảnh | 0 |
| `Configuration` | text (YAML của config) | `add_text()` | 0 |

#### 2.4.4 Checkpoint `*_best.pth`

| Khóa | Kiểu | Ghi chú |
| :--- | :--- | :--- |
| `epoch` | int | Epoch đạt metric tốt nhất. |
| `model_state_dict` | dict | Trọng số model. |
| `optimizer_state_dict` | dict | Có khi truyền `optimizer`; dùng để resume. |
| `metrics` | dict | Hiện là `{"best_metric": <test accuracy>}`. |
| `model_name` | str | **Yêu cầu của spec**, code hiện tại chưa ghi. |
| `config` | dict | **Yêu cầu của spec**, code hiện tại chưa ghi. |

Đường dẫn: `checkpoints/cifar10_checkpoints/<model_name>_<YYYYMMDD-HHMMSS>_best.pth`.


## 3. Success Criteria (Tiêu chí thành công & Nghiệm thu)

### Bảng tiêu chí

Pipeline training chỉ được nghiệm thu đạt chuẩn (**PASS**) khi thỏa mãn toàn bộ các tiêu chí sau:

| STT | Tiêu chí (Criteria) | Điều kiện chấp thuận (Acceptance Condition) | Cách kiểm tra |
| :---: | :--- | :--- | :--- |
| **1** | **Kiểu và miền giá trị metric** | `training_loop()` và `validation()` trả `(loss, acc)` kiểu `float`; loss hữu hạn và `>= 0`; `0 <= acc <= 100`. | `math.isfinite(loss)`, `0 <= acc <= 100`. |
| **2** | **Trung bình theo mẫu** | `loss = Σ(loss_batch × batch_size) / N`, đúng cả khi batch cuối nhỏ hơn; `acc = 100 × correct / N`. | So `validation()` với `F.cross_entropy(model(X), y)` trên toàn bộ tập nhỏ (sai số `< 1e-5`). |
| **3** | **Train/Eval mode** | `training_loop()` đặt `model.train()`; `validation()` đặt `model.eval()`, dùng `torch.no_grad()` và không đổi trọng số. | Kiểm tra `model.training` sau mỗi hàm; so `state_dict` trước/sau `validation()`. |
| **4** | **Cập nhật đúng tham số** | Sau `training_loop()`, tham số trainable thay đổi; tham số frozen (parameters) không đổi. | `torch.equal` trên weight của head và của backbone trước/sau. |
| **5** | **Optimizer chỉ nhận trainable params** | Số tensor trong `optimizer.param_groups` bằng số tensor có `requires_grad=True`. | `sum(len(g["params"]) for g in opt.param_groups)` so với đếm `requires_grad`. |
| **6** | **Optimizer và Device hợp lệ** | `optimizer_name` ngoài `{Adam, SGD}` ném `ValueError`; `device="cuda"` mà không có CUDA ném `RuntimeError` rõ ràng; model và batch cùng device. | `pytest.raises(ValueError)`; gọi `resolve_device("cuda")` trên máy không có GPU. |
| **7** | **TensorBoard scalar** | 4 tag `Loss/train`, `Loss/test`, `Accuracy/train`, `Accuracy/test`, mỗi tag đúng `E` điểm, step từ 1 đến `E`. | Đọc event file bằng `EventAccumulator`. |
| **8** | **TensorBoard artefact** | Event file có model graph, ảnh `Samples/train` và text `Configuration`. | `EventAccumulator.Tags()` có `graph`, `images`, `tensors`. |
| **9** | **Best checkpoint** | Chỉ ghi `<experiment>_best.pth` khi test accuracy cao hơn best trước; chứa `epoch`, `model_state_dict`, `optimizer_state_dict`, `metrics`, `model_name`, `config`. | Gọi `save_best_model` với chuỗi metric `75 → 70 → 82` (kỳ vọng `True, False, True`); `torch.load` và kiểm tra khóa. |
| **10** | **Load và Resume** | File không tồn tại ném `FileNotFoundError`; state không tương thích ném `RuntimeError` có thông báo rõ; sau khi load, trọng số khớp. | `pytest.raises`; `torch.allclose` giữa hai model. |
| **11** | **Cleanup và an toàn log** | `flush()` sau mỗi epoch; `close()` luôn được gọi (`finally` hoặc context manager); log chỉ chứa hyperparameter, không có secret. | Đọc `main.py` và event file. |
| **12** | **Tái lập (Reproducibility)** | `seed` nằm trong config, được set cho `torch` (và CUDA nếu có) và được ghi lại cùng kết quả. | Chạy hai lần cùng seed và so metric epoch 1. |
| **13** | **Tài liệu hóa mã nguồn** | 100% class và method có docstring mô tả chức năng, `Args`, `Returns`, `Raises`. | Đọc đầu mỗi class và hàm trong `src/trainer.py`, `src/logger.py`. |
| **14** | **Test suite** | `pytest tests/test_trainer.py tests/test_logger.py` kết thúc với exit code 0. | Lệnh 1 ở mục 4.4. |


---

## 4. How to do (Cách thực hiện & Hướng dẫn triển khai)

### 4.1 Quy trình kỹ thuật từng bước (Step-by-step Implementation)

1. **Bước 1: Khởi tạo `ModelTrainer`:**
   - Chọn device (`cuda` nếu có, ngược lại `cpu`), chuyển model lên device, tạo `nn.CrossEntropyLoss()`.
   - Optimizer chỉ nhận tham số trainable:
   ```python
   trainable_params = [p for p in self.model.parameters() if p.requires_grad]
   if optimizer_name == "Adam":
       self.optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
   elif optimizer_name == "SGD":
       self.optimizer = optim.SGD(trainable_params, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
   else:
       raise ValueError("optimizer_name phải là 'Adam' hoặc 'SGD'.")
   ```
2. **Bước 2: Viết `training_loop()` theo 5 bước:**
   ```python
   self.model.train()
   for images, labels in self.train_dataloader:
       images, labels = images.to(self.device), labels.to(self.device)
       
       self.optimizer.zero_grad()
       
       outputs = self.model(images)
       
       loss = self.criterion(outputs, labels)
       
       loss.backward()
       
       self.optimizer.step()
       
       total_loss += loss.item() * images.size(0)
       correct += (outputs.argmax(dim=1) == labels).sum().item()
       total += labels.size(0)
   return total_loss / total, 100.0 * correct / total
   ```
3. **Bước 3: Viết `validation()`:** giống bước 2 nhưng đặt `model.eval()`, bọc vòng lặp trong `with torch.no_grad():` và bỏ `zero_grad`, `backward`, `step`. Gán `evaluate = validation` để tương thích test suite.
4. **Bước 4: Hoàn thiện `TensorBoardLogger`:**
   - `log_epoch()` ghi 4 tag scalar; `add_graph()`, `add_image_grid()`, `add_text()` ghi graph, ảnh mẫu và config.
   - `save_best_model()` chỉ lưu khi metric tốt hơn; `load_checkpoint()` khôi phục model và optimizer.
   - `flush()` và `close()` (hoặc dùng `with TensorBoardLogger(...) as logger`).
5. **Bước 5: Lắp ráp trong `main.py`:**
   - `load_config()` → `resolve_device()` → data module → `TransferModelBuilder(...).build()` → `ModelTrainer` → `TensorBoardLogger`.
   - Trước vòng lặp: ghi graph (2 ảnh), image grid (tối đa 16 ảnh), text config.
   - Mỗi epoch: `training_loop()` → `validation()` → `log_epoch()` → `save_best_model(metric=test_accuracy)` → `flush()` → in kết quả.
   - `logger.close()` đặt trong `finally`.
6. **Bước 6: Resume (tùy chọn):** `python main.py --checkpoint <file>_best.pth`. `main.py` kiểm tra file tồn tại, nạp trọng số, cố nạp optimizer state (cảnh báo nếu không tương thích) và tiếp tục từ `epoch + 1`.
7. **Bước 7: Chạy thí nghiệm hyperparameter:** dùng `tests/experiments.py` (xem Lệnh 6 ở mục 4.4); mỗi cấu hình khởi tạo lại model và seed để so sánh công bằng.

---

### 4.2 Build Prompt 1: Trainer (Independent Prompt)

Dán đoạn sau vào AI khi cần tạo mới hoặc cập nhật `ModelTrainer`:

```text
Repository root:
D:\UTH\Deeplearning\lab\Practice-2---Hands-on-practice-with-pre-trained-neural-network-architectures

Implement exactly one task: hoàn thiện training và evaluation loop trong ModelTrainer.

Chỉ được sửa:
- src/trainer.py

Yêu cầu:
1. Khởi tạo ModelTrainer nhận model, train_dataloader, test_dataloader, learning_rate,
   optimizer_name ("Adam" hoặc "SGD"), device, weight_decay. Nếu device là None thì tự chọn cuda/cpu.
2. Hàm mất mát nn.CrossEntropyLoss(), nhận trực tiếp raw logits.
3. Optimizer chỉ nhận trainable parameters:
   filter(lambda p: p.requires_grad, self.model.parameters()).
   SGD dùng momentum=0.9. Cả hai optimizer nhận weight_decay.
4. Phương thức training_loop(): đặt model.train(); mỗi batch đưa lên device, zero_grad, forward,
   CrossEntropyLoss, backward, optimizer.step; trả (avg_loss, accuracy %) tính trên toàn bộ train_dataloader.
5. Phương thức validation(): đặt model.eval() và dùng torch.no_grad(); trả (avg_loss, accuracy %)
   trên toàn bộ test_dataloader, không cập nhật trọng số.
6. Loss trung bình phải cộng dồn loss.item() * batch_size rồi chia cho tổng số mẫu
   (đúng cả khi batch cuối nhỏ hơn).
7. Cung cấp alias evaluate = validation và phương thức fit(num_epochs) trả history dict
   gồm train_loss, train_accuracy, test_losses, test_accuracy.
8. Kiểm tra loss hữu hạn (không NaN/Inf), nếu không thì ném ValueError có thông báo rõ.
9. optimizer_name không hợp lệ phải ném ValueError nêu rõ các giá trị hỗ trợ; không dùng broad exception.
10. Viết docstring chi tiết ở đầu class và mỗi method (mô tả chức năng, Args, Returns, Raises).

Sau khi sửa:
1. Chạy import test:
   .\venv\Scripts\python.exe -c "from src.trainer import ModelTrainer; print('Import PASS')"
2. Chạy test của trainer:
   .\venv\Scripts\python.exe -m pytest tests/test_trainer.py -v
3. Báo train loss, train accuracy, test loss, test accuracy, command và exit code.
4. Không sửa file ngoài phạm vi.
```

---

### 4.3 Build Prompt 2: TensorBoard Logger & Checkpoint (Independent Prompt)

Dán đoạn sau vào AI khi cần tạo mới hoặc cập nhật `TensorBoardLogger`:

```text
Repository root:
D:\UTH\Deeplearning\lab\Practice-2---Hands-on-practice-with-pre-trained-neural-network-architectures

Implement exactly one task: hoàn thiện TensorBoard logging và checkpoint trong TensorBoardLogger.

Chỉ được sửa:
- src/logger.py

Yêu cầu:
1. Khởi tạo TensorBoardLogger với log_dir, experiment_name, checkpoint_dir; tự tạo thư mục nếu chưa có.
2. log_epoch() ghi 4 scalar: Loss/train, Loss/test, Accuracy/train, Accuracy/test.
3. add_graph() ghi model graph từ sample input; không được nuốt lỗi im lặng.
4. add_image_grid() ghi grid ảnh mẫu (tự un-normalize theo mean/std của ImageNet); add_text() ghi config.
5. save_best_model(): chỉ lưu khi test accuracy cao hơn best hiện tại (hỗ trợ higher_is_better=False cho loss);
   file <experiment_name>_best.pth.
6. Checkpoint dict gồm: epoch, model_state_dict, optimizer_state_dict, metrics, model_name, config.
7. load_checkpoint(): ném FileNotFoundError nếu file không tồn tại; ném RuntimeError có thông báo rõ nếu
   file hỏng hoặc state không tương thích với model; có thể nạp optimizer state.
8. Hỗ trợ flush(), close() và context manager (with ... as logger).
9. Không ghi secret hoặc dữ liệu nhạy cảm vào log.
10. Viết docstring chi tiết ở đầu class và mỗi method (mô tả chức năng, Args, Returns, Raises).

Sau khi sửa:
1. Chạy test logger:
   .\venv\Scripts\python.exe -m pytest tests/test_logger.py -v
2. Xác nhận event file TensorBoard và file checkpoint .pth được tạo thành công.
3. Báo command, output và exit code.
4. Không sửa file ngoài phạm vi.
```

---

### 4.4 Smoke Test và Lệnh kiểm tra (Verification Commands)

Tất cả lệnh chạy từ repository root bằng PowerShell. Smoke test phải nhanh, dùng dữ liệu nhỏ trên CPU và không thay thế test đầy đủ. Nếu smoke test thất bại, **không** chạy normal mode và **không** kết luận PASS.

#### Lệnh 1: Chạy unit test của Trainer và Logger
```powershell
.\venv\Scripts\python.exe -m pytest tests/test_trainer.py tests/test_logger.py -v
```
*(Kết quả mong đợi: 13 test passed, exit code 0. Nếu thư mục `data/` chưa có CIFAR-10, test data module sẽ tự tải về nên chạy lần đầu có thể lâu.)*

#### Lệnh 2: Smoke test Trainer (CPU, dữ liệu giả, kiểm tra tiêu chí 1 đến 5)
```powershell
@'
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.trainer import ModelTrainer

torch.manual_seed(42)
x = torch.randn(32, 3, 8, 8)
y = torch.randint(0, 10, (32,))
loader = DataLoader(TensorDataset(x, y), batch_size=10)  # batch: 10, 10, 10, 2 (batch cuối lẻ)

model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 10))
before = model[1].weight.detach().clone()
trainer = ModelTrainer(model, loader, loader, learning_rate=0.01,
                       optimizer_name="Adam", device=torch.device("cpu"), weight_decay=1e-4)

train_loss, train_acc = trainer.training_loop()
test_loss, test_acc = trainer.validation()

assert all(math.isfinite(v) for v in (train_loss, train_acc, test_loss, test_acc))
assert 0 <= train_acc <= 100 and 0 <= test_acc <= 100
assert not torch.equal(before, model[1].weight), "Trọng số không đổi sau training_loop"
assert not model.training, "validation() phải để model ở eval mode"
with torch.no_grad():
    expected = F.cross_entropy(model(x), y).item()
assert abs(test_loss - expected) < 1e-5, "Loss không phải trung bình theo mẫu"

# Tiêu chí 5: optimizer chỉ nhận trainable params
frozen = nn.Sequential(nn.Flatten(), nn.Linear(192, 16), nn.Linear(16, 10))
for p in frozen[1].parameters():
    p.requires_grad = False
t2 = ModelTrainer(frozen, loader, loader, device=torch.device("cpu"))
n_opt = sum(len(g["params"]) for g in t2.optimizer.param_groups)
n_trainable = sum(1 for p in frozen.parameters() if p.requires_grad)
assert n_opt == n_trainable, f"Optimizer có {n_opt} tensor, trainable chỉ {n_trainable}"

print("Trainer smoke PASS", train_loss, train_acc, test_loss, test_acc)
'@ | .\venv\Scripts\python.exe -
```
*(Kỳ vọng sau khi sửa theo Build Prompt 1: in `Trainer smoke PASS`. Với code hiện tại, phần tiêu chí 5 sẽ báo lỗi `AssertionError` vì optimizer đang nhận cả tham số frozen.)*

#### Lệnh 3: Smoke test Logger và Checkpoint (kiểm tra tiêu chí 7 đến 10)
```powershell
@'
import tempfile
from pathlib import Path
import torch
from torch import nn
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from src.logger import TensorBoardLogger

tmp = Path(tempfile.mkdtemp())
model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 10))
opt = torch.optim.Adam(model.parameters())

with TensorBoardLogger(log_dir=tmp / "runs", experiment_name="smoke", checkpoint_dir=tmp / "ckpt") as lg:
    lg.add_graph(model, torch.randn(2, 3, 8, 8))
    lg.add_image_grid("Samples/train", torch.randn(4, 3, 8, 8), nrow=2)
    lg.add_text("Configuration", "learning_rate: 0.001")
    saved = []
    for epoch, acc in enumerate([50.0, 60.0, 55.0], start=1):
        lg.log_epoch(epoch, 1.0 / epoch, acc, 1.2 / epoch, acc - 5)
        saved.append(lg.save_best_model(model, acc, opt, epoch))
        lg.flush()
    assert saved == [True, True, False], saved

    best = tmp / "ckpt" / "smoke_best.pth"
    assert best.exists()
    new_model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 10))
    ckpt = lg.load_checkpoint(new_model, best)
    assert ckpt["epoch"] == 2
    for a, b in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(a, b)

ea = EventAccumulator(str(tmp / "runs" / "smoke"), size_guidance={"scalars": 0, "images": 0, "tensors": 0})
ea.Reload()
tags = ea.Tags()
for t in ("Loss/train", "Loss/test", "Accuracy/train", "Accuracy/test"):
    assert t in tags["scalars"] and len(ea.Scalars(t)) == 3, t
assert tags["graph"], "Thiếu model graph"
assert "Samples/train" in tags["images"], "Thiếu image grid"
assert any(t.startswith("Configuration") for t in tags["tensors"]), "Thiếu text config"
print("Logger smoke PASS")
'@ | .\venv\Scripts\python.exe -
```
*(Kỳ vọng: in `Logger smoke PASS`. Nếu `add_graph` lỗi âm thầm, dòng `Thiếu model graph` sẽ báo ra.)*

#### Lệnh 4: Chạy pipeline thật (Normal Mode) và mở TensorBoard
```powershell
.\venv\Scripts\python.exe main.py --config configs\config.yaml
.\venv\Scripts\tensorboard.exe --logdir runs
Get-ChildItem -Recurse runs
Get-ChildItem -Recurse checkpoints
```
*(Mở `http://localhost:6006`. Kiểm tra thấy 4 biểu đồ Loss/Accuracy, tab Graphs, tab Images và tab Text; ghi lại command, device, seed, metric từng epoch và đường dẫn checkpoint. Đường dẫn do `main.py` in ra có thể chưa đúng thư mục)*

#### Lệnh 5: Smoke test Resume
```powershell
.\venv\Scripts\python.exe main.py --config configs\config.yaml --checkpoint checkpoints\cifar10_checkpoints\<tên_file>_best.pth
```
*(Kỳ vọng: in `Loaded checkpoint: ... (epoch=N)` và tiếp tục từ epoch `N+1`. Nếu `N` đã bằng `epochs` trong config thì vòng lặp không chạy thêm epoch nào.)*

#### Lệnh 6: Smoke test thí nghiệm Hyperparameter
```powershell
.\venv\Scripts\python.exe -m tests.experiments --epochs 1 --train-samples 256 --val-samples 128
.\venv\Scripts\python.exe -m tests.experiments --sweep --epochs 3 --train-samples 1000 --val-samples 500
```
*(Lệnh đầu chỉ chạy baseline để kiểm tra pipeline; với 256 ảnh và 1 epoch, accuracy chỉ quanh mức ngẫu nhiên (~10%), nên **không** dùng để đánh giá chất lượng. Lệnh sau chạy cả 7 cấu hình và ghi `experiments.csv`; lưu ý `ROOT` trong `tests/experiments.py`.)*
