Bạn nên ra lệnh cho AI theo từng task nhỏ, không yêu cầu viết toàn bộ project trong một prompt. Quy trình:

`Description → Generate → Smoke Test → Run Normal → Check`

## 0. Quy tắc chạy chung

Tất cả command phải được chạy từ repository root:

```text
D:\UTH\Deeplearning\lab\Practice-2---Hands-on-practice-with-pre-trained-neural-network-architectures
```

### Cấu trúc file cần tuân thủ

```text
src\data_module.py
src\model_builder.py
src\trainer.py
src\logger.py
src\eda.py
src\evaluation.py
configs\config.yaml
main.py
tests\
docs\
```

Khi yêu cầu AI viết code, luôn ghi rõ:

- Chỉ sửa file được liệt kê trong task.
- Không tạo file mới ngoài phạm vi đã cho phép.
- Không sửa các thay đổi chưa liên quan của người khác.
- Dùng đường dẫn Windows với dấu `\`.
- Sau mỗi thay đổi phải báo file đã sửa, command đã chạy và output.
- Nếu smoke test thất bại, không được chạy normal mode hoặc kết luận PASS.

### Prompt context dùng chung

Dán đoạn sau vào đầu mỗi prompt viết code:

```text
Repository root:
D:\UTH\Deeplearning\lab\Practice-2---Hands-on-practice-with-pre-trained-neural-network-architectures

Làm việc từ repository root và tuân thủ cấu trúc hiện tại:
- src\data_module.py
- src\model_builder.py
- src\trainer.py
- src\logger.py
- src\eda.py
- src\evaluation.py
- configs\config.yaml
- main.py
- tests\
- docs\

Chỉ sửa các file được nêu trong task.
Không tạo file ngoài phạm vi.
Sau khi sửa, chạy test nhỏ nhất phù hợp và báo command, output, exit code.
```

### Lệnh kiểm tra nền tảng

```powershell
Get-Location
Get-ChildItem
.\venv\Scripts\python.exe --version
.\venv\Scripts\python.exe -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__)"
```

Nếu repository không dùng `venv`, phải xác định Python interpreter thực tế trước
khi chạy các lệnh tiếp theo; không tự ý tạo môi trường mới trong task viết code.

## 1. Prompt bắt đầu: Description

Dán vào Copilot Chat:

```text
Hãy đọc repository hiện tại và mô tả task Practice 2.

Phân tích:
- Cấu trúc project hiện tại
- Các file source đang có
- Chức năng đã hoàn thành
- Chức năng còn thiếu
- Dependency và entry point
- Rủi ro khi chỉnh sửa
- Minimum test cần chạy

Chưa được sửa file nào. Chỉ báo cáo kết quả và đề xuất task nhỏ đầu tiên.
```

Sau Description, yêu cầu AI kiểm tra phạm vi trước khi Generate:

```text
Dựa trên Description vừa tạo, hãy xác nhận:
- Task nhỏ duy nhất sẽ thực hiện
- File được phép sửa
- File không được sửa
- Expected behavior
- Smoke test tương ứng

Chưa được viết code.
```

## 2. Viết module dữ liệu

```text
Implement exactly one task: hoàn thiện data module cho CIFAR-10.

Làm việc từ repository root và tuân thủ pattern OOP hiện có.
Chỉ được sửa:
- src/data_module.py

Yêu cầu:
- Tải CIFAR-10 bằng torchvision.datasets.CIFAR10 gồm 2 split:
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
1. Chạy import test
2. Chạy one-batch smoke test cho cả train_dataloader và test_dataloader
3. Báo image shape, label shape, số lượng mẫu từng split (50.000 train, 10.000 test) và exit code
4. Không sửa file ngoài phạm vi
```

Smoke test tối thiểu cho data module:

```powershell
.\venv\Scripts\python.exe -c "from src.data_module import CIFAR10DataModule; dm = CIFAR10DataModule(batch_size=4); dm.setup(); print('Train:', len(dm.train_dataloader().dataset)); print('Test:', len(dm.test_dataloader().dataset))"
```

## 3. Smoke test module dữ liệu

```text
Chạy smoke test cho data module vừa triển khai.

Dùng subset nhỏ hoặc batch đầu tiên, không chạy full training.
Kiểm tra:
- Import không lỗi
- Có đủ 2 loader: train_dataloader (50.000 ảnh), test_dataloader (10.000 ảnh)
- Batch image là tensor 4 chiều [batch_size, 3, 224, 224]
- Label là tensor 1 chiều [batch_size]
- Batch size khớp
- Label nằm trong khoảng 0 đến 9

Nếu lỗi, giải thích nguyên nhân trước rồi mới sửa đúng file liên quan.
Báo chính xác command và output.
```

## 4. Viết model builder

```text
Implement exactly one task: hoàn thiện model builder cho transfer learning.

Chỉ được sửa:
- src/model_builder.py

Yêu cầu:
- Hỗ trợ ResNet-18, VGG-16 và DenseNet-121
- Dùng pretrained weights theo API torchvision hiện tại
- Thay classifier cuối thành 10 classes
- Freeze backbone mặc định
- Có policy fine-tune rõ ràng
- Model không hợp lệ phải báo ValueError
- Không dùng softmax trước CrossEntropyLoss

Sau khi sửa, chạy forward smoke test cho cả 3 model.
Xác nhận output có shape [batch_size, 10].
Báo số parameter trainable.
```

Smoke test tối thiểu cho model:

```powershell
.\venv\Scripts\python.exe -c "from src.model_builder import TransferModelBuilder; print('model module import passed')"
```

Nếu model cần tải pretrained weights, phải ghi rõ việc tải weights có thể cần
Internet và không được coi lỗi mạng là lỗi logic của model nếu chưa kiểm tra
được nguyên nhân.

## 5. Viết trainer

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

Sau khi sửa, chạy one-epoch smoke test trên subset nhỏ bằng CPU.
Báo train loss, train accuracy, test loss và test accuracy.
```

Không chạy full CIFAR-10 trong Smoke Test. Chỉ chuyển sang normal mode sau khi
one-epoch smoke test hoàn tất và metric là số hữu hạn.

## 6. Viết TensorBoard và checkpoint

```text
Implement exactly one task: hoàn thiện TensorBoard logging và checkpoint trong TensorBoardLogger.

Làm việc từ repository root và tuân thủ pattern OOP hiện có.
Chỉ được sửa:
- src/logger.py

Yêu cầu:
- Ghi train/test loss
- Ghi train/test accuracy
- Ghi model graph từ sample input
- Ghi ảnh mẫu và text config
- Lưu best checkpoint dựa trên test metric (test accuracy cao nhất hoặc test loss thấp nhất)
- Checkpoint chứa: model_state_dict, optimizer_state_dict, epoch, metric, model_name, config
- Hỗ trợ load_checkpoint() an toàn, báo lỗi rõ ràng nếu file hỏng hoặc không tồn tại
- Flush sau mỗi epoch và close writer khi kết thúc
- Không ghi secret hoặc dữ liệu nhạy cảm

Sau khi sửa, chạy test tạo event file và checkpoint.
Xác nhận cả hai artifact tồn tại và checkpoint có thể load lại.
```

Kiểm tra artifact bằng PowerShell:

```powershell
Get-ChildItem -Recurse runs
Get-ChildItem -Recurse checkpoints
```

## 7. Kết nối vào `main.py`

```text
Implement exactly one task: kết nối data, model, trainer và logger trong main.py.

Làm việc từ repository root và tuân thủ pattern hiện có.
Chỉ được sửa:
- main.py

Yêu cầu:
- Đọc config hiện có (batch_size, learning_rate, epochs, model_name, device, weight_decay).
- Khởi tạo data module và lấy train_loader (50.000 ảnh), test_loader (10.000 ảnh).
- Khởi tạo TransferModelBuilder với model_name (ResNet-18, VGG-16 hoặc DenseNet-121).
- Khởi tạo ModelTrainer với train_loader và test_loader.
- Khởi tạo TensorBoardLogger.
- Chạy training loop từng epoch, tính train/test loss và accuracy.
- Ghi log TensorBoard và lưu best checkpoint theo test metric.
- Hỗ trợ resume checkpoint nếu được chỉ định (--checkpoint).
- Báo lỗi rõ ràng khi config hoặc checkpoint không hợp lệ.

Không thay đổi behavior ngoài yêu cầu.
Sau khi sửa, chạy smoke run 1 epoch trên subset nhỏ.
```

Smoke run phải thực hiện từ repository root:

```powershell
.\venv\Scripts\python.exe main.py --config configs\config.yaml
```

Nếu config hiện tại chạy full dataset hoặc nhiều epoch, AI phải dùng một config
test được cho phép hoặc test entry point tương đương; không tự ý thay đổi config
production để làm cho smoke test thành công.

## 8. Chạy chế độ normal

Sau khi smoke test passed, yêu cầu AI chạy thật:

```text
Chuyển sang Run Normal Mode.

Chạy pipeline bằng config hiện tại:
- Dùng model trong configs/config.yaml
- Chạy training bình thường
- Ghi đầy đủ TensorBoard log
- Tạo best checkpoint
- In metric của từng epoch
- Ghi device, seed, model và hyperparameters

Không tự ý đổi config để che lỗi.
Sau khi chạy xong, báo:
- Command đã chạy
- Exit code
- Train và test metrics
- TensorBoard directory
- Checkpoint path
- Các lỗi hoặc giới hạn còn lại
```
- TensorBoard directory
- Checkpoint path
- Các lỗi hoặc giới hạn còn lại
```

Nếu chạy bằng PowerShell, command thường là:

```powershell
.\venv\Scripts\python.exe main.py --config configs\config.yaml
```

Mở TensorBoard:

```powershell
.\venv\Scripts\tensorboard.exe --logdir runs
```

Kiểm tra trạng thái sau normal run:

```powershell
Get-ChildItem -Recurse runs
Get-ChildItem -Recurse checkpoints
git status --short
```

## 9. EDA dữ liệu hình ảnh

Chỉ chuyển sang bước này sau khi data module smoke test passed.

```text
Implement exactly one task: hoàn thiện và chạy EDA cho CIFAR-10.

Làm việc từ repository root.
Chỉ được sửa:
- src/eda.py
- notebooks/cifar10_eda.ipynb (nếu cập nhật notebook phân tích)

Không sửa source training hoặc dataset gốc.
Không thay đổi split, label, transform production hay hyperparameter.

Kiểm tra và báo cáo:
- Resolution gốc (32x32x3), sau transform (224x224x3), số kênh, dtype và shape.
- Class distribution theo từng lớp ở 2 split: train (50.000) và test (10.000).
- Pixel min, max, mean, standard deviation theo từng kênh RGB trước normalize.
- Kiểm tra không có NaN, Inf hoặc giá trị pixel ngoài khoảng [0, 255].
- Xuất kết quả ra artifact (JSON/CSV) hoặc visualization.

Sau khi sửa:
1. Chạy smoke test:
   .\venv\Scripts\python.exe -m src.eda --max-samples 100
2. Chạy full EDA:
   .\venv\Scripts\python.exe -m src.eda
3. Báo command, output, exit code, seed, nguồn dữ liệu và artifact path.
Không kết luận PASS khi thiếu số liệu thực tế.
```

Điều kiện chuyển bước:

- Resolution thực tế khớp contract dataset.
- Có đủ phân phối nhãn của mọi split.
- Có pixel statistics theo kênh và không gian giá trị rõ ràng.
- Không phát hiện lỗi dữ liệu chưa được giải thích.

## 10. Evaluation metrics

### 10.1 Xây dựng module evaluation

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

### 10.2 Đánh giá checkpoint trên test set

Chỉ chạy sau khi normal training đã tạo checkpoint hợp lệ và đã khóa siêu tham số.

```text
Implement exactly one task: đánh giá checkpoint CIFAR-10 trên test set.

Không sửa model, trainer, data split hoặc config training.
Không dùng test set để tuning hay chọn lại checkpoint.

Lệnh thực hiện từ repository root:
.\venv\Scripts\python.exe -m src.evaluation --checkpoint <đường_dẫn_checkpoint_best.pth> --config configs\config.yaml --device auto

Sau khi chạy xong, báo:
- Confusion matrix 10x10, hàng là true label và cột là predicted label.
- Precision theo lớp, macro average và weighted average.
- Recall theo lớp, macro average và weighted average.
- F1-score theo lớp, macro average và weighted average.
- AUC-ROC one-vs-rest cho 10 lớp bằng score/xác suất trước argmax.
- Báo checkpoint, model, seed, device, command, output, exit code và artifact path.
```

Smoke test evaluation:

```powershell
.\venv\Scripts\python.exe -m pytest tests/test_evaluation.py
```

Normal evaluation:

```powershell
.\venv\Scripts\python.exe -m src.evaluation --checkpoint checkpoints\cifar10_checkpoints\<experiment>_best.pth --config configs\config.yaml --device auto
```

## 11. Check with Description

```text
Check with Description.

Đối chiếu code và kết quả vừa chạy với yêu cầu Practice 2:
- Có đủ ResNet-18, VGG-16 và DenseNet-121 chưa?
- Classifier đã đổi thành 10 classes chưa?
- Backbone freeze đúng policy chưa?
- Có đủ 2 split train (50.000) và test (10.000) chưa?
- Có tuning learning rate, batch size và weight decay chưa?
- EDA có resolution, class distribution và pixel statistics chưa?
- Evaluation có confusion matrix, precision, recall, F1-score và AUC-ROC chưa?
- AUC-ROC có dùng score/xác suất trước argmax và one-vs-rest chưa?
- TensorBoard có scalar, graph, image và config chưa?
- Checkpoint có lưu và load được chưa?
- Metric có hữu hạn và hợp lý không?
- Có thay đổi ngoài phạm vi không?
- Definition of Done đã đạt chưa?

Chỉ kết luận PASS hoặc FAIL kèm bằng chứng command/output.
Không sửa code trong bước review này.
```

## 12. Khi có lỗi

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

## 13. Review trước khi commit

```text
Review toàn bộ diff hiện tại theo yêu cầu Practice 2.

Kiểm tra:
- Data leakage
- Sai classifier shape
- Sai freeze/fine-tune
- Sai device
- Sai loss/accuracy
- Sai confusion matrix hoặc đảo true/predicted axis
- Precision/recall/F1 chỉ có aggregate, thiếu kết quả theo lớp
- AUC-ROC tính từ argmax thay vì score/xác suất
- EDA làm thay đổi dữ liệu hoặc dùng test để tuning
- Sai train/eval mode
- Thiếu TensorBoard artifact
- Checkpoint không đầy đủ
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

Điểm quan trọng: mỗi prompt chỉ giao **một nhiệm vụ**, luôn yêu cầu AI chạy test sau khi viết code, và không chấp nhận câu trả lời “đã hoàn thành” nếu chưa có command, output và artifact cụ thể.

## 14. Mẫu lệnh theo quy trình

### Description

```text
Đọc repository từ repository root.
Chưa sửa file.
Mô tả task, expected behavior, file liên quan, file không được sửa,
dependency, rủi ro và minimum test.
```

### Generate

```text
Implement exactly one task: <một task duy nhất>.
Chỉ sửa: <đường dẫn file cụ thể>.
Không tạo file ngoài phạm vi.
Sau khi sửa, dừng lại và báo diff tóm tắt.
```

### Smoke Test

```text
Chạy smoke test nhỏ nhất cho thay đổi vừa thực hiện.
Không chạy full training.
Kiểm tra import, shape, output và metric finite.
Báo command, output và exit code.
```

### Run Normal

```text
Chỉ chạy normal mode nếu smoke test đã PASS.
Dùng config chính thức và command từ repository root.
Không tự ý đổi hyperparameter để che lỗi.
Báo metric, log directory, checkpoint path và lỗi còn lại.
```

### Check with Description

```text
Đối chiếu kết quả với Description ban đầu.
Kiểm tra đúng behavior, đúng file scope, test passed và artifact đầy đủ.
Chỉ trả lời PASS khi có bằng chứng command/output.
Không sửa code trong bước review.
```

## 15. Điều kiện kết thúc một task

Một task chỉ được xem là hoàn tất khi:

- Đúng một task đã được thực hiện.
- Không có file ngoài phạm vi bị thay đổi.
- Smoke test đã chạy và passed.
- Normal run đã chạy nếu task yêu cầu.
- Output có command, exit code và kết quả thực tế.
- TensorBoard/checkpoint tồn tại nếu task liên quan.
- EDA có đủ resolution, class distribution và pixel statistics.
- Evaluation có confusion matrix, precision, recall, F1-score và AUC-ROC hợp lệ.
- Kết quả đã được đối chiếu với Description.
- AI không báo PASS chỉ dựa trên việc code không báo lỗi cú pháp.