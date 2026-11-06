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

Trước khi sửa, đọc các file liên quan và giữ nguyên pattern hiện có.
Chỉ được sửa:
- src/data_module.py

Yêu cầu:
- Tải CIFAR-10 bằng torchvision
- Có train transform và test/validation transform riêng
- Tạo DataLoader
- Kiểm tra batch_size hợp lệ
- Train loader shuffle=True
- Validation/test loader shuffle=False
- Không dùng test set để tuning

Sau khi sửa:
1. Chạy import test
2. Chạy one-batch smoke test
3. Báo image shape, label shape và số lượng mẫu
4. Không sửa file ngoài phạm vi
```

Smoke test tối thiểu cho data module:

```powershell
.\venv\Scripts\python.exe -c "from src.data_module import CIFAR10DataModule; print('data module import passed')"
```

## 3. Smoke test module dữ liệu

```text
Chạy smoke test cho data module vừa triển khai.

Dùng subset nhỏ hoặc batch đầu tiên, không chạy full training.
Kiểm tra:
- Import không lỗi
- Batch image là tensor 4 chiều
- Label là tensor 1 chiều
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
Implement exactly one task: hoàn thiện training và evaluation loop.

Chỉ được sửa:
- src/trainer.py

Yêu cầu:
- CrossEntropyLoss
- Optimizer configurable
- Hỗ trợ CPU và CUDA
- Dùng model.train() khi training
- Dùng model.eval() và torch.no_grad() khi evaluation
- Tính loss và accuracy trên toàn bộ dataset
- Không trả về NaN hoặc Inf
- Không dùng broad exception để che lỗi

Sau khi sửa, chạy one-epoch smoke test trên subset nhỏ bằng CPU.
Báo train loss, train accuracy, validation loss và validation accuracy.
```

Không chạy full CIFAR-10 trong Smoke Test. Chỉ chuyển sang normal mode sau khi
one-epoch smoke test hoàn tất và metric là số hữu hạn.

## 6. Viết TensorBoard và checkpoint

```text
Implement exactly one task: hoàn thiện TensorBoard logging và checkpoint.

Chỉ được sửa:
- src/logger.py

Yêu cầu:
- Ghi train/validation loss
- Ghi train/validation accuracy
- Ghi model graph
- Ghi ảnh mẫu
- Ghi config
- Lưu best checkpoint
- Có model state, optimizer state, epoch và metric
- Flush sau mỗi epoch
- Close writer khi kết thúc
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

Chỉ được sửa:
- main.py

Yêu cầu:
- Đọc config hiện có
- Khởi tạo data module
- Khởi tạo ResNet-18, VGG-16 hoặc DenseNet-121
- Chạy training/evaluation
- Ghi TensorBoard
- Lưu best checkpoint
- Hỗ trợ resume checkpoint
- Báo lỗi rõ ràng khi config hoặc checkpoint không hợp lệ

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
- Train/validation/test metrics
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
Implement exactly one task: chạy EDA cho CIFAR-10.

Chỉ sửa các file EDA được nêu trong task; không sửa source training hoặc
dataset. Không thay đổi split, label, transform production hay hyperparameter.

Kiểm tra và báo cáo:
- Resolution, height, width, số kênh, dtype và shape.
- Class distribution theo từng lớp ở train/validation/test.
- Pixel min, max, mean, standard deviation theo kênh trước normalize.
- Không có NaN, Inf hoặc giá trị pixel ngoài khoảng hợp lệ.

Chạy trước trên subset nhỏ để kiểm tra shape, sau đó chạy đầy đủ.
Báo command, output, exit code, seed, nguồn dữ liệu và artifact path.
Không kết luận PASS khi thiếu số liệu thực tế.
```

Điều kiện chuyển bước:

- Resolution thực tế khớp contract dataset.
- Có đủ phân phối nhãn của mọi split.
- Có pixel statistics theo kênh và không gian giá trị rõ ràng.
- Không phát hiện lỗi dữ liệu chưa được giải thích.

## 10. Evaluation metrics

Chỉ chạy sau khi normal training đã tạo checkpoint hợp lệ.

```text
Implement exactly one task: đánh giá checkpoint CIFAR-10 trên test set.

Chỉ sửa các file evaluation được nêu rõ; không sửa model, trainer, data split
hoặc config training. Không dùng test set để tuning.

Sau khi khóa checkpoint, tính:
- Confusion matrix 10x10, hàng là true label và cột là predicted label.
- Precision theo lớp, macro average và weighted average.
- Recall theo lớp, macro average và weighted average.
- F1-score theo lớp, macro average và weighted average.
- AUC-ROC one-vs-rest cho 10 lớp bằng score/xác suất trước argmax.

Kiểm tra logits/score shape, label range, tổng confusion matrix và metric
finite. Ghi rõ zero-division, label binarization, threshold/quy ước và các
lớp thiếu mẫu nếu có. Lưu figure/report không ghi đè artifact cũ.
Báo checkpoint, model, seed, device, command, output, exit code và artifact.
```

Smoke test evaluation:

```text
Chạy evaluation smoke test trên subset test nhỏ.
Kiểm tra confusion matrix đúng shape, mỗi metric có output theo lớp,
macro/weighted average tồn tại, AUC nhận score đa lớp và không có NaN/Inf.
Không chạy full evaluation nếu smoke test thất bại.
```

Normal evaluation:

```text
Chạy evaluation đầy đủ trên test set bằng checkpoint đã khóa.
Không đổi checkpoint, threshold, split hoặc hyperparameter trong lúc chạy.
Báo confusion matrix, precision, recall, F1-score, AUC-ROC, command,
exit code và đường dẫn report/figure.
```

## 11. Check with Description

```text
Check with Description.

Đối chiếu code và kết quả vừa chạy với yêu cầu Practice 2:
- Có đủ ResNet-18, VGG-16 và DenseNet-121 chưa?
- Classifier đã đổi thành 10 classes chưa?
- Backbone freeze đúng policy chưa?
- Có validation riêng với test chưa?
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