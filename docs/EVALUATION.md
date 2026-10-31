# Đánh giá mô hình phân loại CIFAR-10

## Chạy ngay

Chạy từ thư mục gốc dự án, với môi trường đã cài `requirements.txt`:

```powershell
.\venv\Scripts\python.exe evaluate.py
```

Nếu môi trường ảo đã được kích hoạt, dùng `python evaluate.py` trên Windows/Linux/macOS.
Lệnh mặc định dùng toàn bộ **10.000 ảnh test**, chọn checkpoint `*_best.pth` có thời gian sửa
đổi mới nhất trong `checkpoints/`, và tự nhận diện ResNet-18, VGG16 hoặc DenseNet121 từ trọng số.
Nếu không có file `*_best.pth`, chương trình tìm checkpoint `.pth`/`.pt` mới nhất.
Tên file không quyết định kiến trúc. Trọng số phải khớp đầy đủ với builder tương ứng.

Máy hiện tại đã có dữ liệu và checkpoint nên không cần huấn luyện lại.
Máy mới cần một checkpoint đã huấn luyện; có thể tạo bằng `python main.py`.
Nếu dữ liệu chưa có:

```powershell
python evaluate.py --download
```

Kiểm tra nhanh trên một mẫu ngẫu nhiên cố định gồm 128 ảnh:

```powershell
python evaluate.py --max-samples 128 --batch-size 16 --device cpu
```

Chỉ định checkpoint và xác minh kiến trúc:

```powershell
python evaluate.py --checkpoint checkpoints/resnet18_20261030-222811_best.pth --model-name resnet18
python evaluate.py --checkpoint checkpoints/vgg16_best.pth --model-name vgg16
python evaluate.py --checkpoint checkpoints/densenet121_best.pth --model-name densenet121
```

Hai lệnh cuối yêu cầu bạn đã có checkpoint của kiến trúc tương ứng.
Chương trình không tự tạo trọng số ngẫu nhiên để thay thế checkpoint bị thiếu.

## Các file kết nối với nhau thế nào?

```text
configs/evaluation.yaml
         |
     evaluate.py
         |-- src/checkpoint_utils.py --> src/model_builder.py --> checkpoint đã huấn luyện
         |-- src/data_module.py --> CIFAR-10 test + tiền xử lý hiện có
         |-- src/evaluator.py --> src/classification_metrics.py
         `-- src/evaluation_report.py --> JSON, TXT, CSV, PNG
```

`main.py` và `src/logger.py` tiếp tục tạo checkpoint theo định dạng hiện tại;
`evaluate.py` đọc trực tiếp định dạng đó. Bộ đánh giá cũng đọc được raw `state_dict`,
`{"state_dict": ...}`, và trọng số có tiền tố `module.` từ DataParallel.
Model được dựng bằng `pretrained=False` rồi nạp **toàn bộ** checkpoint, nên không tải ImageNet.

`CIFAR10DataModule.setup()` vẫn nạp cả train/test như trước. Lệnh đánh giá gọi
`setup(stage="test", download=False)` để chỉ tạo dataset test, không dùng augmentation ngẫu nhiên.
Ảnh được resize 224×224 và chuẩn hóa bằng mean/std ImageNet đúng như luồng huấn luyện hiện có.
Checkpoint cũ không lưu thứ tự nhãn được hiểu là dùng thứ tự CIFAR-10 chuẩn của dự án:
`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`.
Nếu checkpoint có `class_names`, chương trình kiểm tra thứ tự này trước khi đánh giá.

## Cấu hình

Sửa `configs/evaluation.yaml` hoặc dùng `--config đường_dẫn.yaml`. File cấu hình tùy chỉnh
cần có đủ các khóa như file mẫu; tham số dòng lệnh ghi đè giá trị trong YAML.
Mọi đường dẫn tương đối đều tính từ thư mục gốc dự án, kể cả khi chạy từ thư mục khác.

| Khóa / tham số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `checkpoint` / `--checkpoint` | `null` | Tự chọn checkpoint hoặc chỉ định file |
| `checkpoint_dir` / `--checkpoint-dir` | `checkpoints` | Thư mục tìm checkpoint tự động |
| `model_name` / `--model-name` | `null` | Tự nhận diện hoặc kiểm tra kiến trúc yêu cầu |
| `data_dir` / `--data-dir` | `data` | Thư mục dữ liệu torchvision |
| `output_dir` / `--output-dir` | `results/evaluation` | Thư mục cha của báo cáo |
| `batch_size` / `--batch-size` | `64` | Giảm xuống 16/32 nếu thiếu RAM/VRAM |
| `num_workers` / `--num-workers` | `0` | Mặc định phù hợp Windows |
| `image_size` / `--image-size` | `224` | Phải phù hợp tiền xử lý khi huấn luyện |
| `device` / `--device` | `auto` | `auto`, `cpu`, `cuda` |
| `download` / `--download` | `false` | Cho phép torchvision tải dữ liệu còn thiếu |
| `max_samples` / `--max-samples` | `0` | 0 = toàn bộ test; số dương = giới hạn số ảnh |
| `seed` / `--seed` | `42` | Chọn cùng tập con khi đánh giá nhiều checkpoint |
| `top_k` / `--top-k` | `5` | Top-k accuracy, k tối đa bằng số lớp |
| `plots` / `--no-plots` | `true` | Tắt PNG nếu chỉ cần số liệu |

Tập con được lấy ngẫu nhiên không hoàn lại, rồi sắp lại theo chỉ số gốc.
`predictions.csv` lưu chỉ số gốc; metadata lưu seed và SHA-256 danh sách chỉ số để đối chiếu.
`max_samples` lớn hơn kích thước test sẽ dùng toàn bộ test.

## Kết quả xuất ra

Mỗi lần chạy tạo một thư mục mới:

```text
results/evaluation/<tên_checkpoint>_<thời_gian>/
    metrics.json
    classification_report.txt
    per_class_metrics.csv
    predictions.csv
    confusion_matrix.csv
    confusion_matrix.png
    confusion_matrix_normalized.png
    roc_curves.png
    precision_recall_curves.png
```

- `metrics.json`: loss, accuracy, balanced accuracy, top-k accuracy, precision/recall/F1
  macro và weighted, ROC AUC OvR, average precision, chỉ số từng lớp và metadata.
- `classification_report.txt`: bảng dễ đọc cùng cách tính và lưu ý về tập đánh giá.
- `per_class_metrics.csv`: precision, recall, F1, specificity, support, TP/FP/FN/TN,
  AUC và AP của từng lớp. Mở được bằng Excel với tên lớp và tiêu đề UTF-8.
- `predictions.csv`: nhãn thật, nhãn dự đoán, độ tin cậy, đúng/sai và xác suất đủ 10 lớp.
- Ma trận nhầm lẫn: **hàng là nhãn thật, cột là dự đoán**; bản chuẩn hóa chia theo tổng hàng.
- ROC/PR: mỗi đường là một lớp so với tất cả lớp còn lại.

Các score trong JSON/CSV nằm trên thang **0–1**; màn hình và accuracy trong TXT hiển thị phần trăm.
Loss là cross-entropy trung bình theo **số ảnh**, kể cả khi batch cuối nhỏ hơn.
`confidence` là xác suất softmax lớn nhất; không phải độ tin cậy đã được hiệu chuẩn.

Precision/recall/F1 không xác định được ghi là 0. Macro P/R/F1 tính đủ 10 lớp;
weighted dùng support làm trọng số; balanced accuracy lấy recall trung bình trên các lớp có mẫu.
ROC AUC và AP của lớp thiếu mẫu dương hoặc âm được ghi `null`, có cảnh báo và không vẽ đường đó.
Macro AUC/AP chỉ lấy các lớp tính được; xem thêm `auc_valid_classes`.
AP dùng tổng precision theo mức tăng recall, không dùng diện tích hình thang của đường PR.
Các mẫu bằng điểm được gom cùng ngưỡng khi tính ROC/PR; top-k hòa điểm ưu tiên chỉ số lớp nhỏ hơn.

## Dùng trong mã Python

```python
from pathlib import Path
from src.checkpoint_utils import load_classifier
from src.data_module import CIFAR10DataModule
from src.evaluator import ClassificationEvaluator
from src.evaluation_report import save_evaluation_report

data = CIFAR10DataModule(data_dir="./data", batch_size=32, num_workers=0)
data.setup(stage="test", download=False)
model, metadata = load_classifier(Path("checkpoints/resnet18_20261030-222811_best.pth"))
classes = data.test_dataset.classes
result = ClassificationEvaluator(model, classes, device="cpu").evaluate(data.test_dataloader())
save_evaluation_report(result, "results/evaluation/manual", classes, metadata)
```

Với DataLoader khác, truyền đúng danh sách tên lớp theo thứ tự đầu ra của model.
Model cần trả logits `[batch, classes]`, nhãn cần là tensor `torch.long` một chiều.
Nếu loader có shuffle, `sample_indices` truyền cho hàm xuất báo cáo phải theo đúng thứ tự thu thập.
CLI luôn dùng `shuffle=False` và tự nối đúng chỉ số.
Hàm đánh giá dùng inference mode, tạm chuyển model sang eval và khôi phục chế độ train/eval ban đầu.

## Kiểm thử

Các kiểm thử mới dùng thư viện chuẩn `unittest`, không cần tải dữ liệu hay trọng số:

```powershell
python -m unittest discover -s tests -p test_evaluation.py -v
```

Chạy toàn bộ bộ kiểm thử hiện có của dự án (cần CIFAR-10 cục bộ cho test dữ liệu):

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

## Diễn giải và so sánh mô hình

`main.py` hiện chọn checkpoint theo **test accuracy từng epoch**. Nếu checkpoint đến từ quy trình này,
báo cáo test mô tả checkpoint hiện có nhưng chưa phải ước lượng độc lập trên dữ liệu được giữ kín.
Nên chọn kiến trúc/siêu tham số/checkpoint trên validation tách từ train rồi đánh giá test một lần cuối.
`experiments.py` đã tách validation và là cơ sở phù hợp cho bước lựa chọn này.

Để so sánh ba kiến trúc, cần checkpoint thực của từng mô hình. Giữ nguyên dữ liệu test,
tiền xử lý và danh sách chỉ số mẫu; ưu tiên kết quả toàn bộ test.
So sánh macro F1, recall từng lớp và các cặp nhầm lẫn bên cạnh accuracy.
Kết quả từ 64/128 ảnh chỉ dùng kiểm tra luồng chạy, không thay thế báo cáo toàn bộ test.
