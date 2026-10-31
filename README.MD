
# Practice 2: Hands-on Practice with Pre-trained Neural Network Architectures

Bài thực hành chuyên sâu về **Học chuyển giao (Transfer Learning)** và **Tinh chỉnh (Fine-tuning)** các kiến trúc mạng nơ-ron huấn luyện sẵn (Pre-trained Models) sử dụng framework **PyTorch**. Dự án được xây dựng theo mô hình **Lập trình hướng đối tượng (OOP)**, hỗ trợ theo dõi tiến trình huấn luyện bằng **TensorBoard**.

**Bắt đầu tại [README bài lab — giải thích toàn bộ dự án và hướng dẫn từng bước chạy](docs/README_LAB.md).**

| Tài liệu | Nội dung |
| --- | --- |
| [README bài lab](docs/README_LAB.md) | Cài đặt, vai trò các file, dữ liệu, thử nghiệm, huấn luyện, đánh giá và báo cáo |
| [Hướng dẫn đánh giá](docs/EVALUATION.md) | Tham số, API Python và ý nghĩa chỉ số |
| [Nhận xét dự án](docs/PROJECT_REVIEW.md) | Điểm mạnh, hạn chế và kết quả chạy thực tế |
| [Chỉ dẫn đọc kết quả](results/README.md) | CSV thí nghiệm, báo cáo chạy thử và báo cáo đầy đủ |
| [Tài liệu tham khảo](docs/references/README.md) | Tài liệu Word gốc của nhóm |

## 📌 Mục Tiêu Dự Án

1. **Ứng dụng Transfer Learning:** Tái sử dụng bộ trích xuất đặc trưng (Feature Extractor) từ các mô hình đã huấn luyện trên ImageNet (`ResNet-18`, `VGG16`, `DenseNet121`) cho bài toán phân loại ảnh đa lớp trên tập dữ liệu **CIFAR-10** (10 lớp).
2. **Kỹ thuật Đóng băng trọng số (Freezing Layers):** Thực hành đóng băng các lớp cũ và thay thế lớp phân loại cuối (`model.fc` / `model.classifier`).
3. **Huấn luyện & Tinh chỉnh (Training & Hyperparameter Tuning):** Thực thi vòng lặp 5 bước chuẩn PyTorch, điều chỉnh `learning_rate`, `batch_size`, và áp dụng L2 Regularization chống Overfitting.
4. **Giám sát MLOps:** Trực quan hóa độ lỗi (Loss), độ chính xác (Accuracy), cấu trúc mô hình (Model Graph) và hình ảnh mẫu bằng **TensorBoard**.
## 📂 Cấu Trúc Thư Mục (Project Structure)
```markdown
Practice-2---Hands-on-practice-with-pre-trained-neural-network-architectures/
│
├── configs/
│   ├── config.yaml             # Cấu hình siêu tham số (batch_size, lr, epochs, model_name, device)
│   ├── lab.yaml                # Bài lab đầu tiên: ResNet18, 1 epoch, batch 16
│   └── evaluation.yaml         # Cấu hình đánh giá và xuất báo cáo
│
├── data/                       # Thư mục lưu dữ liệu CIFAR-10 (tự động tải về qua torchvision)
│
├── src/                        # Mã nguồn chính (Được thiết kế theo chuẩn OOP)
│   ├── __init__.py             # Biến src thành Python Package
│   ├── data_module.py          # Class CIFAR10DataModule (Transforms, Dataset, DataLoader)
│   ├── model_builder.py        # Class TransferModelBuilder (ResNet18, VGG16, DenseNet121)
│   ├── trainer.py              # Class ModelTrainer (Vòng lặp Training 5 bước & Validation)
│   ├── logger.py               # Class TensorBoardLogger (add_scalar, add_graph, add_image)
│   ├── checkpoint_utils.py     # Tìm/nạp checkpoint và nhận diện kiến trúc
│   ├── evaluator.py            # Class ClassificationEvaluator (suy luận trên test)
│   ├── classification_metrics.py # Các chỉ số phân loại, ROC/PR
│   └── evaluation_report.py    # Xuất JSON/TXT/CSV và biểu đồ PNG
│
├── runs/                       # Thư mục lưu log TensorBoard (SummaryWriter)
│
├── checkpoints/                # Thư mục lưu trọng số mô hình đã huấn luyện (.pth)
│
├── notebooks/                  # Jupyter Notebooks phục vụ thử nghiệm & phân tích kết quả
│   ├── experiment_demo.ipynb
│   └── src_output_demo.ipynb
│
├── main.py                     # Entry point chính chạy toàn bộ Pipeline
├── experiments.py              # Thử nghiệm siêu tham số, lưu CSV validation
├── evaluate.py                 # Entry point đánh giá checkpoint
├── scripts/check_data.py       # Kiểm tra dữ liệu qua python -m scripts.check_data
├── tests/                      # Kiểm thử tự động
├── results/
│   ├── experiments/            # CSV thí nghiệm theo từng lần chạy
│   └── evaluation/             # JSON/TXT/CSV/PNG đánh giá theo checkpoint
├── docs/
│   ├── README_LAB.md           # Giải thích toàn dự án và hướng dẫn bài lab
│   ├── EVALUATION.md           # Hướng dẫn đánh giá chi tiết
│   ├── PROJECT_REVIEW.md       # Nhận xét và phân tích kết quả
│   └── references/             # Tài liệu Word gốc
├── requirements.txt            # Thư viện phụ thuộc (PyTorch, Torchvision, Tensorboard,...)
├── requirements-dev.txt        # Thư viện chạy dự án và pytest
└── README.MD                   # Tổng quan và chỉ mục tài liệu
```

---

## 🛠️ Hướng Dẫn Cài Đặt (Installation)

### 1. Cài đặt Python 3.11 và khởi tạo môi trường ảo (Khuyên dùng)

Dự án sử dụng **Python 3.11**. Hãy đảm bảo Python 3.11 đã được cài đặt và kiểm tra phiên bản:

```bash
python --version
```

Kết quả cần có dạng:

```text
Python 3.11.x
```

Nếu máy có nhiều phiên bản Python, có thể sử dụng `py -3.11` thay cho `python` trên Windows.

```powershell
# Windows PowerShell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

```bash
# Linux/macOS
python3.11 -m venv venv
source venv/bin/activate
```

### 2. Cài đặt các thư viện cần thiết
```bash
python -m pip install -r requirements-dev.txt
```

Nếu PowerShell không cho kích hoạt môi trường, dùng interpreter trực tiếp:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

*Nội dung file `requirements.txt`:*
```text
--extra-index-url https://download.pytorch.org/whl/cu121
torch>=2.0.0
torchvision>=0.15.0
tensorboard>=2.12.0
matplotlib>=3.7.0
numpy>=1.24.0
pyyaml>=6.0
```

---

## 🚀 Hướng Dẫn Sử Dụng (How to Run)

Các lệnh sau chạy ở thư mục gốc dự án, sau khi kích hoạt môi trường ảo.
Trên PowerShell, có thể thay `python` bằng `.\venv\Scripts\python.exe` để không cần kích hoạt.

Kiểm tra dữ liệu và chạy thử với tập con trước:

```bash
python -m scripts.check_data --download
python -m pytest tests -q
python experiments.py --epochs 1 --train-samples 32 --val-samples 16
```

### 1. Chạy huấn luyện mô hình (Training Pipeline)
Chạy file `main.py` để bắt đầu quy trình tải dữ liệu, khởi tạo mô hình pre-trained, huấn luyện và lưu trọng số:

```bash
python main.py
```

Mặc định là 10 epoch trên toàn bộ 50.000 ảnh train. Với bài lab đầu tiên, dùng cấu hình 1 epoch:

```bash
python main.py --config configs/lab.yaml
```

Một epoch vẫn dùng toàn bộ train; nếu chỉ muốn kiểm tra nhanh luồng chạy, dùng lệnh
`experiments.py` với tập con bên trên. Script thí nghiệm chỉ lưu CSV, còn `main.py` lưu checkpoint.

### 2. Đánh giá mô hình phân loại

Chạy ngay với checkpoint tốt nhất mới nhất trong `checkpoints/` và toàn bộ CIFAR-10 test:

```bash
python evaluate.py
```

Kiểm tra nhanh hoặc chỉ định checkpoint:

```bash
python evaluate.py --max-samples 128 --batch-size 16
python evaluate.py --checkpoint checkpoints/resnet18_20261030-222811_best.pth
```

Cấu hình tại `configs/evaluation.yaml`. Báo cáo được lưu trong `results/evaluation/`, gồm
accuracy, precision/recall/F1, ROC AUC, AP, confusion matrix, dự đoán từng ảnh và biểu đồ PNG.
Nếu chưa có CIFAR-10, thêm `--download`; nếu chưa có checkpoint, chạy huấn luyện trước.

Xem [hướng dẫn đầy đủ và kết nối các file](docs/EVALUATION.md) và
[nhận xét dự án, các hạn chế của quy trình thực nghiệm](docs/PROJECT_REVIEW.md).
Lưu ý: `main.py` hiện chọn checkpoint bằng test accuracy; báo cáo trên checkpoint này cần
được diễn giải cùng hạn chế đó.

### 3. Mở giao diện trực quan hóa TensorBoard
Để theo dõi biểu đồ Loss, Accuracy và cấu trúc mạng theo thời gian thực:

```bash
python -m tensorboard.main --logdir runs --port 6006
```
Sau đó truy cập đường dẫn `http://localhost:6006` trên trình duyệt web.

---

## 👥 Phân Công Công Việc Nhóm (Group Member Roles)

| Thành viên | Phụ trách Mã Nguồn (Code Module) | Phụ trách Bài Tiểu Luận (Report Section) |
| :--- | :--- | :--- |
| **Lê Minh Tân** | `src/data_module.py` (Data Pipeline, Transforms, DataLoaders) | **Chương 1 & 2:** Giới thiệu bài toán CIFAR-10 & Quy trình Tiền xử lý dữ liệu. |
| **Lê Quốc Thái** | `src/model_builder.py` (ResNet-18 Architecture & Freezing) | **Chương 3.1:** Lý thuyết Transfer Learning & Kiến trúc ResNet. |
| **Lưu Nhựt Tân** | `src/model_builder.py` (VGG16 & DenseNet121 Extensions) | **Chương 3.2:** So sánh các kiến trúc Pre-trained (ResNet vs VGG vs DenseNet). |
| **Nguyễn Vĩnh Phú** | `src/trainer.py` (Training Engine, 5-Step Loop, CrossEntropy) | **Chương 4:** Quy trình huấn luyện, Backpropagation & Optimizer (SGD vs Adam). |
| **Hoàng Minh Tâm** | Tuning & Experiments (`learning_rate`, `batch_size`, Weight Decay) | **Chương 5:** Tinh chỉnh siêu tham số & Kỹ thuật phòng chống Overfitting. |
| **Phạm Ngọc Phước** | `src/logger.py` (TensorBoard Integration & Checkpointing) | **Chương 6:** Giám sát MLOps với TensorBoard, Quản lý trọng số & Kết luận. |

---

## 📊 Kết quả đã chạy trên checkpoint hiện có

Checkpoint `resnet18_20261030-222811_best.pth`, đánh giá toàn bộ 10.000 ảnh test:

| Chỉ số | Giá trị |
| --- | ---: |
| Accuracy | 79,13% |
| F1 macro | 0,7912 |
| ROC AUC macro OvR | 0,9770 |

Kết quả của mỗi lần huấn luyện có thể khác. Xem [nhận xét dự án](docs/PROJECT_REVIEW.md)
về quy trình chọn checkpoint và [chỉ dẫn kết quả](results/README.md) để phân biệt chạy thử với toàn bộ test.

