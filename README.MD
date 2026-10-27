
# Practice 2: Hands-on Practice with Pre-trained Neural Network Architectures

Bài thực hành chuyên sâu về **Học chuyển giao (Transfer Learning)** và **Tinh chỉnh (Fine-tuning)** các kiến trúc mạng nơ-ron huấn luyện sẵn (Pre-trained Models) sử dụng framework **PyTorch**. Dự án được xây dựng theo mô hình **Lập trình hướng đối tượng (OOP)**, hỗ trợ theo dõi tiến trình huấn luyện bằng **TensorBoard**.

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
│   └── config.yaml             # Cấu hình siêu tham số (batch_size, lr, epochs, model_name, device)
│
├── data/                       # Thư mục lưu dữ liệu CIFAR-10 (tự động tải về qua torchvision)
│
├── src/                        # Mã nguồn chính (Được thiết kế theo chuẩn OOP)
│   ├── __init__.py             # Biến src thành Python Package
│   ├── data_module.py          # Class CIFAR10DataModule (Transforms, Dataset, DataLoader)
│   ├── model_builder.py        # Class TransferModelBuilder (ResNet18, VGG16, DenseNet121)
│   ├── trainer.py              # Class ModelTrainer (Vòng lặp Training 5 bước & Validation)
│   └── logger.py               # Class TensorBoardLogger (add_scalar, add_graph, add_image)
│
├── runs/                       # Thư mục lưu log TensorBoard (SummaryWriter)
│
├── checkpoints/                # Thư mục lưu trọng số mô hình đã huấn luyện (.pth)
│
├── notebooks/                  # Jupyter Notebooks phục vụ thử nghiệm & phân tích kết quả
│   └── experiment_demo.ipynb
│
├── main.py                     # Entry point chính chạy toàn bộ Pipeline
├── requirements.txt            # Thư viện phụ thuộc (PyTorch, Torchvision, Tensorboard,...)
└── README.md                   # Tài liệu hướng dẫn dự án
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

```bash
# Windows
py -3.11 -m venv venv
venv\Scripts\activate

# Linux/macOS
python3.11 -m venv venv
source venv/bin/activate
```

### 2. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

*Nội dung file `requirements.txt`:*
```text
torch>=2.0.0
torchvision>=0.15.0
tensorboard>=2.12.0
matplotlib>=3.7.0
pyyaml>=6.0
```

---

## 🚀 Hướng Dẫn Sử Dụng (How to Run)

### 1. Chạy huấn luyện mô hình (Training Pipeline)
Chạy file `main.py` để bắt đầu quy trình tải dữ liệu, khởi tạo mô hình pre-trained, huấn luyện và lưu trọng số:

```bash
python main.py
```

### 2. Mở giao diện trực quan hóa TensorBoard
Để theo dõi biểu đồ Loss, Accuracy và cấu trúc mạng theo thời gian thực:

```bash
tensorboard --logdir=runs
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

## 📊 Kết Quả Thực Nghiệm Kỳ Vọng (Expected Results)

