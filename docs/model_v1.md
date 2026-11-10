# Model Specification Plan (Version 1.0)

> **Document Metadata**
> - **Module Name:** `src/model_builder.py` (`TransferModelBuilder`)
> - **Created Date:** 07/11/2026
> - **Last Updated:** 07/11/2026
> - **Target Dataset:** CIFAR-10 (10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

---

## 1. Purpose (Mục đích)

Mục đích chính của module `model_builder` là xây dựng kiến trúc mạng nơ-ron thông qua kỹ thuật **Transfer Learning (Chuyển giao học tập)** từ các mô hình học sâu hiện đại đã được tiền huấn luyện (pre-trained) trên tập dữ liệu quy mô lớn ImageNet-1k, nhằm giải quyết bài toán phân loại 10 lớp ảnh của CIFAR-10.

### Lý do & Vai trò kỹ thuật:
1. **Tận dụng đặc trưng đã học (Feature Reusability):** ImageNet chứa hơn 1.4 triệu ảnh với 1.000 danh mục đa dạng. Các tầng tích chập sâu (Backbone) đã học được những bộ trích xuất đặc trưng thị giác khái quát rất tốt (từ cạnh, góc, vân bề mặt cho tới các hình thái phức tạp).
2. **Đóng băng Backbone (Freeze Feature Extractor):** CIFAR-10 có kích thước nhỏ (50.000 ảnh train). Việc đóng băng toàn bộ backbone giúp:
   - Ngăn chặn hiện tượng phá hủy các trọng số tốt đã học (*catastrophic forgetting*).
   - Giảm đáng kể số lượng tham số cần tối ưu (giảm từ hàng chục triệu tham số xuống chỉ còn vài nghìn tham số ở lớp cuối).
   - Tăng tốc độ huấn luyện trên mỗi epoch và chống hiện tượng quá khớp (*overfitting*).
3. **Thay thế Head phân loại (Classifier Adaptation):** Thay thế lớp Linear cuối cùng (vốn có 1.000 đầu ra trên ImageNet) thành lớp Linear mới có đúng 10 đầu ra tương ứng với 10 lớp của CIFAR-10.
4. **Hỗ trợ đa kiến trúc (Multi-architecture Support):** Module cung cấp interface thống nhất theo mô hình Factory Pattern để khởi tạo linh hoạt 3 kiến trúc bắt buộc:
   - **ResNet-18** (Residual Connections)
   - **VGG-16** (Deep Sequential Convolutional Network)
   - **DenseNet-121** (Dense Feature Reuse Connections)

---

## 2. Model Architecture & Input/Output Design (Kiến trúc mô hình & Thiết kế I/O)

Mục này cung cấp bức tranh toàn cảnh về kiến trúc mạng nơ-ron: mô tả chi tiết các khối tầng bên trong, cơ chế trích xuất đặc trưng, bảng thông số kỹ thuật, cũng như quy định rõ mã nguồn sau khi AI sinh ra sẽ gồm những Class nào và nằm ở đâu trong dự án.

---

### 2.1 Bảng đặc tả kiến trúc kỹ thuật của 3 mô hình (Architecture Comparison Spec)

| Thuộc tính kiến trúc | ResNet-18 | VGG-16 | DenseNet-121 |
| :--- | :--- | :--- | :--- |
| **Nguyên lý cốt lõi** | **Residual Learning:** Dùng đường tắt (Skip Connection) $F(x) + x$ giải quyết triệt tiêu đạo hàm (*vanishing gradient*). | **Deep Sequential:** Xếp chồng các bộ lọc nhỏ $3\times3$ liên tiếp tạo trường tiếp nhận lớn và tăng tính phi tuyến. | **Dense Connectivity:** Mỗi tầng nhận đặc trưng ghép nối (*concatenation*) từ tất cả các tầng phía trước nó. |
| **Số tầng có trọng số** | 18 tầng (17 Conv + 1 FC) | 16 tầng (13 Conv + 3 FC) | 121 tầng (120 Conv + 1 FC) |
| **Cấu trúc Backbone (Feature Extractor)** | `conv1` (7x7, stride 2) $\rightarrow$ `bn1` $\rightarrow$ `relu` $\rightarrow$ `maxpool`<br>$\rightarrow$ 4 Residual Stages (`layer1` - `layer4`)<br>$\rightarrow$ `avgpool` (Global Average Pooling) | 5 khối tích chập liên tiếp (`features[0..30]`):<br>Block 1 (2 conv) $\rightarrow$ Block 2 (2 conv)<br>$\rightarrow$ Block 3 (3 conv) $\rightarrow$ Block 4 (3 conv)<br>$\rightarrow$ Block 5 (3 conv) xen kẽ `MaxPool2d` | `conv0` (7x7) $\rightarrow$ `pool0`<br>$\rightarrow$ 4 Dense Blocks (6, 12, 24, 16 dense layers)<br>xen kẽ 3 Transition Layers (1x1 Conv + AvgPool)<br>$\rightarrow$ `norm5` $\rightarrow$ `relu` $\rightarrow$ `avgpool` |
| **Bottleneck Features ra khỏi Backbone** | **512** chiều | **4.096** chiều (qua 2 tầng Linear ẩn đầu tiên) | **1.024** chiều |
| **Classification Head gốc (ImageNet)** | `model.fc = Linear(512, 1000)` | `model.classifier[6] = Linear(4096, 1000)` | `model.classifier = Linear(1024, 1000)` |
| **Classification Head mới (CIFAR-10)** | `model.fc = Linear(512, 10)` | `model.classifier[6] = Linear(4096, 10)` | `model.classifier = Linear(1024, 10)` |
| **Trọng số Pre-trained chính thức** | `models.ResNet18_Weights.DEFAULT` | `models.VGG16_Weights.DEFAULT` | `models.DenseNet121_Weights.DEFAULT` |
| **Tổng số tham số (Total Params)** | ~11.181.642 (~11.18M) | ~134.301.514 (~134.30M) | ~6.964.106 (~6.96M) |
| **Số tham số học mới (Trainable)** | **5.130** (chiếm ~0.05%) | **40.970** (chiếm ~0.03%) | **10.250** (chiếm ~0.15%) |

---

### 2.2 Chi tiết cấu tạo từng mô hình (Detailed Architecture Anatomy)

#### 1. Kiến trúc ResNet-18 (Residual Network 18)
* **Cơ chế Skip Connection:** Khi mạng nơ-ron càng sâu, việc học ánh xạ đồng nhất $H(x) = x$ trở nên khó khăn và gradient dễ bị triệt tiêu khi lan truyền ngược. ResNet chia nhỏ mạng thành các khối dư (*Residual Blocks*), ép mạng học ánh xạ thặng dư $F(x) = H(x) - x$, sau đó cộng trực tiếp đầu vào ban đầu qua đường tắt: $H(x) = F(x) + x$.
* **Dòng chảy dữ liệu bên trong:**
  1. **Stem:** Ảnh $3 \times 224 \times 224 \xrightarrow{\text{Conv 7x7, s=2}} 64 \times 112 \times 112 \xrightarrow{\text{MaxPool 3x3, s=2}} 64 \times 56 \times 56$.
  2. **Layer 1:** 2 BasicBlocks, giữ nguyên 64 channels ($64 \times 56 \times 56$).
  3. **Layer 2:** 2 BasicBlocks, tăng lên 128 channels, giảm chiều ($128 \times 28 \times 28$).
  4. **Layer 3:** 2 BasicBlocks, tăng lên 256 channels, giảm chiều ($256 \times 14 \times 14$).
  5. **Layer 4:** 2 BasicBlocks, tăng lên 512 channels, giảm chiều ($512 \times 7 \times 7$).
  6. **Global AvgPool:** Thu gọn không gian về vector đặc trưng kích thước **512**.
  7. **Classifier Adaptation:** Thay thế `model.fc = nn.Linear(512, 10)`.

#### 2. Kiến trúc VGG-16 (Visual Geometry Group 16)
* **Cơ chế xếp chồng Conv $3\times3$:** 2 tầng Conv $3\times3$ liên tiếp có cùng trường tiếp nhận tương đương 1 tầng Conv $5\times5$, 3 tầng $3\times3$ tương đương 1 tầng $7\times7$. Việc dùng các bộ lọc nhỏ $3\times3$ giúp giảm mạnh số tham số tính toán và bổ sung thêm hàm kích hoạt phi tuyến sau mỗi tầng.
* **Dòng chảy dữ liệu bên trong:**
  1. **Features Backbone (`model.features`):** Gồm 13 tầng Conv2d chia làm 5 nhóm, mỗi nhóm kết thúc bằng `MaxPool2d(2, 2)`:
     - Group 1: 2 Conv (64 channels) $\rightarrow 64 \times 112 \times 112$
     - Group 2: 2 Conv (128 channels) $\rightarrow 128 \times 56 \times 56$
     - Group 3: 3 Conv (256 channels) $\rightarrow 256 \times 28 \times 28$
     - Group 4: 3 Conv (512 channels) $\rightarrow 512 \times 14 \times 14$
     - Group 5: 3 Conv (512 channels) $\rightarrow 512 \times 7 \times 7$
  2. **AdaptiveAvgPool2d:** Đưa về kích thước $512 \times 7 \times 7 = 25.088$.
  3. **Sequential Classifier (`model.classifier`):**
     - `Linear(25088, 4096)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.5)` (Đóng băng)
     - `Linear(4096, 4096)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.5)` (Đóng băng)
     - `classifier[6]`: Thay thế `nn.Linear(4096, 10)` (Mở trainable).

#### 3. Kiến trúc DenseNet-121 (Densely Connected Network 121)
* **Cơ chế ghép nối Dense Block:** Thay vì cộng giá trị $F(x) + x$ như ResNet, DenseNet thực hiện **ghép kênh (Channel Concatenation)**: tầng thứ $l$ nhận đầu vào là tensor ghép từ output của tất cả các tầng $0, 1, ..., l-1$. Cơ chế này giúp tối đa hóa việc tái sử dụng đặc trưng, dòng gradient chảy thông suốt tới từng tầng sớm nhất.
* **Dòng chảy dữ liệu bên trong:**
  1. **Stem:** Conv 7x7 (64 channels, s=2) $\rightarrow$ MaxPool $\rightarrow 64 \times 56 \times 56$.
  2. **Dense Block 1 (6 layers) + Transition Layer 1:** Nén chiều kênh và giảm nửa kích thước không gian.
  3. **Dense Block 2 (12 layers) + Transition Layer 2:** Tiếp tục học đặc trưng và giảm kích thước.
  4. **Dense Block 3 (24 layers) + Transition Layer 3:** Học các đặc trưng thị giác cấp cao.
  5. **Dense Block 4 (16 layers):** Cho ra tensor đặc trưng $1024 \times 7 \times 7$.
  6. **Global AvgPool:** Thu gọn về vector đặc trưng **1.024**.
  7. **Classifier Adaptation:** Thay thế `model.classifier = nn.Linear(1024, 10)`.

---

### 2.3 Cấu trúc Code AI sinh ra và Vị trí lưu trữ (What & Where)

Để người đọc và các thành viên trong nhóm hiểu rõ **sau khi AI viết code nó sẽ ra cái gì và nằm ở đâu**, quy ước cấu trúc mã nguồn được định nghĩa chính xác như sau:

* **Vị trí file mã nguồn:** [src/model_builder.py](file:///d:/UTH/Deeplearning/lab/Practice-2---Hands-on-practice-with-pre-trained-neural-network-architectures/src/model_builder.py)
* **Kiến trúc các Class hướng đối tượng (OOP Architecture):**
  ```text
  src/model_builder.py
  ├── class TransferModelBuilder_ResNet18      # Đóng gói logic cho ResNet-18
  │     ├── __init__(num_classes=10, pretrained=True)
  │     └── build() -> nn.Module              # Freeze backbone, thay model.fc
  │
  ├── class TransferModelBuilder_VGG16         # Đóng gói logic cho VGG-16
  │     ├── __init__(num_classes=10, pretrained=True)
  │     └── build() -> nn.Module              # Freeze backbone, thay model.classifier[6]
  │
  ├── class TransferModelBuilder_DenseNet121   # Đóng gói logic cho DenseNet-121
  │     ├── __init__(num_classes=10, pretrained=True)
  │     └── build() -> nn.Module              # Freeze backbone, thay model.classifier
  │
  └── class TransferModelBuilder               # Factory Builder điều phối chung
        ├── SUPPORTED_MODELS = {"resnet18": ..., "vgg16": ..., "densenet121": ...}
        ├── __init__(model_name, num_classes=10, pretrained=True)
        └── build() -> nn.Module              # Chuẩn hóa tên và trả về model PyTorch
  ```
* **Đối tượng trả về (Output Object):** Đối tượng `nn.Module` của PyTorch đã sẵn sàng nạp lên GPU/CPU và truyền vào `ModelTrainer` để huấn luyện.

---

### 2.4 Chi tiết Input / Output Contract

| Thuộc tính | Đầu vào (Input Tensor) | Đầu ra (Output Tensor / Logits) |
| :--- | :--- | :--- |
| **Shape** | `[batch_size, 3, 224, 224]` | `[batch_size, 10]` |
| **Data Type** | `torch.float32` | `torch.float32` |
| **Chuẩn hóa / Giá trị** | Đã Resize `224x224`, chuẩn hóa theo ImageNet:<br>`mean = [0.485, 0.456, 0.406]`<br>`std = [0.229, 0.224, 0.225]` | Raw Logits unnormalized ($(-\infty, +\infty)$)<br>*(Tuyệt đối không áp dụng Softmax trước khi đưa vào Loss)* |
| **Mục đích sử dụng** | Batch ảnh truyền từ DataLoader | Đưa trực tiếp vào `nn.CrossEntropyLoss()` để tính mất mát |

---


---

## 3. Success Criteria (Tiêu chí thành công & Nghiệm thu)

Một mô hình do `TransferModelBuilder` tạo ra chỉ được nghiệm thu đạt chuẩn (**PASS**) khi thỏa mãn toàn bộ các tiêu chí định lượng sau:

| STT | Tiêu chí (Criteria) | Điều kiện chấp thuận (Acceptance Condition) | Cách kiểm tra |
| :---: | :--- | :--- | :--- |
| **1** | **Output Classes** | `out_features == 10` cho cả 3 kiến trúc (`resnet18`, `vgg16`, `densenet121`). | Kiểm tra layer cuối của model. |
| **2** | **Freeze Backbone** | 100% parameter thuộc backbone có `requires_grad == False`. | Quét `for name, p in model.named_parameters()`. |
| **3** | **Trainable Classifier** | 100% parameter của head mới có `requires_grad == True`. | `p.requires_grad == True` chỉ xuất hiện ở `fc` hoặc `classifier`. |
| **4** | **Forward Shape** | Input `[B, 3, 224, 224]` cho đúng Output `[B, 10]`. | `model(torch.randn(B, 3, 224, 224)).shape == (B, 10)`. |
| **5** | **Tính hữu hạn** | Đầu ra không chứa `NaN`, không chứa `Inf`. | `torch.isfinite(output).all() == True`. |
| **6** | **Tỷ lệ tham số học** | Số tham số học (*trainable params*) chiếm tỷ lệ nhỏ (< 1% tổng tham số):<br>- **ResNet-18:** ~5.130 / ~11.18M params (~0.05%)<br>- **VGG-16:** ~40.970 / ~134.27M params (~0.03%)<br>- **DenseNet-121:** ~10.250 / ~6.96M params (~0.15%) | Tính tổng: `sum(p.numel() for p in model.parameters() if p.requires_grad)`. |
| **7** | **Bẫy lỗi hợp lệ** | Khi nhập sai `model_name` hoặc `num_classes <= 0`, ném ra `ValueError` rõ ràng. | Test với pytest raises `ValueError`. |
| **8** | **Không dùng Softmax** | Model trả về Logits thô; không bọc `nn.Softmax` ở cuối. | Kiểm tra kiến trúc module con cuối cùng là `nn.Linear`. |
| **9** | **Tài liệu hóa mã nguồn (Docstrings)** | 100% các Class và Method đều có Docstring chuẩn mực (PEP 257) mô tả nguồn gốc weights ImageNet-1k, bài toán phân loại 10 lớp CIFAR-10 và giải thích rõ kiểu dữ liệu tham số vào/ra. | Đọc docstring tại đầu mỗi class và hàm trong `src/model_builder.py`. |

---

## 4. How to do (Cách thực hiện & Hướng dẫn triển khai)

### 4.1 Quy trình kỹ thuật từng bước (Step-by-step Implementation)

1. **Bước 1: Nạp kiến trúc và trọng số chính thức:**
   - Dùng API mới nhất của torchvision (`torchvision.models.*_Weights.DEFAULT`).
   - Tránh dùng cờ cũ `pretrained=True` (đã bị deprecated trong các phiên bản mới của torchvision).
2. **Bước 2: Đóng băng toàn bộ trọng số (Freezing):**
   ```python
   for param in model.parameters():
       param.requires_grad = False
   ```
3. **Bước 3: Thay thế lớp phân loại tương ứng theo từng kiến trúc:**
   - **ResNet-18:**
     ```python
     in_features = model.fc.in_features
     model.fc = nn.Linear(in_features, num_classes)
     ```
   - **VGG-16:**
     ```python
     in_features = model.classifier[6].in_features
     model.classifier[6] = nn.Linear(in_features, num_classes)
     ```
   - **DenseNet-121:**
     ```python
     in_features = model.classifier.in_features
     model.classifier = nn.Linear(in_features, num_classes)
     ```
4. **Bước 4: Thiết lập lớp phân loại mới ở chế độ học:**
   - Mặc định khi khởi tạo `nn.Linear`, các tham số của nó đã có `requires_grad = True`.
5. **Bước 5: Đóng gói thành Factory Builder & Viết Docstring đầy đủ:**
   - Xây dựng lớp trung gian `TransferModelBuilder(model_name, num_classes, pretrained)` chuẩn hóa tên và phân phối builder tương ứng.
   - Bổ sung docstring chi tiết ở đầu từng Class và method giải thích ngữ cảnh Transfer Learning.

---

### 4.2 Build Prompt cho AI (Independent Prompt)

Dán đoạn sau vào AI khi cần tạo mới hoặc cập nhật module model:

```text
Repository root:
D:\UTH\Deeplearning\lab\Practice-2---Hands-on-practice-with-pre-trained-neural-network-architectures

Implement exactly one task: hoàn thiện model builder cho transfer learning trong src/model_builder.py.

Chỉ được sửa:
- src/model_builder.py

Yêu cầu:
1. Hỗ trợ đầy đủ 3 kiến trúc: ResNet-18, VGG-16, DenseNet-121.
2. Dùng API weights hiện hành của torchvision (ResNet18_Weights.DEFAULT, VGG16_Weights.DEFAULT, DenseNet121_Weights.DEFAULT).
3. Đóng băng (requires_grad = False) toàn bộ backbone của mô hình.
4. Thay thế lớp classification head cuối cùng có out_features = 10 classes:
   - ResNet: model.fc
   - VGG: model.classifier[6]
   - DenseNet: model.classifier
5. Đảm bảo chỉ có classifier mới là trainable (requires_grad = True).
6. Tên model không hợp lệ phải ném ValueError kèm danh sách các lựa chọn được hỗ trợ.
7. Model trả về raw logits 2 chiều [batch_size, 10], tuyệt đối không dùng softmax nội bộ.
8. Viết docstring chi tiết ở đầu mỗi Class (nêu rõ bộ dữ liệu nguồn ImageNet-1k gồm 1.000 lớp, bài toán đích phân loại 10 lớp CIFAR-10, cơ chế freeze) và đầu mỗi hàm/method (mô tả chức năng, Args, Returns, Raises).


Sau khi sửa:
1. Chạy import test:
   .\venv\Scripts\python.exe -c "from src.model_builder import TransferModelBuilder; print('Import PASS')"
2. Chạy forward smoke test cho cả 3 kiến trúc:
   .\venv\Scripts\python.exe -c "import torch; from src.model_builder import TransferModelBuilder; x = torch.randn(2, 3, 224, 224); [print(m, TransferModelBuilder(m, 10, pretrained=False).build()(x).shape) for m in ['resnet18', 'vgg16', 'densenet121']]"
3. Báo logits shape [2, 10], tổng số tham số và số tham số trainable của từng mô hình.
4. Không sửa file ngoài phạm vi.
```

---

### 4.3 Lệnh kiểm tra và xác minh (Verification Commands)

#### Lệnh 1: Chạy toàn bộ Unit Tests chuyên biệt cho Model Builder
```powershell
.\venv\Scripts\python.exe -m pytest tests/test_model_builder.py -v
```

#### Lệnh 2: Kiểm tra số lượng tham số tổng và tham số Trainable
```powershell
.\venv\Scripts\python.exe -c "from src.model_builder import TransferModelBuilder; [print(f'=== {m.upper()} ===\nTotal params: {sum(p.numel() for p in model.parameters()):,}\nTrainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}') for m in ['resnet18', 'vgg16', 'densenet121'] for model in [TransferModelBuilder(m, 10, pretrained=False).build()]]"
```
*(Kết quả mong đợi: Toàn bộ 11 tests passed, Trainable params chỉ từ vài nghìn đến vài chục nghìn).*

