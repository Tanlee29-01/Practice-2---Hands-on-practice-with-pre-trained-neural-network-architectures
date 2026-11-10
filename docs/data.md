# Kế hoạch đặc tả dữ liệu CIFAR-10 (Phiên bản 1.0)

> **Thông tin tài liệu**
>
> - **Module chính:** `src/data_module.py` — lớp `CIFAR10DataModule`.
> - **Module phân tích dữ liệu liên quan:** `src/eda.py`.
> - **Ngày soạn:** 10/11/2026.
> - **Khung tham chiếu:** `docs/model_v1.md`.
> - **Trạng thái:** Bản nháp để kiểm tra trước khi đưa vào `docs/data.md`.

Tài liệu được tổ chức theo flow **Purpose → Input/Output → Success Criteria → How to do (Prompt + Smoke Test)**. Các điều kiện PASS dưới đây là yêu cầu nghiệm thu; không phải thông báo rằng mã nguồn đã vượt qua kiểm thử.

---

## 1. Purpose (Mục đích)

Module `data_module` chuẩn bị dữ liệu CIFAR-10 cho bài toán phân loại ảnh 10 lớp bằng kỹ thuật học chuyển giao với **ResNet-18, VGG-16 và DenseNet-121**. Module chuyển ảnh gốc thành các batch tensor đúng giao diện đầu vào đã quy định trong `model_v1.md`.

Các nhiệm vụ chính:

1. **Nạp dữ liệu thống nhất:** Dùng `torchvision.datasets.CIFAR10` để lấy hai tập train và test chuẩn, lưu tại `data/` và sử dụng lại dữ liệu hợp lệ đã có.
2. **Tiền xử lý phù hợp mô hình:** Đổi kích thước ảnh từ `32×32` thành `224×224`, chuyển sang tensor và chuẩn hóa theo các hằng số ImageNet mà project đã thống nhất. Resize giúp đáp ứng contract đầu vào, không khôi phục chi tiết đã thiếu trong ảnh gốc.
3. **Tăng cường dữ liệu train:** Lật ngang ngẫu nhiên để tạo thêm biến thể khi học. Tập test dùng tiền xử lý xác định, không có phép biến đổi ngẫu nhiên.
4. **Cung cấp DataLoader:** Đóng gói việc chia batch và lấy mẫu, để module huấn luyện và đánh giá dùng cùng một giao diện.
5. **Hỗ trợ kiểm tra dữ liệu:** Cung cấp dữ liệu cho EDA nhằm xác nhận kích thước ảnh, phân phối nhãn và thống kê pixel trước khi huấn luyện.

Phạm vi của data module gồm tải dữ liệu, transforms, Dataset và DataLoader. Phân tích EDA nằm trong module riêng; xây dựng model, tính loss, cập nhật trọng số và tính các metric phân loại thuộc các module khác.

---

## 2. Input/Output (Thiết kế đầu vào và đầu ra)

### 2.1. Nguồn dữ liệu và quy ước nhãn

CIFAR-10 có **60.000 ảnh màu RGB**, kích thước gốc **32×32 pixel**, chia thành 10 lớp. Hai split chính thức gồm **50.000 ảnh train** và **10.000 ảnh test**; toàn bộ train có 5.000 ảnh mỗi lớp và test có 1.000 ảnh mỗi lớp. Nguồn: [trang CIFAR-10 chính thức](https://www.cs.toronto.edu/~kriz/cifar.html).

| Thuộc tính | Tập train | Tập test |
| :--- | :--- | :--- |
| Tham số khi nạp | `train=True` | `train=False` |
| Số ảnh | 50.000 | 10.000 |
| Số ảnh mỗi lớp | 5.000 | 1.000 |
| Ảnh gốc trong `dataset.data` | NumPy `uint8`, `[N, 32, 32, 3]` | NumPy `uint8`, `[N, 32, 32, 3]` |
| Tăng cường ngẫu nhiên | Có, lật ngang | Không |
| Xáo trộn khi lấy batch | Có | Không |
| Mục đích trong pipeline hiện tại | Cập nhật trọng số mô hình | Theo dõi mỗi epoch và chạy đánh giá sau huấn luyện |

Giữ thứ tự nhãn thống nhất với `src/evaluation.py`:

| Nhãn | Tên lớp | Diễn giải tiếng Việt |
| :---: | :--- | :--- |
| 0 | `airplane` | Máy bay |
| 1 | `automobile` | Ô tô |
| 2 | `bird` | Chim |
| 3 | `cat` | Mèo |
| 4 | `deer` | Hươu |
| 5 | `dog` | Chó |
| 6 | `frog` | Ếch |
| 7 | `horse` | Ngựa |
| 8 | `ship` | Tàu thủy |
| 9 | `truck` | Xe tải |

**Lưu ý về cách dùng test set:** `main.py` hiện truyền test loader vào bước `validation()` và dùng test accuracy để lưu best checkpoint. Vì vậy, kết quả test trong pipeline này đã tham gia chọn mô hình, không còn là ước lượng từ một tập kiểm tra độc lập hoàn toàn. Bản đặc tả giữ hai split chuẩn để tương thích project. Nếu nhóm muốn tách validation từ train để tuning và chọn checkpoint, cần cập nhật đồng bộ data, training và evaluation trong một nhiệm vụ riêng.

### 2.2. Tham số đầu vào của data module

Giữ nguyên chữ ký khởi tạo hiện có:

```python
CIFAR10DataModule(
    data_dir="./data",
    batch_size=64,
    num_workers=2,
    image_size=224,
)
```

| Tham số | Kiểu dữ liệu | Mặc định | Ý nghĩa và điều kiện hợp lệ |
| :--- | :--- | :--- | :--- |
| `data_dir` | `str` | `"./data"` | Thư mục lưu CIFAR-10; cần ghi được nếu tải mới. Đường dẫn tương đối tính từ thư mục chạy lệnh. |
| `batch_size` | `int` | `64` | Số ảnh tối đa mỗi batch; phải lớn hơn 0. |
| `num_workers` | `int` | `2` | Số tiến trình hỗ trợ nạp dữ liệu; phải từ 0 trở lên. Dùng `0` khi smoke test trên Windows. |
| `image_size` | `int` | `224` | Kích thước ảnh vuông sau resize; phải lớn hơn 0. Pipeline theo `model_v1.md` sử dụng `224`. |

Không thêm `seed`, `download` hoặc `val_dataloader()` vào contract hiện tại. Seed của phép thử được đặt trong chương trình kiểm tra; `setup()` hiện dùng `download=True` khi nạp dữ liệu.

### 2.3. Thiết kế transforms

| Thứ tự | Train transform | Test transform |
| :---: | :--- | :--- |
| 1 | `Resize((image_size, image_size))` | `Resize((image_size, image_size))` |
| 2 | `RandomHorizontalFlip(p=0.5)` | Không có augmentation ngẫu nhiên |
| 3 | `ToTensor()` | `ToTensor()` |
| 4 | `Normalize(mean, std)` | `Normalize(mean, std)` |

Các hằng số dùng chung cho hai tập theo contract của project:

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

Với ảnh RGB `uint8` đang dùng, `ToTensor()` đổi thứ tự chiều từ HWC sang CHW và đưa giá trị từ `[0, 255]` về `[0, 1]`. Bước chuẩn hóa tiếp theo tính theo từng kênh:

```text
pixel_normalized[c] = (pixel_0_1[c] - mean[c]) / std[c]
```

Đây là phép chuẩn hóa được mô tả trong [tài liệu torchvision Normalize](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Normalize.html). Ảnh sau bước này có thể chứa giá trị âm hoặc lớn hơn 1. Không kiểm tra chúng bằng điều kiện thuộc `[0, 1]`, và không yêu cầu mean/std thực đo của CIFAR-10 sau chuẩn hóa phải đúng bằng 0/1.

### 2.4. Giao diện và vị trí mã nguồn

```text
src/data_module.py
└── class CIFAR10DataModule
    ├── __init__(data_dir="./data", batch_size=64,
    │            num_workers=2, image_size=224)
    ├── setup() -> None
    │   ├── train_transform / test_transform
    │   └── train_dataset / test_dataset
    ├── train_dataloader() -> DataLoader
    └── test_dataloader() -> DataLoader

src/eda.py                         # Phân tích ảnh gốc, xuất báo cáo EDA
notebooks/cifar10_eda.ipynb         # Notebook phân tích liên quan
data/cifar-10-batches-py/           # Dữ liệu CIFAR-10 đã giải nén
```

`setup()` chuẩn bị transforms và hai dataset; không trả về loader. Sau khi gọi `setup()`, lấy loader bằng hai phương thức tương ứng. Loader chỉ nạp và biến đổi dữ liệu, không gọi lại `setup()` trong mỗi epoch.

| Đầu ra | Contract |
| :--- | :--- |
| `train_dataset` | Dataset CIFAR-10 train có train transform, 50.000 mẫu trong chế độ đầy đủ. |
| `test_dataset` | Dataset CIFAR-10 test có test transform, 10.000 mẫu trong chế độ đầy đủ. |
| `train_dataloader()` | `shuffle=True`, `drop_last=False`. |
| `test_dataloader()` | `shuffle=False`, `drop_last=False`. |
| Batch `(images, labels)` | Ảnh `torch.float32` dạng `[B, 3, 224, 224]`; nhãn `torch.int64` dạng `[B]`, giá trị từ 0 đến 9. |
| Thiết bị của batch | CPU tại đầu ra data module; trainer/evaluation phụ trách chuyển lên thiết bị tính toán. |
| `pin_memory` | Dùng `torch.cuda.is_available()` như implementation hiện có. |

`B` là số ảnh thực tế trong batch, có thể nhỏ hơn `batch_size` ở batch cuối. Ví dụ, với batch size 64, train có 782 batch và test có 157 batch; batch cuối của mỗi tập có 16 ảnh. Nhãn giữ dạng chỉ số lớp, không chuyển thành one-hot khi dùng `CrossEntropyLoss` theo thiết kế của project.

### 2.5. Luồng dữ liệu

```text
CIFAR-10 gốc: RGB 32×32
    │
    ├── Train 50.000 ảnh → Resize → RandomHorizontalFlip → ToTensor → Normalize
    │                     → train_dataloader (shuffle) → training loop
    │
    └── Test 10.000 ảnh  → Resize → ToTensor → Normalize
                          → test_dataloader (giữ thứ tự) → validation/evaluation

Ảnh gốc, chưa transform → src/eda.py → Thống kê và biểu đồ EDA
```

### 2.6. Đầu ra EDA

EDA làm việc với ảnh gốc trước resize và normalize để mô tả đúng dữ liệu nguồn. Với CLI hiện có, kết quả được lưu tại `results/eda/<timestamp>/`:

| Artifact | Nội dung |
| :--- | :--- |
| `eda_summary.json` | Nguồn dữ liệu, số mẫu đã phân tích, shape, dtype, thống kê RGB và kiểm tra giá trị bất thường. |
| `class_distribution.csv` | Tên lớp, số lượng và tỷ lệ phần trăm theo từng split. |
| `class_distribution.png` | Biểu đồ phân phối nhãn train/test. |
| `split_indices.npz` | Chỉ số của toàn bộ hai split chuẩn, theo implementation hiện tại. |

Chế độ `--max-samples 100` hiện chọn 100 ảnh đầu của mỗi split. Phân phối nhãn và thống kê pixel trong báo cáo khi đó chỉ mô tả subset; không dùng để khẳng định toàn bộ dữ liệu cân bằng. Seed được ghi lại trong báo cáo nhưng không làm cách chọn subset này trở thành lấy mẫu ngẫu nhiên.

---

## 3. Success Criteria (Tiêu chí thành công và nghiệm thu)

Data pipeline được nghiệm thu khi có bằng chứng kiểm tra các điều kiện sau:

| STT | Tiêu chí | Điều kiện chấp thuận | Cách xác minh |
| :---: | :--- | :--- | :--- |
| 1 | Nguồn và split | CIFAR-10 chính thức; 50.000 train và 10.000 test. | Kiểm tra `len(dataset)` và tham số `train`. |
| 2 | Nhãn nhất quán | Đủ 10 lớp theo thứ tự ở mục 2.1; nhãn trong `[0, 9]`. | Kiểm tra `classes`, `class_to_idx`, `targets`. |
| 3 | Cân bằng toàn bộ dữ liệu | Mỗi lớp có 5.000 train và 1.000 test. | Đếm nhãn trên toàn bộ mỗi split. |
| 4 | Batch đúng contract | Ảnh `float32 [B, 3, 224, 224]`, nhãn `int64 [B]`; số ảnh và số nhãn khớp nhau. | Lấy một batch từ mỗi loader. |
| 5 | Transforms đúng | Train có lật ngang; test không random; dùng đúng mean/std và thứ tự xử lý. | Đọc pipeline và đối chiếu một ảnh test với phép tính tham chiếu. |
| 6 | Giá trị hợp lệ | Tensor ảnh hữu hạn; ảnh sau hoàn nguyên normalize nằm trong `[0, 1]` với sai số số thực nhỏ. | Dùng `torch.isfinite()` và phép chuẩn hóa ngược. |
| 7 | Chính sách lấy mẫu | Train xáo trộn, test giữ thứ tự, không bỏ batch cuối. | Kiểm tra sampler, `drop_last` và số batch. |
| 8 | Test có tính xác định | Cùng một mẫu test, cùng môi trường và cấu hình, cho tensor giống nhau khi đọc lặp lại. | So sánh hai lần đọc `test_dataset[0]`. |
| 9 | Báo lỗi đúng | Loader trước `setup()` gây `RuntimeError`; tham số số nguyên ngoài miền hợp lệ gây `ValueError` rõ ràng. | Thử gọi loader sớm và thử giá trị 0/âm theo mục 4.4. |
| 10 | Lỗi dữ liệu có thể chẩn đoán | Không trả dataset giả/rỗng thay cho lỗi tải, quyền truy cập hoặc dữ liệu hỏng; giữ nguyên nhân lỗi. | Đọc xử lý exception; tái hiện lỗi cụ thể khi phát sinh. |
| 11 | Tài liệu mã nguồn | Class và các phương thức có docstring giải thích chức năng, tham số, đầu ra và lỗi liên quan. | Review `src/data_module.py`. |
| 12 | EDA đầy đủ | Có resolution gốc, phân phối 10 lớp và min/max/mean/std theo RGB, nêu rõ thang `[0, 255]` và số mẫu phân tích. | Đọc artifact của lần chạy EDA đầy đủ. |

**Đối chiếu hiện trạng:** Mã nguồn đã có hai dataset, hai loader, transforms và lỗi gọi loader trước `setup()`. Cần bổ sung kiểm tra tham số ngay khi khởi tạo, kiểm tra dataset rỗng và docstring. Các mục này là yêu cầu của prompt bên dưới, không được ghi là đã hoàn thành trước khi triển khai và kiểm thử.

Smoke test trên một batch chỉ xác nhận một phần contract. EDA đầy đủ và review mã nguồn bổ sung bằng chứng cho các mục còn lại; các kiểm tra này không đo accuracy của mô hình.

---

## 4. How to do (Cách thực hiện: Prompt + Smoke Test)

### 4.1. Quy trình thực hiện

1. Đọc `src/data_module.py`, `main.py`, `src/evaluation.py` và `docs/model_v1.md` để xác nhận contract trước khi sửa.
2. Giữ nguyên API công khai và hai split chuẩn. Hoàn thiện phần validation đầu vào, kiểm tra dữ liệu và docstring còn thiếu.
3. Chạy import test, smoke test dữ liệu thật và kiểm tra các trường hợp lỗi; không chạy huấn luyện để thay thế smoke test dữ liệu.
4. Chạy EDA bằng nhiệm vụ riêng, trước ở subset rồi trên toàn bộ dữ liệu.
5. Đối chiếu kết quả với mục 3; ghi command, output thực tế, exit code và đường dẫn artifact. Nếu chưa chạy được, ghi rõ lý do và phần chưa kiểm tra.

### 4.2. Prompt độc lập: Hoàn thiện data module

Dán prompt sau vào AI khi bắt đầu nhiệm vụ triển khai. Đây là hướng dẫn cho lần triển khai sau, không phải yêu cầu thực thi mã nguồn trong lúc duyệt bản nháp tài liệu này.

```text
Nhiệm vụ duy nhất: hoàn thiện data module CIFAR-10 trong src/data_module.py.

Làm việc từ thư mục gốc repository:
Practice-2---Hands-on-practice-with-pre-trained-neural-network-architectures

Trước khi sửa, đọc:
- src/data_module.py
- main.py
- src/evaluation.py
- docs/model_v1.md
- docs/data.md

Chỉ được sửa src/data_module.py.
Giữ nguyên các thay đổi cục bộ không thuộc nhiệm vụ.

Yêu cầu:
1. Giữ class CIFAR10DataModule và chữ ký:
   __init__(data_dir="./data", batch_size=64, num_workers=2, image_size=224).
   Giữ setup(), train_dataloader(), test_dataloader(), train_dataset,
   test_dataset, train_transform và test_transform sau setup().
2. Kiểm tra ngay tại __init__: batch_size và image_size là số nguyên > 0;
   num_workers là số nguyên >= 0. Từ chối cả bool và giá trị không phải số
   nguyên bằng ValueError; thông báo nêu tên tham số và điều kiện hợp lệ.
3. setup() nạp torchvision.datasets.CIFAR10 vào data_dir, download=True,
   train=True cho train_dataset và train=False cho test_dataset.
   Dùng lại dữ liệu hợp lệ đã có, không tải dataset trong mỗi epoch.
4. Train transform đúng thứ tự:
   Resize((image_size, image_size)), RandomHorizontalFlip(p=0.5),
   ToTensor(), Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).
5. Test transform: Resize((image_size, image_size)), ToTensor(), Normalize
   với cùng mean/std; không có augmentation ngẫu nhiên.
6. train_dataloader: shuffle=True; test_dataloader: shuffle=False.
   Cả hai dùng batch_size, num_workers đã cấu hình, drop_last=False,
   pin_memory=torch.cuda.is_available(). Không chuyển tensor lên GPU ở đây.
7. Gọi loader trước setup() phải gây RuntimeError hướng dẫn gọi setup().
   Sau khi nạp, kiểm tra dataset không rỗng, báo RuntimeError nếu rỗng.
   Khi có lỗi tải, dữ liệu hỏng hoặc lỗi quyền truy cập, giữ exception gốc
   hoặc bổ sung ngữ cảnh bằng raise ... from exc; không nuốt lỗi hay dùng
   dữ liệu giả làm fallback. Không xóa dữ liệu đang có để thử tải lại.
8. Batch mặc định: ảnh float32 [B,3,224,224], nhãn int64 [B] thuộc [0,9].
   Giữ thứ tự lớp CIFAR-10. Không one-hot nhãn, không đổi split hoặc nhãn.
9. Viết docstring cho class và mọi phương thức: chức năng, Args, Returns,
   Raises khi có; giải thích train/test transforms và ImageNet normalization.
10. Không thêm validation split, seed parameter hoặc thay đổi consumer.
    Không sửa model, trainer, evaluation, logger, main, config hoặc EDA.

Sau khi sửa:
- Chạy các lệnh ở mục 4.3 và 4.4 của docs/data.md từ repository root.
- Dùng CPU, num_workers=0, seed đặt trong script; không chạy full training.
- Báo file đã sửa, command, output, exit code, số mẫu, shape/dtype của batch
  và kết quả từng kiểm tra. Không kết luận PASS nếu chưa chạy kiểm thử.
- Nếu môi trường hoặc mạng chưa đáp ứng, báo traceback và phần chưa kiểm tra;
  không tự đổi dependency/config hoặc giảm yêu cầu để tạo kết quả PASS.
```

### 4.3. Smoke test với dữ liệu thật

Các lệnh PowerShell dưới đây chạy tại repository root, sau khi môi trường `venv` có Python, PyTorch và torchvision hoạt động. Nếu dùng môi trường khác, thay đường dẫn interpreter cho phù hợp. `setup()` có thể tải CIFAR-10 trong lần đầu nên cần kết nối mạng và quyền ghi thư mục dữ liệu.

**Bước 1 — Import module**

```powershell
.\venv\Scripts\python.exe -c "from src.data_module import CIFAR10DataModule; print('Import PASS')"
$LASTEXITCODE
```

**Bước 2 — Kiểm tra split, batch và transforms**

Đoạn này nạp hai split chuẩn, đếm nhãn từ metadata và chỉ lấy một batch mỗi loader; không duyệt toàn bộ ảnh sau resize và không huấn luyện mô hình.

```powershell
@'
from collections import Counter

import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler
from torchvision.transforms import functional as TF

from src.data_module import CIFAR10DataModule

torch.manual_seed(42)
classes = "airplane automobile bird cat deer dog frog horse ship truck".split()
dm = CIFAR10DataModule(batch_size=4, num_workers=0, image_size=224)
dm.setup()

for name, dataset, size, per_class in [
    ("train", dm.train_dataset, 50000, 5000),
    ("test", dm.test_dataset, 10000, 1000),
]:
    assert len(dataset) == size, (name, len(dataset))
    assert dataset.data.shape == (size, 32, 32, 3)
    assert str(dataset.data.dtype) == "uint8"
    assert list(dataset.classes) == classes
    assert dataset.class_to_idx == {label: i for i, label in enumerate(classes)}
    assert Counter(dataset.targets) == {i: per_class for i in range(10)}
    print(name, "samples:", size, "counts:", dict(sorted(Counter(dataset.targets).items())))

train_loader = dm.train_dataloader()
test_loader = dm.test_dataloader()
assert isinstance(train_loader.sampler, RandomSampler)
assert isinstance(test_loader.sampler, SequentialSampler)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

for name, loader in [("train", train_loader), ("test", test_loader)]:
    assert loader.drop_last is False
    assert len(loader) == (len(loader.dataset) + 3) // 4
    images, labels = next(iter(loader))
    assert images.shape == (4, 3, 224, 224), images.shape
    assert labels.shape == (4,), labels.shape
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64
    assert images.device.type == "cpu" and labels.device.type == "cpu"
    assert torch.isfinite(images).all().item()
    assert ((labels >= 0) & (labels <= 9)).all().item()
    restored = images * std + mean
    assert restored.min().item() >= -1e-5
    assert restored.max().item() <= 1 + 1e-5
    print(name, "batch:", tuple(images.shape), tuple(labels.shape), images.dtype, labels.dtype)

first, label = dm.test_dataset[0]
again, repeated_label = dm.test_dataset[0]
assert label == repeated_label and torch.equal(first, again)
raw_image = Image.fromarray(dm.test_dataset.data[0])
reference = TF.to_tensor(TF.resize(raw_image, [224, 224]))
reference = (reference - mean) / std
assert torch.allclose(first, reference, atol=1e-6, rtol=1e-5)

dm.batch_size = 64
for loader, expected_batches in [(dm.train_dataloader(), 782), (dm.test_dataloader(), 157)]:
    assert loader.drop_last is False
    assert len(loader) == expected_batches
    last_indices = None
    for indices in loader.batch_sampler:
        last_indices = indices
    assert len(last_indices) == 16

print("Data smoke test PASS | seed=42 | num_workers=0 | final batch=16 at batch_size=64")
'@ | .\venv\Scripts\python.exe -
$LASTEXITCODE
```

**Kết quả mong đợi:** Hai split có đúng số lượng và phân phối lớp; mỗi batch thử có ảnh `(4, 3, 224, 224)` và nhãn `(4,)`; các assertion đạt, in `Data smoke test PASS`, exit code bằng `0`. Kết quả này vẫn cần review augmentation train và các trường hợp lỗi ở mục tiếp theo.

Không bắt buộc hai lần đọc cùng một ảnh train phải khác nhau: phép lật ngang ngẫu nhiên có thể cho cùng kết quả ở cả hai lần.

### 4.4. Kiểm tra các trường hợp lỗi sau khi hoàn thiện module

Đoạn này không tải dữ liệu. Phần kiểm tra tham số là yêu cầu bổ sung, có thể FAIL trên mã nguồn hiện tại nếu chưa áp dụng prompt mục 4.2.

```powershell
@'
from src.data_module import CIFAR10DataModule

dm = CIFAR10DataModule(num_workers=0)
for method in (dm.train_dataloader, dm.test_dataloader):
    try:
        method()
    except RuntimeError as exc:
        assert "setup" in str(exc).lower()
        print(method.__name__, "PASS:", str(exc))
    else:
        raise AssertionError("Loader must reject calls before setup()")

for name, invalid_values in [
    ("batch_size", [0, -1, 1.5, True]),
    ("image_size", [0, -1, 1.5, True]),
    ("num_workers", [-1, 1.5, True]),
]:
    for value in invalid_values:
        try:
            CIFAR10DataModule(**{name: value})
        except ValueError as exc:
            assert name in str(exc)
            print(name, repr(value), "PASS:", str(exc))
        else:
            raise AssertionError(f"Expected ValueError for {name}={value!r}")

print("Invalid-input checks PASS")
'@ | .\venv\Scripts\python.exe -
$LASTEXITCODE
```

Kỳ vọng exit code bằng `0`. Không tạo lỗi quyền truy cập hoặc dữ liệu hỏng bằng cách sửa/xóa dataset thật; nếu cần tái hiện các lỗi đó, dùng thư mục thử riêng.

### 4.5. Prompt riêng: Kiểm tra và hoàn thiện EDA

Thực hiện sau khi smoke test data module đạt yêu cầu. Nếu `src/eda.py` đã đáp ứng contract, dùng lại implementation và chỉ chạy xác minh.

```text
Nhiệm vụ duy nhất: kiểm tra và hoàn thiện EDA cho ảnh CIFAR-10.

Làm việc từ repository root. Đọc src/eda.py trước; giữ phần đã đáp ứng yêu cầu.
Nếu cần sửa, chỉ được sửa:
- src/eda.py
- notebooks/cifar10_eda.ipynb, nếu cần cập nhật notebook liên quan.

Yêu cầu:
1. Phân tích ảnh gốc train/test trước transforms: RGB, [N,32,32,3], uint8.
2. Ghi số mẫu đã phân tích và tổng số mẫu của từng split.
3. Báo đủ 10 lớp: số lượng và tỷ lệ phần trăm mỗi lớp ở từng split.
4. Tính min, max, mean và standard deviation theo RGB trên thang [0,255].
   Nêu rõ quy ước population standard deviation nếu dùng ddof=0.
5. Kiểm tra NaN, Inf và pixel ngoài [0,255]; báo lỗi nếu phát hiện bất thường.
6. Xuất eda_summary.json, class_distribution.csv, class_distribution.png
   và split_indices.npz vào results/eda/<timestamp>/ như CLI hiện tại.
7. Với --max-samples 100, báo rõ đây là subset; không kết luận toàn bộ dữ
   liệu cân bằng từ subset. Chỉ số trong split_indices.npz hiện mô tả toàn
   bộ split; không gọi đó là chỉ số riêng của subset đã phân tích.
8. Không sửa data module, transforms dùng cho huấn luyện, split, nhãn,
   model, trainer, evaluation hoặc config. Không tính lại mean/std để thay
   các hằng số ImageNet đã thống nhất trong project.

Sau khi kiểm tra hoặc sửa:
- Chạy EDA smoke test và full EDA theo mục 4.6 của docs/data.md.
- Báo command, output, exit code, số mẫu thực tế và đường dẫn artifact.
- Đối chiếu shape ảnh gốc trong EDA với shape sau transform đã xác minh ở
  data smoke test; không coi metadata hard-code là bằng chứng đo thực tế.
- Không báo PASS nếu chưa chạy hoặc thiếu artifact cần nghiệm thu.
```

### 4.6. Lệnh chạy EDA và bằng chứng bàn giao

Sau khi data module đã chuẩn bị dữ liệu ở `data/`, chạy EDA subset:

```powershell
.\venv\Scripts\python.exe -m src.eda --data-dir data --max-samples 100
$LASTEXITCODE
```

Sau khi smoke test thành công, chạy EDA toàn bộ hai split:

```powershell
.\venv\Scripts\python.exe -m src.eda --data-dir data
$LASTEXITCODE
```

CLI EDA hiện không tự tải dữ liệu theo mặc định. Nếu chạy EDA độc lập khi chưa có CIFAR-10, thêm `--download` vào lệnh. Mỗi lần chạy tạo một thư mục kết quả mới; đọc đường dẫn được in ra để xác định đúng artifact của lần chạy đó.

Khi bàn giao phần data, cung cấp:

- Command đã chạy, exit code và output thực tế của các kiểm tra.
- Số mẫu train/test, mapping 10 lớp, shape và dtype của batch.
- Kết quả kiểm tra transforms, sampler, batch cuối và báo lỗi.
- Thư mục EDA đầy đủ cùng các artifact để thành viên khác kiểm tra.
- Phiên bản Python/PyTorch/torchvision, seed của phép thử và những mục chưa xác minh.

Chỉ đánh dấu hoàn tất khi các bằng chứng phù hợp tiêu chí ở mục 3. Nếu mới duyệt tài liệu hoặc chưa thực thi được lệnh, giữ trạng thái **chưa kiểm chứng runtime**.
