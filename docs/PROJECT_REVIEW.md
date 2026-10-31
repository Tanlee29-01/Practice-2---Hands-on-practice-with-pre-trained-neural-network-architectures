# Nhận xét dự án và quy trình đánh giá

Đánh giá dựa trên mã nguồn tại workspace và repository GitHub
[Practice 2](https://github.com/Tanlee29-01/Practice-2---Hands-on-practice-with-pre-trained-neural-network-architectures).
Các nhận xét bên dưới phân biệt việc mã nguồn đã hỗ trợ với kết quả thực nghiệm đã có.

## Những phần đã làm tốt

- Tách dữ liệu, builder, trainer và logger; có factory chung cho ResNet-18, VGG16, DenseNet121.
- Thay đúng lớp phân loại cuối, đóng băng tham số backbone và giữ head có thể học.
- Train/test dùng cùng resize và chuẩn hóa; chỉ train có RandomHorizontalFlip.
- Loss của trainer đã cộng theo số ảnh trước khi lấy trung bình.
- Có TensorBoard, checkpoint model/optimizer/epoch và kiểm thử cho các module chính.
- `experiments.py` tách validation từ train, dùng transform không ngẫu nhiên cho validation,
  giữ seed và chỉ thay một siêu tham số so với baseline.

## Các hạn chế cần xử lý theo mức ưu tiên

| Mức | Vị trí | Nhận xét và tác động | Hướng xử lý |
| --- | --- | --- | --- |
| Cao | `main.py`: vòng lặp epoch, `save_best_model(metric=test_accuracy)` | Tập test được dùng để chọn checkpoint tốt nhất; kết quả sau lựa chọn có thể lạc quan | Tách train/validation, chọn best trên validation; chỉ dùng test ở bước cuối |
| Cao | `src/trainer.py`: `self.model.train()` | Freeze `requires_grad` không ngăn running mean/variance của BatchNorm thay đổi; backbone chưa được giữ cố định hoàn toàn về trạng thái | Nếu mục tiêu là feature extraction cố định, giữ BatchNorm ở eval như `FrozenBackboneTrainer` trong `experiments.py`; nếu muốn thích nghi thống kê, ghi rõ lựa chọn |
| Vừa | `experiments.py` | Chỉ ghi CSV, chưa lưu checkpoint theo từng cấu hình; không thể khôi phục trọng số từ CSV để đánh giá chi tiết | Lưu best checkpoint theo validation cho mỗi cấu hình, kèm model name, seed, chỉ số split và tiền xử lý |
| Vừa | `src/logger.py`: checkpoint hiện tại | Có state_dict/optimizer/epoch/metrics nhưng thiếu kiến trúc, thứ tự lớp, cấu hình và giao thức chọn best | Bổ sung metadata trong các lần huấn luyện sau; bộ đánh giá mới vẫn đọc được checkpoint cũ nhờ keys và shapes |
| Vừa | Quy trình thử nghiệm | Chưa có kết quả nhiều seed và checkpoint đầy đủ của cả ba kiến trúc trong workspace | Huấn luyện với cùng split, so sánh trên cùng test; báo cáo biến thiên qua nhiều seed khi cần kết luận ổn định |
| Đã xử lý | Script kiểm tra dữ liệu, trước đây là `test.py` ở thư mục gốc | Trước đây tạo dataset ngay khi import, dễ gây tác dụng phụ khi pytest thu thập file | Đã chuyển thành `scripts/check_data.py`, có main guard, tham số CLI và chỉ tải dữ liệu khi có `--download`; chạy bằng `python -m scripts.check_data` |

Phần đánh giá thêm `stage="test"` vào data module. Khi sắp xếp dự án, script kiểm tra dữ liệu
được chuyển vào `scripts/`, CSV thí nghiệm được gom vào `results/experiments/`.
Các hạn chế về chọn checkpoint, BatchNorm và lưu thí nghiệm được ghi nhận để không diễn giải quá mức số liệu.

## Đọc các thí nghiệm cũ

File `results/experiments/20261028_222705_829922/experiments.csv` ghi sweep 3 epoch trên 1.000 ảnh train
và 500 ảnh validation. Ở epoch cuối, `lr_high` (`lr=0.01`) đạt validation accuracy 77,8%,
baseline đạt 73,8%; `lr_low` mới đạt 17,0%.
Đây là kết quả trên tập con và ngân sách huấn luyện ngắn; chưa đủ để kết luận learning rate
tối ưu cho huấn luyện đầy đủ, hoặc mô hình nào trong ba kiến trúc tốt nhất.

Các cấu hình L2 trong sweep cho kết quả rất gần nhau sau 3 epoch.
Chưa đủ cơ sở để kết luận L2 không có tác dụng; cần quan sát train/validation qua nhiều epoch và seed.
Các số này là validation, không phải kết quả test của checkpoint hiện tại.

## Bộ đánh giá bổ sung

Các file độc lập đã có luồng chạy chung qua `evaluate.py`, dùng checkpoint hiện có và
dataset test của dự án. Có accuracy, balanced accuracy, top-k accuracy, precision/recall/F1
macro/weighted, specificity từng lớp, confusion matrix, ROC AUC OvR và average precision.
Kết quả được xuất đồng thời dạng văn bản, JSON, CSV và ảnh PNG.

Xem [hướng dẫn đánh giá](EVALUATION.md) để chạy, đọc quy ước chỉ số và dùng các lớp trong mã Python.
Báo cáo tự lưu đường dẫn checkpoint, epoch, tiền xử lý, seed, tập mẫu và lưu ý về việc dùng test chọn best.

## Kết quả chạy thực tế trên checkpoint hiện có

Đã chạy `python evaluate.py` trên CPU với checkpoint
`resnet18_20261030-222811_best.pth` (metadata ghi epoch 1), toàn bộ **10.000 ảnh test**,
224×224, batch size 64, không augmentation và không huấn luyện thêm.
Báo cáo hoàn tất ngày 31/10/2026, lúc 08:01 theo múi giờ Việt Nam.

| Chỉ số | Kết quả |
| --- | ---: |
| Số dự đoán đúng | 7.913 / 10.000 |
| Cross-entropy loss | 0,629299 |
| Accuracy | 79,13% |
| Balanced accuracy | 79,13% |
| Precision macro | 0,798518 |
| Recall macro | 0,791300 |
| F1 macro | 0,791172 |
| Top-5 accuracy | 98,69% |
| ROC AUC macro OvR | 0,977007 |
| Average precision macro | 0,872522 |

Mỗi lớp có 1.000 ảnh nên macro và weighted P/R/F1 bằng nhau ở lần đánh giá này.
Cả 10 lớp đều có mẫu dương và âm nên AUC/AP đều tính được.

| Lớp | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| airplane | 0,7418 | 0,8590 | 0,7961 |
| automobile | 0,8950 | 0,8610 | 0,8777 |
| bird | 0,7920 | 0,6740 | 0,7283 |
| cat | 0,6347 | 0,7470 | 0,6863 |
| deer | 0,7452 | 0,7840 | 0,7641 |
| dog | 0,8566 | 0,6330 | 0,7280 |
| frog | 0,7848 | 0,8790 | 0,8292 |
| horse | 0,7802 | 0,8270 | 0,8029 |
| ship | 0,8923 | 0,7950 | 0,8408 |
| truck | 0,8626 | 0,8540 | 0,8583 |

`automobile` có F1 cao nhất; `cat` có F1 thấp nhất và precision khoảng 63,47%.
`dog` có recall thấp nhất, 63,30%: 205/1.000 ảnh chó bị dự đoán thành mèo.
Một hướng nhầm đáng chú ý khác là 113/1.000 ảnh tàu bị dự đoán thành máy bay.
Đây là các nhóm ảnh nên kiểm tra khi xây dựng thí nghiệm cải thiện trên validation.

Top-5 accuracy cao hơn nhiều so với top-1 cho thấy nhãn thật thường vẫn nằm trong nhóm năm
đầu ra có xác suất cao nhất, nhưng chưa giải quyết được sai nhãn khi hệ thống chỉ chọn một lớp.
Các số trên mô tả checkpoint hiện tại và chịu hạn chế chọn checkpoint bằng test đã nêu ở trên.
Chưa có checkpoint VGG16/DenseNet121 trong workspace để đưa ra so sánh thực nghiệm giữa ba kiến trúc.

Các file chi tiết nằm tại:

```text
results/evaluation/resnet18_20261030-222811_best_20261031_080111_789486/
```

Đã đối chiếu CSV có đủ 10.000 chỉ số ảnh duy nhất và 7.913 dòng dự đoán đúng.
Bộ kiểm thử dự án đạt 38 test; 14 test mới của phần đánh giá cũng được chạy lại thành công
sau chỉnh sửa cuối cùng. Có hai FutureWarning từ phần TensorBoard graph cũ về `torch.jit.trace`.
