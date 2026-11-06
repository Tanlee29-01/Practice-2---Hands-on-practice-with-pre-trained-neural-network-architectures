# Model plan

## Define

Mục tiêu là dùng weights ImageNet làm feature extractor cho CIFAR-10 rồi thay
classification head thành 10 output classes. Các model bắt buộc: ResNet-18,
VGG-16 và DenseNet-121.

## Design

### Model adaptation

- ResNet: thay `model.fc`.
- VGG: thay phần cuối `model.classifier`.
- DenseNet: thay `model.classifier`.
- Kiểm tra model trả logits, không tự softmax trước CrossEntropyLoss.
- Tên model không hỗ trợ phải báo `ValueError` có danh sách lựa chọn.

### Transfer learning policy

Baseline freeze toàn bộ backbone rồi mở `requires_grad` cho classifier mới.
Fine-tuning là một experiment riêng: unfreeze một số layer cuối với learning
rate nhỏ hơn classifier. Không trộn hai policy mà không ghi rõ config.

### Inspection

Trước training phải in hoặc ghi: tên kiến trúc, classifier cũ/mới, tổng số
parameter, số parameter trainable, device và pretrained weights. Kiểm tra
classifier có đúng `out_features=10`.

### Missing implementation checklist

- [ ] Factory/builder dùng API weights hiện hành của torchvision.
- [ ] Hỗ trợ ba kiến trúc và cùng `num_classes`.
- [ ] Freeze/unfreeze có policy rõ ràng.
- [ ] Không dùng `pretrained=True` nếu API version yêu cầu `weights=`.
- [ ] Có kiểm tra forward bằng tensor mẫu.
- [ ] Đảm bảo optimizer chỉ nhận parameter cần train.

## Independent prompt

```text
Implement exactly one task: hoàn thiện model builder cho transfer learning.

Làm việc từ repository root và tuân thủ pattern OOP hiện có.
Chỉ được sửa:
- src/model_builder.py

Yêu cầu:
- Hỗ trợ cả 3 kiến trúc: ResNet-18, VGG-16 và DenseNet-121.
- Dùng API weights hiện hành của torchvision (ví dụ: ResNet18_Weights.DEFAULT).
- Thay classifier cuối thành 10 output classes:
  + ResNet: thay model.fc
  + VGG: thay model.classifier[6]
  + DenseNet: thay model.classifier
- Freeze backbone mặc định (requires_grad = False), chỉ cho phép classifier mới requires_grad = True.
- Có policy fine-tune rõ ràng (cho phép unfreeze các layer cuối khi được cấu hình).
- Tên model không hợp lệ phải báo ValueError kèm danh sách các lựa chọn được hỗ trợ.
- Model trả về raw logits, không dùng softmax trước CrossEntropyLoss.

Sau khi sửa:
1. Chạy import test:
   .\venv\Scripts\python.exe -c "from src.model_builder import TransferModelBuilder; print('Import PASS')"
2. Chạy forward smoke test cho cả 3 kiến trúc:
   .\venv\Scripts\python.exe -c "import torch; from src.model_builder import TransferModelBuilder; x = torch.randn(2, 3, 224, 224); [print(m, TransferModelBuilder(m, 10, pretrained=False).build()(x).shape) for m in ['resnet18', 'vgg16', 'densenet121']]"
3. Báo logits shape [batch_size, 10], số lượng parameter tổng và trainable parameter cho từng mô hình.
4. Không sửa file ngoài phạm vi.
```

## Verify

Với mỗi model, forward một batch giả lập đúng input shape và xác nhận output
`[batch_size, 10]`. Xác nhận không có parameter backbone ngoài policy được
update, classifier mới có gradient sau backward thử nghiệm, và model chuyển
đúng sang CPU/GPU.

## Explain

Giải thích trong report vì sao classifier cuối phải thay đổi, vì sao freeze giúp
giảm số parameter cần học, và vì sao fine-tuning cần learning rate nhỏ hơn.