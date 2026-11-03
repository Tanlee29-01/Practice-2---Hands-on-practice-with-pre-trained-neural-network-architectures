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
Implement only model construction and adaptation.
Support ResNet18, VGG16, and DenseNet121 with 10 output classes.
Freeze the backbone by default and expose an explicit fine-tune option.
Do not modify data, trainer, logger, or README.
Run a forward smoke test and report logits shape plus trainable parameter count.
```

## Verify

Với mỗi model, forward một batch giả lập đúng input shape và xác nhận output
`[batch_size, 10]`. Xác nhận không có parameter backbone ngoài policy được
update, classifier mới có gradient sau backward thử nghiệm, và model chuyển
đúng sang CPU/GPU.

## Explain

Giải thích trong report vì sao classifier cuối phải thay đổi, vì sao freeze giúp
giảm số parameter cần học, và vì sao fine-tuning cần learning rate nhỏ hơn.