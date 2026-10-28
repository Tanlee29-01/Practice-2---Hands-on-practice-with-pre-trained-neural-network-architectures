from torch import nn
from torchvision import models


class TransferModelBuilder_ResNet18:
    """Xây dựng ResNet-18 cho bài toán CIFAR-10."""

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes phải lớn hơn 0.")

        self.num_classes = num_classes
        self.pretrained = pretrained

    def build(self) -> nn.Module:
        """Tạo ResNet-18, freeze backbone và thay model.fc."""

        if self.pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        model = models.resnet18(weights=weights)

        # Freeze toàn bộ trọng số hiện có của ResNet-18.
        for parameter in model.parameters():
            parameter.requires_grad = False

        # Lấy số lượng input của lớp fc cũ.
        input_features = model.fc.in_features

        # Thay classification head cho CIFAR-10.
        model.fc = nn.Linear(
            in_features=input_features,
            out_features=self.num_classes,
        )

        # Chỉ cho phép lớp fc được cập nhật khi training.
        for parameter in model.fc.parameters():
            parameter.requires_grad = True

        return model
