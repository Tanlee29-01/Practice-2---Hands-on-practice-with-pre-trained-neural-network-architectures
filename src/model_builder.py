from typing import Dict, Type
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


class TransferModelBuilder_VGG16:
    """Xây dựng VGG-16 cho bài toán CIFAR-10."""

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
        """Tạo VGG-16, freeze feature extractor và thay model.classifier[6]."""

        if self.pretrained:
            weights = models.VGG16_Weights.DEFAULT
        else:
            weights = None

        model = models.vgg16(weights=weights)

        # Freeze toàn bộ trọng số hiện có của VGG-16.
        for parameter in model.parameters():
            parameter.requires_grad = False

        # Lấy số lượng input của lớp Linear cuối trong sequential classifier (vị trí index 6).
        input_features = model.classifier[6].in_features

        # Thay classification head cho CIFAR-10.
        model.classifier[6] = nn.Linear(
            in_features=input_features,
            out_features=self.num_classes,
        )

        # Chỉ cho phép lớp phân loại cuối được cập nhật khi training.
        for parameter in model.classifier[6].parameters():
            parameter.requires_grad = True

        return model


class TransferModelBuilder_DenseNet121:
    """Xây dựng DenseNet-121 cho bài toán CIFAR-10."""

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
        """Tạo DenseNet-121, freeze backbone và thay model.classifier."""

        if self.pretrained:
            weights = models.DenseNet121_Weights.DEFAULT
        else:
            weights = None

        model = models.densenet121(weights=weights)

        # Freeze toàn bộ trọng số hiện có của DenseNet-121.
        for parameter in model.parameters():
            parameter.requires_grad = False

        # Lấy số lượng input của lớp classifier cũ.
        input_features = model.classifier.in_features

        # Thay classification head cho CIFAR-10.
        model.classifier = nn.Linear(
            in_features=input_features,
            out_features=self.num_classes,
        )

        # Chỉ cho phép lớp classifier được cập nhật khi training.
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

        return model


class TransferModelBuilder:
    """Factory builder hỗ trợ khởi tạo linh hoạt các kiến trúc: ResNet-18, VGG16, DenseNet121."""

    SUPPORTED_MODELS: Dict[str, Type] = {
        "resnet18": TransferModelBuilder_ResNet18,
        "vgg16": TransferModelBuilder_VGG16,
        "densenet121": TransferModelBuilder_DenseNet121,
    }

    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 10,
        pretrained: bool = True,
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes phải lớn hơn 0.")

        normalized_name = (
            model_name.lower().replace("-", "").replace("_", "").strip()
        )
        builder_cls = self.SUPPORTED_MODELS.get(normalized_name)

        if builder_cls is None:
            supported = ", ".join(self.SUPPORTED_MODELS.keys())
            raise ValueError(
                f"Mô hình '{model_name}' không được hỗ trợ. Các mô hình hỗ trợ: {supported}"
            )

        self.model_name = normalized_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self._builder = builder_cls(
            num_classes=num_classes,
            pretrained=pretrained,
        )

    def build(self) -> nn.Module:
        """Khởi tạo và trả về mô hình PyTorch đã cấu hình."""
        return self._builder.build()

