from typing import Dict, Type
from torch import nn
from torchvision import models


class TransferModelBuilder_ResNet18:
    """Xây dựng mô hình ResNet-18 chuyển giao (Transfer Learning) cho CIFAR-10.

    - Nguồn tiền huấn luyện (Pre-trained Source): ImageNet-1k (1.000 danh mục tổng quát).
    - Bài toán đích (Target Task): Phân loại ảnh 10 đối tượng CIFAR-10.
    - Chiến lược chuyển giao (Transfer Policy): Đóng băng (freeze) toàn bộ trọng số
      của backbone trích xuất đặc trưng (conv1 -> layer4), thay thế lớp Linear cuối
      (model.fc: 512 -> 10) và mở requires_grad = True để chỉ huấn luyện lớp này.
    """

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
    ) -> None:
        """Khởi tạo cấu hình cho builder ResNet-18.

        Args:
            num_classes (int): Số lượng lớp đầu ra của bài toán phân loại (mặc định: 10).
            pretrained (bool): Có tải trọng số pre-trained ImageNet hay không (mặc định: True).

        Raises:
            ValueError: Nếu num_classes <= 0.
        """
        if num_classes <= 0:
            raise ValueError("num_classes phải lớn hơn 0.")

        self.num_classes = num_classes
        self.pretrained = pretrained

    def build(self) -> nn.Module:
        """Tạo đối tượng ResNet-18, đóng băng backbone và thay thế classification head.

        Returns:
            nn.Module: Mô hình PyTorch đã cấu hình sẵn sàng cho huấn luyện với output logits [B, 10].
        """
        if self.pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        model = models.resnet18(weights=weights)

        # Freeze toàn bộ trọng số backbone hiện có của ResNet-18.
        for parameter in model.parameters():
            parameter.requires_grad = False

        # Lấy số lượng input của lớp fc cũ (512 chiều).
        input_features = model.fc.in_features

        # Thay classification head cho CIFAR-10.
        model.fc = nn.Linear(
            in_features=input_features,
            out_features=self.num_classes,
        )

        # Chỉ cho phép lớp fc mới được cập nhật khi training.
        for parameter in model.fc.parameters():
            parameter.requires_grad = True

        return model


class TransferModelBuilder_VGG16:
    """Xây dựng mô hình VGG-16 chuyển giao (Transfer Learning) cho CIFAR-10.

    - Nguồn tiền huấn luyện (Pre-trained Source): ImageNet-1k (1.000 danh mục tổng quát).
    - Bài toán đích (Target Task): Phân loại ảnh 10 đối tượng CIFAR-10.
    - Chiến lược chuyển giao (Transfer Policy): Đóng băng (freeze) 13 tầng tích chập
      (model.features) và 2 tầng Linear ẩn đầu tiên (model.classifier[0..5]),
      thay thế tầng Linear cuối cùng (model.classifier[6]: 4096 -> 10) và mở
      requires_grad = True cho tầng này.
    """

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
    ) -> None:
        """Khởi tạo cấu hình cho builder VGG-16.

        Args:
            num_classes (int): Số lượng lớp đầu ra của bài toán phân loại (mặc định: 10).
            pretrained (bool): Có tải trọng số pre-trained ImageNet hay không (mặc định: True).

        Raises:
            ValueError: Nếu num_classes <= 0.
        """
        if num_classes <= 0:
            raise ValueError("num_classes phải lớn hơn 0.")

        self.num_classes = num_classes
        self.pretrained = pretrained

    def build(self) -> nn.Module:
        """Tạo đối tượng VGG-16, đóng băng feature extractor và thay thế classification head.

        Returns:
            nn.Module: Mô hình PyTorch đã cấu hình sẵn sàng cho huấn luyện với output logits [B, 10].
        """
        if self.pretrained:
            weights = models.VGG16_Weights.DEFAULT
        else:
            weights = None

        model = models.vgg16(weights=weights)

        # Freeze toàn bộ trọng số hiện có của VGG-16.
        for parameter in model.parameters():
            parameter.requires_grad = False

        # Lấy số lượng input của lớp Linear cuối trong sequential classifier (vị trí index 6, 4096 chiều).
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
    """Xây dựng mô hình DenseNet-121 chuyển giao (Transfer Learning) cho CIFAR-10.

    - Nguồn tiền huấn luyện (Pre-trained Source): ImageNet-1k (1.000 danh mục tổng quát).
    - Bài toán đích (Target Task): Phân loại ảnh 10 đối tượng CIFAR-10.
    - Chiến lược chuyển giao (Transfer Policy): Đóng băng (freeze) toàn bộ 4 Dense Blocks
      và các Transition Layers (model.features), thay thế tầng phân loại
      (model.classifier: 1024 -> 10) và mở requires_grad = True cho tầng này.
    """

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
    ) -> None:
        """Khởi tạo cấu hình cho builder DenseNet-121.

        Args:
            num_classes (int): Số lượng lớp đầu ra của bài toán phân loại (mặc định: 10).
            pretrained (bool): Có tải trọng số pre-trained ImageNet hay không (mặc định: True).

        Raises:
            ValueError: Nếu num_classes <= 0.
        """
        if num_classes <= 0:
            raise ValueError("num_classes phải lớn hơn 0.")

        self.num_classes = num_classes
        self.pretrained = pretrained

    def build(self) -> nn.Module:
        """Tạo đối tượng DenseNet-121, đóng băng backbone và thay thế classifier head.

        Returns:
            nn.Module: Mô hình PyTorch đã cấu hình sẵn sàng cho huấn luyện với output logits [B, 10].
        """
        if self.pretrained:
            weights = models.DenseNet121_Weights.DEFAULT
        else:
            weights = None

        model = models.densenet121(weights=weights)

        # Freeze toàn bộ trọng số hiện có của DenseNet-121.
        for parameter in model.parameters():
            parameter.requires_grad = False

        # Lấy số lượng input của lớp classifier cũ (1024 chiều).
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
    """Factory builder hỗ trợ khởi tạo linh hoạt các kiến trúc: ResNet-18, VGG16, DenseNet121.

    Cung cấp giao diện thống nhất để khởi tạo mô hình học sâu theo tên kiến trúc,
    kiểm tra tính hợp lệ của tham số và điều phối đến Builder cụ thể.
    """

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
        """Khởi tạo Factory Builder cho mô hình được chỉ định.

        Args:
            model_name (str): Tên kiến trúc ('resnet18', 'vgg16', 'densenet121').
            num_classes (int): Số lượng lớp phân loại đầu ra (mặc định: 10).
            pretrained (bool): Có sử dụng trọng số ImageNet-1k tiền huấn luyện hay không.

        Raises:
            ValueError: Nếu num_classes <= 0 hoặc model_name không thuộc danh sách hỗ trợ.
        """
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
        """Khởi tạo và trả về mô hình PyTorch đã được cấu hình transfer learning hoàn chỉnh.

        Returns:
            nn.Module: Mô hình PyTorch với backbone đã freeze và classifier head mới.
        """
        return self._builder.build()

