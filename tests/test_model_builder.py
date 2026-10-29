import pytest
import torch
from torch import nn

from src.model_builder import (
    TransferModelBuilder,
    TransferModelBuilder_DenseNet121,
    TransferModelBuilder_ResNet18,
    TransferModelBuilder_VGG16,
)


@pytest.mark.parametrize(
    ("builder_cls", "fc_name"),
    [
        (TransferModelBuilder_ResNet18, "fc"),
        (TransferModelBuilder_DenseNet121, "classifier"),
    ],
)
def test_direct_builders_output_shape_and_freezing(builder_cls, fc_name):
    builder = builder_cls(num_classes=10, pretrained=False)
    model = builder.build()

    assert isinstance(model, nn.Module)

    # Test forward pass output shape
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 10)

    # Verify classification head parameters are trainable
    head_module = getattr(model, fc_name)
    assert all(p.requires_grad for p in head_module.parameters())

    # Verify backbone parameters are frozen
    for name, param in model.named_parameters():
        if not name.startswith(f"{fc_name}."):
            assert not param.requires_grad, f"Parameter {name} should be frozen"


def test_vgg16_builder_output_shape_and_freezing():
    builder = TransferModelBuilder_VGG16(num_classes=10, pretrained=False)
    model = builder.build()

    assert isinstance(model, nn.Module)

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 10)

    # In VGG-16, classifier[6] is the final linear layer
    final_layer = model.classifier[6]
    assert all(p.requires_grad for p in final_layer.parameters())

    # Verify all other parameters are frozen
    for name, param in model.named_parameters():
        if not name.startswith("classifier.6."):
            assert not param.requires_grad, f"Parameter {name} should be frozen"


@pytest.mark.parametrize("model_name", ["resnet18", "vgg16", "densenet121", "ResNet-18", "VGG-16", "DenseNet-121"])
def test_factory_builder_supported_models(model_name):
    builder = TransferModelBuilder(model_name=model_name, num_classes=10, pretrained=False)
    model = builder.build()
    assert isinstance(model, nn.Module)

    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 10)


def test_invalid_num_classes():
    with pytest.raises(ValueError, match="num_classes phải lớn hơn 0"):
        TransferModelBuilder_ResNet18(num_classes=0)

    with pytest.raises(ValueError, match="num_classes phải lớn hơn 0"):
        TransferModelBuilder_VGG16(num_classes=-1)

    with pytest.raises(ValueError, match="num_classes phải lớn hơn 0"):
        TransferModelBuilder_DenseNet121(num_classes=0)

    with pytest.raises(ValueError, match="num_classes phải lớn hơn 0"):
        TransferModelBuilder(model_name="resnet18", num_classes=0)


def test_unknown_model_name():
    with pytest.raises(ValueError, match="không được hỗ trợ"):
        TransferModelBuilder(model_name="alexnet")
