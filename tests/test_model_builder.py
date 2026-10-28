import pytest
import torch
from torch import nn

from src.model_builder import (
    AbstractModelBuilder,
    DenseNet121Builder,
    InvalidModelConfigurationError,
    ModelArchitecture,
    ModelBuilderError,
    ModelBuilderFactory,
    ModelNotBuiltError,
    ResNet18Builder,
    ResNetAdaptationStrategy,
    UnsupportedArchitectureError,
    VGG16Builder,
    resolve_device,
)


@pytest.mark.parametrize(
    ("architecture", "builder_type"),
    [
        (ModelArchitecture.RESNET18, ResNet18Builder),
        (ModelArchitecture.VGG16, VGG16Builder),
        (ModelArchitecture.DENSENET121, DenseNet121Builder),
    ],
)
def test_factory_creates_polymorphic_builders(architecture, builder_type):
    builder = ModelBuilderFactory.create(
        architecture=architecture,
        num_classes=10,
        pretrained=False,
        freeze_backbone=True,
    )

    assert isinstance(builder, AbstractModelBuilder)
    assert isinstance(builder, builder_type)

    model = builder.build()

    assert isinstance(model, nn.Module)
    assert builder.model is model


@pytest.mark.parametrize(
    "architecture",
    [
        ModelArchitecture.RESNET18,
        ModelArchitecture.VGG16,
        ModelArchitecture.DENSENET121,
    ],
)
def test_built_model_exposes_ten_class_logits(architecture):
    builder = ModelBuilderFactory.create(
        architecture=architecture,
        num_classes=10,
        pretrained=False,
        freeze_backbone=False,
    )
    model = builder.build()

    with torch.no_grad():
        output = model(torch.randn(1, 3, 224, 224))

    assert output.shape == (1, 10)


def test_freezing_keeps_only_classifier_trainable_for_resnet():
    builder = ResNet18Builder(
        num_classes=10,
        pretrained=False,
        freeze_backbone=True,
    )
    model = builder.build()

    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("fc.")
    )
    assert all(parameter.requires_grad for parameter in model.fc.parameters())

    builder.unfreeze_feature_extractor()

    assert all(parameter.requires_grad for parameter in model.parameters())


def test_classifier_parameters_are_limited_to_vgg_classifier():
    builder = VGG16Builder(
        num_classes=10,
        pretrained=False,
        freeze_backbone=True,
    )
    model = builder.build()
    classifier_parameters = set(builder.classifier_parameters())

    assert classifier_parameters
    assert classifier_parameters == set(model.classifier.parameters())
    assert all(
        parameter.requires_grad
        for parameter in classifier_parameters
    )


def test_operations_requiring_a_model_fail_before_build():
    builder = DenseNet121Builder(pretrained=False)

    with pytest.raises(ModelNotBuiltError, match="Call build"):
        builder.freeze_feature_extractor()

    with pytest.raises(ModelNotBuiltError, match="Call build"):
        builder.replace_classifier()


@pytest.mark.parametrize("num_classes", [0, -1, True, 10.0, "10"])
def test_invalid_num_classes_are_rejected(num_classes):
    with pytest.raises(InvalidModelConfigurationError):
        ResNet18Builder(num_classes=num_classes, pretrained=False)


@pytest.mark.parametrize("architecture", ["unknown", "", None, 123])
def test_factory_rejects_unknown_architecture(architecture):
    with pytest.raises(UnsupportedArchitectureError):
        ModelBuilderFactory.create(architecture, pretrained=False)


def test_build_wraps_dependency_failure_with_model_builder_error(monkeypatch):
    builder = ResNet18Builder(pretrained=False)

    def failing_backbone():
        raise RuntimeError("backbone unavailable")

    monkeypatch.setattr(builder, "_create_backbone", failing_backbone)

    with pytest.raises(ModelBuilderError, match="backbone unavailable"):
        builder.build()


def test_strategy_rejects_an_incompatible_model():
    strategy = ResNetAdaptationStrategy()
    model = VGG16Builder(pretrained=False).build()

    with pytest.raises(TypeError, match="requires a ResNet"):
        strategy.replace_classifier(model, num_classes=10)


def test_resolve_device_auto_returns_available_device():
    device = resolve_device("auto")

    assert device.type in {"cpu", "cuda"}
    assert device.type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_resolve_device_rejects_unavailable_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(InvalidModelConfigurationError, match="not available"):
        resolve_device("cuda")
