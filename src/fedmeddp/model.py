from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ConvNeXt_Small_Weights,
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
    MobileNet_V2_Weights,
    ResNet18_Weights,
)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_model(backbone: str, num_classes: int, dropout: float = 0.2, pretrained: bool = True) -> nn.Module:
    backbone = backbone.lower()

    if backbone == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, dropout=dropout)

    if backbone == "mobilenet_v2":
        weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        try:
            model = models.mobilenet_v2(weights=weights, dropout=dropout)
        except Exception as exc:
            print(f"警告：MobileNetV2 预训练权重加载失败，已退回随机初始化。原因：{exc}")
            model = models.mobilenet_v2(weights=None, dropout=dropout)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        return model

    if backbone == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            model = models.resnet18(weights=weights)
        except Exception as exc:
            print(f"警告：ResNet18 预训练权重加载失败，已退回随机初始化。原因：{exc}")
            model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if backbone == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            model = models.efficientnet_b0(weights=weights, dropout=dropout)
        except Exception as exc:
            print(f"警告：EfficientNet-B0 预训练权重加载失败，已退回随机初始化。原因：{exc}")
            model = models.efficientnet_b0(weights=None, dropout=dropout)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if backbone == "efficientnet_b3":
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            model = models.efficientnet_b3(weights=weights, dropout=dropout)
        except Exception as exc:
            print(f"警告：EfficientNet-B3 预训练权重加载失败，已退回随机初始化。原因：{exc}")
            model = models.efficientnet_b3(weights=None, dropout=dropout)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if backbone == "efficientnet_b4":
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            model = models.efficientnet_b4(weights=weights, dropout=dropout)
        except Exception as exc:
            print(f"警告：EfficientNet-B4 预训练权重加载失败，已退回随机初始化。原因：{exc}")
            model = models.efficientnet_b4(weights=None, dropout=dropout)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if backbone == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            model = models.convnext_tiny(weights=weights)
        except Exception as exc:
            print(f"警告：ConvNeXt-Tiny 预训练权重加载失败，已退回随机初始化。原因：{exc}")
            model = models.convnext_tiny(weights=None)
        if dropout > 0:
            model.classifier[2] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(model.classifier[2].in_features, num_classes),
            )
        else:
            model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model

    if backbone == "convnext_small":
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            model = models.convnext_small(weights=weights)
        except Exception as exc:
            print(f"警告：ConvNeXt-Small 预训练权重加载失败，已退回随机初始化。原因：{exc}")
            model = models.convnext_small(weights=None)
        if dropout > 0:
            model.classifier[2] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(model.classifier[2].in_features, num_classes),
            )
        else:
            model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model

    raise ValueError(f"Unsupported backbone: {backbone}")
