"""MobileNetV3-Small model and preprocessing contract."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms import v2

from .config import ModelConfig
from .constants import IMAGE_NET_MEAN, IMAGE_NET_STD


def build_model(config: ModelConfig, *, pretrained: bool | None = None) -> nn.Module:
    use_pretrained = config.pretrained if pretrained is None else pretrained
    weights = MobileNet_V3_Small_Weights.DEFAULT if use_pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, config.num_classes)
    return model


def training_transform(image_size: int) -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((image_size, image_size), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08, hue=0.02),
            v2.RandomAffine(degrees=5, translate=(0.03, 0.03)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
        ]
    )


def evaluation_transform(image_size: int) -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((image_size, image_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
        ]
    )
