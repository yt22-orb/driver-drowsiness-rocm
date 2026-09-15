"""Repository-owned drowsiness CNN and its RGB preprocessing contract."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.transforms import v2

from .config import ModelConfig
from .constants import INPUT_MEAN, INPUT_STD

CHECKPOINT_FORMAT = "drowsiness-cnn-v1"


class ConvNormActivation(nn.Sequential):
    """Convolution followed by batch normalization and SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    """Two standard convolutions with a learned projection when shape changes."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvNormActivation(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride == 1 and in_channels == out_channels:
            self.shortcut: nn.Module = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv2(self.conv1(inputs)) + self.shortcut(inputs))


class DrowsinessCNN(nn.Module):
    """Compact binary image classifier designed and initialized in this repository.

    The network uses only ordinary convolutions, residual connections, global
    average pooling, and a two-logit classification head. No external backbone
    or downloaded parameters are involved.
    """

    def __init__(self, *, num_classes: int = 2, dropout: float = 0.25) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormActivation(3, 24, kernel_size=5, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.features = nn.Sequential(
            ResidualBlock(24, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 96, stride=2),
            ResidualBlock(96, 96),
            ResidualBlock(96, 160, stride=2),
            ResidualBlock(160, 160),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(160, num_classes)
        )
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(self.stem(inputs))))


def build_model(config: ModelConfig) -> DrowsinessCNN:
    """Create a new randomly initialized repository-owned CNN."""

    if config.architecture != CHECKPOINT_FORMAT.replace("-", "_"):
        raise ValueError(f"Unsupported architecture: {config.architecture}")
    return DrowsinessCNN(num_classes=config.num_classes, dropout=config.dropout)


def architecture_metadata(model: nn.Module) -> dict[str, object]:
    """Return a portable description of the architecture saved with exports."""

    return {
        "name": "drowsiness_cnn_v1",
        "family": "repository-defined residual CNN",
        "parameterCount": sum(parameter.numel() for parameter in model.parameters()),
        "initialization": "random Kaiming-normal convolution weights; no external weights",
        "stem": "5x5 stride-2 convolution, batch normalization, SiLU, 3x3 max pool",
        "stages": [
            {"channels": 32, "blocks": 2, "firstStride": 1},
            {"channels": 64, "blocks": 2, "firstStride": 2},
            {"channels": 96, "blocks": 2, "firstStride": 2},
            {"channels": 160, "blocks": 2, "firstStride": 2},
        ],
        "head": "global average pool, dropout, two-logit linear classifier",
    }


def validate_checkpoint(checkpoint: dict[str, object], config: ModelConfig) -> None:
    """Reject weights that were not produced by this repository's CNN trainer."""

    if checkpoint.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"Checkpoint is not a {CHECKPOINT_FORMAT!r} repository checkpoint; "
            "retrain the custom CNN"
        )
    if checkpoint.get("architecture") != config.architecture:
        raise ValueError("Checkpoint architecture does not match the configured architecture")
    if checkpoint.get("weights_origin") != "trained_from_scratch":
        raise ValueError("Checkpoint weights were not recorded as trained from scratch")


def training_transform(image_size: int) -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((image_size, image_size), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08, hue=0.02),
            v2.RandomAffine(degrees=5, translate=(0.03, 0.03)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=INPUT_MEAN, std=INPUT_STD),
        ]
    )


def evaluation_transform(image_size: int) -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((image_size, image_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=INPUT_MEAN, std=INPUT_STD),
        ]
    )
