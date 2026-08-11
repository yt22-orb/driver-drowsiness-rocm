"""Typed project configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .constants import DATASET_ID, DATASET_REVISION


@dataclass(slots=True)
class DataConfig:
    dataset_id: str = DATASET_ID
    revision: str = DATASET_REVISION
    validation_fraction: float = 0.10
    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    cache_dir: str = "data/huggingface"
    dedup_manifest: str = "artifacts/audit/dedup-manifest.json"


@dataclass(slots=True)
class ModelConfig:
    architecture: str = "mobilenet_v3_small"
    pretrained: bool = True
    image_size: int = 224
    num_classes: int = 2


@dataclass(slots=True)
class TrainingConfig:
    output_dir: str = "artifacts/runs/mobilenet-v3-small"
    epochs: int = 15
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    amp: bool = True
    device: str = "auto"


@dataclass(slots=True)
class DecisionConfig:
    beta: float = 2.0
    window_seconds: float = 2.0
    minimum_valid_frames: int = 15
    clear_margin: float = 0.15
    clear_seconds: float = 1.0


@dataclass(slots=True)
class ExportConfig:
    opset: int = 18
    parity_samples: int = 100
    onnx_path: str = "artifacts/export/drowsiness-mobilenet-v3-small.onnx"
    metadata_path: str = "artifacts/export/model-metadata.json"
    web_model_dir: str = "web/public/models"


@dataclass(slots=True)
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = ProjectConfig(
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        decision=DecisionConfig(**raw.get("decision", {})),
        export=ExportConfig(**raw.get("export", {})),
    )
    validate_config(config)
    return config


def validate_config(config: ProjectConfig) -> None:
    if config.data.dataset_id != DATASET_ID:
        raise ValueError(f"dataset_id must remain pinned to {DATASET_ID!r}")
    if config.data.revision != DATASET_REVISION:
        raise ValueError(f"revision must remain pinned to {DATASET_REVISION!r}")
    if not 0 < config.data.validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    if config.data.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if config.model.architecture != "mobilenet_v3_small":
        raise ValueError("Only mobilenet_v3_small is supported by the MVP export contract")
    if config.model.num_classes != 2:
        raise ValueError("The dataset contract requires exactly two classes")
    if config.training.epochs < 1 or config.training.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if config.decision.minimum_valid_frames < 1:
        raise ValueError("minimum_valid_frames must be positive")
