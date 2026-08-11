"""Hugging Face dataset loading, validation, splitting, and PyTorch adapters."""

from __future__ import annotations

import json
import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import ClassLabel, Dataset, DatasetDict, load_dataset
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

from .config import DataConfig, ModelConfig
from .constants import LABELS
from .model import evaluation_transform, training_transform


class ImageClassificationDataset(TorchDataset):
    def __init__(self, dataset: Dataset, transform: Callable[[Image.Image], Any]) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        row = self.dataset[index]
        image = row["image"]
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL image at index {index}, got {type(image).__name__}")
        return self.transform(image.convert("RGB")), int(row["label"])


@dataclass(slots=True)
class PreparedDatasets:
    train: ImageClassificationDataset
    validation: ImageClassificationDataset
    test: ImageClassificationDataset
    train_labels: list[int]
    sizes: dict[str, int]


def load_raw_dataset(config: DataConfig) -> DatasetDict:
    raw = load_dataset(
        config.dataset_id,
        revision=config.revision,
        cache_dir=config.cache_dir,
    )
    if not isinstance(raw, DatasetDict):
        raise TypeError("Expected Hugging Face DatasetDict")
    validate_dataset_contract(raw)
    return raw


def validate_dataset_contract(raw: DatasetDict) -> None:
    if set(raw) != {"train", "test"}:
        raise ValueError(f"Expected train/test splits, found {sorted(raw)}")
    for split_name in ("train", "test"):
        split = raw[split_name]
        if set(split.column_names) != {"image", "label"}:
            raise ValueError(f"Unexpected columns in {split_name}: {split.column_names}")
        label_feature = split.features["label"]
        if not isinstance(label_feature, ClassLabel) or tuple(label_feature.names) != LABELS:
            actual_names = getattr(label_feature, "names", None)
            raise ValueError(f"Unexpected label contract in {split_name}: {actual_names}")


def _load_train_drop_indices(path: str | Path, revision: str) -> set[int]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        warnings.warn(
            f"Dedup manifest {manifest_path} is missing; run the full audit before final training.",
            stacklevel=2,
        )
        return set()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("dataset_revision") != revision:
        raise ValueError("Dedup manifest revision does not match the pinned dataset revision")
    return {int(index) for index in payload.get("train_drop_indices", [])}


def prepare_datasets(data_config: DataConfig, model_config: ModelConfig) -> PreparedDatasets:
    raw = load_raw_dataset(data_config)
    train_drop = _load_train_drop_indices(data_config.dedup_manifest, data_config.revision)
    train_raw = raw["train"]
    if train_drop:
        keep = [index for index in range(len(train_raw)) if index not in train_drop]
        train_raw = train_raw.select(keep)

    split = train_raw.train_test_split(
        test_size=data_config.validation_fraction,
        seed=data_config.seed,
        stratify_by_column="label",
    )
    train_labels = [int(value) for value in split["train"]["label"]]
    train_transform = training_transform(model_config.image_size)
    eval_transform = evaluation_transform(model_config.image_size)
    return PreparedDatasets(
        train=ImageClassificationDataset(split["train"], train_transform),
        validation=ImageClassificationDataset(split["test"], eval_transform),
        test=ImageClassificationDataset(raw["test"], eval_transform),
        train_labels=train_labels,
        sizes={
            "train": len(split["train"]),
            "validation": len(split["test"]),
            "test": len(raw["test"]),
        },
    )


def class_counts(labels: list[int]) -> dict[int, int]:
    counts = Counter(labels)
    return {label: int(counts.get(label, 0)) for label in range(len(LABELS))}
