import pytest
import torch

from drowsiness.config import ModelConfig
from drowsiness.model import (
    CHECKPOINT_FORMAT,
    DrowsinessCNN,
    architecture_metadata,
    build_model,
    evaluation_transform,
    validate_checkpoint,
)


def test_model_output_contract() -> None:
    model = build_model(ModelConfig()).eval()
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 224, 224))
    assert logits.shape == (1, 2)
    assert isinstance(model, DrowsinessCNN)


def test_model_is_repository_defined_and_compact() -> None:
    model = build_model(ModelConfig())
    details = architecture_metadata(model)
    assert details["name"] == "drowsiness_cnn_v1"
    assert 1_000_000 < details["parameterCount"] < 2_000_000


def test_external_or_legacy_checkpoint_is_rejected() -> None:
    with pytest.raises(ValueError, match="repository checkpoint"):
        validate_checkpoint({"model_state_dict": {}}, ModelConfig())
    validate_checkpoint(
        {
            "checkpoint_format": CHECKPOINT_FORMAT,
            "architecture": "drowsiness_cnn_v1",
            "weights_origin": "trained_from_scratch",
        },
        ModelConfig(),
    )


def test_evaluation_transform_contract() -> None:
    image = torch.zeros(3, 227, 227, dtype=torch.uint8)
    transformed = evaluation_transform(224)(image)
    assert transformed.shape == (3, 224, 224)
    assert transformed.dtype == torch.float32
    assert torch.all(transformed == -1)
