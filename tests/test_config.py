from pathlib import Path

import pytest

from drowsiness.config import load_config
from drowsiness.constants import DATASET_REVISION, LABELS


def test_mvp_config_is_pinned() -> None:
    config = load_config(Path("configs/mvp.yaml"))
    assert config.data.revision == DATASET_REVISION
    assert config.model.num_classes == len(LABELS)
    assert config.model.image_size == 224
    assert config.model.architecture == "drowsiness_cnn_v1"
    assert config.model.camera_crop_fraction == 0.70
    assert config.data.pin_memory is False
    assert config.data.persistent_workers is False


def test_invalid_validation_fraction_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("data:\n  validation_fraction: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="validation_fraction"):
        load_config(path)
