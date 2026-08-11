import torch

from drowsiness.config import ModelConfig
from drowsiness.model import build_model, evaluation_transform


def test_model_output_contract() -> None:
    model = build_model(ModelConfig(pretrained=False), pretrained=False).eval()
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 224, 224))
    assert logits.shape == (1, 2)


def test_evaluation_transform_contract() -> None:
    image = torch.zeros(3, 227, 227, dtype=torch.uint8)
    transformed = evaluation_transform(224)(image)
    assert transformed.shape == (3, 224, 224)
    assert transformed.dtype == torch.float32
