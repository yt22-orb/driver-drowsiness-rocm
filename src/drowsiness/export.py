"""Export a trained checkpoint to ONNX and validate PyTorch/ONNX Runtime parity."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

from .config import ProjectConfig, load_config
from .constants import IMAGE_NET_MEAN, IMAGE_NET_STD, LABELS
from .data import prepare_datasets
from .model import build_model
from .utils import write_json


def export_onnx_model(
    model: torch.nn.Module,
    destination: str | Path,
    image_size: int,
    opset: int,
) -> Path:
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    sample = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    torch.onnx.export(
        model,
        sample,
        output,
        input_names=["images"],
        output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(output)
    onnx.checker.check_model(onnx_model)
    return output


def verify_parity(
    model: torch.nn.Module,
    onnx_path: str | Path,
    batches: Iterable[torch.Tensor],
    max_samples: int,
) -> dict[str, object]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model = model.cpu().eval()
    maximum_difference = 0.0
    prediction_mismatches = 0
    compared = 0
    with torch.inference_mode():
        for images in batches:
            for image in images:
                single = image.unsqueeze(0)
                pytorch_logits = model(single).numpy()
                onnx_logits = session.run(["logits"], {"images": single.numpy()})[0]
                maximum_difference = max(
                    maximum_difference,
                    float(np.max(np.abs(pytorch_logits - onnx_logits))),
                )
                prediction_mismatches += int(
                    np.sum(np.argmax(pytorch_logits, axis=1) != np.argmax(onnx_logits, axis=1))
                )
                compared += 1
                if compared >= max_samples:
                    break
            if compared >= max_samples:
                break
    if maximum_difference >= 1e-4 or prediction_mismatches:
        raise RuntimeError(
            "ONNX parity failed: "
            f"max_abs_diff={maximum_difference}, mismatches={prediction_mismatches}"
        )
    return {
        "samples": compared,
        "maximum_absolute_logit_difference": maximum_difference,
        "prediction_mismatches": prediction_mismatches,
    }


def metadata_payload(
    config: ProjectConfig,
    checkpoint: dict[str, object],
    model_filename: str,
    metrics: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "modelVersion": "0.1.0",
        "modelFile": model_filename,
        "architecture": config.model.architecture,
        "dataset": {
            "id": config.data.dataset_id,
            "revision": config.data.revision,
            "license": None,
        },
        "labels": list(LABELS),
        "drowsyLabelIndex": 0,
        "input": {
            "name": "images",
            "layout": "NCHW",
            "dtype": "float32",
            "width": config.model.image_size,
            "height": config.model.image_size,
            "mean": list(IMAGE_NET_MEAN),
            "std": list(IMAGE_NET_STD),
        },
        "output": {"name": "logits", "shape": [1, 2]},
        "decision": {
            "drowsyThreshold": float(checkpoint["threshold"]),
            "beta": config.decision.beta,
            "windowSeconds": config.decision.window_seconds,
            "minimumValidFrames": config.decision.minimum_valid_frames,
            "clearMargin": config.decision.clear_margin,
            "clearSeconds": config.decision.clear_seconds,
        },
        "testMetrics": metrics,
        "warning": "Experimental thesis prototype; not a certified automotive safety system.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--metrics", default="artifacts/evaluation/test-metrics.json")
    parser.add_argument("--skip-test-parity", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint_path = Path(args.checkpoint or Path(config.training.output_dir) / "best.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(config.model, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    onnx_path = export_onnx_model(
        model, config.export.onnx_path, config.model.image_size, config.export.opset
    )

    parity: dict[str, object] | None = None
    if not args.skip_test_parity:
        prepared = prepare_datasets(config.data, config.model)
        loader = DataLoader(prepared.test, batch_size=8, shuffle=False, num_workers=0)
        batches = (images for images, _ in loader)
        parity = verify_parity(model, onnx_path, batches, config.export.parity_samples)

    metrics_path = Path(args.metrics)
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
    )
    metadata = metadata_payload(config, checkpoint, onnx_path.name, metrics)
    metadata["onnxParity"] = parity
    metadata_path = write_json(config.export.metadata_path, metadata)

    web_dir = Path(config.export.web_model_dir)
    web_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, web_dir / onnx_path.name)
    shutil.copy2(metadata_path, web_dir / "model-metadata.json")
    print(f"ONNX: {onnx_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Browser assets: {web_dir}")


if __name__ == "__main__":
    main()
