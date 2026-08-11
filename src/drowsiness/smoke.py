"""CPU-only synthetic smoke test for model training and ONNX export."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from .config import ModelConfig
from .export import export_onnx_model, verify_parity
from .model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/smoke")
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)
    config = ModelConfig(pretrained=False, image_size=args.image_size)
    model = build_model(config, pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()
    images = torch.rand(2, 3, args.image_size, args.image_size)
    labels = torch.tensor([0, 1])
    before = criterion(model(images), labels)
    optimizer.zero_grad(set_to_none=True)
    before.backward()
    optimizer.step()
    after = criterion(model(images), labels)
    if not torch.isfinite(after):
        raise RuntimeError("Synthetic training produced a non-finite loss")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    onnx_path = export_onnx_model(model, output / "smoke.onnx", args.image_size, 18)
    parity = verify_parity(model, onnx_path, [images], max_samples=2)
    print(
        {
            "loss_before": float(before.detach()),
            "loss_after": float(after.detach()),
            "onnx": str(onnx_path),
            "parity": parity,
        }
    )


if __name__ == "__main__":
    main()
