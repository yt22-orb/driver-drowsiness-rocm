"""Calibrate on validation, then evaluate once against the official test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data import prepare_datasets
from .engine import run_inference
from .metrics import classification_metrics, select_fbeta_threshold
from .model import build_model
from .utils import resolve_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--output", default="artifacts/evaluation/test-metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint_path = Path(args.checkpoint or Path(config.training.output_dir) / "best.pt")
    device = resolve_device(args.device or config.training.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(config.model, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    prepared = prepare_datasets(config.data, config.model)
    loader_args = {
        "batch_size": config.training.batch_size,
        "shuffle": False,
        "num_workers": config.data.num_workers,
        "pin_memory": config.data.pin_memory and device.type == "cuda",
        "persistent_workers": (config.data.persistent_workers and config.data.num_workers > 0),
    }
    validation_loader = DataLoader(prepared.validation, **loader_args)
    test_loader = DataLoader(
        prepared.test,
        **loader_args,
    )
    use_amp = bool(config.training.amp and device.type == "cuda")
    validation = run_inference(
        model,
        validation_loader,
        device,
        use_amp=use_amp,
    )
    threshold, _ = select_fbeta_threshold(
        validation.labels,
        validation.drowsy_scores,
        config.decision.beta,
    )
    validation_metrics = classification_metrics(
        validation.labels,
        validation.drowsy_scores,
        threshold,
        config.decision.beta,
    )
    result = run_inference(model, test_loader, device, use_amp=use_amp)
    metrics = classification_metrics(
        result.labels,
        result.drowsy_scores,
        threshold,
        config.decision.beta,
    )
    metrics.update(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "dataset_revision": config.data.revision,
            "device": str(device),
            "validation_metrics": validation_metrics,
        }
    )
    checkpoint["threshold"] = threshold
    checkpoint["threshold_calibrated"] = True
    checkpoint["validation_metrics"] = validation_metrics
    torch.save(checkpoint, checkpoint_path)
    write_json(args.output, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
