"""Fine-tune MobileNetV3-Small on the pinned drowsiness dataset."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import load_config
from .data import prepare_datasets
from .engine import run_inference, train_one_epoch
from .metrics import classification_metrics, select_fbeta_threshold
from .model import build_model
from .utils import append_jsonl, resolve_device, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--device", help="Override config device, for example cpu or cuda")
    parser.add_argument("--epochs", type=int, help="Override configured epoch count")
    parser.add_argument("--batch-size", type=int, help="Override configured batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device:
        config.training.device = args.device
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size

    seed_everything(config.data.seed)
    device = resolve_device(config.training.device)
    prepared = prepare_datasets(config.data, config.model)
    pin_memory = device.type == "cuda"
    common_loader = {
        "num_workers": config.data.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": config.data.num_workers > 0,
    }
    generator = torch.Generator().manual_seed(config.data.seed)
    train_loader = DataLoader(
        prepared.train,
        batch_size=config.training.batch_size,
        shuffle=True,
        generator=generator,
        **common_loader,
    )
    validation_loader = DataLoader(
        prepared.validation,
        batch_size=config.training.batch_size,
        shuffle=False,
        **common_loader,
    )

    model = build_model(config.model).to(device)
    counts = Counter(prepared.train_labels)
    total = sum(counts.values())
    class_weights = torch.tensor(
        [total / (len(counts) * counts[label]) for label in range(config.model.num_classes)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.epochs)
    use_amp = bool(config.training.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "resolved-config.json", config.to_dict())
    write_json(output_dir / "split-sizes.json", prepared.sizes)
    history_path = output_dir / "history.jsonl"
    history_path.unlink(missing_ok=True)
    checkpoint_path = output_dir / "best.pt"

    best_score = -1.0
    stale_epochs = 0
    started = time.time()
    for epoch in range(1, config.training.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp=use_amp,
        )
        validation = run_inference(model, validation_loader, device, criterion)
        threshold, _ = select_fbeta_threshold(
            validation.labels, validation.drowsy_scores, config.decision.beta
        )
        metrics = classification_metrics(
            validation.labels,
            validation.drowsy_scores,
            threshold,
            config.decision.beta,
        )
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "validation_loss": validation.mean_loss,
            **metrics,
        }
        append_jsonl(history_path, record)
        print(record)

        score = float(metrics["drowsy_fbeta"])
        if score > best_score:
            best_score = score
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "threshold": threshold,
                    "validation_metrics": metrics,
                    "config": config.to_dict(),
                    "epoch": epoch,
                    "labels": ["Drowsy", "Non Drowsy"],
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= config.training.patience:
            print(f"Early stopping after epoch {epoch}")
            break

    write_json(
        output_dir / "training-summary.json",
        {
            "best_validation_fbeta": best_score,
            "checkpoint": str(checkpoint_path),
            "device": str(device),
            "torch_version": torch.__version__,
            "torch_hip_version": torch.version.hip,
            "elapsed_seconds": time.time() - started,
        },
    )
    print(f"Best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
