"""Reusable training and inference loops."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass(slots=True)
class InferenceResult:
    labels: np.ndarray
    drowsy_scores: np.ndarray
    mean_loss: float


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    use_amp: bool,
) -> float:
    model.train()
    running_loss = 0.0
    sample_count = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(labels.shape[0])
        running_loss += float(loss.detach()) * batch_size
        sample_count += batch_size
    return running_loss / max(sample_count, 1)


@torch.inference_mode()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    *,
    use_amp: bool = False,
) -> InferenceResult:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    running_loss = 0.0
    sample_count = 0
    for images, labels in tqdm(loader, desc="evaluate", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            batch_loss = criterion(logits, labels) if criterion is not None else None
        probabilities = torch.softmax(logits.float(), dim=1)
        batch_size = int(labels.shape[0])
        if batch_loss is not None:
            running_loss += float(batch_loss) * batch_size
        sample_count += batch_size
        all_labels.append(labels.cpu().numpy())
        all_scores.append(probabilities[:, 0].cpu().numpy())
    return InferenceResult(
        labels=np.concatenate(all_labels),
        drowsy_scores=np.concatenate(all_scores),
        mean_loss=running_loss / max(sample_count, 1),
    )
